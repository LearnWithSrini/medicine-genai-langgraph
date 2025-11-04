from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from openai import OpenAI
from pinecone import Pinecone

from backend.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    DATABRICKS_TOKEN,
    DATABRICKS_SQL_URL,
    DATABRICKS_WAREHOUSE_ID,
    SPARQL_ENDPOINT,
    FRONTEND_ORIGIN,
)

oai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("bi-medical-rag")


class AgentState(BaseModel):
    messages: List[Dict[str, Any]]
    query: str
    graph_results: Optional[List[Dict[str, Any]]] = None
    sql_results: Optional[List[Dict[str, Any]]] = None
    rag_results: Optional[List[Dict[str, Any]]] = None
    answer: Optional[str] = None


async def graph_retriever(state: AgentState) -> AgentState:
    q = state.query
    sparql = f"""
    PREFIX bi:   <http://example.com/bi#>
    PREFIX dis:  <http://example.com/bi/disease/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?drugLabel ?diseaseLabel ?targetLabel
    WHERE {{
      ?drug a bi:Drug ;
            rdfs:label ?drugLabel ;
            bi:indicatedFor ?disease ;
            bi:hasActiveIngredient ?ai .
      ?disease rdfs:label ?diseaseLabel .
      ?ai bi:inhibits ?target .
      ?target rdfs:label ?targetLabel .
      FILTER(CONTAINS(LCASE(?diseaseLabel), LCASE("{q}")))
    }}
    LIMIT 10
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            SPARQL_ENDPOINT,
            data={"query": sparql},
            headers={"Accept": "application/sparql-results+json"},
            timeout=20,
        )
    data = resp.json()
    rows: List[Dict[str, Any]] = []
    for b in data.get("results", {}).get("bindings", []):
        rows.append(
            {
                "drug": b["drugLabel"]["value"],
                "disease": b["diseaseLabel"]["value"],
                "target": b["targetLabel"]["value"],
            }
        )
    state.graph_results = rows
    return state


async def sql_retriever(state: AgentState) -> AgentState:
    sql = f"""
    SELECT d.brand_name,
           ds.name as disease_name,
           f.trial_id,
           f.endpoint_code,
           f.hr,
           f.ci_low,
           f.ci_high,
           f.p_value
    FROM bi.fact_trial_outcome f
    JOIN bi.dim_drug d     ON f.drug_id = d.drug_id
    JOIN bi.dim_disease ds ON f.disease_id = ds.disease_id
    WHERE LOWER(ds.name) LIKE LOWER('%{state.query}%')
    LIMIT 10
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{DATABRICKS_SQL_URL}/api/2.0/sql/statements",
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json={"statement": sql, "warehouse_id": DATABRICKS_WAREHOUSE_ID},
            timeout=30,
        )
    stmt = resp.json()
    rows: List[Dict[str, Any]] = []
    result = stmt.get("result", {})
    for row in result.get("data_array", []):
        rows.append(
            {
                "brand_name": row[0],
                "disease_name": row[1],
                "trial_id": row[2],
                "endpoint_code": row[3],
                "hr": row[4],
                "ci_low": row[5],
                "ci_high": row[6],
                "p_value": row[7],
            }
        )
    state.sql_results = rows
    return state


def embed_query(text: str) -> list[float]:
    emb = oai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return emb.data[0].embedding


async def rag_retriever(state: AgentState) -> AgentState:
    vec = embed_query(state.query)
    res = pinecone_index.query(
        vector=vec,
        top_k=5,
        include_values=False,
        include_metadata=True,
    )
    rows: List[Dict[str, Any]] = []
    for m in res.get("matches", []):
        meta = m["metadata"]
        rows.append(
            {
                "text": meta.get("text", ""),
                "source": meta.get("url", ""),
                "doc_id": meta.get("doc_id", ""),
            }
        )
    state.rag_results = rows
    return state


async def answer_node(state: AgentState) -> AgentState:
    system_prompt = (
        "You are a cautious medical assistant for internal pharmaceutical staff. "
        "Use only the provided graph results, SQL results and document snippets. "
        "Explain mechanisms and trial outcomes clearly. "
        "If data is missing, say that clearly."
    )

    parts: List[str] = []
    if state.graph_results:
        parts.append(
            "GRAPH RESULTS:\n"
            + "\n".join(
                f"- Drug {r['drug']} for {r['disease']} with target {r['target']}"
                for r in state.graph_results
            )
        )
    if state.sql_results:
        parts.append(
            "TRIAL DATA:\n"
            + "\n".join(
                f"- Trial {r['trial_id']} endpoint {r['endpoint_code']}, HR {r['hr']}, "
                f"CI [{r['ci_low']}, {r['ci_high']}], p={r['p_value']}"
                for r in state.sql_results
            )
        )
    if state.rag_results:
        parts.append(
            "DOCUMENT SNIPPETS:\n"
            + "\n".join(
                f"- Doc {r['doc_id']}: {r['text'][:400]}"
                for r in state.rag_results
            )
        )

    context = "\n\n".join(parts) if parts else "No extra external data was retrieved."

    resp = oai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User question: {state.query}\n\nContext:\n{context}",
            },
        ],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    state.answer = answer
    state.messages.append({"role": "assistant", "content": answer})
    return state


def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("graph_retriever", graph_retriever)
    workflow.add_node("sql_retriever", sql_retriever)
    workflow.add_node("rag_retriever", rag_retriever)
    workflow.add_node("answer", answer_node)

    workflow.set_entry_point("graph_retriever")
    workflow.add_edge("graph_retriever", "sql_retriever")
    workflow.add_edge("sql_retriever", "rag_retriever")
    workflow.add_edge("rag_retriever", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()


graph_app = build_workflow()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(req: ChatRequest):
    init_state = AgentState(
        messages=[{"role": "user", "content": req.query}],
        query=req.query,
    )
    result_state = await graph_app.ainvoke(init_state)
    return {
        "answer": result_state.answer,
        "graph_results": result_state.graph_results,
        "sql_results": result_state.sql_results,
        "rag_results": result_state.rag_results,
    }
