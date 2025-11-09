from typing import List, Dict, Any, Optional

import json
import sqlite3

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from openai import OpenAI

import chromadb
from chromadb.utils import embedding_functions

from backend.config import (
    OPENAI_API_KEY,
    SPARQL_ENDPOINT,
    LOCAL_SQLITE_PATH,
    FRONTEND_ORIGIN,
    VECTOR_STORE_DIR,
    VECTOR_COLLECTION_NAME,
)

# OpenAI client
oai_client = OpenAI(api_key=OPENAI_API_KEY)

# Local Chroma vector store
chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
chroma_embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-large",
)
vector_collection = chroma_client.get_or_create_collection(
    name=VECTOR_COLLECTION_NAME,
    embedding_function=chroma_embedding_fn
)


class AgentState(BaseModel):
    messages: List[Dict[str, Any]]
    query: str
    extracted_drug: Optional[str] = None  # NEW: normalized drug / formula name
    graph_results: Optional[List[Dict[str, Any]]] = None
    sql_results: Optional[List[Dict[str, Any]]] = None  # will hold citations
    rag_results: Optional[List[Dict[str, Any]]] = None
    answer: Optional[str] = None


# -------------------------
# Drug extractor node (LLM)
# -------------------------
async def extract_drug(state: AgentState) -> AgentState:
    """
    Use the LLM to pull out a single medicine / drug / formula name
    from the user question and store it in state.extracted_drug.
    """
    prompt = (
        "Extract the main medicine, drug, or chemical compound name from the question.\n"
        "Respond with ONLY the name and nothing else.\n"
        "If there is no clear drug name, respond with UNKNOWN.\n\n"
        f"Question: {state.query}"
    )

    resp = oai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You extract drug names from questions."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()

    # Normalise a bit
    if not raw or raw.upper() == "UNKNOWN":
        extracted = None
    else:
        extracted = raw.strip()

    state.extracted_drug = extracted
    return state


# -------------------------
# GraphDB retriever (RDF)
# -------------------------
async def graph_retriever(state: AgentState) -> AgentState:
    # Prefer extracted drug; fall back to full query if extraction failed
    q = (state.extracted_drug or state.query).strip()
    # very small escape to avoid breaking the SPARQL with quotes
    q_escaped = q.replace('"', '\\"')

    sparql = f"""
    PREFIX drug: <http://example.org/drug/>
    PREFIX dbo:  <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?drug ?label ?synonym ?moa ?indication
    WHERE {{
      ?drug a dbo:Drug ;
            rdfs:label ?label .
      OPTIONAL {{ ?drug dbo:synonym ?synonym . }}
      OPTIONAL {{ ?drug dbo:mechanismOfAction ?moa . }}
      OPTIONAL {{ ?drug dbo:indication ?indication . }}

      FILTER(
        CONTAINS(LCASE(?label), LCASE("{q_escaped}")) ||
        CONTAINS(LCASE(COALESCE(?synonym, "")), LCASE("{q_escaped}"))
      )
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
                "iri": b["drug"]["value"],
                "label": b.get("label", {}).get("value", ""),
                "synonym": b.get("synonym", {}).get("value", ""),
                "mechanism_of_action": b.get("moa", {}).get("value", ""),
                "indication": b.get("indication", {}).get("value", ""),
            }
        )

    state.graph_results = rows
    return state


# -------------------------
# SQLite citations retriever
# -------------------------
def _run_citations_query(drug_term: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(LOCAL_SQLITE_PATH)
    cur = conn.cursor()

    # If we somehow got an empty term, use a pattern that matches nothing
    term = (drug_term or "").strip().lower()
    if not term:
        term = "__no_match__"

    pattern = f"%{term}%"

    sql = """
    SELECT id, title, abstract, source, keywords
    FROM citations
    WHERE
        LOWER(title)    LIKE ? OR
        LOWER(abstract) LIKE ? OR
        LOWER(keywords) LIKE ?
    LIMIT 10;
    """

    cur.execute(sql, (pattern, pattern, pattern))
    rows = cur.fetchall()
    conn.close()

    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "id": r[0],
                "title": r[1],
                "abstract": r[2],
                "source": r[3],
                "keywords": r[4],
            }
        )
    return results


async def sql_retriever(state: AgentState) -> AgentState:
    # Prefer extracted drug; fall back to full query if needed
    term = (state.extracted_drug or state.query).strip()
    rows = _run_citations_query(term)
    state.sql_results = rows
    return state


# -------------------------
# Chroma RAG retriever
# -------------------------
async def rag_retriever(state: AgentState) -> AgentState:
    """
    Retrieve RAG context from a local Chroma collection.

    Uses the extracted drug / formula name as the query text.
    """
    query_text = (state.extracted_drug or state.query).strip()

    res = vector_collection.query(
        query_texts=[query_text],
        n_results=5,
    )

    rows: List[Dict[str, Any]] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    for i in range(len(ids)):
        meta = metas[i] or {}
        rows.append(
            {
                "text": docs[i],
                "source": meta.get("source", ""),
                "topic": meta.get("topic", ""),
                "chunk_index": meta.get("chunk_index", i),
                "doc_id": ids[i],
            }
        )

    state.rag_results = rows
    return state


# -------------------------
# Answer node
# -------------------------
async def answer_node(state: AgentState) -> AgentState:
    system_prompt = (
        "You are a cautious internal medical assistant. "
        "Use only the provided graph results (from an RDF knowledge graph), "
        "citations (from the SQLite 'citations' table), and document snippets (from a vector store). "
        "Explain mechanisms and clinical points clearly. "
        "If data is missing, say that clearly instead of guessing."
    )

    parts: List[str] = []

    if state.graph_results:
        parts.append(
            "GRAPH RESULTS:\n"
            + "\n".join(
                f"- Drug {r.get('label') or '[no label]'} "
                f"(synonym: {r.get('synonym') or 'n/a'}) "
                f"indication: {r.get('indication') or 'n/a'} "
                f"MOA: {r.get('mechanism_of_action') or 'n/a'}"
                for r in state.graph_results
            )
        )

    if state.sql_results:
        parts.append(
            "CITATIONS:\n"
            + "\n".join(
                f"- [{r['id']}] {r['title']} "
                f"(source: {r['source']}) "
                f"keywords: {r['keywords']}\n"
                f"  abstract: {r['abstract'][:300]}..."
                for r in state.sql_results
            )
        )

    if state.rag_results:
        parts.append(
            "DOCUMENT SNIPPETS:\n"
            + "\n".join(
                f"- Doc {r['doc_id']} (source: {r['source']}, topic: {r['topic']}): "
                f"{r['text'][:400]}"
                for r in state.rag_results
            )
        )

    context = "\n\n".join(parts) if parts else "No external data was retrieved."

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


# -------------------------
# LangGraph workflow
# -------------------------
def build_workflow():
    print("SPARQL_ENDPOINT =", SPARQL_ENDPOINT)
    print("SQLITE_DB_PATH  =", LOCAL_SQLITE_PATH)
    print("VECTOR_STORE_DIR =", VECTOR_STORE_DIR)
    print("VECTOR_COLLECTION_NAME =", VECTOR_COLLECTION_NAME)

    workflow = StateGraph(AgentState)

    # NEW: extraction node
    workflow.add_node("extract_drug", extract_drug)

    workflow.add_node("graph_retriever", graph_retriever)
    workflow.add_node("sql_retriever", sql_retriever)
    workflow.add_node("rag_retriever", rag_retriever)
    workflow.add_node("answer_step", answer_node)

    # NEW entry point and edges
    workflow.set_entry_point("extract_drug")
    workflow.add_edge("extract_drug", "graph_retriever")
    workflow.add_edge("graph_retriever", "sql_retriever")
    workflow.add_edge("sql_retriever", "rag_retriever")
    workflow.add_edge("rag_retriever", "answer_step")
    workflow.add_edge("answer_step", END)

    return workflow.compile()


graph_app = build_workflow()


# -------------------------
# FastAPI app
# -------------------------
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
    init_state = {
        "messages": [],
        "query": req.query,
    }

    result_state = await graph_app.ainvoke(init_state)

    return {
        "answer": result_state.get("answer"),
        "graph_results": result_state.get("graph_results"),
        "sql_results": result_state.get("sql_results"),   # citations
        "rag_results": result_state.get("rag_results"),
        "extracted_drug": result_state.get("extracted_drug"),
    }


@app.post("/chat_debug")
async def chat_debug(req: ChatRequest):
    init_state = {
        "messages": [],
        "query": req.query,
    }

    events_summary = []

    async for ev in graph_app.astream_events(init_state, version="v2"):
        if ev.get("event") == "on_end" and ev.get("metadata", {}).get("source") == "node":
            node_name = ev["metadata"]["node"]
            state = ev["data"]["state"]

            events_summary.append(
                {
                    "node": node_name,
                    "query": state.get("query"),
                    "extracted_drug": state.get("extracted_drug"),
                    "graph_results_len": len(state.get("graph_results") or []),
                    "sql_results_len": len(state.get("sql_results") or []),
                    "rag_results_len": len(state.get("rag_results") or []),
                    "answer": state.get("answer"),
                }
            )

    return JSONResponse(content={"events": events_summary})
