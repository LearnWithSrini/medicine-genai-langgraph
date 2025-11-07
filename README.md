Here is a full `README.md` you can drop into the root of your `medicine-genai-langgraph` repo.

You can adjust paths and names if your files are a little different, but this gives you a complete, runnable story from “git clone” to “ask a question in the UI”.

---

````markdown
# Medicine GenAI with LangGraph, Pinecone, Databricks SQL and RDF

End to end demo project for medical and pharmaceutical question answering.

It combines:

- **LangGraph** as the orchestration layer  
- **OpenAI** (or your internal LLM endpoint) as the model  
- **Pinecone** for vector search over large PDFs  
- **Databricks SQL** (or a local SQL engine) for structured citation and metadata  
- **An RDF graph store** for protein, compound and relation triples  
- **React** frontend for the user interface  

The goal is to show how you would build a realistic architecture for a pharma company such as Boehringer Ingelheim.

---

## 1. Repository layout

The project is organised as follows:

```text
medicine-genai-langgraph/
  backend/
    app/
      __init__.py
      config.py
      main.py             # FastAPI app entrypoint
      graph.py            # LangGraph definition
      nodes/
        __init__.py
        query_router.py
        vector_retriever.py
        rdf_retriever.py
        sql_retriever.py
        answer_synthesizer.py
      services/
        llm_client.py
        pinecone_client.py
        rdf_client.py
        sql_client.py
      models/
        schemas.py        # Pydantic models for requests / responses
    data/
      pdfs/               # Sample medical PDFs
      rdf/
        biomed_graph.ttl  # Sample RDF triples
      sql/
        citations.csv     # Sample structured metadata
    scripts/
      ingest_pdfs.py      # Load PDFs into Pinecone
      load_rdf.py         # Load RDF triples into a local store (or remote)
      seed_sql_demo.py    # Load CSV metadata into Databricks or local SQL
    requirements.txt
    .env.example
  frontend/
    package.json
    vite.config.* or webpack.config.*
    src/
      main.(tsx|jsx)
      App.(tsx|jsx)
      components/
        ChatPanel.(tsx|jsx)
        SourceViewer.(tsx|jsx)
        QueryHistory.(tsx|jsx)
  README.md
````

If your folder names differ slightly, adapt the commands below to your actual layout.

---

## 2. Prerequisites

You will need:

1. **Python** 3.10 or later
2. **Node.js** 18 or later and **npm** or **yarn**
3. A way to call an LLM

   * Easiest for local work: OpenAI API key
   * In a corporate environment: Databricks model serving or your own internal endpoint
4. Optional but recommended:

   * **Pinecone** account and index
   * **Databricks SQL Warehouse**
   * **RDF store** (can be local with `rdflib`, or remote such as GraphDB or Neptune)

If you do not have Pinecone or Databricks, you can still run the project in **demo mode** using:

* A local FAISS or Chroma store instead of Pinecone
* SQLite or DuckDB instead of Databricks SQL

You can wire these up behind the same service interfaces.

---

## 3. Environment variables

Create a copy of the example environment file.

From the `backend` folder:

```bash
cd backend
cp .env.example .env
```

Open `.env` and fill in the values that apply to you.

Example:

```dotenv
# LLM
OPENAI_API_KEY=sk-...
LLM_MODEL_NAME=gpt-4o-mini

# Pinecone (or leave blank and use local vector store)
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=biomed-pdfs

# Databricks SQL (for real warehouse)
DATABRICKS_SERVER_HOSTNAME=adb-xxxx.azuredatabricks.net
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xxxx
DATABRICKS_TOKEN=your_pat_or_bearer

# If you do not have Databricks, set this to true and the code can fall back to SQLite
USE_LOCAL_SQLITE=true
LOCAL_SQLITE_PATH=./data/sql/biomed.db

# RDF store
RDF_BACKEND=local              # "local" or "remote"
RDF_LOCAL_FILE=./data/rdf/biomed_graph.ttl
# If remote:
RDF_ENDPOINT_URL=
RDF_USERNAME=
RDF_PASSWORD=

# General app settings
APP_HOST=0.0.0.0
APP_PORT=8000
```

If tokens are disabled in your organisation, use `USE_LOCAL_SQLITE=true` for the demo on your laptop and skip the Databricks section.

---

## 4. Backend setup

### 4.1 Create a virtual environment

From the project root:

```bash
cd backend

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

### 4.2 Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` should include at least:

```text
fastapi
uvicorn[standard]
langgraph
openai
pinecone-client
pydantic
python-dotenv
pandas
rdflib
sqlalchemy
databricks-sql-connector
```

Add any libraries you used for FAISS / Chroma or your preferred vector store as well.

---

## 5. Load demo data

The `scripts` folder contains helper scripts to load demo content.

You can run them one by one.

> Make sure your `.env` file is configured before this step.
> Activate the virtual environment for all script commands.

### 5.1 Ingest PDFs into Pinecone (or local vector store)

From `backend`:

```bash
source .venv/bin/activate

python -m scripts.ingest_pdfs
```

The script should:

1. Load all PDFs from `data/pdfs/`
2. Chunk them into passages
3. Embed each chunk using your LLM embedding model
4. Upsert vectors into the Pinecone index (or into your local vector store)

Check the script for the exact embedding model name. Typical choices:

* `text-embedding-3-large` for OpenAI
* Or your internal embedding endpoint

### 5.2 Load RDF triples

From `backend`:

```bash
python -m scripts.load_rdf
```

The script should:

1. Read `data/rdf/biomed_graph.ttl`
2. Either:

   * Load it into an in memory `rdflib.Graph` and persist a local store, or
   * Push to your remote RDF endpoint if `RDF_BACKEND=remote`

Keep the ontology simple for the demo. For example:

* `Drug`
* `Target`
* `Indication`
* `ClinicalTrial`
* Relations such as `treats`, `inhibits`, `hasAdverseEvent`, `testedIn`

### 5.3 Seed SQL metadata

If you have Databricks:

```bash
python -m scripts.seed_sql_demo
```

The script should:

1. Read `data/sql/citations.csv`
2. Connect to Databricks using the connector and `.env`
3. Create a database such as `biomed_demo`
4. Create a table, for example `biomed_demo.citations`
5. Insert the data from the CSV

If `USE_LOCAL_SQLITE=true`, the same script can instead:

* Create `data/sql/biomed.db`
* Create a `citations` table
* Insert the CSV rows into SQLite using SQLAlchemy

---

## 6. LangGraph application

The LangGraph is typically defined in `backend/app/graph.py`.

The general pattern is:

* State is a dictionary with fields such as:

  * `question`
  * `intent`
  * `kg_entities`
  * `vector_context`
  * `sql_rows`
  * `final_answer`
* Nodes:

  1. `query_router`

     * Classifies question type
     * Sets `intent` to `pdf`, `kg`, `sql` or `mixed`
  2. `vector_retriever`

     * Runs Pinecone (or local) search
     * Writes passages into `vector_context`
  3. `rdf_retriever`

     * Runs SPARQL queries
     * Writes canonical URIs and triples into `kg_entities`
  4. `sql_retriever`

     * Runs SQL against Databricks or SQLite
     * Writes structured rows into `sql_rows`
  5. `answer_synthesizer`

     * Calls the LLM with all retrieved context
     * Produces a grounded `final_answer` plus a list of cited sources

`backend/app/main.py` then:

* Builds the graph using `graph.py`
* Starts a FastAPI app with endpoints such as:

  * `POST /api/chat` to run a single question through the graph
  * `GET /api/health` for a basic health check

---

## 7. Run the backend API

From the `backend` folder, with the environment active:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see something like:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Test in a browser or with `curl`:

```bash
curl http://localhost:8000/api/health
```

and

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the mechanism of action of drug X and what trials support it?"}'
```

If that returns a JSON answer, the backend is working.

---

## 8. Frontend setup

From the `frontend` folder:

```bash
cd ../frontend
npm install
```

Check `package.json` includes at least:

* `react` and `react-dom`
* Your bundler (`vite`, `webpack` or `create-react-app`)
* A HTTP client such as `axios` or `fetch` polyfill if needed

In `src/App.(tsx|jsx)` the main flow should be:

* Textarea or input for the question
* Button to send
* Call to `POST http://localhost:8000/api/chat` with JSON body
* Display:

  * Final answer text
  * Sources grouped by type: PDFs, RDF entities, SQL rows

Example API call in React (TypeScript):

```ts
const res = await fetch("http://localhost:8000/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question }),
});
const data = await res.json();
```

---

## 9. Run the frontend

From `frontend`:

```bash
npm run dev
# or
npm start
```

Then open the printed URL, for example:

```text
http://localhost:3000
```

You should see the chat UI.

Try queries such as:

* “Summarise the key efficacy findings for Drug A from the trial PDFs”
* “Which proteins does Compound B target, and what are known adverse events”
* “Show the citation details for the paper that reports the primary endpoint of Trial C”

The UI should:

1. Send the question to the backend
2. The backend runs the LangGraph workflow
3. The UI displays the final answer and a list of sources

---

## 10. Typical troubleshooting

**Backend cannot import LangGraph**

* Check `langgraph` is in `requirements.txt`
* Run `pip install langgraph` inside the virtual environment

**Pinecone errors**

* Check your `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT` and index name
* Make sure the index exists before you run `ingest_pdfs.py`

**Databricks connection errors**

* Verify `DATABRICKS_SERVER_HOSTNAME`, `DATABRICKS_HTTP_PATH` and `DATABRICKS_TOKEN`
* For local demos, set `USE_LOCAL_SQLITE=true` and re run `seed_sql_demo.py`

**RDF store errors**

* If you do not have a remote endpoint, use `RDF_BACKEND=local`
* Confirm the file path in `RDF_LOCAL_FILE` is correct

**CORS issues from frontend**

* In `backend/app/main.py`, enable CORS for `http://localhost:3000`
  using FastAPI’s `CORSMiddleware`

---

## 11. How this maps to an interview story

When you describe this project in an interview, focus on:

* **Architecture**
  One orchestration layer (LangGraph) that coordinates three data backends and an LLM.

* **Grounding and provenance**
  Each answer is built from explicit sources
  You can show which PDF pages, which graph triples, and which SQL rows supported the answer.

* **Extensibility**
  You can swap Pinecone for a different vector store
  You can swap OpenAI for Databricks model serving
  You can move from a local RDF file to a managed graph store.

---

## 12. Quick start checklist

1. Clone the repo
2. `cd backend`, create and activate `.venv`
3. `cp .env.example .env` and fill values
4. `pip install -r requirements.txt`
5. `python -m scripts.ingest_pdfs`
6. `python -m scripts.load_rdf`
7. `python -m scripts.seed_sql_demo`
8. `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
9. `cd ../frontend && npm install && npm run dev`
10. Open the frontend in the browser and ask a question

Once you can pass through those ten steps without errors, the project is “workable” end to end.
