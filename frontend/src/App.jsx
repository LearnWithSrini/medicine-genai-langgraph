import React, { useState } from "react";

const BACKEND_URL = "http://localhost:8000";

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [debugData, setDebugData] = useState(null);
  const [error, setError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim()) {
      return;
    }
    setLoading(true);
    setAnswer("");
    setError("");
    setDebugData(null);

    try {
      const resp = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const data = await resp.json();
      setAnswer(data.answer || "");
      setDebugData({
        graph: data.graph_results,
        sql: data.sql_results,
        rag: data.rag_results
      });
    } catch (e) {
      setError(`Request failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        maxWidth: 960,
        margin: "0 auto",
        padding: 24,
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
      }}
    >
      <h1>Boehringer style Medical Assistant</h1>
      <p style={{ color: "#555", fontSize: 14 }}>
        Ask about a drug, disease, target, or trial outcome. This is a demo using an RDF graph,
        Databricks SQL, and a vector store.
      </p>

      <form onSubmit={handleSubmit} style={{ marginTop: 16 }}>
        <textarea
          rows={3}
          style={{ width: "100%", padding: 8 }}
          placeholder="For example: Explain how Jardiance works in heart failure."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          type="submit"
          disabled={loading}
          style={{
            marginTop: 8,
            padding: "8px 16px",
            cursor: loading ? "wait" : "pointer"
          }}
        >
          {loading ? "Working" : "Ask"}
        </button>
      </form>

      {error && (
        <div
          style={{
            marginTop: 16,
            padding: 12,
            border: "1px solid #cc0000",
            borderRadius: 4,
            color: "#cc0000",
            fontSize: 14
          }}
        >
          {error}
        </div>
      )}

      {answer && (
        <div
          style={{
            marginTop: 24,
            padding: 16,
            border: "1px solid "#ccc",
            borderRadius: 4,
            backgroundColor: "#fafafa"
          }}
        >
          <h2>Answer</h2>
          <p style={{ whiteSpace: "pre-wrap" }}>{answer}</p>
        </div>
      )}

      {debugData && (
        <div style={{ marginTop: 24 }}>
          <details>
            <summary>Show retrieved graph, SQL, and document context</summary>
            <h3>Graph results</h3>
            <pre style={{ fontSize: 12 }}>
              {JSON.stringify(debugData.graph, null, 2)}
            </pre>
            <h3>SQL trial results</h3>
            <pre style={{ fontSize: 12 }}>
              {JSON.stringify(debugData.sql, null, 2)}
            </pre>
            <h3>Document snippets (RAG)</h3>
            <pre style={{ fontSize: 12 }}>
              {JSON.stringify(debugData.rag, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
}

export default App;
