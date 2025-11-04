# BI GenAI Molecule Graph Demo

This project shows a small end to end GenAI assistant for a pharmaceutical setting.

It combines:

- An RDF knowledge graph for molecule target disease relationships.
- A vector database (Pinecone) for medical and clinical documents.
- Databricks SQL tables for structured trial metrics.
- A LangGraph based backend exposed via FastAPI.
- A React frontend acting as a simple medical assistant UI.
