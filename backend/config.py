import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

LOCAL_SQLITE_PATH = os.getenv("LOCAL_SQLITE_PATH", "")

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "/home/srini/PycharmProjects/medicine-genai-langgraph/chroma_db")

SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")


VECTOR_COLLECTION_NAME = "medical_docs"