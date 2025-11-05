import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_SQL_URL = os.getenv("DATABRICKS_SQL_URL", "")
DATABRICKS_WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")

SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")


def VECTOR_STORE_DIR():
    return None


def VECTOR_COLLECTION_NAME():
    return None