import os
import chromadb
from chromadb.utils import embedding_functions

from backend.config import (
    OPENAI_API_KEY
)


# 1. Define the SAME embedding function that was used during ingestion
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-large",  # 3072-dim
)

# 2. Connect to the same persistent DB
client = chromadb.PersistentClient(path="../chroma_db")

# 3. Get the EXISTING collection and attach the embedding function
#    Use get_collection(), not get_or_create_collection(), since it already exists
col = client.get_collection(
    name="medical_docs",
    embedding_function=openai_ef,
)

print("Count:", col.count())

# 4. Query using query_texts (it will use openai_ef)
res = col.query(
    query_texts=["paracetamol mechanism of action"],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

print(res)
