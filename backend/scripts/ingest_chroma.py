import os
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

# 1. Config
CHROMA_DIR = "/home/srini/PycharmProjects/medicine-genai-langgraph/chroma_db"
COLLECTION_NAME = "medical_docs"
PDF_DIR = "data/pdfs"
PDF_FILES = [
    "/home/srini/PycharmProjects/medicine-genai-langgraph/data/pdfs/paracetamol_mechanism.pdf",
    "/home/srini/PycharmProjects/medicine-genai-langgraph/data/pdfs/paracetamol_dosage.pdf",
]

from backend.config import (
    OPENAI_API_KEY
)

OPENAI_API_KEY = OPENAI_API_KEY
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment")

# 2. Simple text chunker
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# 3. Read PDF -> full text
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import os

def read_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        try:
            reader = PdfReader(path)
            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
            text = "\n".join(pages_text).strip()
            if text:
                return text
            # if we got an empty string, fall back to raw read
            print(f"[WARN] Empty text from PDF parser, falling back to plain read: {path}")
        except PdfReadError:
            print(f"[WARN] Invalid PDF, falling back to plain read: {path}")

    # Fallback: treat as plain UTF-8 text file
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    # 4. Chroma client, persistent
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 5. Embedding function using OpenAI
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-large",  # or your preferred embedding model
    )

    # 6. Get or create collection (append on top of existing data)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
    )

    all_ids = []
    all_docs = []
    all_metas = []

    for pdf_name in PDF_FILES:
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"WARNING: PDF not found: {pdf_path}")
            continue

        print(f"Reading {pdf_path}")
        full_text = read_document(pdf_path)
        chunks = chunk_text(full_text)

        print(f"  -> {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            doc_id = f"{pdf_name}_chunk_{idx}"
            all_ids.append(doc_id)
            all_docs.append(chunk)
            all_metas.append(
                {
                    "source": pdf_name,
                    "chunk_index": idx,
                    "topic": "paracetamol",
                }
            )

    if not all_docs:
        print("No documents to insert. Check PDF paths.")
        return

    print(f"Inserting {len(all_docs)} chunks into Chroma collection '{COLLECTION_NAME}'")

    collection.add(
        ids=all_ids,
        documents=all_docs,
        metadatas=all_metas,
    )

    print("Done.")

if __name__ == "__main__":
    main()
