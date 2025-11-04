import uuid
from openai import OpenAI
from pinecone import Pinecone
from backend.config import OPENAI_API_KEY, PINECONE_API_KEY  # adjust import if needed

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "bi-medical-rag"
index = pc.Index(INDEX_NAME)

def embed_text(text: str) -> list[float]:
    resp = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return resp.data[0].embedding

def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    words = text.split()
    chunks = []
    current = []
    length = 0
    for w in words:
        if length + len(w) + 1 <= max_chars:
            current.append(w)
            length += len(w) + 1
        else:
            chunks.append(" ".join(current))
            current = [w]
            length = len(w)
    if current:
        chunks.append(" ".join(current))
    return chunks

def ingest_document(doc_id, drug_id, disease_id, text, source_type, url):
    chunks = chunk_text(text)
    vectors = []
    for i, chunk in enumerate(chunks):
        vec = embed_text(chunk)
        metadata = {
            "doc_id": doc_id,
            "drug_id": drug_id,
            "disease_id": disease_id,
            "section": "body",
            "source_type": source_type,
            "url": url,
            "chunk_index": i,
            "text": chunk,
        }
        vectors.append({"id": str(uuid.uuid4()), "values": vec, "metadata": metadata})
    index.upsert(vectors=vectors)
    print(f"Ingested {len(vectors)} chunks for {doc_id}")

if __name__ == "__main__":
    sample_text = (
        "Empagliflozin is a sodium glucose co transporter 2 inhibitor. "
        "In patients with type 2 diabetes and established cardiovascular disease "
        "it reduces the risk of cardiovascular death and hospitalisation for heart failure. "
        "The mechanism involves osmotic diuresis and natriuresis with downstream effects "
        "on preload, afterload and cardiac metabolism."
    )
    ingest_document(
        doc_id="CSR_HF123",
        drug_id="jardiance",
        disease_id="heart_failure",
        text=sample_text,
        source_type="csr",
        url="s3://example-bucket/csr_hf123.pdf",
    )
