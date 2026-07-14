"""
rag_engine.py
-------------
Step 2 of the RAG pipeline: given a user question, retrieve the most
relevant chunks and ask Gemini to answer using ONLY that context.

This is the "R" (retrieval) + "G" (generation) in RAG.
The "A" (augmented) part is: we stuff the retrieved chunks into the
prompt before sending it to the model, instead of relying on the
model's own memory.
"""

import os
import json
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

STORE_PATH = "vector_store.json"
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-flash-latest"
TOP_K = 3  # how many chunks to retrieve per question


def load_vector_store() -> list[dict]:
    if not os.path.exists(STORE_PATH):
        raise RuntimeError("vector_store.json not found. Run `python ingest.py` first.")
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_query(query: str) -> list[float]:
    result = client.models.embed_content(model=EMBED_MODEL, contents=query)
    return result.embeddings[0].values


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_relevant_chunks(query: str, store: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    This is the 'retrieval' step: find the top_k chunks whose embeddings
    are most similar (cosine similarity) to the query's embedding.
    For a small project like this, brute-force comparison is fine.
    (At scale, you'd use FAISS/Pinecone/a proper vector DB with an
    index instead of comparing against every vector one by one.)
    """
    query_vector = embed_query(query)

    scored = []
    for item in store:
        score = cosine_similarity(query_vector, item["embedding"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:top_k]]


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    return f"""You are an office policy assistant. Answer the question using ONLY
the context below. If the answer isn't in the context, say you don't
have that information rather than guessing.

Context:
{context}

Question: {query}

Answer:"""


def answer_question(query: str) -> str:
    store = load_vector_store()
    relevant_chunks = retrieve_relevant_chunks(query, store)
    prompt = build_prompt(query, relevant_chunks)

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )
    return response.text
