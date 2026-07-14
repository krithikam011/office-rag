"""
ingest.py
---------
Step 1 of the RAG pipeline: turn office documents into a searchable
vector store.

What happens here (know this cold for interviews):
1. Load raw text from PDFs/txt files in ./data/
2. Split text into overlapping chunks (so we don't lose context at chunk edges)
3. Send each chunk to Gemini's embedding model -> get a vector (list of floats)
4. Save all chunks + their vectors to a local JSON file (our "vector store")

Run this once (and again whenever documents change):
    python ingest.py
"""

import os
import json
import glob
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Put it in a .env file (see .env.example).")

client = genai.Client(api_key=API_KEY)

DATA_DIR = "data"
STORE_PATH = "vector_store.json"
EMBED_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap so context isn't cut mid-sentence


def load_text_from_file(filepath: str) -> str:
    """Read raw text out of a PDF or plain text file."""
    if filepath.lower().endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return Path(filepath).read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    Overlap matters: if a policy rule spans a chunk boundary, without
    overlap we might retrieve half the rule and give a wrong answer.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_text(text: str) -> list[float]:
    """Call Gemini's embedding model to turn a chunk of text into a vector."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def build_vector_store():
    files = glob.glob(os.path.join(DATA_DIR, "*.pdf")) + glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not files:
        raise RuntimeError(f"No .pdf or .txt files found in ./{DATA_DIR}/. Add your office policy docs there.")

    store = []  # list of {"text": chunk, "embedding": [...], "source": filename}

    for filepath in files:
        print(f"Processing {filepath} ...")
        text = load_text_from_file(filepath)
        chunks = chunk_text(text)
        print(f"  -> {len(chunks)} chunks")

        for chunk in chunks:
            vector = embed_text(chunk)
            store.append({
                "text": chunk,
                "embedding": vector,
                "source": os.path.basename(filepath),
            })

    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f)

    print(f"\nDone. Stored {len(store)} chunks in {STORE_PATH}")


if __name__ == "__main__":
    build_vector_store()

