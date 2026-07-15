# Office Policy Assistant (RAG)

A Retrieval-Augmented Generation assistant that answers questions about
office policy documents using Google Gemini.

## How it works (pipeline)

```
data/*.pdf, *.txt
      |
      v
  ingest.py  -->  chunks text, embeds each chunk, saves vector_store.json
      |
      v
  rag_engine.py  -->  embeds your question, finds top-3 most similar
      |               chunks (cosine similarity), builds a prompt with
      |               that context, sends it to Gemini
      v
  chat.py  -->  simple CLI to ask questions
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and paste your real Gemini API key
   (get one free at https://aistudio.google.com):
   ```
   cp .env.example .env
   ```

3. Put your real office policy documents (PDF or .txt) in `data/`.
   A sample doc is already there so you can test immediately.

4. Build the vector store (run this once, and again whenever docs change):
   ```
   python ingest.py
   ```

5. Chat with your documents:
   ```
   python chat.py
   ```

   Try asking: "How many leave days do I get?" or "What's the WFH policy?"

## Web frontend (Streamlit)

Instead of the command-line chat, you can run a browser-based UI:

```
streamlit run streamlit_app.py
```

This opens a local web page where you can:
- Upload policy PDFs/txt files directly (no manual file copying into `data/`)
- Click a button to build/rebuild the vector store (replaces running `ingest.py` manually)
- Chat with your documents with full conversation history
- See which source document each answer came from

This is the version worth screen-recording or screenshotting for your resume/portfolio, since it demonstrates a working end-to-end product, not just a script.

## Interview talking points

Be ready to explain these, since interviewers will probe design choices:

- **Why chunking?** LLMs and embedding models have limited context;
  splitting docs into ~500-char chunks with overlap keeps each chunk
  focused and searchable, while overlap prevents losing a rule that
  spans a chunk boundary.

- **Why embeddings + cosine similarity?** Embeddings turn text into
  vectors that capture meaning, not just keywords. Cosine similarity
  measures how close two vectors point in the same "direction" of
  meaning — this is how we find "similar" text even if the wording
  differs from the user's question.

- **Why "augmented" generation?** Instead of asking Gemini to answer
  from its own training data (which doesn't know your company's
  specific policies), we retrieve the actual relevant text and put it
  directly in the prompt. This grounds the answer in real documents
  and reduces hallucination.

- **What would you improve at scale?** Swap the brute-force JSON
  vector store for a proper vector database (FAISS, Pinecone,
  Chroma) with an approximate-nearest-neighbor index — brute-force
  cosine similarity over every chunk doesn't scale past a few
  thousand chunks.

- **Known limitation:** if the answer isn't in any retrieved chunk,
  the assistant should say so rather than guess — the prompt in
  `rag_engine.py` explicitly instructs this, but it's worth testing
  and mentioning as a known failure mode (retrieval quality directly
  caps answer quality).

## Project structure

```
rag-office-assistant/
├── data/                       # your policy PDFs/txt files
│   └── sample_office_policy.txt
├── ingest.py                   # builds the vector store
├── rag_engine.py                # retrieval + generation logic
├── chat.py                      # CLI chat interface
├── requirements.txt
├── .env.example
└── vector_store.json            # generated after running ingest.py
```
