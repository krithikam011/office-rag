"""
streamlit_app.py
-----------------
Web frontend for the Office Policy Assistant, built with Streamlit.

Run with:
    streamlit run streamlit_app.py

Features:
- Upload policy documents (PDF/txt) directly from the browser
- Build the vector store with one click (no need to run ingest.py separately)
- Chat interface with conversation history
- Shows which source document each answer was grounded in
"""

import os
import streamlit as st

from ingest import load_text_from_file, chunk_text, embed_text, DATA_DIR, STORE_PATH
from rag_engine import load_vector_store, retrieve_relevant_chunks, build_prompt, client, CHAT_MODEL

import json

st.set_page_config(page_title="Office Policy Assistant", page_icon="📄", layout="centered")

st.title("📄 Office Policy Assistant")
st.caption("Ask questions about your office policy documents — answered using RAG (Retrieval-Augmented Generation) with Gemini.")

os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Sidebar: document upload + index building ----------
with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Upload policy PDFs or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uf in uploaded_files:
            save_path = os.path.join(DATA_DIR, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to {DATA_DIR}/")

    st.header("2. Build the index")
    st.caption("Run this after uploading new documents.")
    if st.button("🔨 Build / Rebuild vector store", use_container_width=True):
        files = [
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if f.lower().endswith((".pdf", ".txt"))
        ]
        if not files:
            st.error(f"No documents found in {DATA_DIR}/. Upload some first.")
        else:
            progress = st.progress(0, text="Starting...")
            store = []
            for i, filepath in enumerate(files):
                progress.progress(
                    (i) / len(files),
                    text=f"Processing {os.path.basename(filepath)}...",
                )
                text = load_text_from_file(filepath)
                chunks = chunk_text(text)
                for chunk in chunks:
                    vector = embed_text(chunk)
                    store.append({
                        "text": chunk,
                        "embedding": vector,
                        "source": os.path.basename(filepath),
                    })
            with open(STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(store, f)
            progress.progress(1.0, text="Done!")
            st.success(f"Indexed {len(store)} chunks from {len(files)} document(s).")

    st.divider()
    existing_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".txt"))]
    if existing_files:
        st.caption("**Documents currently in data/:**")
        for f in existing_files:
            st.text(f"• {f}")

# ---------- Main: chat interface ----------
st.header("Ask a question")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, message, sources in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

query = st.chat_input("e.g. How many days of casual leave do I get?")

if query:
    st.session_state.chat_history.append(("user", query, None))
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        if not os.path.exists(STORE_PATH):
            answer = "No vector store found yet. Upload documents and click 'Build vector store' in the sidebar first."
            sources = []
        else:
            with st.spinner("Searching documents and generating answer..."):
                store = load_vector_store()
                relevant_chunks = retrieve_relevant_chunks(query, store)
                prompt = build_prompt(query, relevant_chunks)
                response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
                answer = response.text
                sources = sorted(set(c["source"] for c in relevant_chunks))

        st.write(answer)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

    st.session_state.chat_history.append(("assistant", answer, sources))
