#!/usr/bin/env python
# coding: utf-8

# In[6]:


# gemini_app.py
# Streamlit UI for Gemini PDF Q&A (RAG).

import os
import asyncio
import nest_asyncio
import streamlit as st
from typing import List

# 🔧 Fix for RuntimeError: no current event loop
nest_asyncio.apply()
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from new_brain import build_index_for_pdfs, clear_all_indexes, PERSIST_BASE_DIR


# ---------- Page / Secrets ----------
st.set_page_config(page_title="Gemini PDF Q&A Chatbot", page_icon="📄", layout="centered")
st.title("📄 Gemini PDF Q&A Chatbot")

# Read API key from Streamlit secrets (recommended)
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
elif not os.getenv("GOOGLE_API_KEY"):
    st.warning("Set your GOOGLE_API_KEY in .streamlit/secrets.toml or environment.")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Model",
        [
            "models/gemini-1.5-pro",     # robust reasoning
            "models/gemini-2.5-flash",   # fast & cheap
            "models/gemini-2.0-flash",   # older but compatible
        ],
        index=1,
        help="Pick the LLM for answering."
    )

    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100,
                           help="Character length of each chunk stored in the vector DB.")
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, step=50,
                              help="Characters that overlap between adjacent chunks.")
    top_k = st.slider("Top-K Retrieved Passages", 1, 10, 4)

    if st.button("🗑️ Clear All Saved Indexes"):
        removed = clear_all_indexes()
        st.success(f"Removed {removed} index folder(s) from `{PERSIST_BASE_DIR}`.")

st.write("Upload one or more PDFs and ask questions about their content. "
         "The app builds a fresh vector index for each upload.")

# ---------- File upload ----------
uploads = st.file_uploader(
    "Upload PDF(s) (limit set by your server; multiple allowed)",
    type=["pdf"],
    accept_multiple_files=True
)

# Persist the uploaded files to disk so PyPDFLoader can read them
def save_uploads(files) -> List[str]:
    saved_paths = []
    if not files:
        return saved_paths
    os.makedirs("uploads", exist_ok=True)
    for f in files:
        path = os.path.join("uploads", f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        saved_paths.append(path)
    return saved_paths

query = st.text_input("Ask a question about your PDF(s):", placeholder="e.g., What is austenite?")

if uploads and query:
    saved_paths = save_uploads(uploads)

    # ---------- Build retriever ----------
    with st.spinner("Indexing your PDF(s)…"):
        vectordb, persist_dir = build_index_for_pdfs(
            saved_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    # ---------- LLM + RetrievalQA ----------
    llm = ChatGoogleGenerativeAI(model=model_choice)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # ---------- Ask ----------
    with st.spinner("Thinking…"):
        result = qa_chain.invoke(query)

    # ---------- Display ----------
    st.subheader("Answer")
    st.write(result["result"])

    # Sources
    with st.expander("Show sources"):
        srcs = result.get("source_documents", [])
        if not srcs:
            st.write("No source documents returned.")
        else:
            for i, doc in enumerate(srcs, 1):
                meta = doc.metadata or {}
                file_name = meta.get("source", "unknown.pdf")
                page = meta.get("page", "N/A")
                st.markdown(f"**Source {i} — {file_name}, page {page}**")
                st.write(doc.page_content[:800] + ("…" if len(doc.page_content) > 800 else ""))

else:
    st.info("Upload at least one PDF and enter a question to begin.")


# In[ ]:




