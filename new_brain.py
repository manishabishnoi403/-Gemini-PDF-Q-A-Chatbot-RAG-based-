import os
import shutil
import uuid
import asyncio
import nest_asyncio
from typing import List, Tuple

nest_asyncio.apply()
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

PERSIST_BASE_DIR = "chroma_dbs"  # all indexes live under this folder


def _ensure_base_dir() -> None:
    if not os.path.exists(PERSIST_BASE_DIR):
        os.makedirs(PERSIST_BASE_DIR, exist_ok=True)


def safe_rmtree(path: str) -> None:
    """Delete a directory tree safely (ignore if missing)."""
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except Exception:
        
        pass


def load_and_split_pdfs(
    pdf_paths: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        # Tag each chunk with the original file name for nicer citations
        for d in chunks:
            d.metadata.setdefault("source", os.path.basename(path))
        docs.extend(chunks)
    return docs


def build_index_for_pdfs(
    pdf_paths: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[Chroma, str]:
    
    _ensure_base_dir()

    persist_dir = os.path.join(PERSIST_BASE_DIR, f"chroma_db_{uuid.uuid4().hex}")

    docs = load_and_split_pdfs(pdf_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    # Chroma persists automatically when using persist_directory; keep path for cleanup.
    return vectordb, persist_dir

def clear_all_indexes(keep: List[str] = None) -> int:
    
    _ensure_base_dir()
    keep = set(keep or [])
    removed = 0
    for name in os.listdir(PERSIST_BASE_DIR):
        path = os.path.join(PERSIST_BASE_DIR, name)
        if name.startswith("chroma_db_") and path not in keep:
            safe_rmtree(path)
            removed += 1
    return removed






