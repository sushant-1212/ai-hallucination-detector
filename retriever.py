"""
retriever.py
-------------
Step 2 of the pipeline: turn a folder of trusted documents (or user-uploaded
files) into a searchable vector store, and retrieve the most relevant chunks
for a given claim.

Documents -> Chunks -> Embeddings -> FAISS vector store -> similarity search
"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from llm_provider import get_embeddings

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def _load_file(path: str):
    """Load a single file (.txt, .md, or .pdf) into LangChain Document objects."""
    if path.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def load_documents_from_folder(folder_path: str) -> list:
    """Load every supported file in a folder into a flat list of Documents."""
    docs = []
    if not os.path.isdir(folder_path):
        return docs
    for filename in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path) and filename.lower().endswith((".txt", ".md", ".pdf")):
            try:
                docs.extend(_load_file(full_path))
            except Exception as e:
                print(f"[retriever] Skipping {filename}: {e}")
    return docs


def build_vector_store(documents: list):
    """Chunk documents, embed them, and build a FAISS index. Returns None if no docs."""
    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def retrieve_evidence(vector_store, claim: str, k: int = 3) -> list[str]:
    """Return the top-k most relevant evidence chunks (as plain text) for a claim."""
    if vector_store is None:
        return []
    results = vector_store.similarity_search(claim, k=k)
    return [doc.page_content.strip() for doc in results]
