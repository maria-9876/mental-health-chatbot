import os
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
os.environ["HF_HOME"] = "D:/huggingface_cache"
# 🔧 Fix for Chroma sqlite3 issue on Streamlit Cloud
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass


import torch
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings  # ✅ Correct one

 # ✅ Required for Streamlit-safe Chroma config

# Constants
DATA_FOLDER = "data/"
CHROMA_DB_DIR = "chroma_db/"

# Safe settings to avoid server-mode crash
from chromadb.config import Settings

from chromadb.config import Settings as ChromaSettings

from chromadb.config import Settings as ChromaSettings

chroma_settings = ChromaSettings(
    persist_directory=CHROMA_DB_DIR,
    anonymized_telemetry=False,
    allow_reset=True
)





device = "cuda" if torch.cuda.is_available() else "cpu"

def get_embeddings():
    """
    Return HuggingFaceEmbeddings using model name string.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_db():
    """
    Load PDFs, split into chunks, embed, and store in Chroma vector DB.
    """
    print("🔄 Creating new vector DB from PDFs...")
    loader = DirectoryLoader(DATA_FOLDER, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DB_DIR,
    client_settings=chroma_settings  # 👈 Required!
)

    vector_db.persist()
    print("✅ Vector DB created and saved.")
    return vector_db

def load_vector_db():
    """
    Load existing Chroma DB or create if not exists.
    """
    if not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
        return create_vector_db()

    print("✅ Loading existing vector DB...")
    embeddings = get_embeddings()

    vector_db = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings,
    client_settings=chroma_settings  # 👈 Required!
)

    return vector_db
