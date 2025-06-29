import os
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
os.environ["HF_HOME"] = "D:/huggingface_cache"

import torch
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


# Constants
DATA_FOLDER = "data/"
CHROMA_DB_DIR = "chroma_db/"

# Detect device (not used anymore but kept in case you want to use it elsewhere)
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_embeddings():
    """
    Return HuggingFaceEmbeddings using model name string as expected by LangChain.
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
        persist_directory=CHROMA_DB_DIR
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
        embedding_function=embeddings
    )
    return vector_db
