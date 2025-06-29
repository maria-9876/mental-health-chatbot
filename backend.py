import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load API key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

def load_llm():
    """
    Initialize the Groq LLM with LLaMA 3 model.
    """
    llm = ChatGroq(
        temperature=0.2,  # low = less hallucination
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL  # llama-3-70b-8192 recommended
    )
    return llm

def build_qa_chain(vector_db):
    """
    Create a RetrievalQA chain with a strict prompt template.
    """
    prompt_template = """
You are a compassionate and helpful mental health assistant.

Your task is to provide **empathetic**, **accurate**, and **safe** responses based ONLY on the context provided below.

🚫 Do NOT make up examples, stories, characters (like "Jack", "Tom", or "your father").
🚫 Never guess — if you don't have enough information, say so.
✅ If the user question is off-topic or not related to mental health, gently redirect them.

---

Context:
{context}

---

User Question:
{question}

---

Answer as a helpful and caring mental health assistant:
"""

    PROMPT = PromptTemplate(
        template=prompt_template.strip(),
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain

def get_response(query, vector_db):
    """
    Get a grounded response to the user's query from the QA chain.
    """
    chain = build_qa_chain(vector_db)
    result = chain.invoke({"query": query})  # ✅ Proper input format
    return result['result']
