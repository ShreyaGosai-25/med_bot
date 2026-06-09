import os
from dotenv import load_dotenv
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from ctransformers import AutoModelForCausalLM

load_dotenv()

INDEX_NAME = "medical-chatbot"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENV"]
)

vectorstore = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = AutoModelForCausalLM.from_pretrained(
    "llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama"
)

def answer_question(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a medical assistant.
Use the context below to answer safely.
Do not diagnose. Add disclaimer.

Context:
{context}

Question:
{query}

Answer:
"""

    return llm(prompt)
