# store_index.py

from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical"

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env")

def store_index():
    print("📄 Loading PDFs...")
    documents = load_pdf("data/")

    print("✂️ Splitting text...")
    text_chunks = text_split(documents)
    print(f"Total chunks created: {len(text_chunks)}")

    print("🔢 Loading embedding model...")
    embeddings = download_hugging_face_embeddings()

    print("⬆️ Uploading vectors to Pinecone...")
    PineconeVectorStore.from_texts(
        texts=[doc.page_content for doc in text_chunks],
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print("✅ Pinecone indexing completed successfully!")

if __name__ == "__main__":
    store_index()
