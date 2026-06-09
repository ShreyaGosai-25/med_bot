import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec

# ----------------------------
# 0. Load environment variables
# ----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "your_api_key_here"
PINECONE_REGION = os.getenv("PINECONE_ENV") or "us-east-1"  # from your dashboard
INDEX_NAME = "medical-chatbot"

# ----------------------------
# 1. PDF data path
# ----------------------------
DATA_PATH = r"C:\Users\aumsh\OneDrive\Documents\New folder\End_Medical_chatbot25\data"

# ----------------------------
# 2. Main function
# ----------------------------
def main():
    # 1️⃣ Load PDFs
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    if not documents:
        print("⚠️ No PDFs found in the data folder.")
        return

    # 2️⃣ Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # 3️⃣ Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # dimension=384
    )

    # ----------------------------
    # 4️⃣ Initialize Pinecone client
    # ----------------------------
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Check if index exists
    if INDEX_NAME not in pc.list_indexes():
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # match MiniLM embeddings
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_REGION
            )
        )
    else:
        print(f"Index '{INDEX_NAME}' already exists. Skipping creation.")

    # 5️⃣ Create index instance
    index = pc.Index(INDEX_NAME)

    # 6️⃣ Upload documents to Pinecone via Langchain
    LangchainPinecone.from_documents(
        docs,
        embeddings,
        index_name=INDEX_NAME,
        client=pc  # pass the Pinecone client
    )

    print("✅ PDFs successfully ingested into Pinecone")

# ----------------------------
# 3. Run
# ----------------------------
if __name__ == "__main__":
    main()
