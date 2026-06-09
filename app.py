from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from src.prompt import prompt_template

app = Flask(__name__)

print("Loading embeddings...")
embeddings = download_hugging_face_embeddings()
print("Embeddings loaded")

# Dummy docs
docs = [
    "Acne is a skin condition that occurs when hair follicles become clogged with oil and dead skin cells."
]

print("Creating FAISS index...")
vectorstore = FAISS.from_texts(docs, embeddings)
print("FAISS ready")

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

from langchain.llms import CTransformers

print("Loading Llama model...")

llm = CTransformers(
    model=r"C:\Users\aumsh\OneDrive\Documents\medical_chatbot\End_Medical_chatbot25\model\llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        "max_new_tokens": 256,
        "temperature": 0.7
    }
)

print("Llama loaded")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

print("QA chain ready")


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]

        print("=" * 50)
        print("USER:", msg)

        result = qa({"query": msg})

        answer = result["result"]

        print("ANSWER:", answer)
        print("=" * 50)

        return answer

    except Exception as e:
        print("ERROR:", str(e))
        return f"ERROR: {str(e)}"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )