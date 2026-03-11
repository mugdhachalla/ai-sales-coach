from huggingface_hub import InferenceClient
from flask import Flask, render_template, request, jsonify
import pdfplumber
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

client = InferenceClient(
    model="google/flan-t5-base",
    token=HF_API_KEY
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)


# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# -------------------------------
# Split brochure into chunks
# -------------------------------
def chunk_text(text, chunk_size=500):

    sentences = text.split(".")
    chunks = []
    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + "."

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# -------------------------------
# Create embeddings
# -------------------------------
def create_embeddings(chunks):

    embeddings = embedding_model.encode(chunks)

    return np.array(embeddings).astype("float32")


# -------------------------------
# Build FAISS vector index
# -------------------------------
def build_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# -------------------------------
# Retrieve relevant context
# -------------------------------
def retrieve_context(query, index, chunks, k=3):

    query_embedding = embedding_model.encode([query]).astype("float32")

    distances, indices = index.search(np.array(query_embedding), k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return " ".join(results)


# -------------------------------
# Generate pitch using HuggingFace
# -------------------------------
def generate_pitch(query, index, chunks):

    context = retrieve_context(query, index, chunks)

    context = context[:2000]

    prompt = f"""
You are a professional sales trainer.

Using the brochure information below create:

1. A 60 second sales pitch
2. Key selling points
3. Five practice questions for a salesman

Brochure Information:
{context}
"""

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=300
        )

        return response

    except Exception as e:
        return f"Error from HuggingFace: {str(e)}"


# -------------------------------
# Evaluate user answers
# -------------------------------
def generate_pitch(query, index, chunks):

    context = retrieve_context(query, index, chunks)

    context = context[:2000]

    prompt = f"""
You are a professional sales trainer.

Using the brochure information below create:

1. A 60 second sales pitch
2. Key selling points
3. Five practice questions for a salesman

Brochure Information:
{context}
"""

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=300
        )

        return response

    except Exception as e:
        return f"Error from HuggingFace: {str(e)}"


# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["brochure"]

    brochure_text = extract_text(file)

    chunks = chunk_text(brochure_text)

    embeddings = create_embeddings(chunks)

    index = build_index(embeddings)

    pitch = generate_pitch(
        "product features benefits pricing use cases target customer",
        index,
        chunks
    )

    return str(pitch)
@app.route("/evaluate", methods=["POST"])
def evaluate():

    answers = ""

    for i in range(1, 6):
        answers += request.form.get(f"answer{i}") + "\n"

    feedback = evaluate_answers(answers)

    return str(feedback)


# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)