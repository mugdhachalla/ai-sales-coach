from huggingface_hub import InferenceClient
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
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
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_API_KEY
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

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
You are an experienced retail salesperson speaking directly to a customer in a store.

Your goal is to explain the product in a natural, friendly, and persuasive way, as if you are helping a customer decide whether to buy it.

Use the product information below to create a short sales pitch that sounds conversational and human.

Guidelines:
- Speak as if you are talking directly to a customer.
- Keep the tone friendly, clear, and helpful.
- Focus on how the product benefits the customer.
- Avoid special characters like asterisks.
- Do not include headings or formatting.
- Keep the pitch around 7 to 10 sentences.

Product information:
{context}

Write the sales pitch now.
"""

    try:

        completion = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=[
                {"role": "system", "content": "You are a professional sales trainer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )

        return completion.choices[0].message.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        return str(e)
    
# -------------------------------
# Evaluate user answers
# -------------------------------
def evaluate_answers(answers):

    prompt = f"""
You are a strict sales trainer grading a trainee.

Product: Fitbit fitness tracker.

Evaluate the trainee's answers below.

Answers:
{answers}

Score the trainee from 0 to 10 using this rubric:

0-2 → Completely incorrect or irrelevant answers
3-4 → Very weak understanding of the product
5-6 → Basic understanding but lacks clarity or sales value
7-8 → Good understanding with reasonable explanation
9-10 → Excellent sales pitch with clear value and persuasion

Be strict when grading.

Return the result in this format:

Score: X/10

Strengths:
- ...

Weaknesses:
- ...

How to Improve:
- ...
"""

    try:

        completion = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=[
                {"role": "system", "content": "You are an expert sales trainer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )

        return completion.choices[0].message.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        return str(e)

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/upload_page")
def upload_page():
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
    session ["pitch"]=pitch

    return render_template("results.html", pitch=pitch)



@app.route("/practice-questions")
def practice_questions():
    return render_template("practice.html")


@app.route("/practice")
def practice():
    return redirect(url_for("practice_questions"))

@app.route("/evaluate", methods=["POST"])
def evaluate():

    answers = ""

    for i in range(1,6):
        answers += request.form.get(f"answer{i}") + "\n"

    feedback = evaluate_answers(answers)

    return render_template("evaluation.html", feedback=feedback)


@app.route("/pitch")
def pitch():

    pitch = session.get("pitch")

    if not pitch:
        return "No pitch available. Please upload a brochure first."

    return render_template("results.html", pitch=pitch)

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)