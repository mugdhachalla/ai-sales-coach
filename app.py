from huggingface_hub import InferenceClient
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pdfplumber
import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()

MODE = (os.getenv("APP_MODE", "production") or "production").lower()
USE_FULL_RAG = MODE in {"local", "dev", "development"}

HF_API_KEY = os.getenv("HF_API_KEY")


embedding_model = None
faiss = None
np = None

if USE_FULL_RAG:
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[WARN] Full RAG disabled due to import/init issue: {e}")
        USE_FULL_RAG = False


client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_API_KEY
)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

print(f"[INFO] APP_MODE={MODE} | FULL_RAG={'enabled' if USE_FULL_RAG else 'disabled'}")


def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text



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


def create_embeddings(chunks):

    if not USE_FULL_RAG or embedding_model is None or np is None:
        raise RuntimeError("Full RAG mode is disabled. Embeddings are unavailable.")

    embeddings = embedding_model.encode(chunks)

    return np.array(embeddings).astype("float32")

def build_index(embeddings):

    if not USE_FULL_RAG or faiss is None:
        raise RuntimeError("Full RAG mode is disabled. FAISS index is unavailable.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def retrieve_context(query, index, chunks, k=3):

    if not USE_FULL_RAG or embedding_model is None or np is None:
        return retrieve_context_lightweight(query, chunks, k=k)

    if not chunks:
        return ""

    k = min(k, len(chunks))

    query_embedding = embedding_model.encode([query]).astype("float32")

    distances, indices = index.search(np.array(query_embedding), k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return " ".join(results)


def retrieve_context_lightweight(query, chunks, k=3):

    if not chunks:
        return ""

    k = min(k, len(chunks))

    query_terms = set(re.findall(r"\w+", query.lower()))
    scored_chunks = []

    for chunk in chunks:
        chunk_terms = set(re.findall(r"\w+", chunk.lower()))
        overlap_score = len(query_terms.intersection(chunk_terms))
        length_score = min(len(chunk) / 1000.0, 0.5)
        score = overlap_score + length_score
        scored_chunks.append((score, chunk))

    top_chunks = [chunk for _, chunk in sorted(scored_chunks, key=lambda x: x[0], reverse=True)[:k]]
    return " ".join(top_chunks)


def generate_pitch(context, industry="General"):

    context = context[:2000]

    prompt = f"""
        You are an experienced sales trainer and a Subject Matter Expert in the {industry} industry.

        You are preparing training material for a new salesperson.

        The salesperson needs to study the product and learn how to confidently explain, position, and sell it to customers within this industry context.

        Using the product information below, create clear and structured training material.

        Ensure your explanations reflect real world industry knowledge, customer expectations, and competitive dynamics in the {industry} space.

        The material should be easy to read and written like a trainer teaching a trainee how to sell the product.

        Write the training material using the following structure:

        PRODUCT OVERVIEW
        Explain what the product is, what it does, and how it fits within the {industry} industry. Provide context on where it stands in the market.

        TARGET CUSTOMER
        Describe the ideal customer profile in the {industry}. Include roles, business types, and typical pain points relevant to this industry.

        KEY CUSTOMER BENEFITS
        Explain the most important ways this product helps customers. Clearly connect benefits to real problems faced in the {industry}.

        IMPORTANT FEATURES
        Explain the most important product features and why they matter specifically for customers in this industry.

        HOW TO EXPLAIN THIS PRODUCT TO A CUSTOMER
        Describe how a salesperson should present this product in a conversation. Include positioning, storytelling, and value articulation tailored to the {industry}.

        COMMON CUSTOMER QUESTIONS
        List realistic questions customers in this industry are likely to ask about the product.

        CUSTOMER OBJECTIONS AND HOW TO HANDLE THEM
        List common objections customers may have in this industry. For each objection, provide a strong, practical response that a salesperson can use confidently.

        SALES TIPS
        Give practical, real world advice to help the salesperson present, differentiate, and close effectively in the {industry}.

        Guidelines:
        Write in clear paragraphs.
        Do not use asterisks or bullet symbols.
        Do not use markdown formatting.
        Keep the tone professional, practical, and instructional.
        Write as if a trainer is coaching a new salesperson to succeed in real customer conversations.
        Focus on clarity, confidence, and industry relevance.
        Ensure all sections are completed fully. Keep explanations concise but complete.

        Product information from the brochure:
        {context}
        """

    try:

        completion = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=[
                {"role": "system", "content": "You are a professional sales trainer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )

        return completion.choices[0].message.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        return str(e)

def evaluate_answers(answers, context):

    prompt = f"""
You are a strict sales trainer evaluating a trainee.

The trainee studied a product brochure and answered questions about the product.

Product information from the brochure:
{context}

Trainee answers:
{answers}

You must grade the trainee strictly.

Scoring Guidelines:

0–2 (Very Poor)
Answers are meaningless, extremely short, unrelated, or show no understanding of the product.
Examples: ".", "ok", "yes", "good", random words, or generic statements.

2–4 (Poor)
Answers show minimal effort but contain vague or generic statements that could apply to any product.
Little to no product knowledge is demonstrated.

4–6 (Average)
Answers show basic understanding but lack detail, clear explanation of benefits, or connection to the product brochure.

6–8 (Good)
Answers clearly demonstrate understanding of the product and explain features or benefits in a reasonably clear way.

8–10 (Excellent)
Answers show strong product knowledge, clearly explain value to customers, and demonstrate good sales thinking.

Important rules:

If answers are extremely short, meaningless, or contain only punctuation like "." or "...", the score MUST be between 0 and 2.

If answers are generic and do not reference the product, the score MUST NOT exceed 4.

Only give scores above 6 if the trainee clearly demonstrates understanding of the product and its benefits.

Output format:

Sales Readiness Score: X/10

Reasoning
Explain briefly why this score was given.

Strengths
Explain what the trainee did well.

Weaknesses
Explain what knowledge or skills are missing.

How to Improve
Give specific advice on how the trainee could improve their answers.

Write clear training feedback for a salesperson.

Do not use asterisks or bullet symbols.
"""

    try:

        completion = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=[
                {"role": "system", "content": "You are an expert sales trainer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600
        )

        return completion.choices[0].message.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        return str(e)

# Routes

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/upload_page")
def upload_page():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("brochure")
    industry = (request.form.get("industry", "General") or "General").strip()

    if not file or file.filename == "":
        return "No file uploaded", 400

    brochure_text = extract_text(file)
    if not brochure_text.strip():
        return "Could not extract text from brochure", 400

    chunks = chunk_text(brochure_text)
    if not chunks:
        return "No usable content found", 400

    query = f"{industry} product features benefits pricing use cases target customer"

    if USE_FULL_RAG:
        embeddings = create_embeddings(chunks)
        index = build_index(embeddings)
        context = retrieve_context(query, index, chunks, k=3)
    else:
        context = retrieve_context_lightweight(query, chunks, k=3)

    pitch = generate_pitch(context=context, industry=industry)
    session["pitch"] = pitch
    session["context"] = context
    session["industry"] = industry

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
    context=session.get("context")

    feedback = evaluate_answers(answers, context)

    return render_template("evaluation.html", feedback=feedback)


@app.route("/pitch")
def pitch():

    pitch = session.get("pitch")

    if not pitch:
        return "No pitch available. Please upload a brochure first."

    return render_template("results.html", pitch=pitch)





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))