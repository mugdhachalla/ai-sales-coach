# SalesCoach AI

SalesCoach AI is a web application designed to help salespeople quickly understand new products and practice delivering effective sales pitches using AI.

The platform analyzes product brochures, generates sales training material, and allows users to practice answering sales questions while receiving AI based feedback on their readiness.

The goal of the project is to simulate a real world sales preparation workflow and help users build confidence before interacting with customers.

---

## Features

- Upload product brochures in PDF format
- Automatically extract and analyze brochure content
- Generate a natural, conversational sales pitch based on product information
- Practice answering realistic sales questions
- Receive AI evaluation and feedback on sales readiness
- Review the generated pitch without reuploading the brochure
- Multi page interface including landing page, pitch page, practice page, and evaluation page

---

## Tech Stack

- Python
- Flask
- Hugging Face Large Language Models
- Sentence Transformers
- FAISS Vector Search
- HTML and CSS frontend
- Retrieval Augmented Generation (RAG)

---

## How It Works

1. A user uploads a product brochure.
2. The system extracts text from the document.
3. The text is divided into smaller chunks.
4. Sentence embeddings are created using Sentence Transformers.
5. FAISS retrieves the most relevant information from the brochure.
6. A Large Language Model generates a natural sales pitch.
7. The user practices answering sales questions.
8. The system evaluates the responses and provides feedback on sales readiness.

---

## Application Flow

Landing Page  
→ Upload Brochure  
→ AI Generated Sales Pitch  
→ Practice Questions  
→ AI Evaluation and Feedback  

Users can revisit the generated pitch during practice without uploading the brochure again.

---

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/mugdhachalla/ai-sales-coach.git
cd ai-sales-coach
```
### 2. Create virtual environment
```bash
python -m venv venv
```
Activate the environment\
Mac/Linux:
```bash
source venv/bin/activate
```
Windows:
```bash
venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Create environment variables
Create a .env file in the project root and add:
```bash
SECRET_KEY=your_secret_key_here
HF_API_KEY=your_huggingface_api_key_here
```
### 5. Run the application
```bash
python app.py
```
### 6. Open the application
Visit: 
```bash
http://127.0.0.1:5000
```

## Status

This project is currently a **work in progress**.

Planned improvements include:

- Voice based pitch practice
- Improved evaluation scoring
- More advanced AI feedback
- Enhanced UI and UX
- Improved pitch generation

---

## Purpose

This project was built as a learning exercise to explore:

- Retrieval Augmented Generation (RAG)
- Document understanding pipelines
- Building AI powered web applications