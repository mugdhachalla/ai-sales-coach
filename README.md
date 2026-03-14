# AI Sales Coach

AI Sales Coach is a web application designed to help salespeople quickly understand new products and practice delivering effective sales pitches using AI.

The platform analyzes product brochures, generates a natural conversational sales pitch, and allows users to practice answering sales questions while receiving AI based feedback on their readiness.

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
- AI assisted productivity tools
- Building AI powered web applications