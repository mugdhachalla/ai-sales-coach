# AI Sales Coach

AI Sales Coach is a web application that helps salespeople quickly understand and practice pitching new products from brochures.

The system analyzes uploaded product brochures and generates a sales pitch, key selling points, and practice questions to help users prepare before meeting customers.

## Features

- Upload product brochures in PDF format
- Extract and analyze brochure content
- Generate a concise sales pitch and key selling points
- Practice answering sales questions
- Receive AI generated feedback on sales readiness

## Tech Stack

- Python
- Flask
- Hugging Face LLMs
- Sentence Transformers
- FAISS Vector Search
- Retrieval Augmented Generation (RAG)

## How It Works

1. User uploads a product brochure.
2. The system extracts and processes the document.
3. Text is split into chunks and converted into embeddings.
4. FAISS retrieves the most relevant sections.
5. An LLM generates a sales pitch and practice questions.

## Status

This project is currently a **work in progress**.  
Additional improvements planned include:

- Improved UI
- Interactive sales simulation
- Better response evaluation
- Enhanced document understanding

## Purpose

This project was built as a learning exercise to explore **RAG pipelines, document understanding, and AI powered productivity tools.**