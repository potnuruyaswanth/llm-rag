# Simple LLM RAG App

A simple Retrieval-Augmented Generation (RAG) project that lets users upload a PDF, create embeddings from the document, retrieve relevant chunks, and generate answers using a lightweight local language model.

## Features

- Upload a PDF document from the frontend
- Extract and split PDF text into chunks
- Generate embeddings with Sentence Transformers
- Store and search embeddings with FAISS
- Answer questions using retrieved document context
- FastAPI backend with a simple HTML, CSS, and JavaScript frontend

## Tech Stack

- Python
- FastAPI
- FAISS
- Sentence Transformers
- Hugging Face Transformers
- PyPDF
- HTML
- CSS
- JavaScript

## Project Structure

```text
LLM_RAG/
├── backend/
│   ├── main.py
│   ├── rag_pipeline.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

## How It Works

1. The user uploads a PDF file.
2. The backend extracts text from the PDF.
3. The text is split into smaller chunks.
4. Embeddings are created for each chunk.
5. FAISS stores the embeddings for similarity search.
6. When the user asks a question, the system retrieves the most relevant chunks.
7. A lightweight local model generates an answer using the retrieved context.

## Run Locally

### 1. Install dependencies

```powershell
cd backend
pip install -r requirements.txt
```

### 2. Start the backend

```powershell
python -m uvicorn main:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

### 3. Open the frontend

Open `frontend/index.html` in your browser.

## API Endpoints

- `GET /health` : Check if the backend is running
- `POST /upload` : Upload and process a PDF
- `POST /ask` : Ask a question about the uploaded PDF

## Example Use Case

- Upload a research paper, notes file, or report
- Ask questions like:
  - "What is the main topic of this document?"
  - "Summarize the introduction"
  - "What conclusions are mentioned?"

## Current Limitations

- Designed as a simple prototype, not a production-ready system
- Supports one uploaded document at a time
- First-time model loading can be slow
- Answer quality depends on extracted PDF text and the small local model
- Frontend is static and minimal

## Future Improvements

- Multi-document support
- Chat history
- Better ranking and retrieval
- Stronger LLM integration
- Deployment with Docker or cloud hosting

