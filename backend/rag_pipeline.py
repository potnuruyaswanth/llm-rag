from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", "gemini-2.5-flash")
STOPWORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "and", "or", "to", "of",
    "in", "on", "for", "with", "how", "why", "when", "where", "who", "which", "does",
    "do", "did", "can", "could", "should", "would", "about", "from", "this", "that",
}
STORAGE_DIR = Path(__file__).resolve().parent / "storage"
INDEX_PATH = STORAGE_DIR / "faiss.index"
METADATA_PATH = STORAGE_DIR / "chunks.json"
MIN_RAG_SCORE = 0.35


@dataclass
class ChunkRecord:
    text: str
    source: str
    page: int


class SimpleRAGPipeline:
    def __init__(self) -> None:
        STORAGE_DIR.mkdir(exist_ok=True)
        self._client = None
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[ChunkRecord] = []
        self._document_name: str | None = None
        self._restore_index()

    def process_pdf(self, file_path: str) -> dict[str, Any]:
        pdf_path = Path(file_path)
        reader = PdfReader(str(pdf_path))

        chunk_records: list[ChunkRecord] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            for chunk in self._split_text(text):
                chunk_records.append(
                    ChunkRecord(
                        text=chunk,
                        source=pdf_path.name,
                        page=page_number,
                    )
                )

        if not chunk_records:
            raise ValueError("No readable text was found in the PDF.")

        self._chunks = chunk_records
        self._document_name = pdf_path.name
        self._build_index()
        self._persist_index()

        return {
            "filename": pdf_path.name,
            "pages_indexed": len({chunk.page for chunk in chunk_records}),
            "chunks_created": len(chunk_records),
        }

    def get_answer(self, query: str, top_k: int = 3) -> dict[str, Any]:
        question = query.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        if self._index is None or not self._chunks:
            answer = self._generate_answer(self._build_general_prompt(question))
            return {
                "answer": answer,
                "sources": [],
                "scores": [],
                "document": None,
                "mode": "llm",
            }

        query_embedding = self._embed_query(question)

        search_k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query_embedding, search_k)

        retrieved_chunks: list[ChunkRecord] = []
        for index in indices[0]:
            if index == -1:
                continue
            retrieved_chunks.append(self._chunks[index])

        if not retrieved_chunks:
            raise ValueError("No relevant context was found for that question.")

        if not self._should_use_rag(question, retrieved_chunks, scores[0][: len(retrieved_chunks)]):
            answer = self._generate_answer(self._build_general_prompt(question))
            return {
                "answer": answer,
                "sources": [],
                "scores": [],
                "document": None,
                "mode": "llm",
            }

        prompt = self._build_prompt(question, retrieved_chunks)
        answer = self._generate_answer(prompt)

        return {
            "answer": answer,
            "sources": [
                {
                    "source": chunk.source,
                    "page": chunk.page,
                    "preview": chunk.text[:180].strip(),
                }
                for chunk in retrieved_chunks
            ],
            "scores": [round(float(score), 4) for score in scores[0][: len(retrieved_chunks)]],
            "document": self._document_name,
            "mode": "rag",
        }

    def warm_up(self) -> None:
        self._get_client()

    def _build_index(self) -> None:
        chunk_texts = [chunk.text for chunk in self._chunks]
        embeddings = self._embed_texts(chunk_texts)

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def _persist_index(self) -> None:
        if self._index is None:
            return

        faiss.write_index(self._index, str(INDEX_PATH))
        metadata = {
            "document_name": self._document_name,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "generation_model": GENERATION_MODEL_NAME,
            "chunks": [
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                }
                for chunk in self._chunks
            ],
        }
        METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    def _restore_index(self) -> None:
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            return

        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            if metadata.get("embedding_model") != EMBEDDING_MODEL_NAME:
                self._index = None
                self._chunks = []
                self._document_name = None
                return
            self._document_name = metadata.get("document_name")
            self._chunks = [
                ChunkRecord(
                    text=item["text"],
                    source=item["source"],
                    page=item["page"],
                )
                for item in metadata.get("chunks", [])
            ]
            if self._chunks:
                self._index = faiss.read_index(str(INDEX_PATH))
        except Exception:
            self._index = None
            self._chunks = []
            self._document_name = None

    def _get_client(self):
        if self._client is None:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set. Add it to backend/.env or your environment.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        embeddings = np.array([item.values for item in response.embeddings], dtype="float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def _embed_query(self, question: str) -> np.ndarray:
        client = self._get_client()
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=[question],
            config=types.EmbedContentConfig(task_type="QUESTION_ANSWERING"),
        )
        embeddings = np.array([item.values for item in response.embeddings], dtype="float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def _generate_answer(self, prompt: str) -> str:
        client = self._get_client()
        response = client.models.generate_content(
            model=GENERATION_MODEL_NAME,
            contents=prompt,
        )
        return (response.text or "").strip()

    @staticmethod
    def _has_query_overlap(question: str, chunks: list[ChunkRecord]) -> bool:
        query_terms = {
            token for token in re.findall(r"[a-zA-Z0-9]+", question.lower())
            if len(token) > 2 and token not in STOPWORDS
        }
        if not query_terms:
            return True

        combined_context = " ".join(chunk.text.lower() for chunk in chunks)
        return any(term in combined_context for term in query_terms)

    @staticmethod
    def _is_general_knowledge_or_math(question: str) -> bool:
        cleaned = question.strip().lower()
        if re.fullmatch(r"[\d\s+\-*/().=]+", cleaned):
            return True

        general_patterns = (
            "who won",
            "what is",
            "who is",
            "calculate",
            "solve",
            "capital of",
            "today",
            "ipl",
            "cricket",
        )
        return any(pattern in cleaned for pattern in general_patterns)

    def _should_use_rag(self, question: str, chunks: list[ChunkRecord], scores: Any) -> bool:
        if self._is_general_knowledge_or_math(question) and not self._has_query_overlap(question, chunks):
            return False

        if not self._has_query_overlap(question, chunks):
            return False

        top_score = float(scores[0]) if len(scores) else 0.0
        return top_score >= MIN_RAG_SCORE

    @staticmethod
    def _split_text(text: str, chunk_size: int = 700, chunk_overlap: int = 120) -> list[str]:
        cleaned = " ".join(text.split())
        if len(cleaned) <= chunk_size:
            return [cleaned]

        chunks: list[str] = []
        start = 0
        text_length = len(cleaned)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = cleaned[start:end]
            if end < text_length:
                last_break = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("! "))
                if last_break > chunk_size // 2:
                    end = start + last_break + 1
                    chunk = cleaned[start:end]

            chunks.append(chunk.strip())
            if end >= text_length:
                break
            start = max(end - chunk_overlap, 0)

        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _build_prompt(question: str, chunks: list[ChunkRecord]) -> str:
        context = "\n\n".join(
            f"[Page {chunk.page}] {chunk.text}"
            for chunk in chunks
        )
        return (
            "You are a helpful assistant answering questions only from the provided context. "
            "Do not use outside knowledge. If the answer is not clearly present in the context, "
            "reply with: I could not find that information in the uploaded PDF.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    @staticmethod
    def _build_general_prompt(question: str) -> str:
        return (
            "You are a helpful assistant. Answer the user's question clearly and concisely.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )


rag_pipeline = SimpleRAGPipeline()


def process_pdf(file_path: str) -> dict[str, Any]:
    return rag_pipeline.process_pdf(file_path)


def get_answer(query: str, top_k: int = 3) -> dict[str, Any]:
    return rag_pipeline.get_answer(query=query, top_k=top_k)
