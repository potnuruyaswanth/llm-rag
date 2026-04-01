from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-small"


@dataclass
class ChunkRecord:
    text: str
    source: str
    page: int


class SimpleRAGPipeline:
    def __init__(self) -> None:
        self._embedder: SentenceTransformer | None = None
        self._tokenizer = None
        self._generator_model = None
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[ChunkRecord] = []
        self._document_name: str | None = None

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

        return {
            "filename": pdf_path.name,
            "pages_indexed": len({chunk.page for chunk in chunk_records}),
            "chunks_created": len(chunk_records),
        }

    def get_answer(self, query: str, top_k: int = 3) -> dict[str, Any]:
        if self._index is None or not self._chunks:
            raise ValueError("Upload a PDF before asking questions.")

        question = query.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        embedder = self._get_embedder()
        query_embedding = embedder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        search_k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query_embedding, search_k)

        retrieved_chunks: list[ChunkRecord] = []
        for index in indices[0]:
            if index == -1:
                continue
            retrieved_chunks.append(self._chunks[index])

        if not retrieved_chunks:
            raise ValueError("No relevant context was found for that question.")

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
        }

    def warm_up(self) -> None:
        self._get_embedder()
        self._get_tokenizer()
        self._get_generator_model()

    def _build_index(self) -> None:
        embedder = self._get_embedder()
        chunk_texts = [chunk.text for chunk in self._chunks]
        embeddings = embedder.encode(
            chunk_texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._embedder

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
        return self._tokenizer

    def _get_generator_model(self):
        if self._generator_model is None:
            self._generator_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)
            self._generator_model.eval()
        return self._generator_model

    def _generate_answer(self, prompt: str) -> str:
        tokenizer = self._get_tokenizer()
        model = self._get_generator_model()
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

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
            "If the answer is not in the context, say that clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )


rag_pipeline = SimpleRAGPipeline()


def process_pdf(file_path: str) -> dict[str, Any]:
    return rag_pipeline.process_pdf(file_path)


def get_answer(query: str, top_k: int = 3) -> dict[str, Any]:
    return rag_pipeline.get_answer(query=query, top_k=top_k)
