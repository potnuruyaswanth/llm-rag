from pathlib import Path
import logging
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_pipeline import process_pdf, get_answer

app = FastAPI(title="Simple LLM + RAG API")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question about the uploaded PDF")
    top_k: int = Field(3, ge=1, le=5, description="Number of chunks to retrieve")


@app.on_event("startup")
def warm_models() -> None:
    try:
        from rag_pipeline import rag_pipeline

        rag_pipeline.warm_up()
    except Exception:
        # Keep the API booting even if model download/warmup is delayed.
        pass


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Simple LLM + RAG API is running."}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    safe_name = f"{uuid4().hex}_{Path(file.filename).name}"
    file_path = UPLOADS_DIR / safe_name

    try:
        contents = await file.read()
        file_path.write_bytes(contents)
        result = process_pdf(str(file_path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF processing failed")
        raise HTTPException(status_code=500, detail=f"Failed to process the PDF: {exc}") from exc

    return {
        "message": "PDF processed successfully.",
        "document": result,
    }


@app.post("/ask")
def ask_question(payload: AskRequest) -> dict[str, object]:
    try:
        result = get_answer(payload.query, top_k=payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Answer generation failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate an answer: {exc}") from exc

    return result
