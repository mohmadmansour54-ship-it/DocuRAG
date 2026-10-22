"""FastAPI entry point."""

from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from openai import OpenAI

from app.config import get_settings
from app.pdf_loader import PDFExtractionError
from app.providers import OpenAIChatModel, OpenAIEmbedder
from app.rag import RAGService
from app.schemas import AskRequest, AskResponse, DocumentInfo, IngestResponse
from app.vector_store import VectorStore

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

app = FastAPI(
    title="DocuRAG",
    description="Upload PDFs and ask questions about them. Answers are grounded in the documents and cite their sources.",
    version="1.0.0",
)


@lru_cache
def get_rag_service() -> RAGService:
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    client = OpenAI(api_key=settings.openai_api_key)
    return RAGService(
        store=VectorStore.persistent(settings.chroma_path, settings.collection_name),
        embedder=OpenAIEmbedder(client, settings.embedding_model),
        llm=OpenAIChatModel(client, settings.chat_model),
        settings=settings,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse, status_code=201)
async def upload_document(file: UploadFile = File(...), rag: RAGService = Depends(get_rag_service)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 20 MB")
    try:
        return rag.ingest_pdf(data, file.filename)
    except PDFExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(rag: RAGService = Depends(get_rag_service)):
    return [DocumentInfo(document_id=doc_id, **info) for doc_id, info in rag.store.list_documents().items()]


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, rag: RAGService = Depends(get_rag_service)):
    if rag.store.delete_document(document_id) == 0:
        raise HTTPException(status_code=404, detail="Document not found")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, rag: RAGService = Depends(get_rag_service)):
    return rag.ask(request.question, request.top_k)
