"""The RAG pipeline: ingest documents, retrieve relevant chunks, generate cited answers."""

import hashlib
import re

from app.chunking import chunk_pages
from app.config import Settings
from app.pdf_loader import extract_pages
from app.prompts import NOT_FOUND_ANSWER, SYSTEM_PROMPT, build_user_prompt
from app.providers import ChatModel, Embedder
from app.schemas import AskResponse, IngestResponse, Source
from app.vector_store import VectorStore

_CITATION = re.compile(r"\[(\d+)\]")


class RAGService:
    def __init__(self, store: VectorStore, embedder: Embedder, llm: ChatModel, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.settings = settings

    # ----- ingestion -------------------------------------------------------

    def ingest_pdf(self, data: bytes, filename: str) -> IngestResponse:
        # Content hash as the ID: uploading the same file twice is a no-op,
        # so we never pay to embed the same text again or store duplicates.
        document_id = hashlib.sha256(data).hexdigest()[:16]
        pages = extract_pages(data)

        if self.store.has_document(document_id):
            chunks = self.store.list_documents()[document_id]["chunks"]
            return IngestResponse(
                document_id=document_id, filename=filename, chunks=chunks, pages=len(pages), already_indexed=True
            )

        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.add(document_id, filename, chunks, embeddings)
        return IngestResponse(
            document_id=document_id, filename=filename, chunks=len(chunks), pages=len(pages), already_indexed=False
        )

    # ----- question answering ---------------------------------------------

    def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        query_vector = self.embedder.embed([question])[0]
        hits = self.store.search(query_vector, top_k or self.settings.top_k)

        # Retrieval guardrail: if nothing is close enough, the documents don't
        # cover this question. Refuse without calling the LLM at all, which
        # both prevents hallucinated answers and saves a model call.
        relevant = [h for h in hits if h.distance <= self.settings.max_distance]
        if not relevant:
            return AskResponse(answer=NOT_FOUND_ANSWER, found=False, sources=[])

        answer = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, [h.text for h in relevant]))

        # Generation guardrail: the prompt tells the model to use a fixed
        # sentence when the passages don't contain the answer.
        if NOT_FOUND_ANSWER.lower() in answer.lower():
            return AskResponse(answer=NOT_FOUND_ANSWER, found=False, sources=[])

        cited = sorted({int(n) for n in _CITATION.findall(answer) if 1 <= int(n) <= len(relevant)})
        # If the model forgot to cite, fall back to every passage it was shown.
        numbers = cited or list(range(1, len(relevant) + 1))
        sources = [
            Source(
                citation=n,
                filename=relevant[n - 1].filename,
                page=relevant[n - 1].page,
                text=relevant[n - 1].text,
                distance=round(relevant[n - 1].distance, 4),
            )
            for n in numbers
        ]
        return AskResponse(answer=answer, found=True, sources=sources)
