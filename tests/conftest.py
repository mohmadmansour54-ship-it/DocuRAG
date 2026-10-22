import hashlib
import math
import re
import uuid
from io import BytesIO

import chromadb
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.config import Settings
from app.rag import RAGService
from app.vector_store import VectorStore


class FakeEmbedder:
    """Deterministic bag-of-words embedding: texts sharing words are close."""

    dims = 256

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) > 2:
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FakeChatModel:
    """Returns a scripted answer and records the prompts it received."""

    def __init__(self, answer: str = "The warranty lasts 24 months [1]."):
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.answer


def make_pdf(pages: list[str]) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            pdf.drawString(50, y, line)
            y -= 14
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@pytest.fixture
def sample_pdf() -> bytes:
    return make_pdf(
        [
            "Acme Robot Vacuum - Product Manual\nThe battery charges fully in 3 hours.",
            "Warranty\nThe warranty lasts 24 months from the date of purchase.\nIt does not cover water damage.",
        ]
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="test", chunk_size=200, chunk_overlap=40, top_k=3, max_distance=0.8)


@pytest.fixture
def store() -> VectorStore:
    # In-memory Chroma; a unique collection per test keeps tests independent.
    return VectorStore(chromadb.EphemeralClient(), f"test-{uuid.uuid4().hex[:8]}")


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeChatModel:
    return FakeChatModel()


@pytest.fixture
def rag(store, embedder, llm, settings) -> RAGService:
    return RAGService(store=store, embedder=embedder, llm=llm, settings=settings)
