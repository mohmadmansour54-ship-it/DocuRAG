"""ChromaDB wrapper that stores chunk vectors together with citation metadata."""

from dataclasses import dataclass

import chromadb
from chromadb.api import ClientAPI

from app.chunking import Chunk


@dataclass(frozen=True)
class SearchHit:
    text: str
    document_id: str
    filename: str
    page: int
    distance: float


class VectorStore:
    def __init__(self, client: ClientAPI, collection_name: str):
        # Embeddings are computed by our own Embedder, so Chroma's built-in
        # embedding function is disabled. Cosine distance matches how OpenAI
        # embeddings are meant to be compared.
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    @classmethod
    def persistent(cls, path: str, collection_name: str) -> "VectorStore":
        return cls(chromadb.PersistentClient(path=path), collection_name)

    def add(self, document_id: str, filename: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.upsert(
            ids=[f"{document_id}:{c.index}" for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"document_id": document_id, "filename": filename, "page": c.page} for c in chunks],
        )

    def has_document(self, document_id: str) -> bool:
        return bool(self._collection.get(where={"document_id": document_id}, limit=1)["ids"])

    def delete_document(self, document_id: str) -> int:
        ids = self._collection.get(where={"document_id": document_id})["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def list_documents(self) -> dict[str, dict]:
        """Return {document_id: {"filename": ..., "chunks": n}}."""
        documents: dict[str, dict] = {}
        for meta in self._collection.get(include=["metadatas"])["metadatas"] or []:
            entry = documents.setdefault(meta["document_id"], {"filename": meta["filename"], "chunks": 0})
            entry["chunks"] += 1
        return documents

    def search(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchHit(
                text=text,
                document_id=meta["document_id"],
                filename=meta["filename"],
                page=int(meta["page"]),
                distance=float(distance),
            )
            for text, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]
