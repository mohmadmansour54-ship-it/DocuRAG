"""Thin interfaces over the model provider.

The RAG pipeline only depends on the two protocols below, so the OpenAI
implementation can be swapped (another provider, a local model, or the fakes
used in the test suite) without touching retrieval logic.
"""

from typing import Protocol

from openai import OpenAI


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAIEmbedder:
    # The embeddings endpoint accepts many inputs per call; batching keeps the
    # number of HTTP requests low for long documents.
    batch_size = 100

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors


class OpenAIChatModel:
    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,  # deterministic, factual answers
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()
