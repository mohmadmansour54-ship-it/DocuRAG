"""Request and response models for the API."""

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int


class IngestResponse(DocumentInfo):
    pages: int
    already_indexed: bool = Field(description="True if this exact file was uploaded before; nothing was re-embedded.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["What is the refund policy?"])
    top_k: int | None = Field(default=None, ge=1, le=10, description="Number of passages to retrieve.")


class Source(BaseModel):
    citation: int = Field(description="The [n] number used in the answer.")
    filename: str
    page: int
    text: str
    distance: float = Field(description="Cosine distance to the question (lower is more relevant).")


class AskResponse(BaseModel):
    answer: str
    found: bool = Field(description="False when the documents do not contain the answer.")
    sources: list[Source]
