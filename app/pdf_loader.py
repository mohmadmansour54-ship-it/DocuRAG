"""Extract text from PDF files, page by page."""

from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFExtractionError(ValueError):
    """Raised when a file is not a readable PDF or contains no text."""


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as a reader would cite it
    text: str


def _clean(text: str) -> str:
    # pypdf keeps layout line breaks; collapse whitespace so chunks read as prose.
    return " ".join(text.split())


def extract_pages(data: bytes) -> list[Page]:
    """Return the non-empty pages of a PDF with normalised whitespace."""
    try:
        reader = PdfReader(BytesIO(data))
        pages = [Page(number=i + 1, text=_clean(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    except (PdfReadError, ValueError, KeyError) as exc:
        raise PDFExtractionError(f"Could not read PDF: {exc}") from exc

    pages = [p for p in pages if p.text]
    if not pages:
        # Usually a scanned PDF (images only). OCR is out of scope for now.
        raise PDFExtractionError("The PDF contains no extractable text (is it a scanned document?)")
    return pages
