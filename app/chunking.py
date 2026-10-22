"""Split page text into overlapping chunks that keep their page number."""

from dataclasses import dataclass

from app.pdf_loader import Page


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    index: int  # position of the chunk inside its document


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window split on word boundaries.

    Each chunk is at most ``chunk_size`` characters (unless a single word is
    longer) and starts ``overlap`` characters before the previous one ended, so
    a sentence cut at a boundary is still complete in one of the two chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start
        length = 0
        while end < len(words) and (length + len(words[end]) + (1 if end > start else 0)) <= chunk_size:
            length += len(words[end]) + (1 if end > start else 0)
            end += 1
        if end == start:  # a single word longer than chunk_size
            end = start + 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break

        # Step back from `end` until we have roughly `overlap` characters.
        back = end
        carried = 0
        while back > start + 1 and carried + len(words[back - 1]) + 1 <= overlap:
            back -= 1
            carried += len(words[back]) + 1
        start = back
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk every page separately so each chunk can be cited by page."""
    chunks: list[Chunk] = []
    for page in pages:
        for text in split_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(text=text, page=page.number, index=len(chunks)))
    return chunks
