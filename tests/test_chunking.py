import pytest

from app.chunking import chunk_pages, split_text
from app.pdf_loader import Page

TEXT = " ".join(f"word{i:03d}" for i in range(200))  # 200 words, 7 chars each


def test_chunks_respect_max_size():
    for chunk in split_text(TEXT, chunk_size=100, overlap=20):
        assert len(chunk) <= 100


def test_consecutive_chunks_overlap():
    chunks = split_text(TEXT, chunk_size=100, overlap=20)
    for previous, current in zip(chunks, chunks[1:]):
        last_words = previous.split()[-2:]
        assert current.split()[:2] == last_words


def test_no_words_are_lost():
    chunks = split_text(TEXT, chunk_size=100, overlap=20)
    seen = {w for c in chunks for w in c.split()}
    assert seen == set(TEXT.split())


def test_zero_overlap_produces_disjoint_chunks():
    chunks = split_text(TEXT, chunk_size=100, overlap=0)
    assert sum(len(c.split()) for c in chunks) == 200


def test_short_text_is_a_single_chunk():
    assert split_text("just a few words", chunk_size=100, overlap=20) == ["just a few words"]


def test_word_longer_than_chunk_size_is_kept_whole():
    assert split_text("a " + "x" * 50 + " b", chunk_size=10, overlap=2) == ["a", "x" * 50, "b"]


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        split_text(TEXT, size, overlap)


def test_chunk_pages_keeps_page_numbers_and_global_index():
    pages = [Page(1, TEXT[:300]), Page(2, TEXT[:300])]
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    assert {c.page for c in chunks} == {1, 2}
    assert [c.index for c in chunks] == list(range(len(chunks)))
