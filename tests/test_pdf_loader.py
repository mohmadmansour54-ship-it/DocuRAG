import pytest

from app.pdf_loader import PDFExtractionError, extract_pages
from tests.conftest import make_pdf


def test_extracts_text_per_page(sample_pdf):
    pages = extract_pages(sample_pdf)
    assert [p.number for p in pages] == [1, 2]
    assert "24 months" in pages[1].text


def test_whitespace_is_normalised(sample_pdf):
    for page in extract_pages(sample_pdf):
        assert "\n" not in page.text
        assert "  " not in page.text


def test_blank_pages_are_skipped():
    pages = extract_pages(make_pdf(["first page", "", "third page"]))
    assert [p.number for p in pages] == [1, 3]


def test_pdf_without_text_raises():
    with pytest.raises(PDFExtractionError, match="no extractable text"):
        extract_pages(make_pdf([""]))


def test_non_pdf_bytes_raise():
    with pytest.raises(PDFExtractionError):
        extract_pages(b"this is not a pdf")
