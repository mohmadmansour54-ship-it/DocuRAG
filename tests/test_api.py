import pytest
from fastapi.testclient import TestClient

from app.main import app, get_rag_service


@pytest.fixture
def client(rag):
    app.dependency_overrides[get_rag_service] = lambda: rag
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client, data: bytes, name: str = "manual.pdf"):
    return client.post("/documents", files={"file": (name, data, "application/pdf")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_list_ask_delete_flow(client, sample_pdf):
    uploaded = upload(client, sample_pdf)
    assert uploaded.status_code == 201
    doc_id = uploaded.json()["document_id"]

    listed = client.get("/documents").json()
    assert [d["document_id"] for d in listed] == [doc_id]

    answer = client.post("/ask", json={"question": "How long does the warranty last?"})
    assert answer.status_code == 200
    body = answer.json()
    assert body["found"] is True
    assert body["sources"][0]["page"] == 2

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get("/documents").json() == []


def test_non_pdf_upload_is_rejected(client):
    response = upload(client, b"hello", name="notes.txt")
    assert response.status_code == 415


def test_corrupt_pdf_is_rejected(client):
    response = upload(client, b"not really a pdf", name="broken.pdf")
    assert response.status_code == 422


def test_deleting_unknown_document_returns_404(client):
    assert client.delete("/documents/doesnotexist").status_code == 404


def test_question_validation(client):
    assert client.post("/ask", json={"question": "?"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "top_k": 50}).status_code == 422
