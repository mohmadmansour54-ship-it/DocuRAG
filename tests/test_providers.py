from types import SimpleNamespace
from unittest.mock import MagicMock

from app.providers import OpenAIChatModel, OpenAIEmbedder


def embedding_response(batch):
    # The API may return items out of order; `index` says where each belongs.
    items = [SimpleNamespace(index=i, embedding=[float(len(text))]) for i, text in enumerate(batch)]
    return SimpleNamespace(data=list(reversed(items)))


def test_embedder_batches_requests_and_keeps_order():
    client = MagicMock()
    client.embeddings.create.side_effect = lambda model, input: embedding_response(input)
    embedder = OpenAIEmbedder(client, "text-embedding-3-small")
    embedder.batch_size = 2

    vectors = embedder.embed(["a", "bb", "ccc", "dddd", "eeeee"])

    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert client.embeddings.create.call_count == 3


def test_chat_model_sends_system_and_user_messages_at_temperature_zero():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  answer [1]  "))]
    )
    model = OpenAIChatModel(client, "gpt-4o-mini")

    assert model.complete("system rules", "question") == "answer [1]"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0
    assert [m["role"] for m in kwargs["messages"]] == ["system", "user"]
