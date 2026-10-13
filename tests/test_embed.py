import pytest
import requests

from cerveau.embed import EmbeddingError, HashEmbedder, OllamaEmbedder


class FakeResponse:
    def __init__(self, status: int, payload: dict | None = None, text: str = ""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def post(self, url, json, timeout):
        self.calls.append((url, json))
        if self.exc:
            raise self.exc
        return self.response


def ollama(session, dim=3):
    return OllamaEmbedder("http://h:11434/", "m", dim, max_chars=50, session=session)


def test_ollama_asks_for_no_truncation_and_returns_vectors():
    session = FakeSession(FakeResponse(200, {"embeddings": [[1, 2, 3], [4, 5, 6]]}))
    assert ollama(session).embed(["a", "b"]) == [[1, 2, 3], [4, 5, 6]]
    url, body = session.calls[0]
    assert url == "http://h:11434/api/embed"
    assert body == {"model": "m", "input": ["a", "b"], "truncate": False}


def test_ollama_too_long_text_is_refused_before_any_call():
    session = FakeSession(FakeResponse(200, {"embeddings": [[1, 2, 3]]}))
    with pytest.raises(EmbeddingError, match="never truncated"):
        ollama(session).embed(["x" * 51])
    assert session.calls == []


def test_ollama_dimension_mismatch_raises():
    session = FakeSession(FakeResponse(200, {"embeddings": [[1, 2]]}))
    with pytest.raises(EmbeddingError, match="dimensions"):
        ollama(session).embed(["a"])


def test_ollama_http_error_and_unreachable_raise():
    with pytest.raises(EmbeddingError, match="HTTP 500"):
        ollama(FakeSession(FakeResponse(500, text="boom"))).embed(["a"])
    with pytest.raises(EmbeddingError, match="cannot reach"):
        ollama(FakeSession(exc=requests.ConnectionError("down"))).embed(["a"])
    with pytest.raises(EmbeddingError, match="number"):
        ollama(FakeSession(FakeResponse(200, {"embeddings": []}))).embed(["a"])


def test_empty_text_is_refused():
    with pytest.raises(EmbeddingError, match="empty"):
        HashEmbedder(8, 100).embed(["  "])


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder(32, 100)
    v1, v2 = e.embed(["bread and salt"])[0], e.embed(["bread and salt"])[0]
    assert v1 == v2 and len(v1) == 32
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-9


def test_ollama_non_json_answer_raises_embedding_error():
    class Html(FakeResponse):
        def json(self):
            raise ValueError("not json")

    with pytest.raises(EmbeddingError, match="JSON"):
        ollama(FakeSession(Html(200))).embed(["a"])


def test_local_ollama_ignores_proxies_remote_one_keeps_them():
    local = OllamaEmbedder("http://127.0.0.1:11434", "m", 3, 50)
    remote = OllamaEmbedder("https://ollama.example", "m", 3, 50)
    assert local.session.trust_env is False and remote.session.trust_env is True


def test_timeout_and_context_errors_explain_what_to_do():
    with pytest.raises(EmbeddingError, match="CERVEAU_OLLAMA_TIMEOUT"):
        ollama(FakeSession(exc=requests.Timeout("slow"))).embed(["a"])
    body = '{"error":"the input length exceeds the context length"}'
    with pytest.raises(EmbeddingError, match="CERVEAU_CHUNK_MAX_CHARS"):
        ollama(FakeSession(FakeResponse(400, text=body))).embed(["a"])
    with pytest.raises(EmbeddingError, match="ollama pull"):
        ollama(FakeSession(FakeResponse(404, text='model "m" not found'))).embed(["a"])
    with pytest.raises(EmbeddingError, match="CERVEAU_OLLAMA_URL"):
        ollama(FakeSession(FakeResponse(404, text="404 page not found"))).embed(["a"])


def test_credentials_in_the_ollama_url_never_appear_in_messages():
    embedder = OllamaEmbedder("http://admin:s3cret@127.0.0.1:2", "m", 3, 50)
    with pytest.raises(EmbeddingError) as caught:
        embedder.embed(["x"])
    assert "s3cret" not in str(caught.value)


def test_errors_say_whether_the_text_or_the_service_is_the_problem():
    from cerveau.embed import EmbeddingError, _check_lengths

    with pytest.raises(EmbeddingError) as caught:
        _check_lengths(["x" * 10], 5)
    assert caught.value.per_text
    with pytest.raises(EmbeddingError) as caught:
        _check_lengths(["  "], 5)
    assert caught.value.per_text
    assert not EmbeddingError("down").per_text


def test_service_failures_are_not_blamed_on_the_text():
    body = '{"error":"the input length exceeds the context length"}'
    cases = [
        (FakeSession(exc=requests.Timeout("slow")), False),
        (FakeSession(exc=requests.ConnectionError("refused")), False),
        (FakeSession(FakeResponse(404, text='model "m" not found')), False),
        (FakeSession(FakeResponse(500, text="boom")), False),
        (FakeSession(FakeResponse(500, text=body)), True),
        (FakeSession(FakeResponse(413, text="too large")), True),
    ]
    for session, per_text in cases:
        with pytest.raises(EmbeddingError) as caught:
            ollama(session).embed(["a"])
        assert caught.value.per_text is per_text
