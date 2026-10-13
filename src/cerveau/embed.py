"""Embedders turn text into vectors.

Two implementations:
- OllamaEmbedder: real semantic embeddings from a local Ollama server.
- HashEmbedder: deterministic bag-of-words hashing. NOT semantic. It exists so the
  test suite and CI run without a GPU or a model download. Never use it for real data.

Rule: no silent truncation. A text longer than the configured limit is rejected,
and Ollama is asked to fail rather than truncate.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol
from urllib.parse import urlparse

import requests

from cerveau.config import LOOPBACK_HOSTS, redact_url


class EmbeddingError(RuntimeError):
    """Raised when a text cannot be embedded as-is.

    `public` is a version of the message that is safe to show to an AI client: it never
    contains what the server answered (which may hold paths or host names).
    """

    def __init__(self, message: str, public: str | None = None, per_text: bool = False) -> None:
        super().__init__(message)
        self.public = public if public is not None else message
        # True: this text is the problem (too long, empty, too many tokens); another text may
        # pass. False: the service is the problem (down, slow, model missing): retrying text by
        # text is pointless.
        self.per_text = per_text


class Embedder(Protocol):
    model: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _check_lengths(texts: list[str], max_chars: int) -> None:
    for i, text in enumerate(texts):
        if not text.strip():
            raise EmbeddingError(f"text #{i} is empty", per_text=True)
        if len(text) > max_chars:
            raise EmbeddingError(
                f"text #{i} has {len(text)} characters, the limit is {max_chars}. "
                "Split it first (it is never truncated silently).",
                per_text=True,
            )


class OllamaEmbedder:
    def __init__(
        self,
        url: str,
        model: str,
        dimension: int,
        max_chars: int,
        timeout: float = 300.0,
        session: requests.Session | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.shown_url = redact_url(self.url)  # for messages: never the user:password part
        self.model = model
        self.dimension = dimension
        self.max_chars = max_chars
        self.timeout = timeout
        if session is None:
            session = requests.Session()
            # MCP clients may call several tools at once: room for them in the connection pool
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=32)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            if urlparse(self.url).hostname in LOOPBACK_HOSTS:
                # A local Ollama must never be reached through HTTP_PROXY / the system proxy:
                # that would send every text to the proxy.
                session.trust_env = False
        self.session = session

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        _check_lengths(texts, self.max_chars)
        try:
            response = self.session.post(
                f"{self.url}/api/embed",
                json={"model": self.model, "input": texts, "truncate": False},
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise EmbeddingError(
                f"Ollama at {self.shown_url} did not answer within {self.timeout:.0f} s "
                "(slow machine? raise CERVEAU_OLLAMA_TIMEOUT)",
                public=f"the embedding service did not answer within {self.timeout:.0f} s",
            ) from exc
        except requests.RequestException as exc:
            raise EmbeddingError(
                f"cannot reach Ollama at {self.shown_url}: is it running? (start it, or check "
                f"CERVEAU_OLLAMA_URL) [{exc.__class__.__name__}]",
                public="cannot reach the embedding service (Ollama is down or unreachable)",
            ) from exc
        if response.status_code != 200:
            code = response.status_code
            detail = response.text[:300]
            hint = ""
            if "context length" in detail:
                hint = (
                    " The text has too many tokens for the model (dense text: code, tables, "
                    "symbols...): split it (remember), or lower CERVEAU_CHUNK_MAX_CHARS (index)."
                )
            elif code == 404 and "model" in detail.lower():
                hint = f" Pull it first: ollama pull {self.model}"
            elif code == 404:
                hint = " No /api/embed here: check CERVEAU_OLLAMA_URL, or update Ollama."
            raise EmbeddingError(
                f"Ollama returned HTTP {code}: {detail}.{hint}",
                public=f"the embedding service refused the request (HTTP {code}).{hint}",
                per_text="context length" in detail or code in (400, 413),
            )
        try:
            vectors = response.json().get("embeddings")
        except (ValueError, AttributeError) as exc:
            raise EmbeddingError(f"Ollama at {self.shown_url} did not answer with JSON") from exc
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError("Ollama returned an unexpected number of embeddings")
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    f"model {self.model!r} returned {len(vector)} dimensions, "
                    f"CERVEAU_VECTOR_SIZE is {self.dimension}"
                )
        return vectors


_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashEmbedder:
    """Deterministic, lexical, non-semantic. For tests and CI only."""

    def __init__(self, dimension: int, max_chars: int) -> None:
        self.model = f"hash-bow-{dimension}"
        self.dimension = dimension
        self.max_chars = max_chars

    def embed(self, texts: list[str]) -> list[list[float]]:
        _check_lengths(texts, self.max_chars)
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]
