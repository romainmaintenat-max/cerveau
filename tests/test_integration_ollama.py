"""The contract with a real Ollama (not run by the CI: it needs the model). Run before a release:

CERVEAU_IT_OLLAMA_URL=http://127.0.0.1:11434 pytest -m integration tests/test_integration_ollama.py

Optional: CERVEAU_IT_OLLAMA_MODEL (default bge-m3) and CERVEAU_IT_OLLAMA_DIM (1024).
"""

import os
import random
import string
import uuid

import pytest

from cerveau.embed import EmbeddingError, OllamaEmbedder

URL = os.environ.get("CERVEAU_IT_OLLAMA_URL")
MODEL = os.environ.get("CERVEAU_IT_OLLAMA_MODEL", "bge-m3")
DIM = int(os.environ.get("CERVEAU_IT_OLLAMA_DIM", "1024"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL, reason="CERVEAU_IT_OLLAMA_URL not set"),
]


def embedder(model: str = MODEL, max_chars: int = 3000) -> OllamaEmbedder:
    return OllamaEmbedder(URL, model, DIM, max_chars, timeout=300)


def test_embeds_with_the_announced_dimension():
    vectors = embedder().embed(["Sourdough needs flour, water and salt.", "Tomatoes need sun."])
    assert len(vectors) == 2 and all(len(v) == DIM for v in vectors)


def test_a_text_over_the_model_context_is_refused_never_truncated():
    # Random printable characters: many tokens per character, far over a 2k-8k token context.
    rng = random.Random(1)
    dense = "".join(
        rng.choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(60_000)
    )
    with pytest.raises(EmbeddingError) as caught:
        embedder(max_chars=100_000).embed([dense])
    assert caught.value.per_text  # the text is the problem, not the service


def test_a_missing_model_says_to_pull_it():
    with pytest.raises(EmbeddingError, match="ollama pull") as caught:
        embedder(model="cerveau-missing-" + uuid.uuid4().hex[:8]).embed(["x"])
    assert not caught.value.per_text
