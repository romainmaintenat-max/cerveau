from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from cerveau.brain import Brain
from cerveau.config import Settings

DIM = 64


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        qdrant_url=":memory:",
        embedder="hash",
        vector_size=DIM,
        max_text_chars=2000,
        chunk_max_chars=400,
        prefix="test",
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(location=":memory:")


@pytest.fixture
def brain(settings: Settings, client: QdrantClient) -> Brain:
    b = Brain(settings, client=client)
    b.init()
    return b


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "bread.md").write_text(
        "# Bread\n\nSourdough needs flour, water, salt and a starter. Bake at 240 degrees.\n",
        encoding="utf-8",
    )
    (root / "garden.md").write_text(
        "# Garden\n\nTomatoes need sun and regular watering. Plant them after the last frost.\n",
        encoding="utf-8",
    )
    return root
