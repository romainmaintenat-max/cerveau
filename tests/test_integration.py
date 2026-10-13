"""Tests against a real Qdrant server. Run with:

CERVEAU_IT_QDRANT_URL=http://127.0.0.1:6333 CERVEAU_IT_QDRANT_API_KEY=... pytest -m integration
"""

import os
import uuid

import pytest
from qdrant_client import QdrantClient

from cerveau.brain import Brain
from cerveau.config import Settings

URL = os.environ.get("CERVEAU_IT_QDRANT_URL")
KEY = os.environ.get("CERVEAU_IT_QDRANT_API_KEY")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL or not KEY, reason="CERVEAU_IT_QDRANT_URL/API_KEY not set"),
]


@pytest.fixture
def real_settings(tmp_path):
    return Settings(
        qdrant_url=URL,
        qdrant_api_key=KEY,
        embedder="hash",
        vector_size=64,
        max_text_chars=2000,
        chunk_max_chars=400,
        prefix="it_" + uuid.uuid4().hex[:8],
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def cleanup(real_settings):
    yield
    client = QdrantClient(url=URL, api_key=KEY)
    for suffix in ("docs", "memories", "meta"):
        name = f"{real_settings.prefix}_{suffix}"
        if client.collection_exists(name):
            client.delete_collection(name)


def test_server_refuses_requests_without_the_key():
    client = QdrantClient(url=URL, api_key=None)
    with pytest.raises(Exception, match="401|403|nauthorized|orbidden"):
        client.get_collections()


def test_full_flow_on_a_real_server(real_settings, cleanup, notes):
    brain = Brain(real_settings)
    report = brain.index_path(notes)
    assert report.files_indexed == 2 and not report.errors
    assert brain.search("sourdough flour", k=1, scope="docs")[0].source == "bread.md"
    memory_id = brain.remember("The spare key is under the blue pot")
    brain.archive(memory_id)
    assert brain.search("spare key", k=3, scope="memories") == []
    brain.restore(memory_id)
    assert brain.search("spare key", k=1, scope="memories")[0].id == memory_id
    assert brain.replay() == 1
    status = brain.status()
    assert status["collections"] == {"docs": 2, "memories": 1}


def test_mcp_server_over_real_stdio(real_settings, cleanup, tmp_path):
    """Starts `cerveau-mcp` as an AI client would, and talks MCP to it."""
    import asyncio
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env_file = tmp_path / "mcp.env"
    env_file.write_text(
        "\n".join(
            [
                f"CERVEAU_QDRANT_URL={URL}",
                f"CERVEAU_QDRANT_API_KEY={KEY}",
                "CERVEAU_EMBEDDER=hash",
                "CERVEAU_VECTOR_SIZE=64",
                f"CERVEAU_PREFIX={real_settings.prefix}",
                f"CERVEAU_DATA_DIR={real_settings.data_dir}",
            ]
        ),
        encoding="utf-8",
    )
    Brain(real_settings).init()
    env = {k: v for k, v in os.environ.items() if not k.startswith("CERVEAU_")}
    env["CERVEAU_ENV_FILE"] = str(env_file)

    async def talk() -> tuple[set[str], str, str]:
        params = StdioServerParameters(command=sys.executable, args=["-m", "cerveau.mcp_server"], env=env)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            await session.call_tool("remember", {"text": "The lighthouse key is in the red box"})
            found = await session.call_tool("search_memory", {"query": "lighthouse key", "k": 1})
            refused = await session.call_tool("remember", {"text": "x" * 10_000})
            return tools, found.content[0].text, refused.content[0].text

    tools, found, refused = asyncio.run(talk())
    assert tools == {"search_memory", "brain_status", "remember"}
    assert "red box" in found
    assert "split" in refused
