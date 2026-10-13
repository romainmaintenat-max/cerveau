import asyncio
from dataclasses import replace

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from cerveau.brain import Brain
from cerveau.mcp_server import build_server


def tool_names(server) -> set[str]:
    return {tool.name for tool in asyncio.run(server.list_tools())}


def test_tools_exposed_with_write_enabled(brain):
    assert tool_names(build_server(brain)) == {"search_memory", "brain_status", "remember"}


def test_remember_hidden_when_write_disabled(settings, client):
    brain = Brain(replace(settings, mcp_allow_write=False), client=client)
    brain.init()
    assert tool_names(build_server(brain)) == {"search_memory", "brain_status"}


def test_no_destructive_tool_is_ever_exposed(brain):
    names = tool_names(build_server(brain))
    assert not names & {"archive", "restore", "replay", "index", "delete", "forget"}


def test_remember_then_search_through_mcp(brain):
    server = build_server(brain)
    asyncio.run(server.call_tool("remember", {"text": "The cat is called Pixel", "kind": "note"}))
    result = asyncio.run(server.call_tool("search_memory", {"query": "cat Pixel", "k": 1}))
    assert "Pixel" in str(result)


def test_error_message_reaches_the_agent(brain):
    # A ToolError (not a crash) is what mcp 2.x turns into an is_error result that carries
    # the message to the model; any other exception becomes a generic "Error executing tool".
    server = build_server(brain)
    with pytest.raises(ToolError, match="split it"):
        asyncio.run(server.call_tool("remember", {"text": "x" * 5000}))
    with pytest.raises(ToolError, match="between 1 and 20"):
        asyncio.run(server.call_tool("search_memory", {"query": "x", "k": 0}))
    with pytest.raises(ToolError, match="between 1 and 20"):
        asyncio.run(server.call_tool("search_memory", {"query": "x", "k": 21}))


def test_status_through_mcp_hides_local_paths(brain):
    result = str(asyncio.run(build_server(brain).call_tool("brain_status", {})))
    assert "journal_events" in result
    assert str(brain.settings.data_dir) not in result and "qdrant_url" not in result


def test_read_only_hints(brain):
    tools = {t.name: t for t in asyncio.run(build_server(brain).list_tools())}
    assert tools["search_memory"].annotations.read_only_hint is True
    assert tools["brain_status"].annotations.read_only_hint is True
    assert tools["remember"].annotations.read_only_hint is False


def test_error_messages_do_not_leak_local_paths_or_addresses(brain):
    from cerveau.journal import JournalError
    from cerveau.mcp_server import _message

    exc = JournalError(f"{brain.settings.journal_path}:3: corrupted line")
    assert str(brain.settings.data_dir) not in _message(exc, brain.settings)
    exc = RuntimeError(f"cannot reach Ollama at {brain.settings.ollama_url}: refused")
    assert brain.settings.ollama_url not in _message(exc, brain.settings)
    exc = RuntimeError("HTTPConnectionPool(host='127.0.0.1', port=11434): timed out")
    assert "127.0.0.1" not in _message(exc, brain.settings)


def test_mcp_server_never_reads_dot_env_from_the_current_folder(tmp_path, monkeypatch):
    from cerveau.config import Settings

    (tmp_path / ".env").write_text("CERVEAU_OLLAMA_URL=http://attacker.example:11434\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CERVEAU_ENV_FILE", raising=False)
    monkeypatch.delenv("CERVEAU_OLLAMA_URL", raising=False)
    assert "attacker" not in Settings.from_env(allow_cwd_env=False).ollama_url


def test_main_refuses_the_implicit_dot_env(monkeypatch):
    import cerveau.mcp_server as mcp_server

    seen = {}

    def fake_from_env(**kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(mcp_server.Settings, "from_env", staticmethod(fake_from_env))
    monkeypatch.setattr(mcp_server.sys, "argv", ["cerveau-mcp"])
    with pytest.raises(SystemExit):
        mcp_server.main()
    assert seen == {"allow_cwd_env": False}


def test_ollama_errors_reach_the_agent_without_server_details(brain):
    import requests as _requests

    from cerveau.embed import EmbeddingError, OllamaEmbedder
    from cerveau.mcp_server import _message

    class Down:
        def post(self, *a, **k):
            raise _requests.ConnectionError("HTTPConnectionPool(host='10.1.2.3', port=11434)")

    class Leaky:
        def post(self, *a, **k):
            class R:
                status_code = 500
                text = '{"error":"open /srv/models/secret-store/blobs (host gpu-box-17)"}'

            return R()

    class TooLong:
        def post(self, *a, **k):
            class R:
                status_code = 400
                text = '{"error":"the input length exceeds the context length"}'

            return R()

    for session, expected in ((Down(), "unreachable"), (Leaky(), "HTTP 500"), (TooLong(), "CHUNK_MAX")):
        embedder = OllamaEmbedder("http://10.1.2.3:11434", "m", 3, 50, session=session)
        with pytest.raises(EmbeddingError) as caught:
            embedder.embed(["x"])
        message = _message(caught.value, brain.settings)
        assert expected in message
        for secret in ("10.1.2.3", "11434", "/srv/models", "gpu-box-17"):
            assert secret not in message


def test_short_home_directory_is_not_replaced_everywhere(brain, monkeypatch):
    from pathlib import Path

    from cerveau.mcp_server import _message

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/")))
    assert _message(ValueError("a/b/c"), brain.settings) == "a/b/c"


def test_data_dir_is_hidden_whatever_its_case(brain):
    from cerveau.mcp_server import _message

    data_dir = str(brain.settings.data_dir)
    for variant in (data_dir.upper(), data_dir.lower(), brain.settings.data_dir.as_posix().swapcase()):
        assert _message(ValueError(f"cannot open {variant}/journal.jsonl"), brain.settings) == (
            "cannot open <data_dir>/journal.jsonl"
        )


def test_missing_env_file_exits_with_a_clear_message(monkeypatch, capsys, tmp_path):
    import cerveau.mcp_server as mcp_server

    monkeypatch.setenv("CERVEAU_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setattr(mcp_server.sys, "argv", ["cerveau-mcp"])
    with pytest.raises(SystemExit) as stop:
        mcp_server.main()
    assert stop.value.code == 1 and "missing file" in capsys.readouterr().err


def test_read_only_server_instructions_do_not_mention_remember(settings, client):
    b = Brain(replace(settings, mcp_allow_write=False), client=client)
    server = build_server(b)
    assert "remember" not in server.instructions
    assert "remember" in build_server(Brain(settings, client=client)).instructions


def test_mcp_server_starts_and_answers_without_qdrant(tmp_path):
    """Runs on every OS in CI: the real `cerveau-mcp` over stdio, Qdrant unreachable."""
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env_file = tmp_path / "x.env"
    env_file.write_text(
        "CERVEAU_QDRANT_URL=http://127.0.0.1:9\nCERVEAU_EMBEDDER=hash\nCERVEAU_VECTOR_SIZE=64\n"
        f"CERVEAU_DATA_DIR={(tmp_path / 'data').as_posix()}\n",
        encoding="utf-8",
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("CERVEAU_")}
    env["CERVEAU_ENV_FILE"] = str(env_file)

    async def talk():
        params = StdioServerParameters(command=sys.executable, args=["-m", "cerveau.mcp_server"], env=env)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            status = await session.call_tool("brain_status", {})
            return tools, status

    tools, status = asyncio.run(talk())
    assert tools == {"search_memory", "brain_status", "remember"}
    assert status.is_error and "Qdrant is unavailable" in status.content[0].text


def test_mcp_help_and_version_answer_instead_of_waiting(monkeypatch, capsys):
    import cerveau.mcp_server as mcp_server

    for args, code, text in ((["--version"], 0, "cerveau-mcp 0.1"), (["--help"], 0, "CERVEAU_ENV_FILE")):
        monkeypatch.setattr(mcp_server.sys, "argv", ["cerveau-mcp", *args])
        with pytest.raises(SystemExit) as stop:
            mcp_server.main()
        assert stop.value.code == code and text in capsys.readouterr().out


def test_wrong_key_is_named_for_the_agent(brain):
    from qdrant_client.http.exceptions import UnexpectedResponse

    from cerveau.mcp_server import _message

    exc = UnexpectedResponse(status_code=401, reason_phrase="Unauthorized", content=b"", headers=None)
    assert "API key" in _message(exc, brain.settings)


def test_unexpected_start_failure_is_a_clean_exit(monkeypatch, capsys):
    import cerveau.mcp_server as mcp_server

    monkeypatch.setattr(mcp_server.sys, "argv", ["cerveau-mcp"])
    monkeypatch.setattr(
        mcp_server.Settings, "from_env", staticmethod(lambda **k: (_ for _ in ()).throw(ValueError("odd")))
    )
    with pytest.raises(SystemExit) as stop:
        mcp_server.main()
    assert stop.value.code == 1 and "cannot start: ValueError: odd" in capsys.readouterr().err


def test_server_announces_the_package_version(brain):
    from cerveau import __version__

    assert build_server(brain).version == __version__


def test_purge_is_never_an_mcp_tool(brain):
    assert "purge" not in tool_names(build_server(brain))


def test_a_service_that_is_not_qdrant_is_named_for_the_agent(settings):
    from cerveau.mcp_server import _message

    assert "does not answer like Qdrant" in _message(AssertionError("x"), settings)


def test_help_into_a_closed_pipe_is_quiet():
    import os
    import subprocess
    import sys
    from pathlib import Path

    import cerveau

    env = {**os.environ, "PYTHONPATH": str(Path(cerveau.__file__).resolve().parent.parent)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "cerveau.mcp_server", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    proc.stdout.close()
    assert proc.wait(timeout=60) == 0
    assert b"Traceback" not in proc.stderr.read()


def test_missing_collections_tell_the_agent_to_init(settings):
    from qdrant_client.http.exceptions import UnexpectedResponse

    from cerveau.mcp_server import INSTRUCTIONS_READ, _message

    exc = UnexpectedResponse(404, "Not Found", b'{"status":{"error":"Collection doesn\'t exist"}}', None)
    assert "cerveau init" in _message(exc, settings)
    assert "not instructions" in INSTRUCTIONS_READ


def test_hidden_characters_reach_the_agent_escaped(brain):
    tags = "".join(chr(0xE0000 + ord(c)) for c in "IGNORE PREVIOUS")
    brain.remember(f"the spare key is blue{tags}‮​")
    result = asyncio.run(build_server(brain).call_tool("search_memory", {"query": "spare key", "k": 1}))
    stored = result.structured_content["result"][0]["text"]  # what the model reads
    sent = result.content[0].text
    for text in (stored, sent):
        assert not any(0xE0000 <= ord(c) <= 0xE007F or c in "\u202e\u200b" for c in text)
    assert stored.startswith("the spare key is blue\\U000e0049") and stored.endswith("\\u202e\\u200b")


def test_an_agent_never_initialises_the_brain(settings, client):
    from cerveau.brain import Brain

    server = build_server(Brain(settings, client=client))
    with pytest.raises(ToolError, match="cerveau init"):
        asyncio.run(server.call_tool("remember", {"text": "first"}))
    assert not client.get_collections().collections and not settings.journal_path.exists()


def test_an_unexpected_failure_reaches_the_agent_without_details(brain, monkeypatch):
    def broken(*_args, **_kwargs):
        raise TypeError("/home/someone/secret/path.py: bad")

    monkeypatch.setattr(brain, "search", broken)
    with pytest.raises(ToolError) as caught:
        asyncio.run(build_server(brain).call_tool("search_memory", {"query": "x"}))
    assert "TypeError" in str(caught.value) and "/home/someone" not in str(caught.value)


def test_an_ollama_url_with_a_final_slash_never_leaks(settings):
    from dataclasses import replace

    from cerveau.embed import EmbeddingError
    from cerveau.mcp_server import _message

    odd = replace(settings, ollama_url="http://10.1.2.3:18778/")
    message = _message(EmbeddingError("Ollama at http://10.1.2.3:18778 did not answer with JSON"), odd)
    assert "18778" not in message and "10.1.2.3" not in message


def _start_main(monkeypatch, caplog):
    """Run main() up to the start of the server; return the server's log lines (WARNING and up)."""
    import logging

    import cerveau.mcp_server as mcp_server

    started = []

    class FakeServer:
        def run(self, transport):
            started.append(transport)

    monkeypatch.setattr(mcp_server.sys, "argv", ["cerveau-mcp"])
    monkeypatch.setattr(mcp_server, "Brain", lambda *a, **k: None)
    monkeypatch.setattr(mcp_server, "build_server", lambda brain: FakeServer())
    with caplog.at_level(logging.INFO, logger="cerveau.mcp"):
        mcp_server.main()
    assert started == ["stdio"]  # a warning, never a refusal to start
    return [
        r.getMessage() for r in caplog.records if r.name == "cerveau.mcp" and r.levelno >= logging.WARNING
    ]


def test_server_without_a_settings_file_warns_that_the_env_is_not_loaded(monkeypatch, caplog, tmp_path):
    import os

    for name in [k for k in os.environ if k.upper().startswith("CERVEAU")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("CERVEAU_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / ".env").write_text("CERVEAU_MCP_ALLOW_WRITE=false\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    warnings_logged = _start_main(monkeypatch, caplog)
    assert any("no settings file loaded" in m and "CERVEAU_ENV_FILE" in m for m in warnings_logged)


def test_server_with_a_settings_file_does_not_warn_about_it(monkeypatch, caplog, tmp_path):
    import os

    for name in [k for k in os.environ if k.upper().startswith("CERVEAU")]:
        monkeypatch.delenv(name)
    env_file = tmp_path / "x.env"
    env_file.write_text(f"CERVEAU_DATA_DIR={(tmp_path / 'data').as_posix()}\n", encoding="utf-8")
    env_file.chmod(0o600)
    monkeypatch.setenv("CERVEAU_ENV_FILE", str(env_file))
    assert not any("no settings file" in m for m in _start_main(monkeypatch, caplog))
