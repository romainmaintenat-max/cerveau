"""The local log file: on by default, off on request, never a text, never in the way."""

import asyncio
import logging
import os
from dataclasses import replace
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from cerveau import logfile
from cerveau.brain import Brain
from cerveau.cli import run
from cerveau.mcp_server import build_server

MEMORY = "SENTINEL-MEMORY-7f3a"
QUERY = "SENTINEL-QUERY-91bc"
DOCUMENT = "SENTINEL-DOCUMENT-c2d4"
KEY = "SENTINEL-KEY-0123456789abcdef"


def log_text(data_dir: Path) -> str:
    """Every log file of the data folder, rotated ones included."""
    return "".join(p.read_text(encoding="utf-8") for p in sorted((data_dir / "logs").glob("*.log*")))


@pytest.fixture(autouse=True)
def no_log_left_open():
    yield
    logfile.stop()


def test_on_by_default_with_a_start_line_and_one_line_per_command(brain, capsys):
    assert run(["status"], brain=brain) == 0
    text = log_text(brain.settings.data_dir)
    assert (brain.settings.data_dir / "logs" / "cli.log").is_file()
    assert "start: cerveau 0.1.0, Python " in text and ", cli status" in text
    assert "INFO status: ok in " in text


def test_off_creates_nothing(settings, client, notes, capsys):
    brain = Brain(replace(settings, log=False), client=client)
    for command in (
        ["init"],
        ["index", str(notes)],
        ["remember", MEMORY],
        ["search", QUERY],
        ["archive", "x"],
    ):
        run(command, brain=brain)
    assert brain.settings.data_dir.is_dir() and not (brain.settings.data_dir / "logs").exists()


def test_no_text_query_document_or_file_name_ever_reaches_the_log(brain, tmp_path, capsys):
    folder = tmp_path / f"notes-{DOCUMENT}"
    folder.mkdir()
    (folder / f"{DOCUMENT}.md").write_text(f"# {DOCUMENT}\n\nThe {DOCUMENT} says hello.\n", encoding="utf-8")
    assert run(["index", str(folder)], brain=brain) == 0
    assert run(["remember", f"{MEMORY} is the colour of the door"], brain=brain) == 0
    assert run(["search", QUERY, "the door"], brain=brain) == 0
    assert run(["search", QUERY, "--json"], brain=brain) == 0
    assert run(["list"], brain=brain) == 0
    assert run(["remember", f"{MEMORY} " * 500], brain=brain) == 2  # too long: an error
    assert run(["archive", MEMORY], brain=brain) == 2  # not an id: the message quotes it
    assert run(["index", str(tmp_path / f"missing-{DOCUMENT}")], brain=brain) == 2
    capsys.readouterr()
    text = log_text(brain.settings.data_dir)
    assert "SENTINEL" not in text
    assert "index: ok in" in text and "files_indexed 1" in text and "chunks_written 1" in text
    assert "search: ok in" in text and "results " in text
    assert "remember: BrainError in" in text and "archive: BrainError in" in text
    assert "Traceback" not in text and 'File "' in text  # frames, not the message


def test_api_key_never_and_paths_masked(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "brain"
    data_dir.mkdir()
    env_file = data_dir / "settings.env"
    env_file.write_text(
        f"CERVEAU_QDRANT_API_KEY={KEY}\nCERVEAU_QDRANT_URL=http://127.0.0.1:9\n"
        f"CERVEAU_OLLAMA_URL=http://10.255.255.1:11434\nCERVEAU_DATA_DIR={data_dir.as_posix()}\n",
        encoding="utf-8",
    )
    if os.name == "posix":
        env_file.chmod(0o644)  # a warning naming the file
    for name in [k for k in os.environ if k.upper().startswith("CERVEAU")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("CERVEAU_ENV_FILE", str(env_file))
    with pytest.warns(UserWarning) as shown:
        assert run(["status"]) == 2  # Qdrant unreachable
    assert any("CERVEAU_OLLAMA_URL points to 10.255.255.1" in str(w.message) for w in shown)  # as before
    text = log_text(data_dir)
    assert KEY not in text and "10.255.255.1" not in text and str(data_dir) not in text
    assert "WARNING CERVEAU_OLLAMA_URL points to <ollama>" in text
    if os.name == "posix":
        assert "<data_dir>/settings.env can be read by other users" in text
    assert "status: ResponseHandlingException in" in text


def test_the_folder_of_the_settings_file_is_masked(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")  # outside the home folder
    folder = tmp_path / "clients" / "dupont"  # outside the data folder too
    folder.mkdir(parents=True)
    env_file = folder / "secret-projet.env"
    env_file.write_text(
        f"CERVEAU_QDRANT_URL=http://127.0.0.1:9\nCERVEAU_DATA_DIR={(tmp_path / 'data').as_posix()}\n"
        "CERVEAU_TOKEN=x\n",  # another program's: a warning that names the file
        encoding="utf-8",
    )
    env_file.chmod(0o600)
    for name in [k for k in os.environ if k.upper().startswith("CERVEAU")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("CERVEAU_ENV_FILE", str(env_file))
    with pytest.warns(UserWarning) as shown:
        assert run(["status"]) == 2  # Qdrant unreachable
    assert any(f"{env_file}: CERVEAU_TOKEN is ignored" in str(w.message) for w in shown)  # console: as before
    text = log_text(tmp_path / "data")
    assert f"WARNING <env_dir>{os.sep}secret-projet.env: CERVEAU_TOKEN is ignored" in text
    assert str(folder) not in text and folder.as_posix() not in text and "dupont" not in text


def test_a_broken_log_never_breaks_a_command(brain, capsys, monkeypatch):
    logs = brain.settings.data_dir / "logs"
    logs.write_text("a file where the folder should be")  # cannot be created
    assert run(["remember", "first"], brain=brain) == 0
    assert capsys.readouterr().err == ""
    logs.unlink()

    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(logfile._Handler, "shouldRollover", broken)  # every write fails
    assert run(["remember", "second"], brain=brain) == 0
    assert run(["archive", "nope"], brain=brain) == 2  # errors too
    out = capsys.readouterr()
    assert "error: 'nope' is not a memory id" in out.err and "disk full" not in out.err + out.out


def test_output_is_the_same_with_or_without_the_log(settings, client, notes, capsys):
    outputs = []
    for on in (True, False):
        brain = Brain(replace(settings, log=on, data_dir=settings.data_dir / str(on)), client=client)
        run(["init"], brain=brain)
        capsys.readouterr()
        for command in (["index", str(notes)], ["search", "bread", "-k", "1"], ["archive", "x"]):
            run(command, brain=brain)
        out = capsys.readouterr()
        outputs.append((out.out, out.err))
        client.delete_collection(brain.store.docs)
        for name in (brain.store.memories, brain.store.meta):
            client.delete_collection(name)
    assert outputs[0] == outputs[1]


def test_rotation_and_private_files(settings):
    logfile.start(settings, "cli", "status")
    handler = next(h for h in logfile.log.handlers if isinstance(h, logfile._Handler))
    assert handler.maxBytes == 5_000_000 and handler.backupCount == 3
    assert Path(handler.baseFilename) == settings.data_dir / "logs" / "cli.log"
    assert not logfile.log.propagate  # never to stderr, never to stdout
    if os.name == "posix":
        assert (settings.data_dir / "logs").stat().st_mode & 0o777 == 0o700
        assert Path(handler.baseFilename).stat().st_mode & 0o777 == 0o600
        handler.doRollover()
        logfile.log.info("after rotation")
        assert Path(handler.baseFilename).stat().st_mode & 0o777 == 0o600
    logfile.stop()
    assert not any(isinstance(h, logfile._Handler) for h in logfile.log.handlers)


def test_mcp_tool_calls_are_logged_without_the_query(brain, monkeypatch):
    brain.remember(f"{MEMORY} lives here")
    logfile.start(brain.settings, "mcp", "server read-write")
    server = build_server(brain)
    asyncio.run(server.call_tool("search_memory", {"query": f"{QUERY} lives", "k": 1}))
    asyncio.run(server.call_tool("brain_status", {}))
    asyncio.run(server.call_tool("remember", {"text": f"{MEMORY} again"}))
    with pytest.raises(ToolError):
        asyncio.run(server.call_tool("remember", {"text": f"{MEMORY} " * 500}))

    message = "/home/someone/" + QUERY + ".py: bad"  # not on the line that raises (the log shows it)

    def broken(*_args, **_kwargs):
        raise TypeError(message)

    monkeypatch.setattr(brain, "search", broken)
    with pytest.raises(ToolError):
        asyncio.run(server.call_tool("search_memory", {"query": QUERY}))
    logfile.stop()
    text = (brain.settings.data_dir / "logs" / "mcp.log").read_text(encoding="utf-8")
    assert "SENTINEL" not in text and "/home/someone" not in text
    assert "mcp server read-write" in text
    assert "search_memory: ok in" in text and "results 1" in text
    assert "brain_status: ok in" in text and "remember: ok in" in text
    assert "remember: BrainError in" in text and "search_memory: TypeError in" in text


def test_mcp_server_over_stdio_logs_to_its_file_only(tmp_path):
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
            await session.initialize()  # stdout is still a clean protocol stream
            return await session.call_tool("search_memory", {"query": QUERY})

    assert asyncio.run(talk()).is_error  # Qdrant unreachable
    text = log_text(tmp_path / "data")
    assert "mcp server read-write" in text and "search_memory: ResponseHandlingException in" in text
    assert "WARNING CERVEAU_EMBEDDER=hash" in text and "SENTINEL" not in text
    assert not list((tmp_path / "data" / "logs").glob("cli.log*"))


def test_no_settings_no_log(tmp_path, monkeypatch, capsys):
    for name in [k for k in os.environ if k.upper().startswith("CERVEAU")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("CERVEAU_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CERVEAU_VECTOR_SIZE", "abc")  # a configuration error
    monkeypatch.chdir(tmp_path)
    assert run(["status"]) == 2
    assert "CERVEAU_VECTOR_SIZE" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()


def test_the_log_never_writes_to_the_console(brain, capsys, caplog):
    with caplog.at_level(logging.DEBUG):
        assert run(["archive", "nope"], brain=brain) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err and 'File "' not in err
