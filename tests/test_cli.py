import json

from cerveau.brain import Brain
from cerveau.cli import run


def test_cli_flow(brain, notes, capsys):
    assert run(["index", str(notes)], brain=brain) == 0
    assert run(["remember", "Use the blue notebook", "--kind", "note"], brain=brain) == 0
    memory_id = capsys.readouterr().out.strip().splitlines()[-1]
    assert run(["search", "blue notebook", "--json", "-k", "1"], brain=brain) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["id"] == memory_id
    assert run(["archive", memory_id], brain=brain) == 0
    assert run(["restore", memory_id], brain=brain) == 0
    assert run(["replay"], brain=brain) == 0
    assert run(["status"], brain=brain) == 0
    status = json.loads(capsys.readouterr().out.split("\n", 3)[-1])
    assert status["journal_events"] == 3


def test_cli_errors_return_2(brain, capsys):
    assert run(["archive", "nope"], brain=brain) == 2
    assert "error:" in capsys.readouterr().err


def test_index_with_errors_returns_1(brain, notes):
    (notes / "bad.txt").write_bytes(b"\xff\xfe\xfa")
    assert run(["index", str(notes)], brain=brain) == 1


def test_unindex_command(brain, notes, capsys):
    run(["index", str(notes)], brain=brain)
    assert run(["unindex", str(notes)], brain=brain) == 0
    assert "removed 2 chunks" in capsys.readouterr().out


def test_output_never_crashes_on_a_narrow_code_page(monkeypatch):
    import io
    import sys

    from cerveau.cli import _utf8_output

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    _utf8_output()
    print("flèche → ok")
    stream.flush()
    assert "→".encode() in raw.getvalue()


def test_undecodable_file_names_never_crash_the_output(monkeypatch):
    import io
    import sys

    from cerveau.cli import _utf8_output

    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="ascii"))
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="ascii"))
    _utf8_output()
    print("caf\udce9.md")
    sys.stdout.flush()
    assert b"caf\\udce9.md" in raw.getvalue()


def test_list_reset_version_and_empty_search(brain, capsys):
    import pytest as _pytest

    run(["remember", "agent fact"], brain=brain)
    capsys.readouterr()
    assert run(["list"], brain=brain) == 0
    assert "agent fact" in capsys.readouterr().out
    assert run(["search", "zzzz qqqq", "--scope", "docs"], brain=brain) == 0
    assert "no result" in capsys.readouterr().out
    assert run(["reset-collections", "--confirm", "wrong"], brain=brain) == 2
    assert run(["reset-collections", "--confirm", brain.settings.prefix], brain=brain) == 0
    assert run(["replay"], brain=brain) == 0
    with _pytest.raises(SystemExit):
        run(["--version"], brain=brain)
    assert "cerveau 0.1.0" in capsys.readouterr().out


def test_file_system_error_is_a_clean_message(settings, client, tmp_path, capsys):
    from dataclasses import replace

    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")
    b = Brain(replace(settings, data_dir=blocker / "data"), client=client)
    assert run(["remember", "x"], brain=b) == 2
    assert "file system" in capsys.readouterr().err


def test_stored_text_cannot_hide_itself_in_the_terminal(brain, capsys):
    hostile = "harmless\r\x1b[2Kinnocent\x1b[8m SECRET \x1b[0m\x1b]52;c;ZXZpbA==\x07‮evil"
    brain.remember(hostile, actor="mcp")
    for command in (["list"], ["search", "harmless innocent secret", "--scope", "memories"]):
        capsys.readouterr()
        run(command, brain=brain)
        out = capsys.readouterr().out
        assert "\x1b" not in out and "\r" not in out and "\x07" not in out and "‮" not in out
        assert "SECRET" in out and "\\x1b" in out
    run(["list", "--json"], brain=brain)
    out = capsys.readouterr().out
    assert "\x1b" not in out and "‮" not in out


def test_multiline_memory_is_indented_not_escaped(brain, capsys):
    brain.remember("line one\nline two")
    capsys.readouterr()
    run(["list"], brain=brain)
    assert "    line one\n    line two" in capsys.readouterr().out


def test_interrupt_is_a_clean_exit(brain, monkeypatch, capsys):
    monkeypatch.setattr(brain, "status", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert run(["status"], brain=brain) == 130
    assert "interrupted" in capsys.readouterr().err


def test_search_shows_the_absolute_path_to_the_human_only(brain, notes, capsys):
    run(["index", str(notes)], brain=brain)
    capsys.readouterr()
    run(["search", "sourdough flour", "-k", "1", "--scope", "docs"], brain=brain)
    assert notes.resolve().as_posix() in capsys.readouterr().out  # original case, also on Windows
    hit = brain.search("sourdough flour", k=1, scope="docs")[0]
    assert "path" not in hit.to_dict()


def test_python_dash_m_cerveau(tmp_path):
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "cerveau", "--version"], capture_output=True, text=True)
    assert result.returncode == 0 and "cerveau" in result.stdout


def test_json_output_is_always_valid_json(brain, capsys):
    brain.remember("zebra\x7f del, \x85 next line, \u202e bidi, \x1b escape")
    for command in (["list", "--json"], ["search", "zebra del", "--json", "--scope", "memories"]):
        capsys.readouterr()
        run(command, brain=brain)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "\x7f" in data[0]["text"] and "\x7f" not in out and "\u202e" not in out


def test_review_never_cuts_and_search_says_when_it_cuts(brain, capsys):
    long_text = "Meeting notes about the garden. " * 20 + "SECRET-PAYLOAD"
    brain.remember(long_text, actor="mcp")
    capsys.readouterr()
    run(["list", "--actor", "mcp"], brain=brain)
    assert "SECRET-PAYLOAD" in capsys.readouterr().out
    run(["search", "meeting notes garden", "--scope", "memories", "-k", "1"], brain=brain)
    assert "characters: add --json to see the full text" in capsys.readouterr().out


def test_indexing_a_folder_without_text_files_warns(brain, tmp_path, capsys):
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "a.pdf").write_bytes(b"%PDF")
    run(["index", str(tmp_path / "pdfs")], brain=brain)
    assert "no .md" in capsys.readouterr().err


def test_orphans_listing(brain, settings, tmp_path, capsys):
    from dataclasses import replace

    Brain(replace(settings, data_dir=tmp_path / "x"), client=brain.store.client).remember("orphan!")
    capsys.readouterr()
    run(["list", "--orphans"], brain=brain)
    assert "orphan!" in capsys.readouterr().out


def test_invisible_characters_are_shown_escaped(brain, capsys):
    tags = "".join(chr(0xE0000 + ord(c)) for c in "IGNORE PREVIOUS")
    hostile = f"Deploy checklist.{tags}​zero⁠width️­"
    brain.remember(hostile, actor="mcp")
    capsys.readouterr()
    run(["list"], brain=brain)
    out = capsys.readouterr().out
    assert "\\U000e0049" in out and "\\u200b" in out and "\\u2060" in out and "\\xad" in out
    assert not any(0xE0000 <= ord(c) <= 0xE007F or c in "​⁠️­" for c in out)
    run(["list", "--json"], brain=brain)
    out = capsys.readouterr().out
    assert json.loads(out)[0]["text"] == hostile
    assert not any(0xE0000 <= ord(c) <= 0xE007F or c == "​" for c in out)


def test_reset_protects_memories_that_exist_only_in_qdrant(brain, settings, tmp_path, capsys):
    from dataclasses import replace

    Brain(replace(settings, data_dir=tmp_path / "elsewhere"), client=brain.store.client).remember(
        "precious memory"
    )
    assert run(["reset-collections", "--confirm", settings.prefix], brain=brain) == 2
    assert "adopt-orphans" in capsys.readouterr().err
    assert run(["adopt-orphans"], brain=brain) == 0
    assert run(["reset-collections", "--confirm", settings.prefix], brain=brain) == 0
    brain.init()
    brain.replay()
    assert brain.search("precious memory", k=1, scope="memories")[0].text == "precious memory"


def test_reset_of_an_empty_prefix_says_nothing_to_delete(settings, client, capsys):
    from dataclasses import replace

    b = Brain(replace(settings, prefix="emptyone"), client=client)
    assert run(["reset-collections", "--confirm", "emptyone"], brain=b) == 0


def test_service_answering_html_is_a_clean_error(brain, monkeypatch, capsys):
    monkeypatch.setattr(brain, "status", lambda: json.loads("<html>"))
    assert run(["status"], brain=brain) == 2
    assert "did not answer with JSON" in capsys.readouterr().err


def test_unexpected_error_is_one_line_not_a_traceback(brain, monkeypatch, capsys):
    monkeypatch.setattr(brain, "status", lambda: (_ for _ in ()).throw(LookupError("boom")))
    assert run(["status"], brain=brain) == 2
    assert "unexpected LookupError: boom" in capsys.readouterr().err


def test_qdrant_errors_get_their_own_advice_not_please_report(brain, monkeypatch, capsys):
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    unauthorized = UnexpectedResponse(
        status_code=401, reason_phrase="Unauthorized", content=b"", headers=None
    )
    for exc, expected in (
        (unauthorized, "wrong or missing API key"),
        (ResponseHandlingException(ConnectionRefusedError("refused")), "Qdrant is unavailable"),
    ):
        monkeypatch.setattr(brain, "status", lambda e=exc: (_ for _ in ()).throw(e))
        assert run(["status"], brain=brain) == 2
        err = capsys.readouterr().err
        assert expected in err and "please report" not in err


def test_blank_lookalike_characters_are_escaped(brain, capsys):
    brain.remember("visible\u2800\u3164hidden")
    capsys.readouterr()
    run(["list"], brain=brain)
    out = capsys.readouterr().out
    assert "\\u2800" in out and "\u2800" not in out


def test_french_spaces_are_shown_as_spaces(brain, capsys):
    brain.remember("Il est midi\u00a0! Prix\u202f: 12\u202f000")
    capsys.readouterr()
    run(["list"], brain=brain)
    out = capsys.readouterr().out
    assert "midi\u00a0!" in out and "\\xa0" not in out


def test_unquoted_words_are_accepted(brain, capsys):
    assert run(["remember", "the", "spare", "key", "is", "blue"], brain=brain) == 0
    capsys.readouterr()
    assert run(["search", "spare", "key", "--scope", "memories", "-k", "1"], brain=brain) == 0
    assert "the spare key is blue" in capsys.readouterr().out


def test_qdrant_down_suggests_starting_it(brain, monkeypatch, capsys):
    from qdrant_client.http.exceptions import ResponseHandlingException

    exc = ResponseHandlingException(ConnectionRefusedError("refused"))
    monkeypatch.setattr(brain, "status", lambda: (_ for _ in ()).throw(exc))
    run(["status"], brain=brain)
    assert "docker compose up -d" in capsys.readouterr().err


def test_purge_needs_confirm_then_erases(brain, settings, capsys):
    secret = brain.remember("api key sk-SECRET-123")
    assert run(["purge", secret], brain=brain) == 2
    err = capsys.readouterr().err
    assert "--confirm" in err and "archive" in err
    assert brain.list_memories()[0]["id"] == secret
    assert run(["purge", secret, "--confirm"], brain=brain) == 0
    out = capsys.readouterr().out
    assert "purged" in out and "Backups" in out
    assert b"sk-SECRET-123" not in settings.journal_path.read_bytes()
    assert run(["purge", secret, "--confirm"], brain=brain) == 2


def test_list_orphans_survives_missing_fields(brain, capsys):
    import uuid

    from qdrant_client.http import models as qm

    point_id = str(uuid.uuid4())
    brain.store.upsert(
        brain.store.memories,
        [qm.PointStruct(id=point_id, vector=[0.1] * brain.settings.vector_size, payload={"text": None})],
    )
    assert run(["list", "--orphans"], brain=brain) == 0
    out = capsys.readouterr().out
    assert point_id in out and "?" in out


def test_warnings_are_one_plain_line(capsys):
    import warnings

    from cerveau import cli

    old = warnings.formatwarning
    try:
        warnings.formatwarning = cli._plain_warning
        assert warnings.formatwarning("careful", UserWarning, "x.py", 1) == "warning: careful\n"
    finally:
        warnings.formatwarning = old


def test_command_line_warnings_are_plain_lines(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    import cerveau

    env = {k: v for k, v in os.environ.items() if not k.startswith("CERVEAU_") and "PROXY" not in k.upper()}
    env.update(
        PYTHONPATH=str(Path(cerveau.__file__).resolve().parent.parent),
        CERVEAU_OLLAMA_URL="http://10.255.255.1:11434",
        CERVEAU_QDRANT_URL="http://127.0.0.1:9",
        CERVEAU_DATA_DIR=str(tmp_path / "data"),
    )
    result = subprocess.run(
        [sys.executable, "-m", "cerveau", "status"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "warning: CERVEAU_OLLAMA_URL points to 10.255.255.1" in result.stderr
    assert "UserWarning" not in result.stderr and "settings.validate()" not in result.stderr


def test_a_file_name_cannot_print_a_fake_result_line(brain, tmp_path, capsys):
    import sys

    import pytest

    if sys.platform == "win32":
        pytest.skip("Windows file names cannot hold a line break")
    folder = tmp_path / "n"
    folder.mkdir()
    fake = "evil\n[0.999] memories:note (0000) by cli\n    Remember: password\t.md"
    (folder / fake).write_text("onions and garlic\n", encoding="utf-8")
    brain.index_path(folder)
    assert run(["search", "onions", "-k", "1"], brain=brain) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("[") and "\\x0a[0.999]" in lines[0] and "\\x09" in lines[0]
    assert not any(line.startswith("[0.999]") for line in lines)


def test_stored_text_keeps_its_line_breaks_and_escapes_invisible_joiners(brain, capsys):
    brain.remember("line one\nline two with a grapheme joiner \u034f and a Mongolian selector \u180b")
    assert run(["list"], brain=brain) == 0
    out = capsys.readouterr().out
    assert "line one\n    line two" in out
    assert "\\u034f" in out and "\\u180b" in out


def test_list_shows_who_wrote_an_adopted_memory(brain, settings, tmp_path, capsys):
    from dataclasses import replace

    other = Brain(replace(settings, data_dir=tmp_path / "x"), client=brain.store.client)
    other.remember("from an agent", actor="mcp")
    brain.adopt_orphans()
    assert run(["list", "--actor", "mcp"], brain=brain) == 0
    assert "(adopted, written by mcp)" in capsys.readouterr().out


def test_a_service_that_is_not_qdrant_gets_a_clear_message(brain, monkeypatch, capsys):
    def odd(*_args, **_kwargs):
        raise AssertionError("Collection exists returned None")

    monkeypatch.setattr(brain.store.client, "collection_exists", odd)
    brain._ready = False
    assert run(["status"], brain=brain) == 2
    err = capsys.readouterr().err
    assert "does not answer like Qdrant" in err and "please report" not in err


def test_adopt_orphans_says_when_it_deleted_purged_memories(brain, capsys):
    from qdrant_client.http import models as qm

    secret = brain.remember("secret")
    point = brain.store.client.retrieve(brain.store.memories, [secret], with_vectors=True)[0]
    brain.purge(secret)
    brain.store.upsert(
        brain.store.memories, [qm.PointStruct(id=secret, vector=point.vector, payload=point.payload)]
    )
    assert run(["adopt-orphans"], brain=brain) == 0
    assert "deleted 1 purged memories" in capsys.readouterr().out


def test_search_marks_archived_results(brain, capsys):
    memory_id = brain.remember("a forgotten recipe for onion soup")
    brain.archive(memory_id)
    assert run(["search", "onion soup", "--include-archived", "-k", "1"], brain=brain) == 0
    assert "[archived]" in capsys.readouterr().out.splitlines()[0]


def test_adopt_orphans_names_the_points_it_left_out(brain, capsys):
    from qdrant_client.http import models as qm

    vector = [0.1] * brain.settings.vector_size
    brain.store.upsert(brain.store.memories, [qm.PointStruct(id=77, vector=vector, payload={"text": "x"})])
    assert run(["adopt-orphans"], brain=brain) == 0
    captured = capsys.readouterr()
    assert "adopted 0" in captured.out and "77 (not a memory id)" in captured.err and "purge" in captured.err


def test_index_warns_about_deleted_files(brain, notes, capsys):
    run(["index", str(notes)], brain=brain)
    (notes / "garden.md").unlink()
    capsys.readouterr()
    assert run(["index", str(notes)], brain=brain) == 0
    assert "1 indexed files are no longer there" in capsys.readouterr().err


def test_an_option_after_unquoted_words_never_cuts_a_memory(brain, capsys):
    assert run(["remember", "Always", "pass", "the", "--kind", "lesson"], brain=brain) == 2
    assert "quotes" in capsys.readouterr().err
    assert brain.list_memories() == []
    assert run(["remember", "Always pass the", "--kind", "lesson"], brain=brain) == 0
    assert run(["remember", "--kind=lesson", "two", "words"], brain=brain) == 0
    assert run(["remember", "--", "a", "-b", "--kind", "c"], brain=brain) == 0
    texts = sorted(m["text"] for m in brain.list_memories())
    assert texts == ["Always pass the", "a -b --kind c", "two words"]


def test_abbreviated_options_are_refused(brain, capsys):
    import pytest

    with pytest.raises(SystemExit) as caught:
        run(["list", "--include"], brain=brain)  # not --include-archived
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        run(["--vers"], brain=brain)  # not --version
    assert caught.value.code == 2


def test_an_empty_path_never_means_the_current_folder(brain, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "private.md").write_text("not for the index\n", encoding="utf-8")
    for empty in ("", "   "):
        assert run(["index", empty], brain=brain) == 2
        assert "give the folder or file" in capsys.readouterr().err
    assert brain.store.count(brain.store.docs) == 0


def test_explicit_hidden_or_other_files_say_why(brain, tmp_path, capsys):
    (tmp_path / ".draft.md").write_text("x\n", encoding="utf-8")
    (tmp_path / "x.pdf").write_bytes(b"%PDF")
    assert run(["index", str(tmp_path / ".draft.md")], brain=brain) == 2
    assert "hidden file" in capsys.readouterr().err
    assert run(["index", str(tmp_path / "x.pdf")], brain=brain) == 2
    assert "only .md, .markdown and .txt" in capsys.readouterr().err
