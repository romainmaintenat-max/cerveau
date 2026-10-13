import json
import os
from pathlib import Path

import pytest

from cerveau.journal import Journal, JournalError


def make(tmp_path: Path) -> Journal:
    (tmp_path / "j").mkdir(exist_ok=True)
    return Journal(tmp_path / "j" / "journal.jsonl")


def remember(journal: Journal, memory_id: str, text: str = "t") -> None:
    with journal.lock:
        journal.append({"op": "remember", "id": memory_id, "text": text, "kind": "note", "actor": "a"})


def ids(journal: Journal) -> list[str]:
    return [e["id"] for e in journal.events()]


def test_append_requires_the_lock(tmp_path):
    journal = make(tmp_path)
    with pytest.raises(JournalError, match="lock"):
        journal.append({"op": "remember", "id": "1", "text": "t", "kind": "note", "actor": "a"})


def test_append_and_read_back_in_order_with_sequence(tmp_path):
    journal = make(tmp_path)
    remember(journal, "1", "é")
    with journal.lock:
        journal.append({"op": "archive", "id": "1", "actor": "a"})
    events = list(journal.events())
    assert [e["op"] for e in events] == ["remember", "archive"]
    assert [e["seq"] for e in events] == [1, 2]
    assert events[0]["text"] == "é" and "ts" in events[0]


def test_unknown_operation_is_refused(tmp_path):
    journal = make(tmp_path)
    with journal.lock, pytest.raises(JournalError):
        journal.append({"op": "delete", "id": "1"})
    with journal.lock, pytest.raises(JournalError):
        journal.append({"op": "repair", "id": "1"})  # internal only


@pytest.mark.parametrize(
    "torn",
    [
        b'{"op": "remember", "id": "2", "te',  # cut in ASCII
        '{"v": 2, "text": "café é'.encode()[:-1],  # cut inside a UTF-8 character
        b'{"actor": "a", "text": "a}',  # cut right after a "}" that belongs to the text
        b'{"v": 2, "text": "x' + b"\x00" * 64,  # NUL padding after a power cut
        b"\x00" * 4096,  # only NUL bytes after the last newline
    ],
)
def test_torn_tail_is_tolerated_and_never_swallows_the_next_record(tmp_path, torn):
    journal = make(tmp_path)
    remember(journal, "1")
    with open(journal.path, "ab") as handle:
        handle.write(torn)
    assert ids(journal) == ["1"]
    remember(journal, "3")
    assert ids(journal) == ["1", "3"]
    remember(journal, "4")
    assert ids(journal) == ["1", "3", "4"]


def test_valid_last_record_without_newline_is_kept(tmp_path):
    journal = make(tmp_path)
    remember(journal, "1")
    remember(journal, "2")
    journal.path.write_bytes(journal.path.read_bytes().rstrip(b"\n"))
    remember(journal, "3")
    assert ids(journal) == ["1", "2", "3"]
    assert journal.health().repairs == 0


def test_crash_after_the_closing_newline_only(tmp_path):
    journal = make(tmp_path)
    remember(journal, "1")
    with open(journal.path, "ab") as handle:
        handle.write(b'{"v": 2, "torn"\n')  # torn line already closed, no repair written
    remember(journal, "2")
    assert ids(journal) == ["1", "2"]


def test_two_crashes_in_a_row_are_recovered(tmp_path):
    journal = make(tmp_path)
    remember(journal, "1")
    with open(journal.path, "ab") as handle:
        handle.write(b'{"v": 2, "first torn')
    # crash while writing "\n" + repair + record: only "\n" and half a repair made it
    with open(journal.path, "ab") as handle:
        handle.write(b'\n{"v": 2, "op": "repair", "se')
    remember(journal, "2")
    remember(journal, "3")
    assert ids(journal) == ["1", "2", "3"]
    health = journal.health()
    assert health.tolerated_torn_lines == 2 and health.repairs == 1


def test_bom_and_crlf_from_a_text_editor_are_accepted(tmp_path):
    journal = make(tmp_path)
    remember(journal, "1")
    remember(journal, "2")
    data = journal.path.read_bytes()
    journal.path.write_bytes(b"\xef\xbb\xbf" + data.replace(b"\n", b"\r\n"))
    assert ids(journal) == ["1", "2"]
    remember(journal, "3")
    assert ids(journal) == ["1", "2", "3"]


def test_append_reads_only_the_end_of_a_large_journal(tmp_path, monkeypatch):
    journal = make(tmp_path)
    for i in range(300):
        remember(journal, str(i), "x" * 400)
    reads = []
    real_open = open

    def spy_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if "rb" in mode:
            real_read = handle.read
            handle.read = lambda n=-1: (reads.append(n), real_read(n))[1]
        return handle

    monkeypatch.setattr("builtins.open", spy_open)
    remember(journal, "last")
    assert reads and all(0 < n <= 64 * 1024 for n in reads)


def test_damaged_line_in_the_middle_raises(tmp_path):
    journal = make(tmp_path)
    for i in range(3):
        remember(journal, str(i))
    lines = journal.path.read_bytes().split(b"\n")
    lines[1] = lines[1][:20]  # damage the middle record, no repair marker after it
    journal.path.write_bytes(b"\n".join(lines))
    with pytest.raises(JournalError, match="middle"):
        list(journal.events())


def test_modified_record_is_detected_by_its_checksum(tmp_path):
    journal = make(tmp_path)
    remember(journal, "1", "keep the server cool")
    remember(journal, "2")
    data = journal.path.read_bytes().replace(b"keep the server cool", b"heat the server more")
    journal.path.write_bytes(data)
    with pytest.raises(JournalError):
        list(journal.events())


def test_missing_record_is_detected_by_the_sequence(tmp_path):
    journal = make(tmp_path)
    for i in range(3):
        remember(journal, str(i))
    lines = journal.path.read_bytes().split(b"\n")
    del lines[1]
    journal.path.write_bytes(b"\n".join(lines))
    with pytest.raises(JournalError, match="sequence"):
        list(journal.events())


def test_empty_or_missing_journal(tmp_path):
    journal = make(tmp_path)
    assert journal.count() == 0
    journal.path.write_bytes(b"")
    assert journal.count() == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_journal_and_data_folder_are_private(settings, client):
    from cerveau.brain import Brain

    brain = Brain(settings, client=client)
    brain.remember("private")
    assert settings.data_dir.stat().st_mode & 0o077 == 0
    assert brain.journal.path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize(
    "line",
    [
        b'{"v": 2, "text": "complete object, no crc"}',  # complete JSON: not a cut write
        '{"v": 2, "op": "remember"}'.encode("utf-16"),  # re-saved in another encoding
        b"random garbage that is not json",
        b'{"a": [1, 2}',  # mismatched brackets
    ],
)
def test_unreadable_line_at_the_end_is_an_error_not_a_torn_write(tmp_path, line):
    journal = make(tmp_path)
    remember(journal, "1")
    with open(journal.path, "ab") as handle:
        handle.write(line + b"\n")
    with pytest.raises(JournalError, match="unreadable"):
        list(journal.events())
    with journal.lock, pytest.raises(JournalError):
        journal.append({"op": "remember", "id": "2", "text": "t", "kind": "note", "actor": "a"})


def _rewrite_record(journal, index, **changes):
    import json as _json

    from cerveau.journal import _crc

    lines = journal.path.read_bytes().split(b"\n")
    record = _json.loads(lines[index])
    record.update(changes)
    record["crc"] = _crc(record)
    lines[index] = _json.dumps(record, sort_keys=True).encode()
    journal.path.write_bytes(b"\n".join(lines))


@pytest.mark.parametrize("index, changes", [(2, {"v": 3}), (2, {"op": "teleport"}), (0, {"seq": True})])
def test_record_from_another_version_is_never_masked(tmp_path, index, changes):
    journal = make(tmp_path)
    for i in range(3):
        remember(journal, str(i))
    _rewrite_record(journal, index, **changes)
    with pytest.raises(JournalError):
        list(journal.events())


def test_altered_last_record_with_wrong_checksum_is_an_error(tmp_path):
    journal = make(tmp_path)
    remember(journal, "1", "keep the server cool")
    data = journal.path.read_bytes().replace(b"keep the server cool", b"heat the server more")
    journal.path.write_bytes(data)
    with pytest.raises(JournalError):
        list(journal.events())


def test_repair_record_without_damage_before_it_is_an_error(tmp_path):
    journal = make(tmp_path)
    for i in range(3):
        remember(journal, str(i))
    _rewrite_record(journal, 1, op="repair")
    with pytest.raises(JournalError, match="repair"):
        list(journal.events())


def test_long_nul_tail_is_handled_quickly(tmp_path):
    import time

    journal = make(tmp_path)
    remember(journal, "1")
    with open(journal.path, "ab") as handle:
        handle.write(b'{"actor": "a", "te' + b"\x00" * (20 * 1024 * 1024))
    start = time.monotonic()
    remember(journal, "2")
    assert time.monotonic() - start < 5
    assert ids(journal) == ["1", "2"]


def test_text_that_is_not_valid_unicode_is_refused_before_writing(tmp_path):
    journal = make(tmp_path)
    with journal.lock, pytest.raises(JournalError, match="Unicode"):
        journal.append({"op": "remember", "id": "1", "text": "bad\udcffbyte", "kind": "n", "actor": "a"})
    assert not journal.path.exists() or journal.path.read_bytes() == b""


@pytest.mark.parametrize(
    "line, expected",
    [
        (b'{"actor": "a", "text": "caf\xc3', "torn"),  # cut inside a UTF-8 character
        (b'{"actor": "a", "te' + b"\x00" * 10, "torn"),
        (b'{"a": 1} {"b": ', "invalid"),  # a complete object followed by more data
        (b'{"text": "a\x01b', "invalid"),  # raw control character inside a string
        (b'{"a": @@', "invalid"),  # characters that never appear in JSON
        (b'"just a string', "invalid"),
        (b"[1, 2", "invalid"),
        (b'{"text": "a\xffb", "x": ', "invalid"),  # invalid byte in the middle
        (b"\x00\x00 \r", "blank"),
    ],
)
def test_line_classification(line, expected):
    from cerveau.journal import classify

    assert classify(line)[0] == expected


def test_purge_rewrites_without_the_records_of_one_id(tmp_path):
    journal = make(tmp_path)
    remember(journal, "a", "keep me")
    remember(journal, "b", "SECRET-TOKEN")
    with journal.lock:
        journal.append({"op": "archive", "id": "b", "actor": "a"})
    remember(journal, "c", "keep me too")
    with journal.lock:
        removed = journal.purge(lambda e: e["id"] == "b", {"op": "purge", "id": "b", "actor": "cli"})
    assert removed == 2
    assert ids(journal) == ["a", "c"]
    raw = journal.path.read_bytes()
    assert b"SECRET-TOKEN" not in raw
    assert not journal.path.with_name(journal.path.name + ".purge-tmp").exists()
    health = journal.health()
    assert (health.events, health.tolerated_torn_lines, health.repairs) == (2, 0, 0)
    remember(journal, "d")  # appending after a purge continues the renumbered sequence
    assert ids(journal) == ["a", "c", "d"]
    seqs = [json.loads(line)["seq"] for line in journal.path.read_text("utf-8").splitlines()]
    assert seqs == [1, 2, 3, 4]


def test_purge_drops_torn_lines_that_may_hold_the_text(tmp_path):
    journal = make(tmp_path)
    remember(journal, "a")
    with open(journal.path, "ab") as handle:
        handle.write(b'{"op": "remember", "id": "b", "text": "SECRET-PART')
    remember(journal, "b", "SECRET-PART of a secret")
    with journal.lock:
        assert journal.purge(lambda e: e["id"] == "b", {"op": "purge", "id": "b", "actor": "cli"}) == 1
    assert b"SECRET" not in journal.path.read_bytes()
    assert ids(journal) == ["a"] and journal.health().repairs == 0


def test_purge_of_an_unknown_id_leaves_the_file_untouched(tmp_path):
    journal = make(tmp_path)
    remember(journal, "a")
    before = journal.path.read_bytes()
    with journal.lock:
        assert journal.purge(lambda e: e["id"] == "zzz", {"op": "purge", "id": "zzz", "actor": "cli"}) == 0
    assert journal.path.read_bytes() == before


def test_purge_requires_the_lock_and_a_note_without_text(tmp_path):
    journal = make(tmp_path)
    remember(journal, "a")
    with pytest.raises(JournalError, match="lock"):
        journal.purge(lambda e: True, {"op": "purge", "id": "a"})
    with journal.lock, pytest.raises(JournalError, match="without text"):
        journal.purge(lambda e: True, {"op": "purge", "id": "a", "text": "leak"})
    with journal.lock, pytest.raises(JournalError, match="without text"):
        journal.purge(lambda e: True, {"op": "remember", "id": "a"})
    assert ids(journal) == ["a"]


def test_purge_refuses_a_damaged_journal(tmp_path):
    journal = make(tmp_path)
    remember(journal, "a")
    remember(journal, "b")
    lines = journal.path.read_bytes().splitlines(keepends=True)
    journal.path.write_bytes(lines[0].replace(b'"t"', b'"x"') + lines[1])
    with journal.lock, pytest.raises(JournalError):
        journal.purge(lambda e: e["id"] == "b", {"op": "purge", "id": "b"})


def test_crash_before_the_rename_keeps_the_old_journal(tmp_path, monkeypatch):
    journal = make(tmp_path)
    remember(journal, "a")
    remember(journal, "b")
    before = journal.path.read_bytes()

    def crash(*_args):
        raise OSError("power cut")

    monkeypatch.setattr("cerveau.journal.os.replace", crash)
    with journal.lock, pytest.raises(OSError, match="power cut"):
        journal.purge(lambda e: e["id"] == "b", {"op": "purge", "id": "b"})
    assert journal.path.read_bytes() == before
    monkeypatch.undo()
    with journal.lock:  # the leftover temporary file is simply overwritten
        assert journal.purge(lambda e: e["id"] == "b", {"op": "purge", "id": "b"}) == 1
    assert ids(journal) == ["a"]


def test_rename_is_retried_on_windows_while_a_reader_holds_the_file(tmp_path, monkeypatch):
    import cerveau.journal as module

    calls = []
    real = os.replace

    def busy_twice(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(13, "in use")
        real(src, dst)

    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(module.os, "replace", busy_twice)
    journal = make(tmp_path)
    remember(journal, "a")
    remember(journal, "b")
    with journal.lock:
        assert journal.purge(lambda e: e["id"] == "b", {"op": "purge", "id": "b"}) == 1
    assert len(calls) == 3 and ids(journal) == ["a"]


def test_rename_is_not_retried_elsewhere(tmp_path, monkeypatch):
    import cerveau.journal as module

    monkeypatch.setattr(module.sys, "platform", "linux")

    def busy(_src, _dst):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(module.os, "replace", busy)
    journal = make(tmp_path)
    remember(journal, "a")
    with journal.lock, pytest.raises(PermissionError):
        journal.purge(lambda e: True, {"op": "purge", "id": "a"})
