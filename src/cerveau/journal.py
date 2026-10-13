"""Append-only journal: the source of truth for memories.

Every change to a memory (remember, archive, restore) is first written here,
flushed and fsync'ed, and only then applied to Qdrant (write-ahead). The Qdrant
collection of memories can therefore always be rebuilt by replaying the journal,
for example after a crash or when switching to another embedding model.

The journal stores text, never vectors: it does not depend on the model.

Format: one JSON record per line, each with a sequence number ("seq") and a CRC32 of
its own content ("crc").

Every non-blank line is one of:
- VALID:   a complete record of this format, with a correct CRC;
- TORN:    the beginning of a JSON object, cut before its end (possibly followed by NUL
           bytes): exactly what an interrupted write leaves behind;
- INVALID: anything else (a complete record with a wrong CRC, an unknown version or
           operation, another encoding, garbage). It is never skipped: JournalError.

Crash recovery
- Before writing, the writer looks at the end of the file (reading backwards, in blocks):
  * bytes after the last newline are closed with a newline first, so a new record never
    ends up glued to them (a complete valid record there is kept as is);
  * if TORN lines follow the last valid record, a "repair" record is written.
- On read, a run of TORN lines is accepted only at the very end of the file or right
  before a repair record; a repair record is accepted only right after TORN lines.
  A gap in sequence numbers raises JournalError.

Purge
- The only change that is not an append: purge() rewrites the whole file without the
  records of one memory (renumbered, new CRCs), plus a "purge" record holding no text,
  through a temporary file, fsync and an atomic rename. Torn lines and repair records are
  dropped by the rewrite (a torn line may hold part of the purged text).

A single writer at a time is enforced by a lock file next to the journal.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import zlib

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

SCHEMA_VERSION = 2
OPERATIONS = {"remember", "archive", "restore"}
_INTERNAL = {"repair", "purge"}
_BOM = b"\xef\xbb\xbf"
_BLANK = b" \t\r\n\x00"
_TAIL_READ = 64 * 1024
# A single interrupted write is one record: far below this size.
_MAX_TORN_BYTES = 4 * 1024 * 1024

BLANK, VALID, TORN, INVALID = "blank", "valid", "torn", "invalid"
_JSON_OUTSIDE_STRINGS = set(' \t\r\n{}[]:,"-+.0123456789eEtrufalsn')


class JournalError(RuntimeError):
    """Raised when the journal is damaged in a way that is not an interrupted write."""


@dataclass
class JournalHealth:
    events: int
    tolerated_torn_lines: int
    repairs: int


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _crc(record: dict) -> int:
    body = {k: v for k, v in record.items() if k != "crc"}
    data = json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return zlib.crc32(data)


def _is_object_prefix(text: str) -> bool:
    """True if `text` is the beginning of a JSON object that is not finished yet."""
    if not text.startswith("{"):
        return False
    stack: list[str] = []
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            elif ord(char) < 0x20:
                return False  # raw control character: not something we ever write
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or {"}": "{", "]": "["}[char] != stack.pop():
                return False
            if not stack:
                return False  # the object is complete: it is not a cut write
        elif char not in _JSON_OUTSIDE_STRINGS:
            return False
    return bool(stack)


def classify(raw: bytes) -> tuple[str, dict | None]:
    """Classify one line of the journal (see the module docstring)."""
    data = raw.strip(_BLANK)
    if data.startswith(_BOM):
        data = data[len(_BOM) :].strip(_BLANK)
    if not data:
        return BLANK, None
    cut_inside_character = False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.reason != "unexpected end of data" or exc.end != len(data):
            return INVALID, None
        try:
            text = data[: exc.start].decode("utf-8")
        except UnicodeDecodeError:
            return INVALID, None
        cut_inside_character = True
    try:
        record = None if cut_inside_character else json.loads(text)
    except (ValueError, RecursionError):
        record = None
    if record is None:
        return (TORN, None) if _is_object_prefix(text) else (INVALID, None)
    if not isinstance(record, dict) or record.get("crc") != _crc(record):
        return INVALID, None
    seq = record.get("seq")
    if record.get("v") != SCHEMA_VERSION or not isinstance(seq, int) or isinstance(seq, bool):
        return INVALID, None
    if record.get("op") not in OPERATIONS | _INTERNAL:
        return INVALID, None
    return VALID, record


class Journal:
    def __init__(self, path: Path, lock_timeout: float = 60.0) -> None:
        self.path = path
        self.lock = FileLock(str(path) + ".lock", timeout=lock_timeout)

    # -- writing ---------------------------------------------------------------

    def append(self, event: dict) -> dict:
        """Durably append one event. The caller must hold self.lock."""
        if event.get("op") not in OPERATIONS:
            raise JournalError(f"unknown operation {event.get('op')!r}")
        if not self.lock.is_locked:
            raise JournalError("Journal.append called without holding the lock")
        record = {**event, "ts": now_iso()}
        try:
            json.dumps(record, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise JournalError("the text contains characters that are not valid Unicode") from exc
        created = not self.path.exists()
        if created:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
        tail_open, last_seq, damaged = self._inspect_end()
        out = b"\n" if tail_open else b""  # close what follows the last newline; keep its bytes
        if damaged:
            last_seq += 1
            out += self._encode({"op": "repair", "seq": last_seq, "ts": now_iso()})
        record["seq"] = last_seq + 1
        out += self._encode(record)
        with open(self.path, "ab") as handle:
            handle.write(out)
            handle.flush()
            _full_fsync(handle.fileno())
        if created:
            _fsync_dir(self.path.parent)
        return dict(record)

    def purge(self, drop: Callable[[dict], bool], note: dict, note_anyway: bool = False) -> int:
        """Rewrite the journal without the user events for which drop(event) is true, then
        add `note` (a "purge" record, which must hold no text). Returns the number of
        records removed; with 0, the file is left untouched (unless note_anyway: the note is
        still added, as a tombstone for a memory that was only in Qdrant). The caller must hold
        self.lock.

        Crash-safe: the new journal is written to a temporary file, fsync'ed, then renamed
        over the old one; at any moment the journal is either the old or the new file.
        """
        if not self.lock.is_locked:
            raise JournalError("Journal.purge called without holding the lock")
        if note.get("op") != "purge" or "text" in note:
            raise JournalError("a purge note is a 'purge' record without text")
        kept: list[dict] = []
        removed = 0
        for _number, record in self._records():  # raises JournalError on a damaged journal
            if record is None or record["op"] == "repair":
                continue  # interrupted writes: may hold part of the text being purged
            if record["op"] in OPERATIONS and drop(record):
                removed += 1
            else:
                kept.append(record)
        if not removed and not note_anyway:
            return 0
        kept.append({**note, "ts": now_iso()})
        temporary = self.path.with_name(self.path.name + ".purge-tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
        with os.fdopen(fd, "wb") as handle:
            for seq, record in enumerate(kept, 1):
                body = {k: v for k, v in record.items() if k not in ("v", "crc", "seq")}
                body["seq"] = seq
                handle.write(self._encode(body))
            handle.flush()
            _full_fsync(handle.fileno())
        _replace(temporary, self.path)
        _fsync_dir(self.path.parent)
        return removed

    @staticmethod
    def _encode(record: dict) -> bytes:
        record = {"v": SCHEMA_VERSION, **record}
        record["crc"] = _crc(record)
        return (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")

    def _inspect_end(self) -> tuple[bool, int, bool]:
        """Look at the end of the journal, reading backwards in blocks.

        Returns (bytes follow the last newline, seq of the last valid record,
        TORN lines follow that record). Raises JournalError on an INVALID line there.
        Reads a constant amount of data, whatever the size of the journal.
        """
        size = self.path.stat().st_size
        tail_open = False
        damaged = False
        first_segment = True
        carry = b""
        position = size
        with open(self.path, "rb") as handle:
            while position > 0:
                step = min(_TAIL_READ, position)
                position -= step
                handle.seek(position)
                parts = (handle.read(step) + carry).split(b"\n")
                carry, complete = (parts[0], parts[1:]) if position > 0 else (b"", parts)
                if not carry.strip(_BLANK):
                    carry_blank_len = len(carry)
                    carry = b""  # a long run of NUL bytes is kept out of memory
                else:
                    carry_blank_len = 0
                    if len(carry) > _MAX_TORN_BYTES:
                        raise JournalError(f"{self.path}: unexpected large unterminated data at the end")
                for raw in reversed(complete):
                    if first_segment:
                        tail_open = tail_open or raw != b""
                        first_segment = False
                    kind, record = classify(raw)
                    if kind == VALID:
                        return tail_open, record["seq"], damaged
                    if kind == INVALID:
                        raise JournalError(f"{self.path}: an unreadable record follows the last valid one")
                    if kind == TORN:
                        damaged = True
                if first_segment and (carry or carry_blank_len):
                    tail_open = True  # no newline seen yet: everything so far is the tail
        return tail_open, 0, damaged

    # -- reading ---------------------------------------------------------------

    def _records(self) -> Iterator[tuple[int, dict | None]]:
        """Yield (line number, record or None for a tolerated torn line). Streams the file."""
        if not self.path.exists():
            return
        expected = 1
        pending: list[int] = []  # line numbers of a run of TORN lines
        with open(self.path, "rb") as handle:
            for number, raw in enumerate(handle, 1):
                kind, record = classify(raw)
                if kind == BLANK:
                    continue
                if kind == INVALID:
                    raise JournalError(
                        f"{self.path}:{number}: unreadable record (not an interrupted write). Restore "
                        "a backup, or see the README: rename this file, then `cerveau adopt-orphans`"
                    )
                if kind == TORN:
                    pending.append(number)
                    continue
                if record["op"] == "repair":
                    if not pending:
                        raise JournalError(f"{self.path}:{number}: repair record without damage before it")
                elif pending:
                    raise JournalError(f"{self.path}:{pending[0]}: damaged line in the middle of the journal")
                for torn in pending:
                    yield torn, None
                pending = []
                if record["seq"] != expected:
                    raise JournalError(
                        f"{self.path}:{number}: sequence {record['seq']}, expected {expected} "
                        "(a record is missing or duplicated)"
                    )
                expected += 1
                yield number, record
        for torn in pending:  # torn lines at the very end: an interrupted write
            yield torn, None

    def events(self) -> Iterator[dict]:
        """Yield the user events (remember/archive/restore) in order."""
        for _number, record in self._records():
            if record is not None and record["op"] in OPERATIONS:
                yield record

    def purged_ids(self) -> set[str]:
        """Ids named by "purge" records: memories erased on purpose, never to come back."""
        return {
            record["id"]
            for _number, record in self._records()
            if record is not None and record["op"] == "purge" and isinstance(record.get("id"), str)
        }

    def health(self) -> JournalHealth:
        events = torn = repairs = 0
        for _number, record in self._records():
            if record is None:
                torn += 1
            elif record["op"] == "repair":
                repairs += 1
            elif record["op"] in OPERATIONS:
                events += 1
        return JournalHealth(events, torn, repairs)

    def count(self) -> int:
        return sum(1 for _ in self.events())


def _full_fsync(fd: int) -> None:
    """fsync, plus F_FULLFSYNC on macOS, where fsync alone does not flush the disk cache."""
    os.fsync(fd)
    full = getattr(fcntl, "F_FULLFSYNC", None) if fcntl is not None else None
    if full is not None:
        with contextlib.suppress(OSError):  # not supported by this file system: fsync ran
            fcntl.fcntl(fd, full)


def _replace(source: Path, target: Path) -> None:
    """os.replace, retried briefly on Windows, where a reader holding the file open
    (the MCP server answering a search) makes the rename fail for a moment."""
    for attempt in range(20):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == 19:
                raise
            time.sleep(0.25)


def _fsync_dir(directory: Path) -> None:
    """Make the creation of a new file durable (POSIX); a no-op where unsupported."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
