"""The brain: index documents, remember things, search both by meaning."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import unicodedata
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from filelock import FileLock, Timeout
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from cerveau.chunking import chunk_text
from cerveau.config import HEADING_BUDGET, Settings, redact_url
from cerveau.embed import Embedder, EmbeddingError, HashEmbedder, OllamaEmbedder
from cerveau.journal import Journal
from cerveau.store import Store, make_client

NAMESPACE = uuid.UUID("6f1c3c2e-6d0b-4c55-9a43-4b8a0f0c6e21")
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
# Folders that hold installed software, never notes (hidden folders are skipped too).
EXCLUDED_DIRS = {"node_modules", "__pycache__", "site-packages", "venv"}
KIND_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,31}")
MAX_FILE_BYTES = 20_000_000  # a note larger than this is almost certainly not a note
EMBED_BATCH = 8  # small batches: a slow CPU must answer each request within the timeout
WRITE_SLICE = 64  # chunks of one file embedded then written together: memory stays flat
SCOPES = {"all", "docs", "memories"}
CHUNKING_VERSION = 3  # bump when chunking changes, so files are re-chunked
# Errors raised by qdrant-client when the server is down or refuses a request.
STORAGE_ERRORS = (
    ResponseHandlingException,
    UnexpectedResponse,
)


class BrainError(RuntimeError):
    """A user-facing error (bad input, unknown id, busy...)."""


@dataclass
class Hit:
    score: float
    collection: str
    id: str
    text: str
    source: str | None = None
    heading: str | None = None
    kind: str | None = None
    state: str | None = None
    ts: str | None = None
    actor: str | None = None
    path: str | None = None  # absolute path: shown to the human, never sent to AI clients

    def to_dict(self, include_path: bool = False) -> dict:
        data = {k: v for k, v in asdict(self).items() if v is not None}
        if not include_path:
            data.pop("path", None)
        return data


@dataclass
class IndexReport:
    root: str
    files_seen: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    chunks_written: int = 0
    chunks_removed: int = 0
    stale_files: int = 0  # indexed files no longer found there (removed only with prune)
    errors: list[str] = field(default_factory=list)


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hash":
        return HashEmbedder(settings.vector_size, settings.max_text_chars)
    return OllamaEmbedder(
        settings.ollama_url,
        settings.embed_model,
        settings.vector_size,
        settings.max_text_chars,
        timeout=settings.ollama_timeout,
    )


def _canonical_id(memory_id: str) -> str:
    try:
        return str(uuid.UUID(memory_id))
    except ValueError:
        return memory_id


def _posix(path: Path) -> str:
    """Identity key of a path: "/" separators, Unicode NFC, and case-folded on Windows and
    macOS, whose file systems ignore case by default (~/notes and ~/Notes are one folder)."""
    key = unicodedata.normalize("NFC", os.path.normcase(str(path)))
    if sys.platform == "darwin":
        key = key.casefold()
    return Path(key).as_posix()


class NotText(ValueError):
    """A file that decodes but is not text (NUL characters): binary, or UTF-16 without BOM."""


def _read_text(file: Path) -> str:
    """UTF-8 (with or without BOM), or UTF-16/UTF-32 with BOM (Windows Notepad "Unicode")."""
    data = file.read_bytes()
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        text = data.decode("utf-32")
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        text = data.decode("utf-8-sig")
    if "\x00" in text:  # valid UTF-8, but no note holds NUL characters
        raise NotText
    return text


# Folders of the system and of installed programs: never notes, and often huge.
_SYSTEM_ROOTS = (
    "/etc",
    "/usr",
    "/var",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/opt",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/System",
    "/Library",
    "/Applications",
)
_SYSTEM_TOPS = ("/home", "/Users", "/mnt", "/media", "/Volumes", "/srv", "/root")
_WINDOWS_ROOTS = ("windows", "program files", "program files (x86)", "programdata", "users")


def _too_broad(path: Path) -> bool:
    """A whole disk, the home folder or the folder of all homes, a mount point holding other
    disks, or a system folder (and what is under it): application data, not notes."""
    home = Path.home().resolve()
    if path in (Path(path.anchor), home, home.parent):
        return True
    if sys.platform == "win32":
        parts = path.parts
        return (
            len(parts) >= 2
            and parts[1].lower() in _WINDOWS_ROOTS
            and (len(parts) == 2 or parts[1].lower() != "users")
        )
    text = path.as_posix()
    return text in _SYSTEM_TOPS or any(text == r or text.startswith(r + "/") for r in _SYSTEM_ROOTS)


def _bounded_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value if len(value) <= limit else value[:limit] + " [cut: longer than CERVEAU_MAX_TEXT_CHARS]"


def _short(value: object) -> str | None:
    """A value read from Qdrant, kept only if it is a short string."""
    return value if isinstance(value, str) and len(value) <= 64 else None


def _is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


class Brain:
    def __init__(
        self,
        settings: Settings,
        client: QdrantClient | None = None,
        embedder: Embedder | None = None,
        read_only: bool = False,
    ) -> None:
        self.settings = settings
        self.read_only = read_only
        self.embedder = embedder or make_embedder(settings)
        if self.embedder.dimension != settings.vector_size:
            raise BrainError("embedder dimension differs from CERVEAU_VECTOR_SIZE")
        model = self.embedder.model
        if model.endswith(":latest"):
            model = model[: -len(":latest")]  # same model for Ollama: one seal for both spellings
        self.store = Store(client or make_client(settings), settings.prefix, model, settings.vector_size)
        self.journal = Journal(settings.journal_path)
        self.docs_lock = FileLock(str(settings.data_dir / f"docs-{settings.prefix}.lock"), timeout=60)
        self._ready = False
        self._journal_checked = False
        self.purged_again = 0  # set by adopt_orphans()
        self.skipped_orphans: list[str] = []  # set by adopt_orphans()
        self.purged_prefixes: list[str] = []  # set by purge()

    @contextmanager
    def _locked(self, journal: bool = True, docs: bool = False) -> Iterator[None]:
        """Two locks: the journal's (memories) and this prefix's documents'. A long `index`
        holds only the second, so an agent can still `remember` while it runs."""
        if self.read_only:
            raise BrainError("this brain is read-only")
        self.settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        locks = [self.journal.lock] if journal else []
        if docs:
            locks.append(self.docs_lock)
        try:
            with ExitStack() as stack:
                for lock in locks:
                    stack.enter_context(lock)
                yield
        except Timeout as exc:
            what = "an index or unindex" if not journal else "another index, replay or write"
            raise BrainError(f"the brain is busy ({what} is running); retry in a moment") from exc

    def init(self) -> None:
        """Connect, and create the collections if needed.

        When the collections already exist, nothing is written and no lock is taken, so
        reading never waits for a long index. Before sealing new collections, the embedding
        model is tried once: a missing model or a wrong vector size seals nothing.
        """
        if self.store.is_complete():
            self.store.check_seal()
        elif self.read_only:
            raise BrainError("the brain is not initialised yet: run `cerveau init` first")
        else:
            self._embed(["cerveau: checking the embedding model"])
            with self._locked():
                self.store.ensure()
        self._ready = True

    def _check_journal_once(self) -> None:
        """Refuse to add to a journal that can no longer be read (checked once per process)."""
        if not self._journal_checked:
            self.journal.health()  # raises JournalError
            self._journal_checked = True

    def _require_ready(self, create: bool = True) -> None:
        """Connect; only writing operations may create the collections."""
        if self._ready:
            return
        if not create and not self.store.is_complete():
            raise BrainError("the brain is not initialised yet: run `cerveau init` first")
        self.init()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            vectors.extend(self.embedder.embed(texts[start : start + EMBED_BATCH]))
        return vectors

    # -- documents -------------------------------------------------------------

    def index_path(
        self, path: Path, prune: bool = False, progress: Callable[[str], None] | None = None
    ) -> IndexReport:
        """Index .md/.markdown/.txt files under `path` (or one file). Idempotent.

        A document is identified by its absolute path, so indexing a folder and one of
        its sub-folders never creates duplicates. A changed file has its old chunks
        replaced; an unchanged file is skipped. With prune=True, chunks of files under
        `path` that no longer exist are removed.
        """
        path = path.expanduser().resolve()
        if not path.exists():  # checked first: a typo creates no collection
            raise BrainError(f"{path} does not exist (to forget an old folder: cerveau unindex)")
        if _too_broad(path):
            raise BrainError(f"refusing to index {path}: too broad (give a folder of notes)")
        self._require_ready()
        walk_errors: list[str] = []
        if path.is_dir():
            files = self._list_files(path, walk_errors)
            root = path
        else:
            if path.name.startswith("."):
                raise BrainError(f"{path.name} is a hidden file: hidden files are never indexed")
            if path.suffix.lower() not in TEXT_SUFFIXES:
                raise BrainError("only .md, .markdown and .txt files can be indexed")
            files = [path]
            root = path.parent
        report = IndexReport(root=path.as_posix())  # as written on disk, not the identity key
        report.errors.extend(walk_errors)
        with self._locked(journal=False, docs=True):
            existing = self._existing_docs(_posix(path))
            seen: set[str] = set()
            service_down = False
            for file in files:
                report.files_seen += 1
                if progress is not None:
                    progress(f"[{report.files_seen}/{len(files)}] {file.relative_to(root).as_posix()}")
                key = _posix(file)
                if key in seen:
                    # Same name once normalised (NFC/NFD, or case where it is ignored): one
                    # identity for two files would make them overwrite each other.
                    report.errors.append(
                        f"{file}: same name as another file of this folder once normalised "
                        "(accents written another way, or case): rename one of them"
                    )
                    continue
                seen.add(key)
                shown = file.as_posix()
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError:
                    report.errors.append(
                        f"{key.encode('utf-8', 'backslashreplace').decode()}: the file name is not "
                        "valid UTF-8 (rename the file)"
                    )
                    continue
                try:
                    self._index_file(file, key, file.relative_to(root).as_posix(), existing, report, progress)
                except EmbeddingError as exc:
                    report.errors.append(f"{shown}: {exc}")
                    if not exc.per_text:  # Ollama down or too slow: every other file would fail too
                        report.errors.append(
                            f"indexing stopped after {report.files_seen} of {len(files)} files: run "
                            "the same command again once the embedding service answers"
                        )
                        service_down = True
                        break
                except UnicodeDecodeError:
                    report.errors.append(f"{shown}: not UTF-8 text (save the file as UTF-8)")
                except NotText:
                    report.errors.append(
                        f"{shown}: looks binary, or UTF-16 without BOM (save the file as UTF-8)"
                    )
                except OSError as exc:
                    report.errors.append(f"{shown}: {exc}")
                except Exception as exc:  # a Qdrant failure must not stop the other files...
                    report.errors.append(f"{shown}: storage error ({exc.__class__.__name__}: {exc})"[:500])
                    if isinstance(exc, ResponseHandlingException):  # ...unless Qdrant is gone
                        report.errors.append(
                            f"indexing stopped after {report.files_seen} of {len(files)} files: Qdrant "
                            "cannot be reached; run the same command again once it is back"
                        )
                        service_down = True
                        break
            stale = [p for p in existing if p not in seen] if path.is_dir() and not service_down else []
            if not prune or service_down:
                report.stale_files = len(stale)  # still searchable: the CLI says so
            elif prune and path.is_dir() and walk_errors:
                # A folder that could not be read looks empty: its files would be pruned.
                report.errors.append(
                    "--prune skipped: some folders could not be read (see above); nothing was removed"
                )
            elif prune and path.is_dir() and not files and existing:
                # An unmounted disk or an emptied folder looks like "everything was deleted".
                report.errors.append(
                    "--prune skipped: no text file found there, but documents are indexed under "
                    f"it (a disk not mounted?). If they are really gone: cerveau unindex {path}"
                )
            elif prune and path.is_dir():
                gone = [pid for p in stale for ids in existing[p].values() for pid in ids]
                self.store.delete_points(self.store.docs, gone)
                report.chunks_removed += len(gone)
        return report

    @staticmethod
    def _list_files(folder: Path, errors: list[str]) -> list[Path]:
        """Text files under `folder`. An unreadable sub-folder is reported, never fatal.
        Hidden entries, software folders and symbolic links are skipped."""
        found: list[Path] = []
        for current, dirs, names in os.walk(
            folder, onerror=lambda e: errors.append(f"{e.filename}: {e.strerror}")
        ):
            dirs[:] = sorted(
                d
                for d in dirs
                if not d.startswith(".")
                and d not in EXCLUDED_DIRS
                and not os.path.islink(os.path.join(current, d))
            )
            for name in sorted(names):
                candidate = Path(current) / name
                try:
                    if (
                        not name.startswith(".")
                        and candidate.suffix.lower() in TEXT_SUFFIXES
                        and not candidate.is_symlink()
                        and candidate.is_file()
                        and candidate.resolve().is_relative_to(folder)
                    ):
                        found.append(candidate)
                except OSError as exc:
                    errors.append(f"{candidate}: {exc.strerror or exc}")
        return found

    def _index_file(
        self,
        file: Path,
        key: str,
        source: str,
        existing: dict[str, dict[str, list[str]]],
        report: IndexReport,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if file.stat().st_size > MAX_FILE_BYTES:
            raise OSError(f"skipped: larger than {MAX_FILE_BYTES // 1_000_000} MB")
        content = _read_text(file)
        fingerprint = f"{CHUNKING_VERSION}|{self.settings.chunk_max_chars}|{content}"
        file_hash = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        old = existing.get(key, {})
        if set(old) == {file_hash} and self._complete(key, file_hash, old[file_hash]):
            report.files_unchanged += 1
            return
        chunks = chunk_text(content, self.settings.chunk_max_chars)
        if not chunks and not old:
            report.files_unchanged += 1  # empty file, nothing stored, nothing to do
            return
        # Embedded and written slice by slice: a large file never holds all its vectors in
        # memory, and an interruption keeps what was written (the next run sees, through
        # n_chunks, that this version is incomplete and writes it again).
        for start in range(0, len(chunks), WRITE_SLICE):
            part = chunks[start : start + WRITE_SLICE]
            vectors = self._embed([self._embed_input(c.heading, c.text) for c in part])
            points = [
                qm.PointStruct(
                    id=str(uuid.uuid5(NAMESPACE, f"{key}|{file_hash}|{i}")),
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "heading": chunk.heading,
                        "source": source,
                        "path": key,  # identity: case-folded on Windows
                        "file": file.as_posix(),  # for display: as written on disk
                        "file_hash": file_hash,
                        "chunk": i,
                        "n_chunks": len(chunks),  # lets the next run see an interrupted write
                        "kind": "doc",
                        "state": "active",
                    },
                )
                for i, (chunk, vector) in enumerate(zip(part, vectors, strict=True), start)
            ]
            self.store.upsert(self.store.docs, points)
            report.chunks_written += len(points)
            if progress is not None and len(chunks) > WRITE_SLICE:
                progress(f"    {source}: {start + len(part)}/{len(chunks)} chunks")
        # The new version is written first, then the old one removed:
        # a crash in between leaves duplicates (fixed by the next run), never a hole.
        stale = [pid for h, ids in old.items() if h != file_hash for pid in ids]
        self.store.delete_points(self.store.docs, stale)
        report.files_indexed += 1
        report.chunks_removed += len(stale)

    @staticmethod
    def _embed_input(heading: str, text: str) -> str:
        if not heading:
            return text
        # The heading gives context to the vector; it is metadata, the chunk text is kept whole.
        return f"{heading[-(HEADING_BUDGET - 1) :]}\n{text}"

    def _existing_docs(self, under: str) -> dict[str, dict[str, list[str]]]:
        """path -> file_hash -> chunk ids. A version is complete only if all its chunks are
        there (see _complete): an interrupted write leaves a partial version behind."""
        result: dict[str, dict[str, list[str]]] = {}
        self._chunk_counts: dict[tuple[str, str], int] = {}
        fields = ["path", "file_hash", "n_chunks"]
        for record in self.store.scroll_payloads(self.store.docs, fields=fields):
            payload = record.payload or {}
            doc_path = payload.get("path", "")
            if _is_under(doc_path, under):
                file_hash = payload.get("file_hash", "")
                result.setdefault(doc_path, {}).setdefault(file_hash, []).append(str(record.id))
                self._chunk_counts[(doc_path, file_hash)] = int(payload.get("n_chunks", -1))
        return result

    def _complete(self, key: str, file_hash: str, ids: list[str]) -> bool:
        return self._chunk_counts.get((key, file_hash)) == len(ids)

    def unindex(self, path: str) -> int:
        """Remove every indexed chunk at or under `path`. The path does not need to exist."""
        self._require_ready(create=False)  # nothing to remove from a brain never initialised
        if not str(path).strip():
            raise BrainError("give the path to unindex")
        resolved = Path(path).expanduser().resolve()
        if _too_broad(resolved):
            raise BrainError(f"refusing to unindex {resolved}: too broad (give a folder of notes)")
        target = _posix(resolved)
        with self._locked(journal=False, docs=True):
            ids = [
                pid
                for by_hash in self._existing_docs(target).values()
                for ids in by_hash.values()
                for pid in ids
            ]
            self.store.delete_points(self.store.docs, ids)
        return len(ids)

    # -- memories --------------------------------------------------------------

    def _check_text(self, text: str, what: str = "text") -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()  # pasted from Windows
        if not text:
            raise BrainError(f"the {what} is empty")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise BrainError(f"the {what} contains characters that are not valid Unicode") from exc
        if len(text) > self.settings.max_text_chars:
            advice = "shorten it" if what == "query" else "split it into several shorter texts"
            raise BrainError(
                f"the {what} has {len(text)} characters, the limit is "
                f"{self.settings.max_text_chars}: {advice}"
            )
        return text

    def remember(self, text: str, kind: str = "note", actor: str = "user", create: bool = True) -> str:
        """Store a memory. create=False (the MCP server): an uninitialised brain is an error,
        as for searches, instead of being set up by an agent with whatever settings it has."""
        self._require_ready(create=create)
        text = self._check_text(text)
        kind = unicodedata.normalize("NFKD", kind.strip().lower().replace(" ", "-"))
        kind = "".join(c for c in kind if not unicodedata.combining(c))  # décision -> decision
        if not KIND_PATTERN.fullmatch(kind):
            raise BrainError(
                "kind must be a short label: a letter a-z, then a-z, 0-9, - or _ (32 characters at "
                "most), like note, decision or lesson"
            )
        vector = self._embed([text])[0]  # fail before journaling if it cannot be embedded
        memory_id = str(uuid.uuid4())
        with self._locked():
            self._check_journal_once()
            record = self.journal.append(
                {"op": "remember", "id": memory_id, "text": text, "kind": kind, "actor": actor}
            )
            try:
                self.store.upsert(
                    self.store.memories,
                    [qm.PointStruct(id=memory_id, vector=vector, payload=self._payload(record))],
                )
            except Exception as exc:
                raise BrainError(
                    f"memory {memory_id} IS saved in the journal, but Qdrant failed "
                    f"({exc.__class__.__name__}). Do not retry (it would create a duplicate): "
                    "run `cerveau replay` once Qdrant is back."
                ) from exc
        return memory_id

    @staticmethod
    def _payload(record: dict, state: str = "active") -> dict:
        return {
            "text": record["text"],
            "kind": record["kind"],
            "ts": record.get("adopted_from_ts") or record["ts"],  # original date of an adopted memory
            "actor": record["actor"],
            "state": state,
        }

    def _set_state(self, memory_id: str, op: str, state: str, actor: str) -> str:
        self._require_ready(create=False)
        try:
            memory_id = str(uuid.UUID(memory_id))  # one spelling only: lower case, with dashes
        except ValueError as exc:
            raise BrainError(f"{memory_id!r} is not a memory id") from exc
        with self._locked():
            self._check_journal_once()
            current = self.store.get(self.store.memories, memory_id)
            folded = self._fold_journal().get(memory_id)
            journal_state = folded["state"] if folded else None
            qdrant_state = (current.payload or {}).get("state") if current is not None else None
            if journal_state == state and qdrant_state == state:
                return memory_id  # already in that state everywhere: nothing to record
            if journal_state == state and current is not None:
                self.store.set_state(self.store.memories, memory_id, state)  # Qdrant catches up
                return memory_id
            known = folded is not None
            if current is None:
                hint = " It is in the journal: run `cerveau replay` first." if known else ""
                raise BrainError(f"no memory with id {memory_id} in Qdrant.{hint}")
            if not known:
                raise BrainError(
                    f"memory {memory_id} is in Qdrant but not in this journal: `cerveau adopt-orphans` first"
                )
            self.journal.append({"op": op, "id": memory_id, "actor": actor})
            try:
                self.store.set_state(self.store.memories, memory_id, state)
            except Exception as exc:
                raise BrainError(
                    f"{op} of {memory_id} IS recorded in the journal, but Qdrant failed "
                    f"({exc.__class__.__name__}): run `cerveau replay` once Qdrant is back."
                ) from exc
        return memory_id

    def archive(self, memory_id: str, actor: str = "user") -> str:
        """Hide a memory from searches. Reversible with restore(); nothing is deleted."""
        return self._set_state(memory_id, "archive", "archived", actor)

    def restore(self, memory_id: str, actor: str = "user") -> str:
        return self._set_state(memory_id, "restore", "active", actor)

    def replay(self, progress: Callable[[str], None] | None = None) -> int:
        """Write every memory of the journal, in its final state, to Qdrant. Idempotent.

        Holds the lock for the whole rebuild. Points that are not in the journal (written
        to Qdrant by something else) are left untouched.
        """
        self._require_ready()
        with self._locked():
            memories = self._fold_journal()
            ids = list(memories)
            failed: dict[str, str] = {}
            written = 0
            for start in range(0, len(ids), EMBED_BATCH * 4):
                batch = ids[start : start + EMBED_BATCH * 4]
                vectors: dict[str, list[float]] = {}
                try:
                    vectors = dict(
                        zip(batch, self._embed([memories[i]["record"]["text"] for i in batch]), strict=True)
                    )
                except EmbeddingError as exc:
                    if not exc.per_text:  # Ollama down or too slow: one by one would only wait more
                        raise BrainError(
                            f"replayed {written} of {len(ids)} memories, then stopped: {exc}. "
                            "Run `cerveau replay` again once the embedding service answers."
                        ) from exc
                    for memory_id in batch:  # find the culprits one by one; keep the others
                        try:
                            vectors[memory_id] = self._embed([memories[memory_id]["record"]["text"]])[0]
                        except EmbeddingError as one:
                            if not one.per_text:
                                raise BrainError(
                                    f"replayed {written} of {len(ids)} memories, then stopped: {one}. "
                                    "Run `cerveau replay` again once the embedding service answers."
                                ) from one
                            failed[memory_id] = str(one)
                points = [
                    qm.PointStruct(
                        id=i,
                        vector=v,
                        payload=self._payload(memories[i]["record"], memories[i]["state"]),
                    )
                    for i, v in vectors.items()
                ]
                self.store.upsert(self.store.memories, points)
                written += len(points)
                if progress is not None:
                    progress(f"replay: {start + len(batch)}/{len(ids)}")
        if failed:
            details = "; ".join(f"{i}: {why}" for i, why in list(failed.items())[:5])
            raise BrainError(
                f"replayed {written} memories, {len(failed)} could not be embedded ({details}). "
                "Journal entries cannot be split: raise CERVEAU_MAX_TEXT_CHARS (or use a model "
                "with a larger context), then run `cerveau replay` again."
            )
        return written

    def _fold_journal(self) -> dict[str, dict]:
        """Final state of every memory of the journal, keyed by canonical id."""
        memories: dict[str, dict] = {}
        for event in self.journal.events():
            memory_id = _canonical_id(event["id"])
            if event["op"] == "remember":
                memories[memory_id] = {"record": event, "state": "active"}
            elif memory_id in memories:
                memories[memory_id]["state"] = "archived" if event["op"] == "archive" else "active"
        return memories

    def list_memories(self, actor: str | None = None, include_archived: bool = False) -> list[dict]:
        """Every memory of the journal, oldest first: what agents (actor "mcp") have written."""
        result = []
        for memory_id, item in self._fold_journal().items():
            record = item["record"]
            if actor is not None and actor not in (record["actor"], record.get("adopted_from_actor")):
                continue
            if item["state"] == "archived" and not include_archived:
                continue
            result.append(
                {
                    "id": memory_id,
                    "ts": record.get("adopted_from_ts") or record["ts"],
                    "actor": record["actor"],
                    "kind": record["kind"],
                    "state": item["state"],
                    "text": record["text"],
                }
            )
            if record.get("adopted_from_actor"):
                result[-1]["adopted_from"] = record["adopted_from_actor"]
        return sorted(result, key=lambda item: item["ts"])

    def orphan_memories(self, folded: dict[str, dict] | None = None) -> list[dict]:
        """Memories present in Qdrant but absent from this journal."""
        self._require_ready(create=False)
        return self._orphans(self._fold_journal() if folded is None else folded)

    def _purged(self) -> set[str]:
        return {_canonical_id(i) for i in self.journal.purged_ids()}

    def _purged_back(self) -> list[str]:
        """Purged memories found again in Qdrant (an older Qdrant backup restored...)."""
        purged = self._purged()
        if not purged:
            return []
        return [
            str(r.id)
            for r in self.store.scroll_payloads(self.store.memories, fields=["state"])
            if str(r.id) in purged
        ]

    def _orphans(self, folded: dict[str, dict], with_text: bool = True) -> list[dict]:
        """Memories in Qdrant that are neither in this journal nor purged from it.
        with_text=False reads only the small fields (to count them: status)."""
        orphans = []
        purged = self._purged()
        fields = None if with_text else ["ts", "actor", "kind", "state"]
        for record in self.store.scroll_payloads(self.store.memories, fields=fields):
            if str(record.id) not in folded and str(record.id) not in purged:
                payload = record.payload or {}
                orphans.append(
                    {
                        "id": str(record.id),
                        "ts": payload.get("ts"),
                        "actor": payload.get("actor"),
                        "kind": payload.get("kind"),
                        "state": payload.get("state"),
                        "text": payload.get("text", ""),
                    }
                )
        return orphans

    def reset_collections(self, discard_orphans: bool = False) -> None:
        """Delete this prefix's Qdrant collections. The journal and your files are untouched:
        `cerveau init`, `cerveau replay` and `cerveau index` rebuild everything.

        Memories that exist only in Qdrant (not in this journal) would be lost for good:
        refused unless discard_orphans is True (adopt them first with adopt_orphans()).
        """
        with self._locked(docs=True):
            if self.store.is_complete() and not discard_orphans:
                orphans = self._orphans(self._fold_journal())  # works even with a wrong seal
                if orphans:
                    raise BrainError(
                        f"{len(orphans)} memories exist only in Qdrant and would be lost: "
                        "`cerveau adopt-orphans` first, or --discard-orphans"
                    )
            self.store.drop()
        self._ready = False

    def adopt_orphans(self) -> int:
        """Copy into this journal the memories that exist only in Qdrant (same id, text,
        kind, author and state), so that the journal is again the complete source of truth.

        A memory that was purged is never adopted: if it is found again in Qdrant (an older
        backup restored), it is deleted from Qdrant again (see self.purged_again)."""
        self._require_ready(create=False)
        self.purged_again = 0
        with self._locked():
            self._check_journal_once()
            back = self._purged_back()
            self.store.delete_points(self.store.memories, back)
            self.purged_again = len(back)
            orphans = []
            self.skipped_orphans = []
            for orphan in self.orphan_memories():
                why = self._unadoptable(orphan)
                if why:
                    self.skipped_orphans.append(f"{orphan['id']} ({why})")
                else:
                    orphans.append(orphan)
            orphans.sort(key=lambda o: str(o.get("ts") or ""))
            for orphan in orphans:
                self.journal.append(
                    {
                        "op": "remember",
                        "id": orphan["id"],
                        "text": orphan["text"],
                        "kind": orphan["kind"] if KIND_PATTERN.fullmatch(str(orphan.get("kind"))) else "note",
                        # Who wrote it is not known for sure (anything with the API key can
                        # write to Qdrant): never adopted as "cli" or "mcp".
                        "actor": "adopted",
                        "adopted_from_actor": _short(orphan.get("actor")),
                        "adopted_from_ts": _short(orphan.get("ts")),
                    }
                )
                if orphan.get("state") == "archived":
                    self.journal.append({"op": "archive", "id": orphan["id"], "actor": "adopt"})
        return len(orphans)

    def _unadoptable(self, orphan: dict) -> str | None:
        """Why a point found in Qdrant cannot become a journal entry (None if it can):
        the journal only takes what Cerveau itself would have written, so replay keeps working."""
        try:
            if str(uuid.UUID(orphan["id"])) != orphan["id"]:
                return "not a memory id"
        except ValueError:
            return "not a memory id"
        text = orphan.get("text")
        if not isinstance(text, str) or not text.strip():
            return "no text"
        if len(text) > self.settings.max_text_chars:
            return "text longer than CERVEAU_MAX_TEXT_CHARS"
        return None

    def purge(self, memory_id: str) -> int:
        """Delete a memory for good: its records are removed from the journal (rewritten)
        and its point from Qdrant, in this prefix and in every other Cerveau prefix of the
        same Qdrant (self.purged_prefixes names them). Unlike archive(), this cannot be undone.

        Returns the number of journal records removed (0 for a memory found only in Qdrant).
        Copies made before (backups of the data folder, Qdrant snapshots) still hold the text.
        """
        point_id: str | int
        try:
            point_id = memory_id = str(uuid.UUID(memory_id))
        except ValueError as exc:
            if not memory_id.isascii() or not memory_id.isdigit() or int(memory_id) >= 2**64:
                raise BrainError(f"{memory_id!r} is not a memory id") from exc
            point_id = int(memory_id)  # a point written to Qdrant by something else
        # Qdrant must answer first: purging only the journal would leave an orphan that
        # `adopt-orphans` could bring back.
        self._require_ready(create=False)
        with self._locked():
            holders = [
                name
                for name in [self.store.memories, *self.store.other_memories()]
                if self.store.get(name, point_id) is not None
            ]
            known = any(_canonical_id(e["id"]) == memory_id for e in self.journal.events())
            if not known and not holders:
                raise BrainError(f"no memory with id {memory_id} (neither in the journal nor in Qdrant)")
            # The tombstone is written even for a memory found only in Qdrant: an older
            # backup that brings it back is then recognised (never adopted again).
            removed = self.journal.purge(
                lambda event: _canonical_id(event["id"]) == memory_id,
                {"op": "purge", "id": memory_id, "actor": "cli"},
                note_anyway=True,
            )
            self._journal_checked = False
            self.purged_prefixes = [n[: -len("_memories")] for n in holders if n != self.store.memories]
            try:
                for name in holders:
                    self.store.delete_points(name, [point_id])
            except Exception as exc:
                raise BrainError(
                    f"memory {memory_id} IS purged from the journal, but Qdrant failed "
                    f"({exc.__class__.__name__}): run `cerveau purge {memory_id} --confirm` again "
                    "once Qdrant is back."
                ) from exc
        return removed

    # -- search ----------------------------------------------------------------

    def search(self, query: str, k: int = 5, scope: str = "all", include_archived: bool = False) -> list[Hit]:
        """Search by meaning. Results from both collections are merged by raw cosine score."""
        self._require_ready(create=False)
        if scope not in SCOPES:
            raise BrainError(f"scope must be one of {sorted(SCOPES)}")
        if not 1 <= k <= 50:
            raise BrainError("k must be between 1 and 50")
        vector = self._embed([self._check_text(query, what="query")])[0]
        collections = {
            "docs": [self.store.docs],
            "memories": [self.store.memories],
            "all": [self.store.docs, self.store.memories],
        }[scope]
        hits: list[Hit] = []
        for collection in collections:
            short = "docs" if collection == self.store.docs else "memories"
            for point in self.store.search(collection, vector, k, include_archived):
                payload = point.payload or {}
                hits.append(
                    Hit(
                        score=round(float(point.score), 4),
                        collection=short,
                        id=str(point.id),
                        # Points written by something else may hold anything: text only, bounded.
                        text=_bounded_text(payload.get("text"), self.settings.max_text_chars),
                        source=payload.get("source"),
                        heading=payload.get("heading") or None,
                        kind=payload.get("kind"),
                        state=payload.get("state"),
                        ts=payload.get("ts"),
                        actor=payload.get("actor"),
                        path=payload.get("file"),
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def status(self, include_paths: bool = True) -> dict:
        self._require_ready(create=False)
        health = self.journal.health()
        status = {
            "embed_model": self.store.model,
            "vector_size": self.settings.vector_size,
            "collections": {
                "docs": self.store.count(self.store.docs),
                "memories": self.store.count(self.store.memories),
            },
            "journal_events": health.events,
        }
        if health.tolerated_torn_lines:
            status["journal_torn_lines"] = health.tolerated_torn_lines
        folded = self._fold_journal()
        in_journal = len(folded)
        status["memories_in_journal"] = in_journal  # journal_events also counts archive/restore
        warnings: list[str] = []
        if in_journal > status["collections"]["memories"]:
            warnings.append(
                f"the journal holds {in_journal} memories, Qdrant only "
                f"{status['collections']['memories']}: run `cerveau replay`"
            )
        orphans = self._orphans(folded, with_text=False)
        if orphans:
            warnings.append(
                f"{len(orphans)} memories in Qdrant are not in this journal (written with another "
                "CERVEAU_DATA_DIR? an older journal restored?): see `cerveau list --orphans`"
            )
        back = self._purged_back()
        if back:
            warnings.append(
                f"{len(back)} purged memories are in Qdrant again (an older backup restored?): "
                "`cerveau adopt-orphans` deletes them again"
            )
        archived_journal = sum(1 for item in folded.values() if item["state"] == "archived")
        archived_qdrant = sum(  # only memories of this journal: orphans are reported above
            1
            for record in self.store.scroll_payloads(self.store.memories, fields=["state"])
            if str(record.id) in folded and (record.payload or {}).get("state") == "archived"
        )
        if archived_journal != archived_qdrant:
            warnings.append(
                f"{archived_journal} memories are archived in the journal, {archived_qdrant} in "
                "Qdrant: run `cerveau replay`"
            )
        if warnings:
            status["warning"] = " | ".join(warnings)
        if include_paths:
            status["qdrant_url"] = redact_url(self.settings.qdrant_url)
            status["journal"] = str(self.settings.journal_path)
            status["env_file"] = str(self.settings.env_file) if self.settings.env_file else None
        return status
