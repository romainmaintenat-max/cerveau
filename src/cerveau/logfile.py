"""The local log file, for bug reports: <data_dir>/logs/cli.log and mcp.log.

It stays on this machine: nothing here ever sends it anywhere (the user reads it, then attaches it
to an issue if he wants). It holds versions, command and tool names, durations, counts, warnings
and errors with their traceback. Never a memory, a query, a document, a file's content or the
API key: messages of errors are left out (they may quote a text), except those known to be safe,
and every line goes through mask() (data folder, settings file's folder, home folder and service
addresses hidden).
CERVEAU_LOG=off: no folder, no file. Logging never breaks a command: its own errors are swallowed.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import time
import traceback
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlparse

from cerveau import __version__
from cerveau.config import Settings
from cerveau.embed import EmbeddingError

MAX_BYTES = 5_000_000  # per file; then cli.log.1 ... cli.log.3, the oldest one is dropped
BACKUP_COUNT = 3
log = logging.getLogger("cerveau.logfile")
log.addHandler(logging.NullHandler())  # off or not started: no "last resort" print to stderr
log.propagate = False  # never to stderr (the MCP server logs there) nor to stdout (MCP protocol)
log.setLevel(logging.INFO)
_pending: list[str] = []  # warnings shown before the log file could be opened


def mask(message: str, settings: Settings) -> str:
    """The text without local paths or service addresses: <data_dir>, <env_dir>, ~, <qdrant>, <ollama>."""
    folders = [(str(settings.data_dir), "<data_dir>"), (settings.data_dir.as_posix(), "<data_dir>")]
    home = Path.home()
    if len(str(home)) > 3:  # never replace "/" or "C:\\" everywhere
        folders += [(str(home), "~"), (home.as_posix(), "~")]
    if settings.env_file is not None and len(str(settings.env_file.parent)) > 3:  # the settings file's folder
        folders += [
            (str(settings.env_file.parent), "<env_dir>"),
            (settings.env_file.parent.as_posix(), "<env_dir>"),
        ]
    # The longest folder first: a folder inside another one is replaced whole, never left half-written.
    # Same folder twice: <data_dir>, then ~, then <env_dir> (sort() keeps the order of equal lengths).
    folders.sort(key=lambda folder: -len(folder[0]))
    replacements = [
        (settings.ollama_url.rstrip("/"), "<ollama>"),  # messages show the URL without its final "/"
        (settings.qdrant_url.rstrip("/"), "<qdrant>"),
        *folders,
        (settings.ollama_url, "<ollama>"),
        (settings.qdrant_url, "<qdrant>"),
    ]
    for url, name in ((settings.ollama_url, "<ollama>"), (settings.qdrant_url, "<qdrant>")):
        host = urlparse(url).hostname
        if host:
            replacements.append((host, name))
    for secret, placeholder in replacements:
        if secret and placeholder in ("<data_dir>", "<env_dir>", "~"):  # Windows paths ignore case
            message = re.sub(re.escape(secret), placeholder, message, flags=re.IGNORECASE)
        elif secret:
            message = message.replace(secret, placeholder)
    return message


def _private(path: str, flags: int) -> int:
    return os.open(path, flags, 0o600)  # readable by its owner only (POSIX)


class _Handler(RotatingFileHandler):
    def _open(self):  # noqa: ANN202
        return open(self.baseFilename, self.mode, encoding=self.encoding, errors=self.errors, opener=_private)

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        pass  # a full disk, a rotation refused by Windows...: the command goes on, unlogged


class _Masking(logging.Formatter):
    def __init__(self, settings: Settings) -> None:
        super().__init__("%(asctime)s [%(process)d] %(levelname)s %(message)s")
        self.settings = settings

    def format(self, record: logging.LogRecord) -> str:
        return mask(super().format(record), self.settings)


def start(settings: Settings, component: str, what: str) -> None:
    """Open <data_dir>/logs/<component>.log and write the start line (unless CERVEAU_LOG=off)."""
    stop()
    pending = _pending[:]
    _pending.clear()
    if not settings.log:
        return
    try:
        folder = settings.data_dir / "logs"
        settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        folder.mkdir(exist_ok=True, mode=0o700)
        handler = _Handler(
            folder / f"{component}.log",
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
            errors="backslashreplace",
        )
        handler.setFormatter(_Masking(settings))
        log.addHandler(handler)
        log.info(
            "start: cerveau %s, Python %s, %s, %s %s",
            __version__,
            platform.python_version(),
            platform.platform(),
            component,
            what,
        )
        for message in pending:
            warning(message)
    except Exception:
        stop()  # no log file (read-only folder...): the command runs the same


def stop() -> None:
    for handler in log.handlers[:]:
        if isinstance(handler, _Handler):
            log.removeHandler(handler)
            with suppress(Exception):  # called in a `finally`: must never raise
                handler.close()


def warning(message: str) -> None:
    log.warning("%s", message)


def describe(exc: BaseException) -> str:
    """The error type, and only what is known not to quote a text (a message may hold a memory)."""
    name = exc.__class__.__name__
    if isinstance(exc, EmbeddingError):
        return f"{name}: {exc.public}"  # written for the AI client: never the service's answer
    if isinstance(exc, OSError) and exc.strerror:
        return f"{name}: [Errno {exc.errno}] {exc.strerror}"  # no file name
    if isinstance(getattr(exc, "status_code", None), int):
        return f"{name}: HTTP {exc.status_code}"  # type: ignore[attr-defined]
    return name


def _frames(exc: BaseException) -> str:
    """Where the error happened, causes included: file, line, function, code. No message."""
    lines: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if lines:
            lines.append(f"cause: {describe(current)}\n")
        lines.extend(traceback.format_list(traceback.extract_tb(current.__traceback__)))
        current = current.__cause__ or current.__context__
    return "".join(lines).rstrip("\n")


def error(exc: BaseException) -> None:
    with suppress(Exception):  # logging never breaks a command
        log.error("%s\n%s", describe(exc), _frames(exc))


@contextmanager
def recording_warnings() -> Iterator[None]:
    """Show warnings as usual, and keep them for the log file (opened by start(), later)."""
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            yield
    finally:
        for item in caught:  # after catch_warnings: shown exactly as without it, error or not
            warnings.showwarning(
                item.message, item.category, item.filename, item.lineno, item.file, item.line
            )
            _pending.append(str(item.message))


def load_settings(load: Callable[[], Settings]) -> Settings:
    with recording_warnings():
        return load()


class Job:
    """One command or tool call: its name, duration, outcome and counts (never its arguments)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.started = time.monotonic()
        self.outcome = "ok"
        self.counts: dict[str, int] = {}

    def failed(self, exc: BaseException) -> None:
        with suppress(Exception):
            self.outcome = describe(exc)
        error(exc)

    def finish(self) -> None:
        with suppress(Exception):
            counts = "".join(f", {key} {value}" for key, value in self.counts.items())
            log.info("%s: %s in %.3f s%s", self.name, self.outcome, time.monotonic() - self.started, counts)
