"""Settings, read from environment variables (and an optional .env file).

Every setting has a CERVEAU_ prefix. Nothing is hard-coded to a machine:
no host name, no user name, no absolute path.
"""

from __future__ import annotations

import os
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
# Every setting Cerveau reads. Another CERVEAU_* name close to one of these is refused: a typo in a
# safety setting (CERVEAU_MCP_ALLOW_WRITE) must not silently fall back to its default.
KNOWN_KEYS = {
    "CERVEAU_ENV_FILE",
    "CERVEAU_QDRANT_URL",
    "CERVEAU_QDRANT_API_KEY",
    "CERVEAU_OLLAMA_URL",
    "CERVEAU_OLLAMA_TIMEOUT",
    "CERVEAU_EMBEDDER",
    "CERVEAU_EMBED_MODEL",
    "CERVEAU_VECTOR_SIZE",
    "CERVEAU_MAX_TEXT_CHARS",
    "CERVEAU_CHUNK_MAX_CHARS",
    "CERVEAU_PREFIX",
    "CERVEAU_DATA_DIR",
    "CERVEAU_MCP_ALLOW_WRITE",
    "CERVEAU_LOG",
    "CERVEAU_QDRANT_PORT",  # read by docker-compose.yml only
}
TEST_KEY_PREFIX = "CERVEAU_IT_"  # used by the integration tests only
HEADING_BUDGET = 256  # characters reserved for the heading path prepended to each chunk
# An unknown name is a typo, and is refused, when it is at most this many edits away from a known one
# (at most half the length of a short one: CERVEAU_DB is not CERVEAU_LOG), when its words are a part or
# a superset of a known one's (CERVEAU_ALLOW_WRITE, CERVEAU_DATA), or when it mentions writing,
# read-only mode or the "remember" tool (CERVEAU_READONLY, CERVEAU_WRITABLE, CERVEAU_RO, CERVEAU_RW,
# CERVEAU_NO_REMEMBER): a missed CERVEAU_MCP_ALLOW_WRITE=false would leave MCP writes on.
# Any other name belongs to another program ("cerveau" is a common word): a warning, not an error.
TYPO_DISTANCE = 3
GUARD_WORDS = {"WRITE", "WRITABLE", "READ", "READONLY", "ONLY", "REMEMBER", "RO", "RW"}  # whole words
NOT_OURS_HINT = "if this variable belongs to another program, rename or unset it"


class ConfigError(ValueError):
    """Raised when the configuration is unsafe or inconsistent."""


def _visible_name(name: str) -> str:
    """The name without the invisible characters a copy from a web page can bring (U+200B...)."""
    return "".join(c for c in name if unicodedata.category(c) != "Cf").strip()


def _misspelt(name: str) -> bool:
    """A name meant for Cerveau that the "CERVEAU_" test would silently skip:
    "CERVEAU-MCP-ALLOW-WRITE", "CERVEAU MCP_...", "CERVEAUMCP_ALLOW_WRITE", "\u200bCERVEAU_...".
    Every name starting with CERVEAU (any case) is checked: see _is_typo."""
    visible = _visible_name(name)
    return visible.upper().startswith("CERVEAU") and not name.upper().startswith("CERVEAU_")


def _distance(a: str, b: str) -> int:
    """Levenshtein distance: insertions, deletions and substitutions needed to turn a into b."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _normal(name: str) -> str:
    """CERVEAU_ followed by the name's words, in capitals, separated by "_"."""
    normal = re.sub(r"[^A-Za-z0-9]+", "_", _visible_name(name)).upper()
    if not normal.startswith("CERVEAU_"):
        normal = "CERVEAU_" + normal[len("CERVEAU") :].lstrip("_")
    return normal


def _words(normal: str) -> set[str]:
    return {word for word in normal[len("CERVEAU_") :].split("_") if word}


def _typo_of(name: str) -> str | None:
    """The setting an unknown name was probably meant to be (see TYPO_DISTANCE), or None."""
    normal = _normal(name)
    close = [
        (distance, known)
        for known in sorted(KNOWN_KEYS)
        if (distance := _distance(normal, known)) <= min(TYPO_DISTANCE, len(known[len("CERVEAU_") :]) // 2)
    ]
    if close:
        return min(close)[1]
    words = _words(normal)
    same_words = [
        (_distance(normal, known), known)
        for known in sorted(KNOWN_KEYS)
        if words and (words <= _words(known) or words >= _words(known))
    ]
    if same_words:
        return min(same_words)[1]
    rest = normal[len("CERVEAU_") :]
    if words & GUARD_WORDS or "WRIT" in rest or "READONLY" in rest or "REMEMBER" in rest:
        return "CERVEAU_MCP_ALLOW_WRITE"
    return None


def _suggest(name: str) -> str | None:
    return _typo_of(name)


def _is_typo(name: str) -> bool:
    return _typo_of(name) is not None


def _ignore(where: str) -> None:
    warnings.warn(
        f"{where} is ignored: it does not look like any Cerveau setting "
        "(if it was meant for Cerveau, rename it)",
        stacklevel=3,
    )


def _strip_inline_comment(value: str) -> str:
    """Remove a trailing ' # comment' from an unquoted value, as docker compose does."""
    match = re.search(r"\s+#", value)
    return value[: match.start()] if match else value


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse the CERVEAU_* lines of a .env file.

    Other lines are ignored, so a project's own .env (multi-line values, other syntax)
    never breaks Cerveau. Handles UTF-8 (with or without BOM), UTF-16 with BOM (what
    PowerShell 5 writes with `>`), quotes, inline comments and `export `.
    """
    data = path.read_bytes()
    try:
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = data.decode("utf-16")
        else:
            text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{path} is not valid UTF-8: save it as UTF-8") from exc
    values: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if not line.upper().startswith("CERVEAU_"):
            name = line.split("=", 1)[0].strip()
            if "=" in line and _misspelt(name):
                if _is_typo(name):
                    raise ConfigError(
                        f"{path}:{lineno}: {name} is not a setting: did you mean {_suggest(name)}? "
                        f"({NOT_OURS_HINT})"
                    )
                _ignore(f"{path}:{lineno}: {name}")
            continue  # other tools' settings are none of our business
        if "=" not in line:
            raise ConfigError(f"{path}:{lineno}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key != key.upper():
            raise ConfigError(f"{path}:{lineno}: {key} must be written in capitals ({key.upper()})")
        if value[:1] in {'"', "'"}:
            closing = value.find(value[0], 1)
            if closing == -1:
                raise ConfigError(f"{path}:{lineno}: unterminated quote")
            value = value[1:closing]
        else:
            value = _strip_inline_comment(value)
        values[key] = value
    return values


_POSIX_VAR = re.compile(r"\$\{(\w+)\}|\$([A-Za-z_]\w*)")
_WINDOWS_VAR = re.compile(r"%(\w+)%")


def _expand_vars(value: str, environ: dict[str, str], windows: bool | None = None) -> str:
    """Expand variables the way the platform's shell does; an undefined one is an error.

    Windows: %VAR%, case-insensitive (a literal "$" in a path is kept).
    Elsewhere: $VAR and ${VAR}.
    """
    windows = os.name == "nt" if windows is None else windows
    if windows:
        lookup = {k.upper(): v for k, v in environ.items()}
        pattern = _WINDOWS_VAR
    else:
        lookup = environ
        pattern = _POSIX_VAR

    def substitute(match: re.Match) -> str:
        name = next(g for g in match.groups() if g)
        key = name.upper() if windows else name
        if key not in lookup:
            raise ConfigError(f"CERVEAU_DATA_DIR uses an undefined variable: {name}")
        return lookup[key]

    return pattern.sub(substitute, value)


def _check_url(url: str, name: str) -> str:
    """http(s) URL with a host and a valid port; returns the host."""
    if "\\" in url or any(c.isspace() for c in url):
        # urlparse and HTTP libraries disagree on such URLs (which host is really used?)
        raise ConfigError(f"{name} must not contain backslashes or spaces: {redact_url(url)!r}")
    try:
        parsed = urlparse(url)
        parsed.port  # noqa: B018  (raises ValueError when out of range)
    except ValueError as exc:
        raise ConfigError(f"{name} is not a valid URL: {redact_url(url)!r}") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(f"{name} must be an http:// or https:// URL, got {redact_url(url)!r}")
    return parsed.hostname


def redact_url(url: str) -> str:
    """The URL without user name or password (for display), even when it is malformed."""
    try:
        parsed = urlparse(url)
        if parsed.username is None and parsed.password is None:
            return url
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError:  # bad port, broken IPv6...: hide whatever comes before an "@"
        return re.sub(r"(//)[^/?#]*@", r"\1***@", url)
    netloc = f"[{host}]" if ":" in host else host
    if port:
        netloc += f":{port}"
    return parsed._replace(netloc=netloc).geturl()


def _bool(value: str, name: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name}: expected a boolean, got {value!r}")


def _int(value: str, name: str, minimum: int) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise ConfigError(f"{name}: expected an integer, got {value!r}") from exc
    if number < minimum:
        raise ConfigError(f"{name}: must be >= {minimum}, got {number}")
    return number


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = field(default=None, repr=False)
    ollama_url: str = "http://127.0.0.1:11434"
    embedder: str = "ollama"
    embed_model: str = "bge-m3"
    vector_size: int = 1024
    max_text_chars: int = 3000
    chunk_max_chars: int = 1500
    ollama_timeout: float = 300.0
    prefix: str = "cerveau"
    data_dir: Path = field(default_factory=lambda: Path.home() / ".cerveau")
    mcp_allow_write: bool = True
    log: bool = True  # local log file in <data_dir>/logs (CERVEAU_LOG=off: none)
    env_file: Path | None = None  # informational: the .env file that was loaded, if any

    @property
    def journal_path(self) -> Path:
        return self.data_dir / "journal.jsonl"

    def validate(self) -> None:
        if self.embedder not in {"ollama", "hash"}:
            raise ConfigError("CERVEAU_EMBEDDER must be 'ollama' or 'hash'")
        if self.qdrant_api_key is not None and not re.fullmatch(r"[\x21-\x7e]+", self.qdrant_api_key):
            raise ConfigError(
                "CERVEAU_QDRANT_API_KEY must be printable ASCII without spaces (a typographic quote or "
                "a no-break space was probably pasted with it)"
            )
        if self.qdrant_api_key and ("$" in self.qdrant_api_key or "\\" in self.qdrant_api_key):
            raise ConfigError(
                "CERVEAU_QDRANT_API_KEY must not contain '$' or '\\': Docker Compose would read another key"
            )
        if self.qdrant_api_key and len(self.qdrant_api_key) < 16:
            warnings.warn(
                "CERVEAU_QDRANT_API_KEY is short: generate one with "
                '`python -c "import secrets; print(secrets.token_urlsafe(32))"`',
                stacklevel=2,
            )
        if not re.fullmatch(r"[A-Za-z0-9_]{1,40}", self.prefix):
            raise ConfigError("CERVEAU_PREFIX must be 1 to 40 ASCII letters, digits or underscores")
        if self.chunk_max_chars + HEADING_BUDGET > self.max_text_chars:
            raise ConfigError(
                f"CERVEAU_CHUNK_MAX_CHARS must be at most CERVEAU_MAX_TEXT_CHARS - {HEADING_BUDGET} "
                "(room for the heading added to each chunk)"
            )
        if not self.data_dir.is_absolute():
            raise ConfigError(f"CERVEAU_DATA_DIR must be an absolute path, got {str(self.data_dir)!r}")
        ollama_host = _check_url(self.ollama_url, "CERVEAU_OLLAMA_URL")
        if self.embedder == "ollama" and ollama_host not in LOOPBACK_HOSTS:
            warnings.warn(
                f"CERVEAU_OLLAMA_URL points to {ollama_host}: every text you index or remember "
                "is sent there. Use HTTPS or a private network.",
                stacklevel=2,
            )
        if self.qdrant_url == ":memory:":
            return
        host = _check_url(self.qdrant_url, "CERVEAU_QDRANT_URL")
        if not self.qdrant_api_key and host not in LOOPBACK_HOSTS:
            raise ConfigError(
                "Refusing to talk to a non-local Qdrant without an API key. Set CERVEAU_QDRANT_API_KEY."
            )
        if host not in LOOPBACK_HOSTS and urlparse(self.qdrant_url).scheme == "http":
            warnings.warn(
                f"CERVEAU_QDRANT_URL uses plain http to {host}: the API key travels unencrypted "
                "(fine on a private encrypted network, otherwise use https)",
                stacklevel=2,
            )

    @classmethod
    def from_env(
        cls,
        env: dict[str, str] | None = None,
        env_file: Path | None | str = "auto",
        allow_cwd_env: bool = True,
    ) -> Settings:
        """Build settings from a .env file, overridden by non-empty environment variables.

        env_file="auto": use CERVEAU_ENV_FILE if set (it must exist), else ./.env if present.
        env_file=None: no file. A Path: that file (it must exist).
        allow_cwd_env=False: never read ./.env implicitly (used by the MCP server, which is
        started from a folder chosen by the AI client, possibly an untrusted project).
        """
        environ = dict(os.environ if env is None else env)
        implicit_env = False
        unknown_env = sorted(
            k
            for k in environ
            if k.upper().startswith("CERVEAU_") and k not in KNOWN_KEYS and not k.startswith(TEST_KEY_PREFIX)
        )
        misspelt = sorted(k for k in environ if _misspelt(k))
        if typos := [k for k in misspelt if _is_typo(k)]:
            raise ConfigError(
                "misspelt setting(s) in the environment: "
                + ", ".join(f"{k} (did you mean {_suggest(k)}?)" for k in typos)
                + f"; {NOT_OURS_HINT}"
            )
        if typos := [k for k in unknown_env if _is_typo(k)]:
            raise ConfigError(
                "unknown setting(s) in the environment: "
                + ", ".join(f"{k} (did you mean {_suggest(k)}?)" for k in typos)
                + f"; {NOT_OURS_HINT}"
            )
        for k in misspelt + unknown_env:
            _ignore(f"the environment variable {k}")
        if env_file == "auto":
            explicit = environ.get("CERVEAU_ENV_FILE")
            if explicit:
                env_file = Path(explicit).expanduser()
                if not env_file.is_absolute():
                    # A relative path would be read from wherever the AI client starts the server,
                    # perhaps a project someone else wrote.
                    raise ConfigError(f"CERVEAU_ENV_FILE must be an absolute path, got {explicit!r}")
                if not env_file.is_file():
                    raise ConfigError(f"CERVEAU_ENV_FILE points to a missing file: {env_file}")
            else:
                env_file = Path(".env") if allow_cwd_env and Path(".env").is_file() else None
                implicit_env = env_file is not None
        file_values: dict[str, str] = {}
        if env_file is not None:
            env_file = Path(env_file).resolve()
            if not env_file.is_file():
                raise ConfigError(f".env file not found: {env_file}")
            try:
                file_values = parse_env_file(env_file)
            except ConfigError as exc:
                if not implicit_env or not isinstance(exc.__cause__, UnicodeDecodeError):
                    raise
                # ./.env of some other project, in another encoding: not ours to judge.
                warnings.warn(f"ignored {env_file}: not UTF-8 (set CERVEAU_ENV_FILE)", stacklevel=2)
                env_file, implicit_env = None, False
            if "CERVEAU_ENV_FILE" in file_values:
                raise ConfigError(
                    f"{env_file}: CERVEAU_ENV_FILE cannot be set inside a .env file (set it in the "
                    "environment, or in the configuration of your AI client)"
                )
            unknown_file = sorted(k for k in file_values if k not in KNOWN_KEYS)
            if typos := [k for k in unknown_file if _is_typo(k)]:
                raise ConfigError(
                    f"unknown setting(s) in {env_file}: "
                    + ", ".join(f"{k} (did you mean {_suggest(k)}?)" for k in typos)
                    + f"; {NOT_OURS_HINT}"
                )
            for k in unknown_file:
                _ignore(f"{env_file}: {k}")
            has_key = bool(file_values.get("CERVEAU_QDRANT_API_KEY"))
            mode = env_file.stat().st_mode if os.name == "posix" and env_file is not None else 0
            if mode & 0o022:  # anyone who can write it decides where texts go and whether MCP may write
                warnings.warn(
                    f"{env_file} can be modified by other users and decides where your texts go: "
                    f"run `chmod 600 {env_file}`",
                    stacklevel=2,
                )
            elif mode & 0o077 and (has_key or not implicit_env):  # a named file is the user's own
                warnings.warn(
                    f"{env_file} can be read by other users"
                    + (" and holds the Qdrant API key" if has_key else "")
                    + f": run `chmod 600 {env_file}`",
                    stacklevel=2,
                )

        def get(name: str) -> str | None:
            key = f"CERVEAU_{name}"
            if environ.get(key):
                return environ[key]
            return file_values.get(key) or None

        kwargs: dict[str, object] = {}
        for name, attr in [
            ("QDRANT_URL", "qdrant_url"),
            ("QDRANT_API_KEY", "qdrant_api_key"),
            ("OLLAMA_URL", "ollama_url"),
            ("EMBEDDER", "embedder"),
            ("EMBED_MODEL", "embed_model"),
            ("PREFIX", "prefix"),
        ]:
            if (value := get(name)) is not None:
                kwargs[attr] = value
        for name, attr, minimum in [
            ("VECTOR_SIZE", "vector_size", 2),
            ("MAX_TEXT_CHARS", "max_text_chars", 100),
            ("CHUNK_MAX_CHARS", "chunk_max_chars", 100),
        ]:
            if (value := get(name)) is not None:
                kwargs[attr] = _int(value, f"CERVEAU_{name}", minimum)
        if (value := get("OLLAMA_TIMEOUT")) is not None:
            kwargs["ollama_timeout"] = float(_int(value, "CERVEAU_OLLAMA_TIMEOUT", 1))
        if (value := get("DATA_DIR")) is not None:
            expanded = _expand_vars(value, environ)
            if "\x00" in expanded:
                raise ConfigError("CERVEAU_DATA_DIR contains a NUL character")
            try:
                data_dir = Path(expanded).expanduser()
            except RuntimeError as exc:  # "~someone" who does not exist
                raise ConfigError(f"CERVEAU_DATA_DIR: {exc}") from exc
            from_file = not environ.get("CERVEAU_DATA_DIR") and env_file is not None
            if not data_dir.is_absolute() and from_file:
                # Relative to the .env file, never to the current directory: the MCP server
                # may be started from anywhere and must find the same journal.
                data_dir = (env_file.parent / data_dir).resolve()
            kwargs["data_dir"] = data_dir
        kwargs["env_file"] = env_file
        # Like every setting, an empty environment variable leaves the file's value in force;
        # an empty value in the file is an error, not "use the default".
        allow_write = environ.get("CERVEAU_MCP_ALLOW_WRITE") or file_values.get("CERVEAU_MCP_ALLOW_WRITE")
        if allow_write is not None:
            kwargs["mcp_allow_write"] = _bool(allow_write, "CERVEAU_MCP_ALLOW_WRITE")
        log = environ.get("CERVEAU_LOG") or file_values.get("CERVEAU_LOG")
        if log is not None:
            kwargs["log"] = _bool(log, "CERVEAU_LOG")
        if kwargs.get("qdrant_url") == ":memory:":
            raise ConfigError("CERVEAU_QDRANT_URL=:memory: is for the test suite only")

        if implicit_env:
            # A .env picked up from the current folder may belong to someone else's project:
            # it may not send your texts (or your API key, set in your shell) to another machine.
            for name in ("CERVEAU_QDRANT_URL", "CERVEAU_OLLAMA_URL"):
                url = file_values.get(name)
                if url and not environ.get(name) and _check_url(url, name) not in LOOPBACK_HOSTS:
                    raise ConfigError(
                        f"{env_file} (found in the current folder) sets {name} to a non-local "
                        f"address. If this is your file, set CERVEAU_ENV_FILE={env_file}"
                    )
        settings = cls(**kwargs)  # type: ignore[arg-type]
        settings.validate()
        return settings
