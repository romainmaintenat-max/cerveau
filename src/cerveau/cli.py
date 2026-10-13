"""Command line: `cerveau <command>`."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import warnings
from dataclasses import asdict
from pathlib import Path

from cerveau import __version__, logfile
from cerveau.brain import STORAGE_ERRORS, Brain, BrainError
from cerveau.config import ConfigError, Settings
from cerveau.embed import EmbeddingError
from cerveau.journal import JournalError
from cerveau.store import SealMismatch

# Printable characters that display as nothing: Braille blank, Hangul fillers, and the
# "default ignorable" marks that Python counts as printable (grapheme joiner, Mongolian
# variation selectors, Khitan filler), plus the blank hieroglyphs.
_BLANK_LOOKALIKES = set(
    "\u2800\u3164\u115f\u1160\uffa0\u17b4\u17b5\u034f\u180b\u180c\u180d\u180f\U00016fe4\U00013441\U00013442"
)


def _hidden(char: str, inline: bool = False) -> bool:
    """A character that is invisible or acts on the display instead of being shown: controls,
    format characters (zero-width, bidi, Unicode "tags"), variation selectors, separators.
    Line breaks and tabs count as hidden only in a one-line field (inline=True)."""
    if char in "\n\t":
        return inline
    if char in _BLANK_LOOKALIKES:
        return True
    if unicodedata.category(char) == "Zs":
        return False  # visible spaces: no-break space of French typography, thin space...
    if not char.isprintable():
        return True
    return unicodedata.category(char) in {"Cf", "Zl", "Zp"} or (
        "\ufe00" <= char <= "\ufe0f" or "\U000e0100" <= char <= "\U000e01ef"
    )


def _escape(char: str) -> str:
    code = ord(char)
    return f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}" if code < 0x10000 else f"\\U{code:08x}"


def safe(text: str) -> str:
    """Text for a terminal: every hidden character is shown as an escape (\\x1b, \\u200b,
    \\U000e0041...), so a stored text can neither hide part of itself nor drive the terminal."""
    return "".join(_escape(c) if _hidden(c) else c for c in text)


def safe_line(text: str) -> str:
    """Like safe(), for a one-line field (a file name, a header): line breaks and tabs are
    escaped too, so a file name cannot print a fake result line."""
    return "".join(_escape(c) if _hidden(c, inline=True) else c for c in text)


def _json_escape(char: str) -> str:
    code = ord(char)
    if code < 0x10000:
        return f"\\u{code:04x}"
    code -= 0x10000  # JSON has no 32-bit escape: write a UTF-16 surrogate pair
    return f"\\u{0xD800 + (code >> 10):04x}\\u{0xDC00 + (code & 0x3FF):04x}"


def _json(data: object) -> str:
    """Valid JSON for scripts, with every hidden character written as a \\u escape."""
    text = json.dumps(data, indent=2, ensure_ascii=False)
    return "".join(_json_escape(c) if c not in "\n\t" and _hidden(c) else c for c in text)


def _excerpt(text: str, limit: int) -> str:
    """At most `limit` characters, and say so when something is left out."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [+{len(text) - limit} characters: add --json to see the full text]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cerveau",
        description="A local, sovereign memory for AI agents.",
        epilog='A text starting with "-" must follow "--": cerveau remember -- "-5 degrees"',
    )
    parser.add_argument("--version", action="version", version=f"cerveau {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the collections and check the configuration")

    p = sub.add_parser("index", help="index .md/.txt files of a folder (or one file)")
    p.add_argument("path")
    p.add_argument(
        "--prune",
        action="store_true",
        help="remove what is no longer indexable there: deleted files, and files now hidden or excluded",
    )

    p = sub.add_parser("unindex", help="remove indexed chunks at or under a path (may no longer exist)")
    p.add_argument("path")

    p = sub.add_parser("remember", help="store a memory")
    p.add_argument("text", nargs="+", help="the text, in quotes if an option follows it")
    p.add_argument("--kind", default="note", help="a short lowercase label: note, decision, lesson...")

    p = sub.add_parser("search", help="search by meaning")
    p.add_argument("query", nargs="+", help="what you are looking for (quotes are optional)")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--scope", choices=["all", "docs", "memories"], default="all")
    p.add_argument("--include-archived", action="store_true")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("archive", help="hide a memory from searches (reversible)")
    p.add_argument("id")
    p = sub.add_parser("restore", help="make an archived memory visible again")
    p.add_argument("id")

    p = sub.add_parser("list", help="list the memories of the journal (to review what agents wrote)")
    p.add_argument("--actor", help='only memories written by this actor ("mcp" = AI clients, "cli")')
    p.add_argument("--include-archived", action="store_true")
    p.add_argument("--orphans", action="store_true", help="memories found in Qdrant but not in this journal")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser(
        "reset-collections",
        help="delete this prefix's Qdrant collections (journal and files are kept)",
    )
    p.add_argument("--confirm", metavar="PREFIX", required=True, help="type the prefix to confirm")
    p.add_argument(
        "--discard-orphans",
        action="store_true",
        help="delete even if Qdrant holds memories that are not in the journal (they will be lost)",
    )

    sub.add_parser(
        "adopt-orphans",
        help="copy into this journal the memories found only in Qdrant (see: list --orphans)",
    )

    p = sub.add_parser(
        "purge",
        help="delete a memory for good, from the journal and Qdrant (archive hides; purge erases)",
    )
    p.add_argument("id")
    p.add_argument("--confirm", action="store_true", help="required: this cannot be undone")

    sub.add_parser("replay", help="rebuild the memories collection from the journal")
    sub.add_parser("status", help="show counts and configuration")
    parser.allow_abbrev = False  # "--include" is not silently "--include-archived"
    for command in sub.choices.values():
        command.allow_abbrev = False
    return parser


_VALUE_OPTIONS = {"--kind"}


def _swallowed_words(argv: list[str]) -> bool:
    """`cerveau remember Always pass the --kind lesson`: several unquoted words, then an
    option that takes a value. argparse would store "Always pass the" with kind "lesson":
    a text silently cut. True when the words and the option cannot be told apart."""
    if not argv or argv[0] != "remember":  # a search loses nothing: only stored text matters
        return False
    rest = argv[1:]
    if "--" in rest:
        rest = rest[: rest.index("--")]
    words = 0
    skip = False
    for token in rest:
        if skip:
            skip = False
        elif token in _VALUE_OPTIONS or token.split("=", 1)[0] in _VALUE_OPTIONS:
            if words > 1:
                return True
            skip = "=" not in token
        elif not token.startswith("-"):
            words += 1
    return False


def run(argv: list[str] | None = None, brain: Brain | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    if _swallowed_words(raw):
        print(
            "error: several words, then an option: put the text in quotes, or the options first "
            '(cerveau remember --kind lesson "the text")',
            file=sys.stderr,
        )
        return 2
    args = build_parser().parse_args(argv)
    job = logfile.Job(args.command)  # the local log file: see logfile.py
    try:
        settings = brain.settings if brain else logfile.load_settings(Settings.from_env)
        logfile.start(settings, "cli", args.command)
        brain = brain or Brain(settings)
        if brain.settings.embedder == "hash" and brain.settings.qdrant_url != ":memory:":
            logfile.warning("CERVEAU_EMBEDDER=hash is a fake, non-semantic embedder for tests only")
            print(
                "warning: CERVEAU_EMBEDDER=hash is a fake, non-semantic embedder for tests only",
                file=sys.stderr,
            )
        if args.command == "init":
            brain.init()
            print(_json(brain.status()))
        elif args.command == "index":
            if not args.path.strip():  # `cerveau index "$NOTES"` with NOTES unset: not the current folder
                raise BrainError("give the folder or file to index")
            args.path = Path(args.path)
            if args.prune and not args.path.expanduser().is_dir():
                print("warning: --prune only applies to a folder", file=sys.stderr)
            show = sys.stderr.isatty()
            report = brain.index_path(
                args.path,
                prune=args.prune,
                progress=(lambda line: print(safe_line(line), file=sys.stderr)) if show else None,
            )
            print(_json(asdict(report)))
            job.counts = {k: v for k, v in asdict(report).items() if isinstance(v, int)}
            job.counts["errors"] = len(report.errors)
            if report.files_seen == 0:
                print(
                    "warning: no .md, .markdown or .txt file found there (PDF and Office files "
                    "are not supported)",
                    file=sys.stderr,
                )
            if report.stale_files:
                where = safe_line(str(args.path))
                print(
                    f"warning: {report.stale_files} indexed files are no longer there and can still be "
                    f"found by searches: `cerveau index {where} --prune` removes them",
                    file=sys.stderr,
                )
            return 1 if report.errors else 0
        elif args.command == "unindex":
            removed = brain.unindex(args.path)
            job.counts["chunks_removed"] = removed
            print(f"removed {removed} chunks")
        elif args.command == "remember":
            print(brain.remember(" ".join(args.text), kind=args.kind, actor="cli"))
        elif args.command == "search":
            hits = brain.search(
                " ".join(args.query), k=args.k, scope=args.scope, include_archived=args.include_archived
            )
            job.counts["results"] = len(hits)
            if args.json:
                print(_json([h.to_dict(include_path=True) for h in hits]))
            elif not hits:
                print("no result")
            else:
                for hit in hits:
                    where = hit.path or hit.kind or ""
                    who = f" by {hit.actor} {hit.ts}" if hit.actor else ""
                    who += " [archived]" if hit.state == "archived" else ""
                    print(safe_line(f"[{hit.score:.3f}] {hit.collection}:{where} ({hit.id}){who}"))
                    print("    " + safe(_excerpt(hit.text, 300)).replace("\n", "\n    "))
        elif args.command == "list":
            if args.orphans:
                items = brain.orphan_memories()
            else:
                items = brain.list_memories(actor=args.actor, include_archived=args.include_archived)
            job.counts["memories"] = len(items)
            if args.json:
                print(_json(items))
            elif not items:
                print("no memory")
            else:
                for item in items:
                    flag = " [archived]" if item["state"] == "archived" else ""
                    ts, actor, kind = (str(item.get(k) or "?") for k in ("ts", "actor", "kind"))
                    origin = item.get("adopted_from")
                    origin = f" (adopted, written by {origin})" if origin else ""
                    print(safe_line(f"{ts} {actor:<4} {kind:<10} {item['id']}{flag}{origin}"))
                    text = str(item.get("text") or "")
                    print("    " + safe(text).replace("\n", "\n    "))  # never cut: this is the review
        elif args.command == "reset-collections":
            if args.confirm != brain.settings.prefix:
                raise BrainError(f"--confirm must be the current prefix ({brain.settings.prefix!r})")
            brain.reset_collections(discard_orphans=args.discard_orphans)
            print(
                f"deleted the collections of prefix {brain.settings.prefix!r}. "
                "Now: cerveau init, cerveau replay, cerveau index <your folders>"
            )
        elif args.command == "adopt-orphans":
            adopted = brain.adopt_orphans()
            job.counts["adopted"] = adopted
            again = (
                f"; deleted {brain.purged_again} purged memories found again in Qdrant"
                if brain.purged_again
                else ""
            )
            print(f"adopted {adopted} memories into the journal{again}")
            if brain.skipped_orphans:
                print(
                    safe_line(
                        f"warning: {len(brain.skipped_orphans)} points of Qdrant are not memories Cerveau "
                        f"could have written, left out: {', '.join(brain.skipped_orphans[:10])}. "
                        "Delete them with: cerveau purge <id> --confirm"
                    ),
                    file=sys.stderr,
                )
        elif args.command == "archive":
            print(f"archived {brain.archive(args.id, actor='cli')}")
        elif args.command == "restore":
            print(f"restored {brain.restore(args.id, actor='cli')}")
        elif args.command == "purge":
            if not args.confirm:
                raise BrainError(
                    "purge deletes the memory for good (to hide it reversibly: cerveau archive). "
                    f"Add --confirm: cerveau purge {args.id} --confirm"
                )
            removed = brain.purge(args.id)
            if brain.purged_prefixes:
                print(safe_line(f"also deleted from the prefixes: {', '.join(brain.purged_prefixes)}"))
            print(
                f"purged {args.id} ({removed} journal records removed). Backups made before "
                "(copies of the data folder, Qdrant snapshots) still hold its text."
            )
        elif args.command == "replay":
            replayed = brain.replay(progress=lambda line: print(line, file=sys.stderr, flush=True))
            job.counts["replayed"] = replayed
            print(f"replayed {replayed} memories from the journal")
        elif args.command == "status":
            print(_json(brain.status()))
    except (BrainError, ConfigError, EmbeddingError, JournalError, SealMismatch) as exc:
        job.failed(exc)
        print(f"error: {safe(str(exc))}", file=sys.stderr)
        return 2
    except KeyboardInterrupt as exc:
        job.failed(exc)
        print(
            "interrupted. What was written so far is consistent: run `cerveau status` (it says when "
            "`cerveau replay` is needed) before running the command again",
            file=sys.stderr,
        )
        return 130
    except BrokenPipeError:
        job.outcome = "output closed by the reader"
        return 0  # output closed by the reader (`| head`): not an error
    except json.JSONDecodeError as exc:
        job.failed(exc)
        print(
            f"error: the service did not answer with JSON ({exc.msg}): check CERVEAU_QDRANT_URL "
            "and CERVEAU_OLLAMA_URL point to Qdrant and Ollama",
            file=sys.stderr,
        )
        return 2
    except OSError as exc:
        job.failed(exc)
        print(f"error: file system: {safe(str(exc))}", file=sys.stderr)
        return 2
    except STORAGE_ERRORS as exc:
        job.failed(exc)
        hint = ""
        if getattr(exc, "status_code", None) in (401, 403):
            hint = (
                " (wrong or missing API key: check CERVEAU_QDRANT_API_KEY, and that the right .env "
                "is loaded: run from the folder that holds it, or set CERVEAU_ENV_FILE. Changed the "
                "key? `docker compose up -d` gives it to Qdrant)"
            )
        elif getattr(exc, "status_code", None) == 404:
            hint = (
                " (the collections do not exist: run `cerveau init`, then `cerveau replay`; or "
                "CERVEAU_QDRANT_URL does not point to Qdrant)"
            )
        elif getattr(exc, "status_code", None) is None:
            hint = " (is Qdrant running? `docker compose up -d` in the cerveau folder)"
        detail = safe(str(exc).splitlines()[0]) if str(exc) else exc.__class__.__name__
        print(f"error: Qdrant is unavailable or refused the request: {detail}{hint}", file=sys.stderr)
        return 2
    except AssertionError as exc:  # qdrant-client checks the shape of the answers it gets
        job.failed(exc)
        print(
            "error: the service at CERVEAU_QDRANT_URL does not answer like Qdrant (check the URL "
            "and the port)",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # last resort: a clear line instead of a traceback
        job.failed(exc)
        print(
            f"error: unexpected {exc.__class__.__name__}: {safe(str(exc))} "
            "(please report it, with the command you ran)",
            file=sys.stderr,
        )
        return 2
    finally:
        job.finish()
        logfile.stop()
    return 0


def _utf8_output() -> None:
    """Never crash on a character the console code page cannot show (Windows)."""
    for stream in (sys.stdout, sys.stderr):
        if not hasattr(stream, "reconfigure"):
            continue
        if stream.isatty():
            stream.reconfigure(errors="replace")
        else:
            # files and pipes get UTF-8; undecodable file names are escaped, never fatal
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def _plain_warning(message, category, filename, lineno, line=None) -> str:  # noqa: ANN001
    return f"warning: {message}\n"


def main() -> None:
    _utf8_output()
    warnings.formatwarning = _plain_warning  # "warning: ..." instead of a source line
    sys.exit(run())


if __name__ == "__main__":
    main()
