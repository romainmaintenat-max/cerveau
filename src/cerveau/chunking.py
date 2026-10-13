"""Split Markdown or plain text into chunks small enough to embed.

Strategy, in order:
1. split on Markdown headings, each chunk remembers its heading path;
2. a section that is still too long is split on blank lines (paragraphs);
3. a paragraph that is still too long is split on whitespace.
No word is ever dropped: joining the chunks gives back every word of the input, except encoded
data (pasted images, long base64 or hexadecimal runs), replaced by a marker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def _heading(line: str) -> tuple[int, str] | None:
    """Parse an ATX heading ("## Title ##") in linear time (no backtracking regex)."""
    level = len(line) - len(line.lstrip("#"))
    if not 1 <= level <= 6:
        return None
    rest = line[level:]
    if rest and not rest[0].isspace():
        return None  # "#hashtag" is not a heading
    title = rest.strip().rstrip("#").rstrip()
    return level, title


@dataclass(frozen=True)
class Chunk:
    heading: str
    text: str


def _sections(text: str) -> list[tuple[str, str]]:
    path: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = [("", [])]
    fence: str | None = None  # the marker that opened the current code block
    for line in text.splitlines():
        stripped = line.lstrip()
        for marker in ("```", "~~~"):
            if stripped.startswith(marker):
                if fence is None:
                    fence = marker
                elif fence == marker:
                    fence = None
                break
        match = None if fence is not None else _heading(line)
        if match:
            level, title = match
            path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, title)]
            sections.append((" > ".join(t for _, t in path), [line]))
        else:
            sections[-1][1].append(line)
    # A heading with no text of its own (immediately followed by a sub-heading) would be
    # a useless chunk: it is carried into the next section instead.
    merged: list[tuple[str, str]] = []
    carried: list[str] = []
    for heading, lines in sections:
        body = [line for line in lines if line.strip()]
        if heading and len(body) == 1:
            carried.append(lines[0])
            continue
        merged.append((heading, "\n".join(carried + lines).strip()))
        carried = []
    if carried:
        merged.append((sections[-1][0], "\n".join(carried).strip()))
    return merged


def _split_words(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _pack(parts: list[str], separator: str, max_chars: int) -> list[str]:
    """Greedily join parts with `separator` without exceeding max_chars."""
    pieces: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{separator}{part}" if current else part
        if len(candidate) > max_chars:
            if current:
                pieces.append(current)
            current = part
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split on line breaks first (keeps code and lists readable), then on words."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    lines: list[str] = []
    for line in paragraph.split("\n"):
        lines.extend([line] if len(line) <= max_chars else _split_words(line, max_chars))
    return _pack([line for line in lines if line.strip()], "\n", max_chars)


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip("\n")
        if paragraph.strip():
            parts.extend(_split_paragraph(paragraph, max_chars))
    return _pack(parts, "\n\n", max_chars)


# Pasted images in Markdown (Obsidian, Typora...): data URIs, and runs of base64 or hexadecimal
# characters long enough to be encoded data, never words. Embedding them is slow and fills
# searches with noise; they are replaced by a short marker. ASCII only: a paragraph of Chinese
# or Japanese, written without spaces, is text and is kept.
_DATA_URI = re.compile(r"data:[\w.+/-]*;base64,[A-Za-z0-9+/=]{64,}")
_BLOB = re.compile(r"[A-Za-z0-9+/=_-]{1000,}")
BLOB_MARKER = "[binary data omitted]"


def strip_blobs(text: str) -> str:
    text = _DATA_URI.sub(lambda m: m.group().split(",", 1)[0] + "," + BLOB_MARKER, text)
    return _BLOB.sub(BLOB_MARKER, text)


def chunk_text(text: str, max_chars: int) -> list[Chunk]:
    if max_chars < 20:
        raise ValueError("max_chars must be at least 20")
    text = strip_blobs(text)
    chunks: list[Chunk] = []
    for heading, body in _sections(text):
        if not body:
            continue
        for piece in _split_long(body, max_chars):
            if piece.strip():
                chunks.append(Chunk(heading=heading, text=piece))
    return chunks
