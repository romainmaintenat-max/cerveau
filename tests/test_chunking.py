import pytest

from cerveau.chunking import chunk_text


def words(text: str) -> list[str]:
    return text.split()


def test_headings_become_paths():
    text = "intro line\n# A\nalpha\n## B\nbeta\n# C\ngamma\n"
    chunks = chunk_text(text, 200)
    assert [c.heading for c in chunks] == ["", "A", "A > B", "C"]


def test_heading_inside_code_fence_is_not_a_heading():
    text = "# Real\n```\n# not a heading\n```\n"
    chunks = chunk_text(text, 200)
    assert len(chunks) == 1 and chunks[0].heading == "Real"


@pytest.mark.parametrize("max_chars", [20, 50, 120])
def test_no_chunk_exceeds_limit_and_no_word_is_lost(max_chars):
    paragraph = " ".join(f"word{i}" for i in range(200))
    text = f"# Title\n\n{paragraph}\n\nshort tail\n\n" + "x" * 130
    chunks = chunk_text(text, max_chars)
    assert all(len(c.text) <= max_chars for c in chunks)
    rebuilt = "".join(c.text for c in chunks)
    assert "".join(words(text)) == "".join(words(rebuilt))


def test_empty_text_gives_no_chunk():
    assert chunk_text("   \n\n", 100) == []


def test_tiny_limit_is_refused():
    with pytest.raises(ValueError):
        chunk_text("abc", 5)


def test_long_paragraph_keeps_its_line_breaks():
    code = "\n".join(f"    line_{i} = compute({i})" for i in range(40))
    chunks = chunk_text(code, 200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all("\n" in c.text for c in chunks[:-1])
    rebuilt = "\n".join(c.text for c in chunks).split("\n")
    assert [x.strip() for x in rebuilt] == [x.strip() for x in code.split("\n")]
    assert rebuilt[1:] == code.split("\n")[1:]  # inner indentation kept


def test_heading_without_text_is_carried_into_the_next_section():
    text = "# Sourdough bread\n\n## Baking\nBake at 240 degrees.\n"
    chunks = chunk_text(text, 200)
    assert len(chunks) == 1
    assert chunks[0].heading == "Sourdough bread > Baking"
    assert chunks[0].text.startswith("# Sourdough bread") and "240" in chunks[0].text


def test_lone_heading_at_the_end_is_kept():
    chunks = chunk_text("intro\n# Last\n", 200)
    assert [c.text for c in chunks] == ["intro", "# Last"]


def test_tilde_fences_protect_comments_in_code():
    text = "# Doc\ntext\n~~~bash\n# install deps\nnpm i\n~~~\nmore\n"
    chunks = chunk_text(text, 500)
    assert len(chunks) == 1 and "# install deps" in chunks[0].text


def test_heading_with_thousands_of_spaces_is_parsed_in_linear_time():
    import time

    start = time.monotonic()
    chunks = chunk_text("# a" + " " * 50_000 + "b\n\ntext\n", 1500)
    assert time.monotonic() - start < 1
    assert chunks and chunks[0].heading.startswith("a")


def test_heading_rules():
    assert chunk_text("#hashtag line\n", 200)[0].heading == ""
    assert chunk_text("## Title ##\nbody\n", 200)[0].heading == "Title"
    assert chunk_text("####### seven\n", 200)[0].heading == ""


def test_blobs_are_replaced_and_words_kept():
    from cerveau.chunking import BLOB_MARKER, strip_blobs

    assert strip_blobs("see data:image/png;base64," + "A" * 100 + " end") == (
        f"see data:image/png;base64,{BLOB_MARKER} end"
    )
    assert strip_blobs("x " + "Q" * 1000 + " y") == f"x {BLOB_MARKER} y"
    long_word = "W" * 999
    assert strip_blobs(long_word) == long_word
    assert strip_blobs("data:text/plain;base64,SGk= short") == "data:text/plain;base64,SGk= short"


def test_text_without_spaces_in_other_scripts_is_kept():
    from cerveau.chunking import BLOB_MARKER, chunk_text

    japanese = "今日は公園を散歩して桜を見ました。" * 80  # 1280 characters, no space
    chunks = chunk_text("# 日記\n\n" + japanese, 400)
    assert BLOB_MARKER not in "".join(c.text for c in chunks)
    assert "".join(c.text for c in chunks).replace("# 日記", "").replace("\n", "") == japanese
    assert all(len(c.text) <= 400 for c in chunks)
    assert BLOB_MARKER in chunk_text("x " + "ab12" * 300, 400)[0].text
