import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from cerveau.brain import Brain, BrainError


def test_index_then_search_finds_the_right_file(brain, notes):
    report = brain.index_path(notes)
    assert report.files_seen == 2 and report.files_indexed == 2 and not report.errors
    hits = brain.search("sourdough flour salt", k=1, scope="docs")
    assert hits[0].source == "bread.md"


def test_reindexing_unchanged_files_writes_nothing(brain, notes):
    brain.index_path(notes)
    report = brain.index_path(notes)
    assert report.files_unchanged == 2 and report.chunks_written == 0


def test_changed_file_replaces_its_old_chunks(brain, notes):
    brain.index_path(notes)
    before = brain.status()["collections"]["docs"]
    (notes / "bread.md").write_text("# Bread\n\nRye bread is dense.\n", encoding="utf-8")
    report = brain.index_path(notes)
    assert report.files_indexed == 1 and report.chunks_removed >= 1
    assert brain.status()["collections"]["docs"] == before
    texts = [h.text for h in brain.search("bread", k=10, scope="docs")]
    assert not any("Sourdough" in t for t in texts)


def test_prune_removes_deleted_files_only_when_asked(brain, notes):
    brain.index_path(notes)
    (notes / "garden.md").unlink()
    brain.index_path(notes)
    assert any(h.source == "garden.md" for h in brain.search("tomatoes", k=10, scope="docs"))
    report = brain.index_path(notes, prune=True)
    assert report.chunks_removed >= 1
    assert not any(h.source == "garden.md" for h in brain.search("tomatoes", k=10, scope="docs"))


def test_hidden_folders_and_other_extensions_are_ignored(brain, notes):
    (notes / ".git").mkdir()
    (notes / ".git" / "x.md").write_text("secret", encoding="utf-8")
    (notes / "image.png").write_bytes(b"\x89PNG")
    assert brain.index_path(notes).files_seen == 2


def test_unreadable_file_is_reported_not_fatal(brain, notes):
    (notes / "latin1.txt").write_bytes("caf\xe9".encode("latin-1"))
    report = brain.index_path(notes)
    assert report.files_indexed == 2 and len(report.errors) == 1


def test_missing_path_raises(brain, tmp_path):
    with pytest.raises(BrainError):
        brain.index_path(tmp_path / "nope")


def test_remember_search_archive_restore(brain):
    memory_id = brain.remember("The backup runs every night at three", kind="decision")
    hit = brain.search("backup night", k=1, scope="memories")[0]
    assert hit.id == memory_id and hit.kind == "decision"
    brain.archive(memory_id)
    assert brain.search("backup night", k=5, scope="memories") == []
    assert brain.search("backup night", k=5, scope="memories", include_archived=True)[0].id == memory_id
    brain.restore(memory_id)
    assert brain.search("backup night", k=1, scope="memories")[0].id == memory_id


def test_remember_rejects_bad_input(brain):
    with pytest.raises(BrainError):
        brain.remember("   ")
    with pytest.raises(BrainError):
        brain.remember("x" * 2001)
    with pytest.raises(BrainError):
        brain.remember("ok", kind="décision!")
    assert brain.journal.count() == 0


def test_archive_unknown_id_raises_and_writes_nothing(brain):
    with pytest.raises(BrainError):
        brain.archive("not-a-uuid")
    with pytest.raises(BrainError):
        brain.archive("00000000-0000-0000-0000-00000000abcd")
    assert brain.journal.count() == 0


def test_journal_is_written_before_qdrant(brain, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(brain.store, "upsert", boom)
    with pytest.raises(RuntimeError):
        brain.remember("survives a qdrant failure")
    assert [e["text"] for e in brain.journal.events()] == ["survives a qdrant failure"]


def test_replay_rebuilds_memories_into_an_empty_qdrant(settings, brain):
    kept = brain.remember("Keep the server cool")
    hidden = brain.remember("Old wrong idea")
    brain.archive(hidden)
    fresh = Brain(settings, client=QdrantClient(location=":memory:"))
    assert fresh.replay() == 2
    assert fresh.search("server cool", k=1, scope="memories")[0].id == kept
    ids = [h.id for h in fresh.search("idea", k=5, scope="memories")]
    assert hidden not in ids
    assert fresh.replay() == 2  # idempotent
    assert fresh.status()["collections"]["memories"] == 2


def test_search_merges_collections_by_score(brain, notes):
    brain.index_path(notes)
    brain.remember("Tomatoes were planted on the south side")
    hits = brain.search("tomatoes", k=3)
    assert {h.collection for h in hits} == {"docs", "memories"}
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


def test_search_validates_arguments(brain):
    with pytest.raises(BrainError):
        brain.search("x", scope="everything")
    with pytest.raises(BrainError):
        brain.search("x", k=0)
    with pytest.raises(BrainError):
        brain.search("   ")


def test_single_file_can_be_indexed(brain, notes: Path):
    report = brain.index_path(notes / "bread.md")
    assert report.files_indexed == 1


def test_subfolder_indexing_creates_no_duplicates(brain, notes):
    (notes / "sub").mkdir()
    (notes / "sub" / "a.md").write_text("# A\n\nThe attic key hangs by the door.\n", encoding="utf-8")
    brain.index_path(notes)
    brain.index_path(notes / "sub")
    sources = [h.source for h in brain.search("attic key", k=10, scope="docs") if "attic" in h.text]
    assert len(sources) == 1


def test_unindex_removes_a_renamed_folder(brain, tmp_path):
    old = tmp_path / "alpha"
    old.mkdir()
    (old / "plan.md").write_text("The deadline is March.\n", encoding="utf-8")
    brain.index_path(old)
    old.rename(tmp_path / "beta")
    with pytest.raises(BrainError, match="unindex"):
        brain.index_path(old, prune=True)
    assert brain.unindex(str(old)) == 1
    assert brain.search("deadline", k=5, scope="docs") == []


def test_unindex_does_not_touch_sibling_prefixes(brain, tmp_path):
    for name in ("notes", "notes2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "x.md").write_text(f"file in {name}\n", encoding="utf-8")
        brain.index_path(tmp_path / name)
    assert brain.unindex(str(tmp_path / "notes")) == 1
    assert brain.status()["collections"]["docs"] == 1


def test_empty_file_is_stable(brain, notes):
    (notes / "empty.md").write_text("  \n", encoding="utf-8")
    brain.index_path(notes)
    report = brain.index_path(notes)
    assert report.files_indexed == 0 and report.files_unchanged == 3


def test_explicit_non_text_or_hidden_file_is_refused(brain, tmp_path):
    (tmp_path / "id_rsa").write_text("secret", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("secret", encoding="utf-8")
    for name in ("id_rsa", ".hidden.md"):
        with pytest.raises(BrainError):
            brain.index_path(tmp_path / name)


def test_changing_chunk_size_reindexes(settings, client, notes):
    from dataclasses import replace

    Brain(settings, client=client).index_path(notes)
    report = Brain(replace(settings, chunk_max_chars=300), client=client).index_path(notes)
    assert report.files_indexed == 2


def test_storage_error_on_one_file_does_not_stop_the_others(brain, notes, monkeypatch):
    real = brain.store.upsert

    def flaky(collection, points):
        if points and points[0].payload["source"] == "bread.md":
            raise RuntimeError("payload too large")
        return real(collection, points)

    monkeypatch.setattr(brain.store, "upsert", flaky)
    report = brain.index_path(notes)
    assert report.files_indexed == 1 and len(report.errors) == 1 and "bread.md" in report.errors[0]


def test_kind_is_normalised_then_checked(brain):
    brain.remember("ok", kind=" Big Decision\n")
    assert brain.list_memories()[0]["kind"] == "big-decision"
    brain.remember("ok", kind="Décision")
    assert brain.list_memories()[1]["kind"] == "decision"
    with pytest.raises(BrainError, match="a letter a-z"):
        brain.remember("ok", kind="note\x00")


def test_archive_of_a_journal_only_memory_suggests_replay(brain, monkeypatch):
    real = brain.store.upsert
    monkeypatch.setattr(brain.store, "upsert", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError):
        brain.remember("written to the journal only")
    monkeypatch.setattr(brain.store, "upsert", real)
    memory_id = next(brain.journal.events())["id"]
    with pytest.raises(BrainError, match="replay"):
        brain.archive(memory_id)


def test_busy_lock_becomes_a_clear_error(brain):
    from filelock import FileLock

    brain.journal.lock = FileLock(str(brain.journal.path) + ".lock", timeout=0.1)
    other = FileLock(str(brain.journal.path) + ".lock")
    with other, pytest.raises(BrainError, match="busy"):
        brain.remember("blocked")


def test_replay_handles_many_memories_in_batches(settings, brain, monkeypatch):
    for i in range(300):
        brain.remember(f"memory number {i}")
    calls = []
    real = brain.store.upsert
    monkeypatch.setattr(brain.store, "upsert", lambda c, p: (calls.append(len(p)), real(c, p)))
    assert brain.replay() == 300
    assert max(calls) <= 128


def test_journal_is_fsynced_before_qdrant_is_written(brain, monkeypatch):
    import os

    order = []
    real_fsync = os.fsync
    monkeypatch.setattr("cerveau.journal.os.fsync", lambda fd: (order.append("fsync"), real_fsync(fd)))
    real_upsert = brain.store.upsert
    monkeypatch.setattr(brain.store, "upsert", lambda c, p: (order.append("qdrant"), real_upsert(c, p)))
    brain.remember("first write creates the file")
    order.clear()
    brain.remember("ordered write")
    assert order == ["fsync", "qdrant"]


def test_qdrant_failure_after_journal_says_do_not_retry(brain, monkeypatch):
    monkeypatch.setattr(brain.store, "upsert", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(BrainError, match="IS saved in the journal.*Do not retry.*replay"):
        brain.remember("half written")


def test_more_documents_than_one_scroll_page(brain, tmp_path):
    folder = tmp_path / "many"
    folder.mkdir()
    for i in range(600):
        (folder / f"n{i}.md").write_text(f"note number {i}\n", encoding="utf-8")
    assert brain.index_path(folder).files_indexed == 600
    report = brain.index_path(folder)
    assert report.files_unchanged == 600 and report.chunks_written == 0
    assert brain.unindex(str(folder)) == 600


def test_symlink_escaping_the_folder_is_not_indexed(brain, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("PRIVATE-TOKEN-12345\n", encoding="utf-8")
    folder = tmp_path / "shared"
    folder.mkdir()
    try:
        (folder / "link.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not available")
    (folder / "ok.md").write_text("fine\n", encoding="utf-8")
    assert brain.index_path(folder).files_seen == 1


def test_bom_in_a_note_does_not_hide_its_heading(brain, tmp_path):
    folder = tmp_path / "win"
    folder.mkdir()
    (folder / "w.md").write_bytes("﻿# Title\n\nBody text.\n".encode())
    brain.index_path(folder)
    hit = brain.search("body text", k=1, scope="docs")[0]
    assert hit.heading == "Title" and not hit.text.startswith("﻿")


def test_long_heading_never_pushes_a_chunk_over_the_limit(settings, client, tmp_path):
    from dataclasses import replace

    s = replace(settings, max_text_chars=600, chunk_max_chars=300)
    b = Brain(s, client=client)
    folder = tmp_path / "h"
    folder.mkdir()
    (folder / "x.md").write_text("# " + "very long heading " * 60 + "\n\n" + "word " * 59 + "\n")
    report = b.index_path(folder)
    assert report.errors == [] and report.files_indexed == 1


def test_status_reports_tolerated_torn_lines(brain):
    brain.remember("before the crash")
    with open(brain.journal.path, "ab") as handle:
        handle.write(b'{"v": 2, "torn')
    assert brain.status()["journal_torn_lines"] == 1


def test_invalid_unicode_is_refused_before_reaching_the_model(brain, monkeypatch):
    calls = []
    monkeypatch.setattr(brain.embedder, "embed", lambda texts: calls.append(texts) or [])
    with pytest.raises(BrainError, match="Unicode"):
        brain.remember("bad\udcffbyte")
    assert calls == []


def test_symlinks_inside_the_folder_are_skipped(brain, tmp_path):
    folder = tmp_path / "linked"
    folder.mkdir()
    (folder / "real.md").write_text("only once\n", encoding="utf-8")
    try:
        (folder / "alias.md").symlink_to(folder / "real.md")
    except OSError:
        pytest.skip("symlinks not available")
    assert brain.index_path(folder).files_seen == 1


def test_archive_with_qdrant_down_says_it_is_recorded(brain, monkeypatch):
    memory_id = brain.remember("to be archived")
    monkeypatch.setattr(brain.store, "set_state", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(BrainError, match="IS recorded in the journal.*replay"):
        brain.archive(memory_id)


def test_symlinked_folder_pointing_outside_is_not_followed(brain, tmp_path):
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "s.md").write_text("PRIVATE\n", encoding="utf-8")
    folder = tmp_path / "public"
    folder.mkdir()
    (folder / "ok.md").write_text("fine\n", encoding="utf-8")
    try:
        (folder / "door").symlink_to(secret, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not available")
    assert brain.index_path(folder).files_seen == 1


@pytest.mark.parametrize("spelling", [str.upper, lambda s: s.replace("-", "")])
def test_archive_with_another_spelling_of_the_id_survives_replay(settings, brain, spelling):
    memory_id = brain.remember("SECRET plan")
    assert brain.archive(spelling(memory_id)) == memory_id
    fresh = Brain(settings, client=QdrantClient(location=":memory:"))
    fresh.replay()
    assert fresh.search("SECRET plan", k=5, scope="memories") == []


def test_wrong_model_seals_nothing(settings, client):
    from cerveau.embed import EmbeddingError

    class Missing:
        model, dimension = "no-such-model", settings.vector_size

        def embed(self, texts):
            raise EmbeddingError("model not found")

    with pytest.raises(EmbeddingError):
        Brain(settings, client=client, embedder=Missing()).init()
    assert client.get_collections().collections == []


def test_reading_does_not_wait_for_a_writer(brain, settings, client):
    from filelock import FileLock

    reader = Brain(settings, client=client)
    with FileLock(str(brain.journal.path) + ".lock"):
        assert reader.search("anything at all", k=1) == []
        assert reader.status()["collections"]["docs"] == 0


def test_read_only_brain_writes_nothing(settings, client):
    reader = Brain(settings, client=client, read_only=True)
    with pytest.raises(BrainError, match="not initialised"):
        reader.status()
    assert client.get_collections().collections == []
    assert not settings.data_dir.exists()


def test_read_only_brain_reads_an_existing_brain(settings, client, brain):
    brain.remember("visible to readers")
    reader = Brain(settings, client=client, read_only=True)
    assert reader.search("visible readers", k=1, scope="memories")[0].text == "visible to readers"
    with pytest.raises(BrainError, match="read-only"):
        reader.remember("nope")


def test_list_shows_who_wrote_what(brain):
    brain.remember("from the agent", actor="mcp")
    human = brain.remember("from the human", actor="cli")
    brain.archive(human)
    agent_items = brain.list_memories(actor="mcp")
    assert [i["text"] for i in agent_items] == ["from the agent"]
    assert len(brain.list_memories()) == 1
    assert len(brain.list_memories(include_archived=True)) == 2
    assert brain.search("from the agent", k=1, scope="memories")[0].actor == "mcp"


def test_status_warns_when_qdrant_lags_behind_the_journal(settings, brain):
    brain.remember("only in the journal for the new qdrant")
    fresh = Brain(settings, client=QdrantClient(location=":memory:"))
    fresh.init()
    assert "cerveau replay" in fresh.status()["warning"]


def test_unreadable_journal_blocks_new_writes(brain):
    brain.remember("one")
    brain.remember("two")
    lines = brain.journal.path.read_bytes().split(b"\n")
    lines[0] = b"garbage"
    brain.journal.path.write_bytes(b"\n".join(lines))
    from cerveau.journal import JournalError

    fresh = Brain(brain.settings, client=brain.store.client)
    with pytest.raises(JournalError):
        fresh.remember("three")


def test_notes_in_utf16_are_read_and_cp1252_is_explained(brain, tmp_path):
    folder = tmp_path / "enc"
    folder.mkdir()
    (folder / "unicode.txt").write_bytes("# Notes\n\nCafé crème\n".encode("utf-16"))
    (folder / "ansi.txt").write_bytes("Café crème\n".encode("cp1252"))
    report = brain.index_path(folder)
    assert report.files_indexed == 1
    assert len(report.errors) == 1 and "UTF-8" in report.errors[0]


def test_software_folders_are_not_indexed(brain, tmp_path):
    folder = tmp_path / "project"
    for sub in ("node_modules/pkg", "venv/lib", "docs"):
        (folder / sub).mkdir(parents=True)
    (folder / "node_modules/pkg/README.md").write_text("package\n")
    (folder / "venv/lib/README.md").write_text("venv\n")
    (folder / "docs/guide.md").write_text("guide\n")
    assert brain.index_path(folder).files_seen == 1


def test_source_keeps_its_original_case(brain, tmp_path):
    folder = tmp_path / "Mes Notes"
    folder.mkdir()
    (folder / "README.md").write_text("Some text here\n", encoding="utf-8")
    brain.index_path(folder)
    assert brain.search("some text", k=1, scope="docs")[0].source == "README.md"


def test_replay_understands_ids_written_in_another_spelling(settings, brain):
    memory_id = brain.remember("old journal entry")
    with brain.journal.lock:
        brain.journal.append({"op": "archive", "id": memory_id.upper(), "actor": "old-tool"})
    fresh = Brain(settings, client=QdrantClient(location=":memory:"))
    fresh.replay()
    assert fresh.search("old journal entry", k=5, scope="memories") == []


def test_list_filters_by_actor(brain):
    brain.remember("from the agent", actor="mcp")
    brain.remember("from the human", actor="cli")
    assert [i["text"] for i in brain.list_memories(actor="cli")] == ["from the human"]


def test_source_keeps_its_case_where_paths_ignore_case(brain, tmp_path, monkeypatch):
    import cerveau.brain as brain_module

    monkeypatch.setattr(brain_module.os.path, "normcase", str.lower)  # what Windows does
    folder = tmp_path / "Notes"
    folder.mkdir()
    (folder / "ReadMe.md").write_text("Mixed case file\n", encoding="utf-8")
    brain.index_path(folder)
    hit = brain.search("mixed case", k=1, scope="docs")[0]
    assert hit.source == "ReadMe.md" and hit.path.endswith("Notes/ReadMe.md")


def test_unindex_refuses_a_whole_disk_or_home(brain):
    from pathlib import Path

    for target in (Path.home(), Path(Path.home().anchor), ""):
        with pytest.raises(BrainError):
            brain.unindex(str(target))


def test_latest_tag_and_bare_model_name_share_one_seal(settings, client):
    class Tagged:
        model, dimension = "hash-bow-64:latest", settings.vector_size

        def embed(self, texts):
            return HashEmbedder(settings.vector_size, settings.max_text_chars).embed(texts)

    from cerveau.embed import HashEmbedder

    Brain(settings, client=client).init()
    Brain(settings, client=client, embedder=Tagged()).init()  # no SealMismatch


def test_reads_never_create_collections(settings, client):
    reader = Brain(settings, client=client)
    for read in (lambda: reader.search("x"), reader.status):
        with pytest.raises(BrainError, match="cerveau init"):
            read()
    assert client.get_collections().collections == []


def test_status_reports_memories_written_with_another_journal(settings, brain, tmp_path):
    from dataclasses import replace

    other = Brain(replace(settings, data_dir=tmp_path / "elsewhere"), client=brain.store.client)
    other.remember("written with another data folder")
    status = brain.status()
    assert "not in this journal" in status["warning"]
    assert [o["text"] for o in brain.orphan_memories()] == ["written with another data folder"]


def test_status_reports_an_archive_missing_in_qdrant(brain, monkeypatch):
    memory_id = brain.remember("archived only in the journal")
    monkeypatch.setattr(brain.store, "set_state", lambda *a: None)  # crash before Qdrant
    brain.archive(memory_id)
    monkeypatch.undo()
    assert "archived in the journal" in brain.status()["warning"]


def test_archiving_twice_records_once(brain):
    memory_id = brain.remember("archive me")
    brain.archive(memory_id)
    brain.archive(memory_id)
    assert brain.journal.count() == 2


def test_huge_file_is_skipped_with_a_reason(brain, tmp_path, monkeypatch):
    import cerveau.brain as brain_module

    monkeypatch.setattr(brain_module, "MAX_FILE_BYTES", 10)
    folder = tmp_path / "big"
    folder.mkdir()
    (folder / "big.md").write_text("x" * 100, encoding="utf-8")
    report = brain.index_path(folder)
    assert report.files_indexed == 0 and "larger than" in report.errors[0]


def test_read_only_brain_never_creates_even_when_asked_to_write(settings, client):
    reader = Brain(settings, client=client, read_only=True)
    with pytest.raises(BrainError):
        reader.remember("attempt")
    with pytest.raises(BrainError):
        reader.init()
    assert client.get_collections().collections == []


def test_archiving_an_orphan_is_refused(settings, brain, tmp_path):
    from dataclasses import replace

    other = Brain(replace(settings, data_dir=tmp_path / "other"), client=brain.store.client)
    orphan = other.remember("not in my journal")
    with pytest.raises(BrainError, match="adopt-orphans"):
        brain.archive(orphan)
    assert brain.adopt_orphans() == 1
    brain.archive(orphan)
    assert "warning" not in brain.status()


def test_adopted_archived_orphan_stays_archived(settings, brain, tmp_path):
    from dataclasses import replace

    other = Brain(replace(settings, data_dir=tmp_path / "other"), client=brain.store.client)
    orphan = other.remember("archived elsewhere")
    other.archive(orphan)
    brain.adopt_orphans()
    assert brain.list_memories(include_archived=True)[0]["state"] == "archived"


def test_orphans_of_an_uninitialised_brain_say_run_init(settings, client):
    with pytest.raises(BrainError, match="cerveau init"):
        Brain(settings, client=client).orphan_memories()


def test_macos_folds_case_in_identity(brain, tmp_path, monkeypatch):
    import cerveau.brain as brain_module

    monkeypatch.setattr(brain_module.sys, "platform", "darwin")
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "a.md").write_text("only once please\n", encoding="utf-8")
    brain.index_path(folder)
    assert brain_module._posix(folder / "A.md") == brain_module._posix(folder / "a.md")
    assert brain.unindex(str(tmp_path / "NOTES")) == 1


def test_too_long_query_says_shorten(brain):
    with pytest.raises(BrainError, match="query.*shorten"):
        brain.search("x" * 5000)


def test_interrupted_indexing_of_a_new_file_is_completed_next_time(brain, tmp_path, monkeypatch):
    folder = tmp_path / "book"
    folder.mkdir()
    (folder / "book.md").write_text(
        "\n\n".join(f"## Part {i}\nmarker-{i} " + "word " * 40 for i in range(300)), encoding="utf-8"
    )
    real = brain.store.client.upsert
    calls = {"n": 0}

    def hiccup(collection, points, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("Qdrant hiccup")
        return real(collection, points, **kwargs)

    monkeypatch.setattr(brain.store.client, "upsert", hiccup)
    first = brain.index_path(folder)
    assert first.errors and first.files_indexed == 0
    monkeypatch.setattr(brain.store.client, "upsert", real)
    second = brain.index_path(folder)
    assert second.files_indexed == 1 and second.files_unchanged == 0
    third = brain.index_path(folder)
    assert third.files_unchanged == 1
    hits = brain.search("marker-250", k=3, scope="docs")
    assert any("marker-250" in h.text for h in hits)


def test_restore_after_a_failed_archive_is_recorded(brain, monkeypatch, settings):
    memory_id = brain.remember("archive half done")
    monkeypatch.setattr(brain.store, "set_state", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(BrainError):
        brain.archive(memory_id)
    monkeypatch.undo()
    brain.restore(memory_id)
    assert [e["op"] for e in brain.journal.events()] == ["remember", "archive", "restore"]
    fresh = Brain(settings, client=QdrantClient(location=":memory:"))
    fresh.init()
    fresh.replay()
    assert fresh.search("archive half done", k=1, scope="memories")[0].id == memory_id


def test_replay_keeps_going_and_names_what_it_could_not_embed(settings, brain):
    from dataclasses import replace

    long_id = brain.remember("y" * 1500)
    brain.remember("short and fine")
    smaller = Brain(
        replace(settings, max_text_chars=1000, chunk_max_chars=500), client=QdrantClient(location=":memory:")
    )
    smaller.init()
    with pytest.raises(
        BrainError, match=f"replayed 1 memories, 1 could not.*{long_id}.*CERVEAU_MAX_TEXT_CHARS"
    ):
        smaller.replay()
    assert smaller.search("short and fine", k=1, scope="memories")[0].text == "short and fine"


def test_reset_works_after_a_wrong_first_setup(settings, client):
    from dataclasses import replace

    wrong = replace(settings, vector_size=32)
    Brain(wrong, client=client).init()
    right = Brain(settings, client=client)
    from cerveau.store import SealMismatch

    with pytest.raises(SealMismatch):
        right.status()
    right.reset_collections()
    right.init()
    assert right.status()["vector_size"] == settings.vector_size


def test_adopted_memories_keep_their_original_date_and_order(settings, brain, tmp_path, monkeypatch):
    from dataclasses import replace

    import cerveau.journal as journal_module

    clock = iter(["2020-01-02T00:00:00+00:00", "2020-01-01T00:00:00+00:00"])
    monkeypatch.setattr(journal_module, "now_iso", lambda: next(clock))
    other = Brain(replace(settings, data_dir=tmp_path / "old"), client=brain.store.client)
    other.remember("second in time")
    other.remember("first in time")
    monkeypatch.setattr(journal_module, "now_iso", lambda: "2030-01-01T00:00:00+00:00")
    brain.adopt_orphans()
    items = brain.list_memories()
    assert [i["text"] for i in items] == ["first in time", "second in time"]
    assert items[0]["ts"] == "2020-01-01T00:00:00+00:00"
    brain.replay()
    assert brain.search("first in time", k=1, scope="memories")[0].ts == "2020-01-01T00:00:00+00:00"


def test_unreadable_sub_folder_is_reported_not_fatal(brain, tmp_path, monkeypatch):
    import os as _os

    folder = tmp_path / "notes"
    (folder / "ok").mkdir(parents=True)
    (folder / "ok" / "a.md").write_text("fine\n", encoding="utf-8")
    real_walk = _os.walk

    def walk_with_a_locked_folder(top, onerror=None, **kwargs):
        if onerror is not None:
            onerror(PermissionError(13, "Permission denied", str(folder / "locked")))
        yield from real_walk(top, onerror=onerror, **kwargs)

    monkeypatch.setattr("cerveau.brain.os.walk", walk_with_a_locked_folder)
    report = brain.index_path(folder)
    assert report.files_indexed == 1 and any("Permission denied" in e for e in report.errors)


def test_hidden_files_are_skipped_too(brain, tmp_path):
    folder = tmp_path / "n"
    folder.mkdir()
    (folder / ".draft.md").write_text("secret draft\n", encoding="utf-8")
    (folder / "public.md").write_text("public\n", encoding="utf-8")
    assert brain.index_path(folder).files_seen == 1


def test_purge_erases_a_memory_from_journal_and_qdrant(settings, brain):
    keep = brain.remember("the blue notebook is on the shelf")
    secret = brain.remember("my password is hunter2-SECRET")
    brain.archive(secret)
    assert brain.purge(secret.upper()) == 2  # any spelling of the id
    assert b"hunter2-SECRET" not in settings.journal_path.read_bytes()
    assert brain.store.get(brain.store.memories, secret) is None
    assert [m["id"] for m in brain.list_memories(include_archived=True)] == [keep]
    assert brain.orphan_memories() == []
    status = brain.status()
    assert "warning" not in status and status["memories_in_journal"] == 1
    brain.replay()  # the purged memory never comes back
    assert brain.store.count(brain.store.memories) == 1
    brain.remember("after the purge")  # the journal is still writable
    assert len(brain.list_memories()) == 2


def test_purge_of_an_unknown_or_bad_id(brain):
    import uuid as _uuid

    with pytest.raises(BrainError, match="not a memory id"):
        brain.purge("nope")
    with pytest.raises(BrainError, match="neither in the journal nor in Qdrant"):
        brain.purge(str(_uuid.uuid4()))


def test_purge_erases_an_orphan_too(settings, brain, tmp_path):
    from dataclasses import replace

    other = Brain(replace(settings, data_dir=tmp_path / "other"), client=brain.store.client)
    orphan = other.remember("leaked elsewhere")
    assert brain.purge(orphan) == 0
    assert brain.orphan_memories() == []


def test_purge_with_qdrant_down_first_touches_nothing(settings, client):
    b = Brain(settings, client=client)
    with pytest.raises(BrainError, match="cerveau init"):
        b.purge("6f1c3c2e-6d0b-4c55-9a43-4b8a0f0c6e21")
    assert not settings.journal_path.exists()


def test_purge_when_qdrant_fails_after_the_journal_says_run_again(settings, brain, monkeypatch):
    secret = brain.remember("secret")

    def down(*_args, **_kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr(brain.store, "delete_points", down)
    with pytest.raises(BrainError, match=f"purge {secret} --confirm"):
        brain.purge(secret)
    assert b"secret" not in settings.journal_path.read_bytes().replace(b'"purge"', b"")
    monkeypatch.undo()
    assert brain.purge(secret) == 0  # now an orphan: the second run removes it from Qdrant
    assert brain.store.get(brain.store.memories, secret) is None


def test_read_only_brain_cannot_purge(settings, client, brain):
    secret = brain.remember("x")
    with pytest.raises(BrainError, match="read-only"):
        Brain(settings, client=client, read_only=True).purge(secret)


def test_prune_is_skipped_when_a_sub_folder_could_not_be_read(brain, tmp_path, monkeypatch):
    import os as _os

    folder = tmp_path / "notes"
    (folder / "locked").mkdir(parents=True)
    (folder / "locked" / "kept.md").write_text("still here\n", encoding="utf-8")
    (folder / "gone.md").write_text("deleted soon\n", encoding="utf-8")
    (folder / "stays.md").write_text("still readable\n", encoding="utf-8")
    brain.index_path(folder)
    (folder / "gone.md").unlink()
    real_walk = _os.walk

    def walk_without_locked(top, onerror=None, **kwargs):
        for current, dirs, names in real_walk(top, onerror=onerror, **kwargs):
            if "locked" in dirs:
                dirs.remove("locked")
                onerror(PermissionError(13, "Permission denied", str(folder / "locked")))
            yield current, dirs, names

    monkeypatch.setattr("cerveau.brain.os.walk", walk_without_locked)
    report = brain.index_path(folder, prune=True)
    assert report.chunks_removed == 0
    assert any("some folders could not be read" in e for e in report.errors)
    assert brain.store.count(brain.store.docs) == 3
    monkeypatch.undo()
    assert brain.index_path(folder, prune=True).chunks_removed == 1


@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="needs a file system that accepts any bytes")
def test_file_name_that_is_not_utf8_is_reported(brain, tmp_path):
    folder = tmp_path / "n"
    folder.mkdir()
    (folder / "ok.md").write_text("fine\n", encoding="utf-8")
    with open(os.path.join(os.fsencode(folder), b"caf\xe9.md"), "wb") as handle:
        handle.write(b"latin-1 name\n")
    report = brain.index_path(folder)
    assert report.files_indexed == 1
    assert any("not valid UTF-8" in e and "rename" in e for e in report.errors)


def test_adopted_memories_are_never_attributed_to_cli_or_mcp(settings, brain, tmp_path):
    from dataclasses import replace

    other = Brain(replace(settings, data_dir=tmp_path / "other"), client=brain.store.client)
    other.remember("written by an agent elsewhere", actor="mcp")
    brain.adopt_orphans()
    (adopted,) = brain.list_memories(actor="adopted")
    assert adopted["text"] == "written by an agent elsewhere" and adopted["adopted_from"] == "mcp"
    # the review of what agents wrote still finds it, labelled as adopted
    assert [(m["actor"], m["adopted_from"]) for m in brain.list_memories(actor="mcp")] == [("adopted", "mcp")]
    assert brain.list_memories(actor="cli") == []
    record = next(brain.journal.events())
    assert record["adopted_from_actor"] == "mcp"


def test_purge_removes_records_written_with_another_spelling_of_the_id(settings, brain):
    memory_id = brain.remember("written by an old tool")
    with brain.journal.lock:
        brain.journal.append({"op": "archive", "id": "{" + memory_id.upper() + "}", "actor": "old-tool"})
    assert brain.purge(memory_id) == 2
    assert memory_id.upper().encode() not in settings.journal_path.read_bytes()


def test_index_of_a_missing_path_creates_nothing(settings, client):
    b = Brain(settings, client=client)
    with pytest.raises(BrainError, match="does not exist"):
        b.index_path(settings.data_dir / "Docuemnts")
    assert not client.get_collections().collections


def test_a_purged_memory_restored_in_qdrant_is_never_adopted(settings, brain):
    from qdrant_client.http import models as qm

    keep = brain.remember("keep this")
    secret = brain.remember("secret-1 token ABC1")
    point = brain.store.client.retrieve(brain.store.memories, [secret], with_vectors=True)[0]
    brain.purge(secret)
    # an older Qdrant backup is restored: the point is back
    brain.store.upsert(
        brain.store.memories, [qm.PointStruct(id=secret, vector=point.vector, payload=point.payload)]
    )
    assert brain.orphan_memories() == []
    assert "purged memories are in Qdrant again" in brain.status()["warning"]
    assert brain.adopt_orphans() == 0 and brain.purged_again == 1
    assert b"secret-1" not in settings.journal_path.read_bytes()
    assert brain.store.get(brain.store.memories, secret) is None
    assert [m["id"] for m in brain.list_memories()] == [keep]
    assert "warning" not in brain.status()
    brain.reset_collections()  # a restored purged point never blocks a reset either


def test_status_archive_count_ignores_orphans(settings, brain, tmp_path):
    from dataclasses import replace

    other = Brain(replace(settings, data_dir=tmp_path / "other"), client=brain.store.client)
    other.archive(other.remember("archived elsewhere"))
    warning = brain.status()["warning"]
    assert "list --orphans" in warning and "archived in the journal" not in warning


def test_large_file_is_written_slice_by_slice(brain, tmp_path, monkeypatch):
    import cerveau.brain as module

    monkeypatch.setattr(module, "WRITE_SLICE", 3)
    folder = tmp_path / "big"
    folder.mkdir()
    (folder / "log.txt").write_text("\n\n".join(f"paragraph {i} " + "word " * 80 for i in range(10)), "utf-8")
    sizes, lines = [], []
    real = brain.store.upsert
    monkeypatch.setattr(brain.store, "upsert", lambda c, pts: (sizes.append(len(pts)), real(c, pts)))
    report = brain.index_path(folder, progress=lines.append)
    assert max(sizes) <= 3 and sum(sizes) == report.chunks_written > 3
    assert any("chunks" in line for line in lines)

    # interrupted after the first slice: the next run completes the file
    brain.unindex(str(folder))
    calls = []

    def fail_second(collection, points):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("Qdrant down")
        real(collection, points)

    monkeypatch.setattr(brain.store, "upsert", fail_second)
    assert brain.index_path(folder).errors
    monkeypatch.setattr(brain.store, "upsert", real)
    again = brain.index_path(folder)
    assert again.files_indexed == 1 and brain.store.count(brain.store.docs) == again.chunks_written
    assert brain.index_path(folder).files_unchanged == 1


@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="these file systems normalise names")
def test_two_names_equal_once_normalised_are_reported(brain, tmp_path):
    import unicodedata

    folder = tmp_path / "n"
    folder.mkdir()
    (folder / unicodedata.normalize("NFC", "café.md")).write_text("one\n", encoding="utf-8")
    (folder / unicodedata.normalize("NFD", "café.md")).write_text("two\n", encoding="utf-8")
    report = brain.index_path(folder)
    assert report.files_indexed == 1
    assert any("same name as another file" in e for e in report.errors)
    assert brain.index_path(folder).files_unchanged == 1


def test_index_refuses_the_home_folder(brain, tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    with pytest.raises(BrainError, match="too broad"):
        brain.index_path(tmp_path)
    with pytest.raises(BrainError, match="too broad"):
        brain.index_path(Path(tmp_path.anchor))


def test_binary_and_bomless_utf16_files_are_reported_not_indexed(brain, tmp_path):
    folder = tmp_path / "enc"
    folder.mkdir()
    (folder / "u16nobom.txt").write_bytes("Turtles live long.\n".encode("utf-16-le"))
    (folder / "zeros.txt").write_bytes(b"\x00" * 2000)
    (folder / "u32.txt").write_bytes("Owls hunt at night.\n".encode("utf-32"))
    report = brain.index_path(folder)
    assert report.files_indexed == 1  # the UTF-32 file, which has a BOM
    assert sum("looks binary" in e for e in report.errors) == 2
    assert brain.search("owls night", k=1, scope="docs")[0].text.startswith("Owls")


def test_adoption_leaves_out_points_cerveau_could_not_have_written(settings, brain):
    import uuid as _uuid

    from qdrant_client.http import models as qm

    vector = [0.1] * settings.vector_size
    good = str(_uuid.uuid4())
    points = [
        qm.PointStruct(id=77, vector=vector, payload={"text": "integer id", "kind": "note"}),
        qm.PointStruct(id=str(_uuid.uuid4()), vector=vector, payload={"text": "", "kind": "note"}),
        qm.PointStruct(id=str(_uuid.uuid4()), vector=vector, payload={"text": "x" * 5000}),
        qm.PointStruct(id=good, vector=vector, payload={"text": "fine", "kind": "UPPER bad", "ts": 3}),
    ]
    brain.store.upsert(brain.store.memories, points)
    assert brain.adopt_orphans() == 1
    assert len(brain.skipped_orphans) == 3 and any(s.startswith("77 ") for s in brain.skipped_orphans)
    (adopted,) = brain.list_memories()
    assert adopted["id"] == good and adopted["kind"] == "note"
    assert brain.replay() == 1  # the journal stays replayable
    assert brain.purge("77") == 0  # an integer id written by something else can be deleted
    assert brain.store.get(brain.store.memories, 77) is None


def test_prune_on_an_empty_folder_removes_nothing(brain, notes):
    brain.index_path(notes)
    for file in notes.iterdir():
        file.unlink()
    report = brain.index_path(notes, prune=True)
    assert report.chunks_removed == 0 and any("cerveau unindex" in e for e in report.errors)
    assert brain.store.count(brain.store.docs) == 2


def test_index_refuses_the_folder_of_all_homes(brain, tmp_path, monkeypatch):
    home = tmp_path / "users" / "me"
    home.mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: home))
    with pytest.raises(BrainError, match="too broad"):
        brain.index_path(home.parent)
    (home / "notes").mkdir()
    brain.index_path(home / "notes")  # a folder of notes is fine


def test_index_counts_deleted_files_that_stay_searchable(brain, notes):
    brain.index_path(notes)
    (notes / "bread.md").unlink()
    report = brain.index_path(notes)
    assert report.stale_files == 1 and report.chunks_removed == 0
    pruned = brain.index_path(notes, prune=True)
    assert pruned.stale_files == 0 and pruned.chunks_removed == 1


def test_purge_reaches_every_prefix_of_the_same_qdrant(settings, brain):
    from dataclasses import replace

    other = Brain(replace(settings, prefix="newmodel"), client=brain.store.client)
    other.init()
    secret = brain.remember("API password is hunter2 (saved by mistake)")
    other.replay()  # the documented model change: same journal, new prefix
    assert other.store.get(other.store.memories, secret) is not None
    brain.purge(secret)
    assert brain.purged_prefixes == ["newmodel"]
    assert other.store.get(other.store.memories, secret) is None
    assert other.search("hunter2 password", k=5, scope="memories") == []


def test_purge_of_a_qdrant_only_memory_leaves_a_tombstone(settings, brain, tmp_path):
    from dataclasses import replace

    from qdrant_client.http import models as qm

    other = Brain(replace(settings, data_dir=tmp_path / "other"), client=brain.store.client)
    orphan = other.remember("leaked secret")
    point = brain.store.client.retrieve(brain.store.memories, [orphan], with_vectors=True)[0]
    assert brain.purge(orphan) == 0
    assert orphan in brain.journal.purged_ids()
    brain.store.upsert(
        brain.store.memories, [qm.PointStruct(id=orphan, vector=point.vector, payload=point.payload)]
    )
    assert brain.adopt_orphans() == 0 and brain.purged_again == 1


def test_purge_refuses_an_integer_qdrant_cannot_hold(brain):
    with pytest.raises(BrainError, match="not a memory id"):
        brain.purge("99999999999999999999")


def test_reports_show_paths_as_written_where_identity_ignores_case(brain, tmp_path, monkeypatch):
    monkeypatch.setattr("cerveau.brain.sys.platform", "darwin")
    folder = tmp_path / "Notes"
    folder.mkdir()
    (folder / "Été.md").write_bytes(b"\xff\xfe\xfa")  # not UTF-8: reported
    report = brain.index_path(folder)
    assert report.root.endswith("/Notes")
    assert any("/Notes/" in e and "UTF-8" in e for e in report.errors)


def test_replay_stops_at_once_when_the_embedding_service_is_down(settings, brain, monkeypatch):
    from cerveau.embed import EmbeddingError

    for i in range(5):
        brain.remember(f"memory {i}")
    calls = []

    def down(texts):
        calls.append(len(texts))
        raise EmbeddingError("cannot reach Ollama", per_text=False)

    monkeypatch.setattr(brain.embedder, "embed", down)
    with pytest.raises(BrainError, match="then stopped") as caught:
        brain.replay()
    assert "MAX_TEXT_CHARS" not in str(caught.value)
    assert len(calls) == 1  # no retry text by text


def test_replay_reports_progress(brain):
    for i in range(3):
        brain.remember(f"memory {i}")
    lines = []
    brain.replay(progress=lines.append)
    assert lines and lines[-1] == "replay: 3/3"


def test_unindex_archive_restore_never_create_collections(settings, client):
    b = Brain(settings, client=client)
    for action in (
        lambda: b.unindex("/tmp/typo"),
        lambda: b.archive(str(uuid4())),
        lambda: b.restore(str(uuid4())),
    ):
        with pytest.raises(BrainError, match="cerveau init"):
            action()
    assert not client.get_collections().collections


def test_status_does_not_read_the_text_of_orphans(settings, brain, tmp_path, monkeypatch):
    from dataclasses import replace

    Brain(replace(settings, data_dir=tmp_path / "o"), client=brain.store.client).remember("orphan")
    asked = []
    real = brain.store.scroll_payloads

    def spy(collection, flt=None, fields=None):
        asked.append(fields)
        return real(collection, flt, fields)

    monkeypatch.setattr(brain.store, "scroll_payloads", spy)
    assert "list --orphans" in brain.status()["warning"]
    assert None not in asked  # never the whole payload (the texts)


def test_remember_works_while_an_index_holds_the_documents_lock(brain, notes):
    with _other_process_lock(brain.docs_lock):  # a long index running in another process
        memory_id = brain.remember("decided while indexing")
    assert brain.list_memories()[0]["id"] == memory_id


def _other_process_lock(lock):
    from filelock import FileLock

    return FileLock(lock.lock_file)  # another file handle: behaves like another process


def test_index_waits_for_another_index_not_for_memories(brain, notes):
    brain.docs_lock.timeout = 0.1
    with _other_process_lock(brain.docs_lock), pytest.raises(BrainError, match="index or unindex"):
        brain.index_path(notes)
    with _other_process_lock(brain.journal.lock):  # a replay or a remember elsewhere
        assert brain.index_path(notes).files_indexed == 2


def test_reset_takes_both_locks(brain):
    brain.docs_lock.timeout = 0.1
    with _other_process_lock(brain.docs_lock), pytest.raises(BrainError, match="busy"):
        brain.reset_collections()


def test_index_stops_at_the_first_service_failure(brain, tmp_path, monkeypatch):
    from cerveau.embed import EmbeddingError

    folder = tmp_path / "many"
    folder.mkdir()
    for i in range(10):
        (folder / f"n{i}.md").write_text(f"note {i}\n", encoding="utf-8")
    calls = []

    def down(texts):
        calls.append(1)
        raise EmbeddingError("cannot reach Ollama", per_text=False)

    monkeypatch.setattr(brain.embedder, "embed", down)
    report = brain.index_path(folder, prune=True)
    assert len(calls) == 1 and report.files_seen == 1
    assert any("indexing stopped after 1 of 10" in e for e in report.errors)
    assert report.chunks_removed == 0


def test_a_text_error_does_not_stop_the_other_files(brain, tmp_path, monkeypatch):
    from cerveau.embed import EmbeddingError

    folder = tmp_path / "some"
    folder.mkdir()
    for i in range(3):
        (folder / f"n{i}.md").write_text(f"note {i}\n", encoding="utf-8")
    real = brain.embedder.embed

    def first_too_dense(texts):
        if "note 0" in texts[0]:
            raise EmbeddingError("too many tokens", per_text=True)
        return real(texts)

    monkeypatch.setattr(brain.embedder, "embed", first_too_dense)
    assert brain.index_path(folder).files_indexed == 2


def test_pasted_images_are_not_indexed_as_text(brain, tmp_path):
    import base64

    folder = tmp_path / "vault"
    folder.mkdir()
    blob = base64.b64encode(bytes(range(256)) * 400).decode()
    (folder / "note.md").write_text(f"# Trip\n\nThe view.\n\n![](data:image/png;base64,{blob})\n", "utf-8")
    report = brain.index_path(folder)
    assert report.chunks_written == 1
    (hit,) = brain.search("view", k=1, scope="docs")
    assert "data:image/png;base64,[binary data omitted]" in hit.text and blob[:100] not in hit.text


def test_a_memory_pasted_with_windows_line_ends_is_stored_with_plain_ones(brain):
    brain.remember("line one\r\nline two\r")
    assert brain.list_memories()[0]["text"] == "line one\nline two"


def test_foreign_points_cannot_break_or_flood_a_search(settings, brain):
    import uuid as _uuid

    from qdrant_client.http import models as qm

    vector = brain._embed(["flood"])[0]
    brain.store.upsert(
        brain.store.memories,
        [
            qm.PointStruct(id=str(_uuid.uuid4()), vector=vector, payload={"text": None}),
            qm.PointStruct(id=str(_uuid.uuid4()), vector=vector, payload={"text": "flood " * 400_000}),
        ],
    )
    hits = brain.search("flood", k=5, scope="memories")
    assert all(isinstance(h.text, str) and len(h.text) <= settings.max_text_chars + 60 for h in hits)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX system folders")
def test_index_refuses_system_folders(brain):
    from cerveau.brain import _too_broad

    for folder in ("/etc", "/usr/share/doc", "/home", "/Users", "/mnt", "/Volumes", "/var/lib"):
        assert _too_broad(Path(folder)), folder
    assert not _too_broad(Path("/mnt/notes")) and not _too_broad(Path("/srv/wiki"))


def test_windows_system_folders_are_refused(monkeypatch):
    from pathlib import PureWindowsPath

    import cerveau.brain as module

    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module.Path, "home", classmethod(lambda cls: Path("/nonexistent/home/me")))
    for folder in (r"C:\Windows\System32", r"C:\Program Files", r"C:\Users"):
        assert module._too_broad(PureWindowsPath(folder)), folder
    assert not module._too_broad(PureWindowsPath(r"C:\Users\me\Notes"))
    assert not module._too_broad(PureWindowsPath(r"D:\Notes"))


def test_index_stops_when_qdrant_is_gone(brain, tmp_path, monkeypatch):
    from qdrant_client.http.exceptions import ResponseHandlingException

    folder = tmp_path / "many"
    folder.mkdir()
    for i in range(10):
        (folder / f"n{i}.md").write_text(f"note {i}\n", encoding="utf-8")
    calls = []

    def gone(*_args, **_kwargs):
        calls.append(1)
        raise ResponseHandlingException(ConnectionRefusedError("refused"))

    monkeypatch.setattr(brain.store, "upsert", gone)
    report = brain.index_path(folder, prune=True)
    assert len(calls) == 1 and any("Qdrant cannot be reached" in e for e in report.errors)
