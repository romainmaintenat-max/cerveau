import pytest
from qdrant_client.http import models as qm

from cerveau.store import SEAL_ID, SealMismatch, Store


def names(client) -> set[str]:
    return {c.name for c in client.get_collections().collections}


def test_ensure_is_idempotent(client):
    store = Store(client, "p", "model-a", 8)
    store.ensure()
    store.ensure()
    assert names(client) == {"p_docs", "p_memories", "p_meta"}


def test_other_model_is_refused(client):
    Store(client, "p", "model-a", 8).ensure()
    with pytest.raises(SealMismatch, match="model-a"):
        Store(client, "p", "model-b", 8).ensure()


def test_other_vector_size_is_refused(client):
    Store(client, "p", "model-a", 8).ensure()
    with pytest.raises(SealMismatch):
        Store(client, "p", "model-a", 16).ensure()


def test_mismatch_is_detected_before_creating_anything(client):
    Store(client, "p", "model-a", 8).ensure()
    client.delete_collection("p_docs")
    with pytest.raises(SealMismatch):
        Store(client, "p", "model-a", 16).ensure()
    assert "p_docs" not in names(client)


def test_foreign_collection_is_refused_and_nothing_is_written(client):
    client.create_collection("p_docs", vectors_config=qm.VectorParams(size=16, distance=qm.Distance.COSINE))
    with pytest.raises(SealMismatch, match="without a Cerveau seal"):
        Store(client, "p", "model-a", 8).ensure()
    assert names(client) == {"p_docs"}
    with pytest.raises(SealMismatch):  # the prefix was not sealed by the failed attempt
        Store(client, "p", "model-a", 16).ensure()
    assert names(client) == {"p_docs"}


def test_lost_meta_with_data_is_refused_not_resealed(client):
    Store(client, "p", "model-a", 8).ensure()
    client.delete_collection("p_meta")
    with pytest.raises(SealMismatch):
        Store(client, "p", "model-other-same-size", 8).ensure()
    assert "p_meta" not in names(client)


def test_interrupted_first_start_recovers(client):
    # crash after creating the meta collection, before writing the seal
    client.create_collection("p_meta", vectors_config=qm.VectorParams(size=1, distance=qm.Distance.DOT))
    Store(client, "p", "model-a", 8).ensure()
    assert client.retrieve("p_meta", [SEAL_ID])
    assert names(client) == {"p_docs", "p_memories", "p_meta"}


def test_meta_without_seal_but_with_data_is_refused(client):
    client.create_collection("p_meta", vectors_config=qm.VectorParams(size=1, distance=qm.Distance.DOT))
    client.create_collection("p_docs", vectors_config=qm.VectorParams(size=8, distance=qm.Distance.COSINE))
    with pytest.raises(SealMismatch, match="no seal"):
        Store(client, "p", "model-a", 8).ensure()


def test_concurrent_creation_is_tolerated(client, monkeypatch):
    store = Store(client, "p", "model-a", 8)
    real_create = client.create_collection

    def racing_create(name, **kwargs):
        real_create(name, **kwargs)  # another process wins the race...
        raise RuntimeError("409 Conflict: already exists")  # ...and we get the conflict

    monkeypatch.setattr(client, "create_collection", racing_create)
    store.ensure()
    assert names(client) == {"p_docs", "p_memories", "p_meta"}


def test_creation_failure_is_not_hidden(client, monkeypatch):
    def failing_create(name, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(client, "create_collection", failing_create)
    with pytest.raises(RuntimeError, match="disk full"):
        Store(client, "p", "model-a", 8).ensure()


def test_another_prefix_is_independent(client):
    Store(client, "p", "model-a", 8).ensure()
    Store(client, "q", "model-b", 16).ensure()


def test_upsert_is_sent_in_batches(client, monkeypatch):
    store = Store(client, "p", "model-a", 8)
    store.ensure()
    sizes = []
    real = client.upsert

    def spy(collection, points, **kwargs):
        sizes.append(len(points))
        return real(collection, points, **kwargs)

    monkeypatch.setattr(client, "upsert", spy)
    store.upsert("p_docs", [qm.PointStruct(id=i, vector=[1.0] * 8, payload={}) for i in range(300)])
    assert sizes == [128, 128, 44]
    assert store.count("p_docs") == 300


def test_existing_collection_with_wrong_size_is_refused_even_with_a_valid_seal(client):
    Store(client, "p", "model-a", 8).ensure()
    client.delete_collection("p_memories")
    client.create_collection(
        "p_memories", vectors_config=qm.VectorParams(size=16, distance=qm.Distance.COSINE)
    )
    with pytest.raises(SealMismatch, match="size 16"):
        Store(client, "p", "model-a", 8).ensure()


def test_delete_is_sent_in_batches(client, monkeypatch):
    store = Store(client, "p", "model-a", 8)
    store.ensure()
    store.upsert("p_docs", [qm.PointStruct(id=i, vector=[1.0] * 8, payload={}) for i in range(2500)])
    sizes = []
    real = client.delete

    def spy(collection, points_selector, **kwargs):
        sizes.append(len(points_selector.points))
        return real(collection, points_selector=points_selector, **kwargs)

    monkeypatch.setattr(client, "delete", spy)
    store.delete_points("p_docs", list(range(2500)))
    assert sizes == [1000, 1000, 500] and store.count("p_docs") == 0


def test_foreign_meta_collections_are_refused(client):
    client.create_collection("p_meta", vectors_config=qm.VectorParams(size=1024, distance=qm.Distance.COSINE))
    with pytest.raises(SealMismatch, match="not created by Cerveau"):
        Store(client, "p", "model-a", 8).ensure()
    client.create_collection("q_meta", vectors_config=qm.VectorParams(size=1, distance=qm.Distance.DOT))
    client.upsert("q_meta", [qm.PointStruct(id=7, vector=[1.0], payload={"someone": "else"})])
    with pytest.raises(SealMismatch, match="not created by Cerveau"):
        Store(client, "q", "model-a", 8).ensure()
    assert names(client) == {"p_meta", "q_meta"}


def test_local_qdrant_client_ignores_proxies(monkeypatch):
    import cerveau.store as store_module
    from cerveau.config import Settings

    seen = {}
    monkeypatch.setattr(store_module, "QdrantClient", lambda **kw: seen.update(kw))
    store_module.make_client(Settings(qdrant_url="http://127.0.0.1:6333"))
    assert seen["trust_env"] is False and seen["check_compatibility"] is False
    seen.clear()
    store_module.make_client(Settings(qdrant_url="https://q.example:6333", qdrant_api_key="k"))
    assert "trust_env" not in seen


def test_drop_removes_only_this_prefix(client):
    Store(client, "p", "model-a", 8).ensure()
    Store(client, "q", "model-a", 8).ensure()
    Store(client, "p", "model-a", 8).drop()
    assert names(client) == {"q_docs", "q_memories", "q_meta"}


def test_reset_never_deletes_collections_cerveau_did_not_create(client):
    client.create_collection("x_docs", vectors_config=qm.VectorParams(size=8, distance=qm.Distance.COSINE))
    client.create_collection(
        "x_memories", vectors_config=qm.VectorParams(size=8, distance=qm.Distance.COSINE)
    )
    with pytest.raises(SealMismatch):
        Store(client, "x", "model-a", 8).drop()
    client.create_collection("x_meta", vectors_config=qm.VectorParams(size=1, distance=qm.Distance.DOT))
    client.upsert("x_meta", [qm.PointStruct(id=SEAL_ID, vector=[1.0], payload={"other": "tool"})])
    with pytest.raises(SealMismatch):
        Store(client, "x", "model-a", 8).drop()
    assert names(client) == {"x_docs", "x_memories", "x_meta"}


def test_reset_works_on_a_wrongly_sealed_prefix(client):
    Store(client, "p", "wrong-model", 768).ensure()
    Store(client, "p", "right-model", 1024).drop()  # different settings: still a Cerveau seal
    assert names(client) == set()


def test_drop_refuses_data_collections_next_to_an_unsealed_meta(client):
    from qdrant_client.http import models as qm

    store = Store(client, "x", "m", 4)
    client.create_collection(store.meta, vectors_config=qm.VectorParams(size=1, distance=qm.Distance.DOT))
    client.create_collection(store.docs, vectors_config=qm.VectorParams(size=4, distance=qm.Distance.COSINE))
    client.upsert(store.docs, [qm.PointStruct(id=1, vector=[1, 0, 0, 0], payload={})])
    with pytest.raises(SealMismatch, match="hold data"):
        store.drop()
    assert client.collection_exists(store.docs)


def test_other_memories_ignores_collections_without_a_cerveau_seal(client):
    from qdrant_client.http import models as qm

    mine = Store(client, "mine", "m", 4)
    mine.ensure()
    sealed = Store(client, "old", "m", 4)
    sealed.ensure()
    client.create_collection("zzapp_meta", vectors_config=qm.VectorParams(size=1, distance=qm.Distance.DOT))
    client.create_collection(
        "zzapp_memories", vectors_config=qm.VectorParams(size=4, distance=qm.Distance.COSINE)
    )
    assert mine.other_memories() == ["old_memories"]


def test_an_unreadable_neighbour_collection_does_not_block(client, monkeypatch):
    mine = Store(client, "mine", "m", 4)
    mine.ensure()
    Store(client, "old", "m", 4).ensure()
    Store(client, "odd", "m", 4).ensure()
    real = mine._read_seal

    def broken(meta=None):
        if meta == "odd_meta":
            raise ValueError("Invalid non-printable ASCII character in URL")
        return real(meta)

    monkeypatch.setattr(mine, "_read_seal", broken)
    assert mine.other_memories() == ["old_memories"]
