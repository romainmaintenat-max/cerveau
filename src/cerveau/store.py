"""Qdrant storage: sealed collections.

Three collections, all named from the configured prefix:
- <prefix>_docs      indexed documents (rebuildable from the files),
- <prefix>_memories  memories (rebuildable from the journal),
- <prefix>_meta      a single "seal" point recording the embedding model and size.

The seal prevents a classic silent failure: querying vectors made by one model
with a query embedded by another model. Any mismatch stops the program.

Only `drop()` deletes collections, on an explicit human command, and only those of
this prefix that carry a Cerveau seal (any model): never collections Cerveau did not create.
"""

from __future__ import annotations

import warnings
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from cerveau.config import LOOPBACK_HOSTS, Settings

SEAL_ID = "00000000-0000-0000-0000-000000000001"
STORE_SCHEMA = 1
UPSERT_BATCH = 128
DELETE_BATCH = 1000


class SealMismatch(RuntimeError):
    """The existing collections were built with another model or vector size."""


def make_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url == ":memory:":
        return QdrantClient(location=":memory:")
    with warnings.catch_warnings():
        # Plain HTTP is expected for a Qdrant bound to 127.0.0.1 (see docker-compose.yml).
        # For any other host, the warning is kept: use HTTPS there.
        if urlparse(settings.qdrant_url).hostname in LOOPBACK_HOSTS:
            warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")
        extra = {}
        if urlparse(settings.qdrant_url).hostname in LOOPBACK_HOSTS:
            # Never send a local Qdrant (and its key) through a proxy. The client's version
            # check opens its own connection that ignores trust_env: it is skipped here
            # (docker-compose.yml pins the server version anyway).
            extra["trust_env"] = False
            extra["check_compatibility"] = False
        return QdrantClient(
            timeout=60,  # a batch of 128 points of 1024 dims can exceed the 5 s default on slow disks
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False,
            **extra,
        )


class Store:
    def __init__(self, client: QdrantClient, prefix: str, model: str, vector_size: int) -> None:
        self.client = client
        self.model = model
        self.vector_size = vector_size
        self.docs = f"{prefix}_docs"
        self.memories = f"{prefix}_memories"
        self.meta = f"{prefix}_meta"

    # -- setup -----------------------------------------------------------------

    def ensure(self) -> None:
        """Check everything, then create what is missing. Idempotent.

        Order of creation: seal first, data collections after. So:
        - data collections without a seal were not made by Cerveau: refused, nothing written;
        - an empty meta collection with no data collections is our own interrupted first
          start: the seal is written and the start continues.
        """
        meta_exists = self.client.collection_exists(self.meta)
        data_exists = [n for n in (self.docs, self.memories) if self.client.collection_exists(n)]
        if meta_exists:
            seal = self._read_seal()
            if seal is None and (data_exists or not self._is_own_empty_meta()):
                raise SealMismatch(
                    f"collection {self.meta!r} exists but has no seal and was not created by Cerveau. "
                    "Use another CERVEAU_PREFIX."
                )
            if seal is not None:
                self.check_seal()
        elif data_exists:
            raise SealMismatch(
                f"collections {data_exists} exist without a Cerveau seal ({self.meta!r} is missing). "
                "Refusing to adopt them. Use another CERVEAU_PREFIX."
            )

        if not meta_exists:
            self._create(self.meta, qm.VectorParams(size=1, distance=qm.Distance.DOT))
        if self._read_seal() is None:
            self.client.upsert(
                self.meta,
                [
                    qm.PointStruct(
                        id=SEAL_ID,
                        vector=[1.0],
                        payload={
                            "embed_model": self.model,
                            "vector_size": self.vector_size,
                            "schema": STORE_SCHEMA,
                        },
                    )
                ],
                wait=True,
            )
        self.check_seal()  # also catches a concurrent first start with another model
        for name in (self.docs, self.memories):
            if not self.client.collection_exists(name):
                self._create(name, qm.VectorParams(size=self.vector_size, distance=qm.Distance.COSINE))
            for field in ("state", "path", "kind"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # local mode: payload indexes are a no-op
                    self.client.create_payload_index(name, field, qm.PayloadSchemaType.KEYWORD)

    def is_complete(self) -> bool:
        return all(self.client.collection_exists(n) for n in (self.meta, self.docs, self.memories))

    def other_memories(self) -> list[str]:
        """Memory collections of the other Cerveau prefixes of this Qdrant: only those whose
        meta collection carries a Cerveau seal, never another application's collections."""
        names = {c.name for c in self.client.get_collections().collections}
        found = []
        for name in sorted(names):
            prefix = name[: -len("_meta")]
            if not name.endswith("_meta") or name == self.meta or f"{prefix}_memories" not in names:
                continue
            try:
                seal = self._read_seal(name)
            except Exception:  # a collection that cannot even be read is not ours to touch
                continue
            if seal is not None and "embed_model" in seal and "vector_size" in seal:
                found.append(f"{prefix}_memories")
        return found

    def drop(self) -> None:
        """Delete this prefix's three collections, only if they were made by Cerveau."""
        if not self.client.collection_exists(self.meta):
            others = [n for n in (self.docs, self.memories) if self.client.collection_exists(n)]
            if not others:
                return  # nothing to delete
            raise SealMismatch(f"{others} exist without {self.meta!r}: not created by Cerveau, left alone")
        seal = self._read_seal()
        if seal is None:
            if not self._is_own_empty_meta():
                raise SealMismatch(f"{self.meta!r} has no Cerveau seal: nothing deleted")
            # An interrupted first start creates no data: points mean someone else's collections.
            filled = [
                n
                for n in (self.docs, self.memories)
                if self.client.collection_exists(n) and self.client.count(n, exact=True).count
            ]
            if filled:
                raise SealMismatch(f"{filled} hold data but {self.meta!r} has no seal: nothing deleted")
        elif "embed_model" not in seal or "vector_size" not in seal:
            raise SealMismatch(f"{self.meta!r} does not hold a Cerveau seal: nothing deleted")
        for name in (self.docs, self.memories, self.meta):
            if self.client.collection_exists(name):
                self.client.delete_collection(name)

    def _create(self, name: str, params: qm.VectorParams) -> None:
        """Create a collection, tolerating another process creating it at the same time."""
        try:
            self.client.create_collection(name, vectors_config=params)
        except Exception:
            if not self.client.collection_exists(name):
                raise

    def _is_own_empty_meta(self) -> bool:
        """An empty 1-dim dot meta collection: what an interrupted first start leaves."""
        params = self.client.get_collection(self.meta).config.params.vectors
        return (
            getattr(params, "size", None) == 1
            and getattr(params, "distance", None) == qm.Distance.DOT
            and self.client.count(self.meta, exact=True).count == 0
        )

    def _read_seal(self, meta: str | None = None) -> dict | None:
        meta = meta or self.meta
        params = self.client.get_collection(meta).config.params.vectors
        if getattr(params, "size", None) != 1:
            return None
        records = self.client.retrieve(meta, [SEAL_ID], with_payload=True)
        return (records[0].payload or {}) if records else None

    def check_seal(self) -> None:
        seal = self._read_seal()
        if seal is None:
            raise SealMismatch(f"collection {self.meta!r} exists but has no seal")
        if seal.get("embed_model") != self.model or seal.get("vector_size") != self.vector_size:
            raise SealMismatch(
                f"these collections were built with model {seal.get('embed_model')!r} "
                f"({seal.get('vector_size')} dims); the configuration says {self.model!r} "
                f"({self.vector_size} dims). Use another CERVEAU_PREFIX to start fresh, "
                "or set the configuration back."
            )
        for name in (self.docs, self.memories):
            if self.client.collection_exists(name):
                params = self.client.get_collection(name).config.params.vectors
                size = getattr(params, "size", None)
                if size != self.vector_size:
                    raise SealMismatch(f"collection {name!r} has vectors of size {size}")

    # -- data ------------------------------------------------------------------

    def upsert(self, collection: str, points: list[qm.PointStruct]) -> None:
        # Batches keep each request far below Qdrant's request size limit (32 MB by default):
        # a 1024-dim point is about 20 KB in JSON.
        for start in range(0, len(points), UPSERT_BATCH):
            self.client.upsert(collection, points[start : start + UPSERT_BATCH], wait=True)

    def set_state(self, collection: str, point_id: str, state: str) -> None:
        self.client.set_payload(collection, {"state": state}, points=[point_id], wait=True)

    def get(self, collection: str, point_id: str | int) -> qm.Record | None:
        records = self.client.retrieve(collection, [point_id], with_payload=True)
        return records[0] if records else None

    def search(
        self, collection: str, vector: list[float], limit: int, include_archived: bool
    ) -> list[qm.ScoredPoint]:
        query_filter = None
        if not include_archived:
            query_filter = qm.Filter(
                must_not=[qm.FieldCondition(key="state", match=qm.MatchValue(value="archived"))]
            )
        return self.client.query_points(
            collection, query=vector, limit=limit, query_filter=query_filter, with_payload=True
        ).points

    def scroll_payloads(
        self, collection: str, flt: qm.Filter | None = None, fields: list[str] | None = None
    ) -> list[qm.Record]:
        records: list[qm.Record] = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection,
                scroll_filter=flt,
                limit=512,
                offset=offset,
                with_payload=fields if fields is not None else True,
                with_vectors=False,
            )
            records.extend(batch)
            if offset is None:
                return records

    def delete_points(self, collection: str, ids: list[str] | list[str | int]) -> None:
        for start in range(0, len(ids), DELETE_BATCH):
            self.client.delete(
                collection,
                points_selector=qm.PointIdsList(points=ids[start : start + DELETE_BATCH]),
                wait=True,
            )

    def count(self, collection: str) -> int:
        return self.client.count(collection, exact=True).count
