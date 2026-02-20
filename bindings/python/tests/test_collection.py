"""Tests for Collection (in-memory, standalone)."""

import pytest
from arrowdb import (
    Collection,
    CollectionConfig,
    Document,
    Space,
    Quantization,
    IndexConfig,
    HNSWParams,
    ArrowDBError,
)


@pytest.fixture
def collection(dim):
    config = CollectionConfig(name="test", dimensions=dim)
    return Collection(config)


class TestInsert:
    def test_insert_auto_id(self, collection, make_vector):
        vec_id = collection.insert(make_vector(seed=1))
        assert isinstance(vec_id, str)
        assert len(vec_id) > 0
        assert len(collection) == 1

    def test_insert_with_id(self, collection, make_vector):
        collection.insert("v1", make_vector(seed=1))
        assert len(collection) == 1

    def test_insert_with_metadata(self, collection, make_vector):
        vec_id = collection.insert(
            make_vector(seed=1),
            metadata={"category": "test", "score": 0.95, "count": 42, "active": True},
        )
        meta = collection.get_metadata(vec_id)
        assert meta["category"] == "test"
        assert meta["score"] == pytest.approx(0.95)
        assert meta["count"] == 42
        assert meta["active"] is True

    def test_insert_duplicate_updates(self, collection, make_vector):
        """HNSW handles duplicate IDs by updating the existing point."""
        collection.insert("v1", make_vector(seed=1))
        collection.insert("v1", make_vector(seed=2))
        assert len(collection) == 1  # still one vector

    def test_insert_wrong_dimension(self, collection):
        with pytest.raises(Exception):
            collection.insert([1.0, 2.0])  # wrong dim


class TestInsertDoc:
    def test_insert_document(self, collection, make_vector):
        doc = Document(embedding=make_vector(seed=1), metadata={"k": "v"}, id="doc1")
        result_id = collection.insert_doc(doc)
        assert result_id == "doc1"
        assert len(collection) == 1

    def test_insert_document_auto_id(self, collection, make_vector):
        doc = Document(embedding=make_vector(seed=1))
        result_id = collection.insert_doc(doc)
        assert len(result_id) > 0


class TestBatchInsert:
    def test_insert_batch(self, collection, make_vector):
        docs = [
            Document(embedding=make_vector(seed=i), id=f"v{i}")
            for i in range(10)
        ]
        result = collection.insert_batch(docs)
        assert result.success_count == 10
        assert result.failure_count == 0
        assert len(collection) == 10


class TestSearch:
    def test_search_basic(self, collection, make_vector):
        for i in range(20):
            collection.insert(f"v{i}", make_vector(seed=i))

        results = collection.search(make_vector(seed=0), k=5)
        assert len(results) == 5
        assert results[0].id == "v0"

    def test_query_with_metadata(self, collection, make_vector):
        for i in range(10):
            collection.insert(
                f"v{i}", make_vector(seed=i), metadata={"idx": i}
            )

        result = collection.query(make_vector(seed=0), k=3)
        assert len(result) == 3
        for hit in result:
            assert "idx" in hit.metadata


class TestGetUpdateRemove:
    def test_get(self, collection, make_vector):
        vec = make_vector(seed=42)
        collection.insert("v1", vec)
        retrieved = collection.get("v1")
        assert len(retrieved) == len(vec)

    def test_update(self, collection, make_vector):
        collection.insert("v1", make_vector(seed=1))
        collection.update("v1", make_vector(seed=2), metadata={"updated": True})
        meta = collection.get_metadata("v1")
        assert meta["updated"] is True

    def test_upsert_new(self, collection, make_vector):
        collection.upsert("v1", make_vector(seed=1))
        assert len(collection) == 1

    def test_upsert_existing(self, collection, make_vector):
        collection.insert("v1", make_vector(seed=1))
        collection.upsert("v1", make_vector(seed=2))
        assert len(collection) == 1

    def test_remove(self, collection, make_vector):
        collection.insert("v1", make_vector(seed=1))
        collection.remove("v1")
        # After remove, get should fail
        with pytest.raises(Exception):
            collection.get("v1")

    def test_remove_nonexistent_raises(self, collection):
        with pytest.raises(Exception):
            collection.remove("nonexistent")


class TestProperties:
    def test_name(self, collection):
        assert collection.name == "test"

    def test_dimension(self, collection, dim):
        assert collection.dimension == dim

    def test_space(self, collection):
        assert collection.space == Space.COSINE

    def test_stats(self, collection, make_vector):
        collection.insert("v1", make_vector(seed=1))
        s = collection.stats()
        assert s.vector_count == 1
        assert s.dimensions == 32
