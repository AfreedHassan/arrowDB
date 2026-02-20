"""Tests for Client (multi-collection management)."""

import pytest
from arrowdb import (
    Client,
    CollectionConfig,
    Space,
)


class TestClientBasics:
    def test_create_and_get(self, tmp_dir, dim, make_vector):
        client = Client(tmp_dir)
        config = CollectionConfig(name="col1", dimensions=dim)
        col = client.create_collection("col1", config)
        assert col.name == "col1"

        col2 = client.get_collection("col1")
        assert col2.name == "col1"
        client.close()

    def test_list_collections(self, tmp_dir, dim):
        client = Client(tmp_dir)
        config = CollectionConfig(name="a", dimensions=dim)
        client.create_collection("a", config)
        config2 = CollectionConfig(name="b", dimensions=dim)
        client.create_collection("b", config2)
        names = client.list_collections()
        assert set(names) == {"a", "b"}
        client.close()

    def test_has_collection(self, tmp_dir, dim):
        client = Client(tmp_dir)
        assert not client.has_collection("x")
        config = CollectionConfig(name="x", dimensions=dim)
        client.create_collection("x", config)
        assert client.has_collection("x")
        client.close()

    def test_drop_collection(self, tmp_dir, dim):
        client = Client(tmp_dir)
        config = CollectionConfig(name="x", dimensions=dim)
        client.create_collection("x", config)
        client.drop_collection("x")
        assert not client.has_collection("x")
        client.close()

    def test_get_or_create(self, tmp_dir, dim, make_vector):
        client = Client(tmp_dir)
        config = CollectionConfig(name="col", dimensions=dim)
        col1 = client.get_or_create_collection("col", config)
        col1.insert("v1", make_vector(seed=1))

        col2 = client.get_or_create_collection("col", config)
        assert len(col2) == 1  # same collection
        client.close()

    def test_get_nonexistent_raises(self, tmp_dir):
        client = Client(tmp_dir)
        with pytest.raises(Exception):
            client.get_collection("nope")
        client.close()


class TestContextManager:
    def test_with_statement(self, tmp_dir, dim, make_vector):
        with Client(tmp_dir) as client:
            config = CollectionConfig(name="ctx", dimensions=dim)
            col = client.create_collection("ctx", config)
            col.insert("v1", make_vector(seed=1))
            assert len(col) == 1


class TestCollectionRefOperations:
    def test_search_via_ref(self, tmp_dir, dim, make_vector):
        with Client(tmp_dir) as client:
            config = CollectionConfig(name="search_test", dimensions=dim)
            col = client.create_collection("search_test", config)
            for i in range(20):
                col.insert(f"v{i}", make_vector(seed=i))

            results = col.search(make_vector(seed=0), k=5)
            assert len(results) == 5
            assert results[0].id == "v0"

    def test_query_via_ref(self, tmp_dir, dim, make_vector):
        with Client(tmp_dir) as client:
            config = CollectionConfig(name="query_test", dimensions=dim)
            col = client.create_collection("query_test", config)
            for i in range(10):
                col.insert(f"v{i}", make_vector(seed=i), metadata={"n": i})

            result = col.query(make_vector(seed=0), k=3)
            assert len(result) == 3

    def test_metadata_via_ref(self, tmp_dir, dim, make_vector):
        with Client(tmp_dir) as client:
            config = CollectionConfig(name="meta_test", dimensions=dim)
            col = client.create_collection("meta_test", config)
            col.insert("v1", make_vector(seed=1))
            col.set_metadata("v1", {"key": "value"})
            meta = col.get_metadata("v1")
            assert meta["key"] == "value"
