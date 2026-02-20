"""Tests for save/load persistence."""

import os
import pytest
from arrowdb import (
    Collection,
    CollectionConfig,
    Client,
)


class TestCollectionPersistence:
    def test_save_and_load(self, tmp_dir, dim, make_vector):
        config = CollectionConfig(name="persist", dimensions=dim)
        col = Collection(config)
        vectors = {}
        for i in range(20):
            vec = make_vector(seed=i)
            col.insert(f"v{i}", vec, metadata={"idx": i})
            vectors[f"v{i}"] = vec

        save_path = os.path.join(tmp_dir, "save_test")
        col.save(save_path)

        loaded = Collection.load(save_path)
        assert loaded.name == "persist"
        assert loaded.dimension == dim
        assert len(loaded) == 20

        # Verify search still works
        results = loaded.search(make_vector(seed=0), k=5)
        assert len(results) == 5
        assert results[0].id == "v0"

    def test_persistent_collection(self, tmp_dir, dim, make_vector):
        path = os.path.join(tmp_dir, "persistent")
        config = CollectionConfig(name="wal_test", dimensions=dim)
        col = Collection.create(config, path)
        for i in range(10):
            col.insert(f"v{i}", make_vector(seed=i))
        col.close()
        del col  # release file lock

        loaded = Collection.load(path)
        assert len(loaded) == 10


class TestClientPersistence:
    def test_client_persistence_roundtrip(self, tmp_dir, dim, make_vector):
        # Create and populate
        with Client(tmp_dir) as client:
            config = CollectionConfig(name="roundtrip", dimensions=dim)
            col = client.create_collection("roundtrip", config)
            for i in range(15):
                col.insert(f"v{i}", make_vector(seed=i), metadata={"n": i})

        # Reopen and verify
        with Client(tmp_dir) as client:
            assert client.has_collection("roundtrip")
            col = client.get_collection("roundtrip")
            assert len(col) == 15
            results = col.search(make_vector(seed=0), k=3)
            assert len(results) == 3
