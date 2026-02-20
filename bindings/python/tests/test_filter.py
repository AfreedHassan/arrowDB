"""Tests for MetadataFilter DSL and Python callable filters."""

import pytest
from arrowdb import (
    Collection,
    CollectionConfig,
    MetadataFilter,
)


@pytest.fixture
def filtered_collection(dim, make_vector):
    config = CollectionConfig(name="filter_test", dimensions=dim)
    col = Collection(config)
    for i in range(50):
        col.insert(
            f"v{i}",
            make_vector(seed=i),
            metadata={
                "category": "even" if i % 2 == 0 else "odd",
                "value": i,
                "name": f"item_{i}",
                "active": i < 25,
            },
        )
    return col


class TestDSLFilters:
    def test_eq(self, filtered_collection, make_vector):
        f = MetadataFilter.eq("category", "even")
        results = filtered_collection.search(make_vector(seed=0), k=10, filter=f)
        for r in results:
            meta = filtered_collection.get_metadata(r.id)
            assert meta["category"] == "even"

    def test_neq(self, filtered_collection, make_vector):
        f = MetadataFilter.neq("category", "even")
        results = filtered_collection.search(make_vector(seed=0), k=10, filter=f)
        for r in results:
            meta = filtered_collection.get_metadata(r.id)
            assert meta["category"] != "even"

    def test_gt(self, filtered_collection, make_vector):
        f = MetadataFilter.gt("value", 40)
        results = filtered_collection.search(make_vector(seed=0), k=10, filter=f)
        for r in results:
            meta = filtered_collection.get_metadata(r.id)
            assert meta["value"] > 40

    def test_lte(self, filtered_collection, make_vector):
        f = MetadataFilter.lte("value", 5)
        results = filtered_collection.search(make_vector(seed=0), k=10, filter=f)
        for r in results:
            meta = filtered_collection.get_metadata(r.id)
            assert meta["value"] <= 5

    def test_in(self, filtered_collection, make_vector):
        f = MetadataFilter.in_("value", [0, 1, 2])
        results = filtered_collection.search(make_vector(seed=0), k=10, filter=f)
        for r in results:
            meta = filtered_collection.get_metadata(r.id)
            assert meta["value"] in [0, 1, 2]


class TestLogicalCombinators:
    def test_and(self, filtered_collection, make_vector):
        f = MetadataFilter.and_(
            MetadataFilter.eq("category", "even"),
            MetadataFilter.lt("value", 10),
        )
        results = filtered_collection.search(make_vector(seed=0), k=10, filter=f)
        for r in results:
            meta = filtered_collection.get_metadata(r.id)
            assert meta["category"] == "even"
            assert meta["value"] < 10

    def test_or(self, filtered_collection, make_vector):
        f = MetadataFilter.or_(
            MetadataFilter.eq("value", 0),
            MetadataFilter.eq("value", 1),
        )
        results = filtered_collection.search(make_vector(seed=0), k=10, filter=f)
        for r in results:
            meta = filtered_collection.get_metadata(r.id)
            assert meta["value"] in [0, 1]

    def test_not(self, filtered_collection, make_vector):
        f = MetadataFilter.not_(MetadataFilter.eq("category", "even"))
        results = filtered_collection.search(make_vector(seed=0), k=10, filter=f)
        for r in results:
            meta = filtered_collection.get_metadata(r.id)
            assert meta["category"] != "even"


class TestCallableFilter:
    def test_python_callable(self, filtered_collection, make_vector):
        f = MetadataFilter(lambda m: m.get("value", 0) < 5)
        results = filtered_collection.search(make_vector(seed=0), k=10, filter=f)
        for r in results:
            meta = filtered_collection.get_metadata(r.id)
            assert meta["value"] < 5

    def test_filter_direct_call(self):
        f = MetadataFilter.eq("x", 1)
        assert f({"x": 1}) is True
        assert f({"x": 2}) is False
        assert f({}) is False
