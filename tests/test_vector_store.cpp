#include <gtest/gtest.h>
#include "arrow_utils.h"
#include "vector_store.h"
#include "flat_search.h"

using namespace arrow;

TEST(VectorStoreTest, InsertAndRetrieve) {
    VectorStore store(3);
    std::vector<float> v = {1.0f, 0.0f, 0.0f};
    auto normalized = validateAndNormalize(v, 3);
    size_t idx = store.insert(42, normalized);

    EXPECT_EQ(idx, 0);
    EXPECT_EQ(store.size(), 1);
    EXPECT_EQ(store.vecIdAt(0), 42);
}

TEST(FlatSearchTest, SimpleCosineSearch) {
    VectorStore store(3);
    store.insert(1, validateAndNormalize({1,0,0},3));
    store.insert(2, validateAndNormalize({0,1,0},3));
    store.insert(3, validateAndNormalize({0,0,1},3));

    auto query = validateAndNormalize({1,0.5,0},3);
    auto results = flatSearch(store, query, 2);

    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].id, 1); // closest
    EXPECT_EQ(results[1].id, 2); // second closest
}
