#include "roaring_test_fixture.h"

namespace arrow {

// ─────────────────────────────────────────────────────────────────────────────
// Iterator (CRoaring: iterator tests, to_uint32_array_test)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, IteratorRangeFor) {
    bm.addRange(10, 20);
    std::vector<uint32_t> collected;
    for (uint32_t v : bm) collected.push_back(v);
    EXPECT_EQ(collected.size(), 10u);
    for (uint32_t i = 0; i < 10; ++i) {
        EXPECT_EQ(collected[i], 10u + i);
    }
}


TEST_F(RoaringBitmapTest, IteratorEmpty) {
    std::vector<uint32_t> collected;
    for (uint32_t v : bm) collected.push_back(v);
    EXPECT_TRUE(collected.empty());
}


TEST_F(RoaringBitmapTest, IteratorMultiChunk) {
    bm.add(0);
    bm.add(65536);
    bm.add(131072);
    auto vec = bm.toVector();
    EXPECT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[0], 0u);
    EXPECT_EQ(vec[1], 65536u);
    EXPECT_EQ(vec[2], 131072u);
}


TEST_F(RoaringBitmapTest, ToVectorCorrectness) {
    bm.addRange(0, 100);
    auto vec = bm.toVector();
    EXPECT_EQ(vec.size(), 100u);
    for (uint32_t i = 0; i < 100; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}


// CRoaring: to_uint32_array_test — verify spacing for different offsets.
TEST_F(RoaringBitmapTest, ToVectorSpacing) {
    for (uint32_t offset : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
        RoaringBitmap rb;
        for (uint32_t x = 0; x < 65536; x += offset) rb.add(x);
        auto vec = rb.toVector();
        EXPECT_EQ(vec.size(), static_cast<size_t>((65536 + offset - 1) / offset));
        for (size_t k = 1; k < vec.size(); ++k) {
            EXPECT_EQ(vec[k] - vec[k - 1], offset);
        }
    }
}


TEST_F(RoaringBitmapTest, ForEachCallback) {
    bm.addRange(0, 10);
    uint32_t sum = 0;
    bm.forEach([&sum](uint32_t v) { sum += v; });
    EXPECT_EQ(sum, 45u);  // 0+1+...+9
}


// ─────────────────────────────────────────────────────────────────────────────
// Iterator over bitmap container (CRoaring: bitset to_uint32_array_test)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, IteratorOverBitmapContainer) {
    // Force bitmap container by adding > 4096 elements.
    bm.addRange(0, 5000);
    auto vec = bm.toVector();
    EXPECT_EQ(vec.size(), 5000u);
    for (uint32_t i = 0; i < 5000; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Iterator over run container
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, IteratorOverRunContainer) {
    bm.addRange(10, 20);
    bm.addRange(100, 200);
    bm.optimize();

    auto vec = bm.toVector();
    EXPECT_EQ(vec.size(), 110u);
    EXPECT_EQ(vec.front(), 10u);
    EXPECT_EQ(vec.back(), 199u);
}


// ─────────────────────────────────────────────────────────────────────────────
// Iterator: complete element-by-element verification (CRoaring: iterator tests)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, IteratorFullVerification) {
    // Multi-chunk bitmap with different container types.
    // Chunk 0: array (sparse)
    for (uint32_t i = 0; i < 100; i += 3) bm.add(i);
    // Chunk 1: bitmap (dense)
    for (uint32_t i = 65536; i < 70536; ++i) bm.add(i);
    // Chunk 2: small range → optimize to run
    bm.addRange(131072, 131172);

    auto vec = bm.toVector();
    EXPECT_EQ(vec.size(), bm.cardinality());

    // Verify sorted.
    for (size_t i = 1; i < vec.size(); ++i) {
        EXPECT_LT(vec[i - 1], vec[i]);
    }

    // Verify every element is in the bitmap.
    for (uint32_t v : vec) {
        EXPECT_TRUE(bm.contains(v));
    }

    // Verify no extra elements.
    size_t countViaForEach = 0;
    bm.forEach([&](uint32_t) { ++countViaForEach; });
    EXPECT_EQ(countViaForEach, vec.size());
}


// ─────────────────────────────────────────────────────────────────────────────
// ReverseIterator tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, ReverseIteratorEmpty) {
    std::vector<uint32_t> result;
    for (auto it = bm.rbegin(); it != bm.rend(); ++it)
        result.push_back(*it);
    EXPECT_TRUE(result.empty());
}


TEST_F(RoaringBitmapTest, ReverseIteratorSingleElement) {
    bm.add(42);
    std::vector<uint32_t> result;
    for (auto it = bm.rbegin(); it != bm.rend(); ++it)
        result.push_back(*it);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 42u);
}


TEST_F(RoaringBitmapTest, ReverseIteratorMultiChunkDescending) {
    // Values in different chunks.
    bm.add(10);
    bm.add(70000);   // chunk 1
    bm.add(200000);  // chunk 3
    std::vector<uint32_t> result;
    for (auto it = bm.rbegin(); it != bm.rend(); ++it)
        result.push_back(*it);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 200000u);
    EXPECT_EQ(result[1], 70000u);
    EXPECT_EQ(result[2], 10u);
}


TEST_F(RoaringBitmapTest, ReverseIteratorMatchesForward) {
    for (uint32_t i = 0; i < 500; ++i) bm.add(i * 137);  // scattered
    auto fwd = bm.toVector();
    std::vector<uint32_t> rev;
    for (auto it = bm.rbegin(); it != bm.rend(); ++it)
        rev.push_back(*it);
    std::reverse(rev.begin(), rev.end());
    EXPECT_EQ(fwd, rev);
}


// ─────────────────────────────────────────────────────────────────────────────
// moveEqualOrLarger tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, MoveEqualOrLargerExactMatch) {
    bm.add(10); bm.add(20); bm.add(30);
    auto it = bm.begin();
    it.moveEqualOrLarger(20);
    EXPECT_EQ(*it, 20u);
}


TEST_F(RoaringBitmapTest, MoveEqualOrLargerBetweenElements) {
    bm.add(10); bm.add(20); bm.add(30);
    auto it = bm.begin();
    it.moveEqualOrLarger(15);
    EXPECT_EQ(*it, 20u);
}


TEST_F(RoaringBitmapTest, MoveEqualOrLargerBeyondMax) {
    bm.add(10); bm.add(20);
    auto it = bm.begin();
    it.moveEqualOrLarger(100);
    EXPECT_EQ(it, bm.end());
}


TEST_F(RoaringBitmapTest, MoveEqualOrLargerBeforeMin) {
    bm.add(10); bm.add(20);
    auto it = bm.begin();
    it.moveEqualOrLarger(5);  // already at 10 >= 5, no-op
    EXPECT_EQ(*it, 10u);
}


TEST_F(RoaringBitmapTest, MoveEqualOrLargerCrossChunk) {
    bm.add(100);      // chunk 0
    bm.add(70000);    // chunk 1
    bm.add(200000);   // chunk 3
    auto it = bm.begin();
    EXPECT_EQ(*it, 100u);
    it.moveEqualOrLarger(70000);
    EXPECT_EQ(*it, 70000u);
    it.moveEqualOrLarger(100000);
    EXPECT_EQ(*it, 200000u);
}


// ─────────────────────────────────────────────────────────────────────────────
// readMany tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, ReadManyAll) {
    for (uint32_t i = 0; i < 100; ++i) bm.add(i);
    auto it = bm.begin();
    std::vector<uint32_t> buf(200);
    uint32_t n = it.readMany(buf.data(), 200);
    EXPECT_EQ(n, 100u);
    buf.resize(n);
    EXPECT_EQ(buf, bm.toVector());
}


TEST_F(RoaringBitmapTest, ReadManyBatches) {
    for (uint32_t i = 0; i < 100; ++i) bm.add(i * 3);
    auto it = bm.begin();
    std::vector<uint32_t> all;
    uint32_t buf[10];
    while (true) {
        uint32_t n = it.readMany(buf, 10);
        if (n == 0) break;
        for (uint32_t i = 0; i < n; ++i) all.push_back(buf[i]);
    }
    EXPECT_EQ(all, bm.toVector());
}


TEST_F(RoaringBitmapTest, ReadManyEmpty) {
    auto it = bm.begin();
    uint32_t buf[10];
    EXPECT_EQ(it.readMany(buf, 10), 0u);
}
} // namespace arrow
