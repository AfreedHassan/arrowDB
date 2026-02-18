#include "roaring_test_fixture.h"

namespace arrow {

// ─────────────────────────────────────────────────────────────────────────────
// cpp_unit.cpp: test_cpp_rank_many equivalent
// Rank = count of elements ≤ x. We implement via AND + cardinality.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RankViaAndCardinality) {
    // {123, 9999, 0xFFFFFFF7, 0xFFFFFFFF}
    bm.add(123);
    bm.add(9999);
    bm.add(0xFFFFFFF7);
    bm.add(0xFFFFFFFF);

    // rank(x) = # elements <= x.
    // rank(123) = 1 (just 123)
    // rank(9999) = 2 (123, 9999)
    // rank(0xFFFFFFF7) = 3
    // rank(0xFFFFFFFF) = 4
    auto rank = [&](uint32_t x) -> uint32_t {
        uint32_t count = 0;
        bm.forEach([&](uint32_t v) { if (v <= x) ++count; });
        return count;
    };

    EXPECT_EQ(rank(123), 1u);
    EXPECT_EQ(rank(9999), 2u);
    EXPECT_EQ(rank(9999), 2u);  // duplicate query
    EXPECT_EQ(rank(0xFFFFFFF7), 3u);
    EXPECT_EQ(rank(0xFFFFFFFF), 4u);
}


// ─────────────────────────────────────────────────────────────────────────────
// cpp_unit.cpp: test_cpp_remove_range equivalent
// We don't have removeRange, but we can test remove in a range via loop.
// 8 sub-cases from CRoaring.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RemoveRangeWiderThanContent) {
    bm.addRange(10, 20);
    for (uint32_t i = 0; i < 30; ++i) bm.remove(i);
    EXPECT_TRUE(bm.empty());
}


TEST_F(RoaringBitmapTest, RemoveRangeLeftPartial) {
    bm.addRange(10, 20);
    for (uint32_t i = 5; i < 15; ++i) bm.remove(i);
    EXPECT_EQ(bm.cardinality(), 5u);  // [15,20)
    EXPECT_TRUE(bm.contains(15));
    EXPECT_FALSE(bm.contains(14));
}


TEST_F(RoaringBitmapTest, RemoveRangeRightPartial) {
    bm.addRange(10, 20);
    for (uint32_t i = 15; i < 25; ++i) bm.remove(i);
    EXPECT_EQ(bm.cardinality(), 5u);  // [10,15)
    EXPECT_TRUE(bm.contains(14));
    EXPECT_FALSE(bm.contains(15));
}


TEST_F(RoaringBitmapTest, RemoveRangeInterior) {
    bm.addRange(10, 20);
    for (uint32_t i = 13; i < 17; ++i) bm.remove(i);
    EXPECT_EQ(bm.cardinality(), 6u);  // [10,13) + [17,20)
    EXPECT_TRUE(bm.contains(12));
    EXPECT_FALSE(bm.contains(13));
    EXPECT_FALSE(bm.contains(16));
    EXPECT_TRUE(bm.contains(17));
}


TEST_F(RoaringBitmapTest, RemoveRangeBelow) {
    bm.addRange(10, 20);
    for (uint32_t i = 0; i < 5; ++i) bm.remove(i);
    EXPECT_EQ(bm.cardinality(), 10u);  // unchanged
}


TEST_F(RoaringBitmapTest, RemoveRangeAbove) {
    bm.addRange(10, 20);
    for (uint32_t i = 25; i < 30; ++i) bm.remove(i);
    EXPECT_EQ(bm.cardinality(), 10u);  // unchanged
}


// ─────────────────────────────────────────────────────────────────────────────
// toplevel_unit.c: full flip (0 to 0x10000) via XOR with full bitmap.
// We don't have flip(), but can simulate: flip(bm) = full ^ bm.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, FlipViaXorWithFull) {
    RoaringBitmap full;
    full.addRange(0, 65536);

    // Original: every 3rd value.
    for (uint32_t x = 0; x < 65536; x += 3) bm.add(x);

    auto flipped = full ^ bm;
    // Flipped should contain values NOT divisible by 3.
    EXPECT_EQ(flipped.cardinality(), 65536u - bm.cardinality());
    for (uint32_t x = 0; x < 65536; ++x) {
        EXPECT_EQ(flipped.contains(x), (x % 3 != 0));
    }
}


TEST_F(RoaringBitmapTest, FlipEmpty) {
    RoaringBitmap full;
    full.addRange(0, 65536);
    auto flipped = full ^ bm;  // bm is empty
    EXPECT_EQ(flipped, full);
}


TEST_F(RoaringBitmapTest, FlipFull) {
    RoaringBitmap full;
    full.addRange(0, 65536);
    auto flipped = full ^ full;
    EXPECT_TRUE(flipped.empty());
}


// ─────────────────────────────────────────────────────────────────────────────
// select() tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, SelectBasic) {
    bm.add(10);
    bm.add(20);
    bm.add(30);
    EXPECT_EQ(bm.select(0), 10u);
    EXPECT_EQ(bm.select(1), 20u);
    EXPECT_EQ(bm.select(2), 30u);
    EXPECT_FALSE(bm.select(3).has_value());
}


TEST_F(RoaringBitmapTest, SelectEmpty) {
    EXPECT_FALSE(bm.select(0).has_value());
}


TEST_F(RoaringBitmapTest, SelectSingleChunk) {
    for (uint32_t i = 0; i < 100; ++i) bm.add(i * 3);
    for (uint32_t i = 0; i < 100; ++i) {
        EXPECT_EQ(bm.select(i), i * 3);
    }
    EXPECT_FALSE(bm.select(100).has_value());
}


TEST_F(RoaringBitmapTest, SelectCrossChunk) {
    bm.add(0);
    bm.add(65536);     // chunk 1
    bm.add(131072);    // chunk 2
    EXPECT_EQ(bm.select(0), 0u);
    EXPECT_EQ(bm.select(1), 65536u);
    EXPECT_EQ(bm.select(2), 131072u);
    EXPECT_FALSE(bm.select(3).has_value());
}


TEST_F(RoaringBitmapTest, SelectBoundary) {
    bm.add(0);
    bm.add(UINT32_MAX);
    EXPECT_EQ(bm.select(0), 0u);
    EXPECT_EQ(bm.select(1), UINT32_MAX);
    EXPECT_FALSE(bm.select(2).has_value());
}


TEST_F(RoaringBitmapTest, SelectBitmapContainer) {
    // Force bitmap container by adding > 4096 elements in one chunk.
    for (uint32_t i = 0; i < 5000; ++i) bm.add(i);
    EXPECT_EQ(bm.select(0), 0u);
    EXPECT_EQ(bm.select(4999), 4999u);
    EXPECT_FALSE(bm.select(5000).has_value());
}


TEST_F(RoaringBitmapTest, SelectRunContainer) {
    bm.addRange(100, 200);
    bm.optimize();  // convert to run container
    EXPECT_EQ(bm.select(0), 100u);
    EXPECT_EQ(bm.select(50), 150u);
    EXPECT_EQ(bm.select(99), 199u);
    EXPECT_FALSE(bm.select(100).has_value());
}


TEST_F(RoaringBitmapTest, SelectMultipleRunContainers) {
    bm.addRange(10, 20);    // 10 elements
    bm.addRange(100, 110);  // 10 elements
    bm.optimize();
    EXPECT_EQ(bm.select(0), 10u);
    EXPECT_EQ(bm.select(9), 19u);
    EXPECT_EQ(bm.select(10), 100u);
    EXPECT_EQ(bm.select(19), 109u);
}


// ─────────────────────────────────────────────────────────────────────────────
// rank() tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RankBasic) {
    bm.add(10);
    bm.add(20);
    bm.add(30);
    EXPECT_EQ(bm.rank(5), 0u);
    EXPECT_EQ(bm.rank(10), 1u);
    EXPECT_EQ(bm.rank(15), 1u);
    EXPECT_EQ(bm.rank(20), 2u);
    EXPECT_EQ(bm.rank(30), 3u);
    EXPECT_EQ(bm.rank(100), 3u);
}


TEST_F(RoaringBitmapTest, RankEmpty) {
    EXPECT_EQ(bm.rank(0), 0u);
    EXPECT_EQ(bm.rank(UINT32_MAX), 0u);
}


TEST_F(RoaringBitmapTest, RankSingleChunk) {
    for (uint32_t i = 0; i < 50; ++i) bm.add(i * 2);  // evens 0..98
    EXPECT_EQ(bm.rank(0), 1u);     // 0 is present
    EXPECT_EQ(bm.rank(1), 1u);     // only 0 <= 1
    EXPECT_EQ(bm.rank(98), 50u);   // all 50 present
    EXPECT_EQ(bm.rank(99), 50u);   // still 50
}


TEST_F(RoaringBitmapTest, RankCrossChunk) {
    bm.add(0);
    bm.add(65536);
    bm.add(131072);
    EXPECT_EQ(bm.rank(0), 1u);
    EXPECT_EQ(bm.rank(65535), 1u);
    EXPECT_EQ(bm.rank(65536), 2u);
    EXPECT_EQ(bm.rank(131072), 3u);
    EXPECT_EQ(bm.rank(200000), 3u);
}


TEST_F(RoaringBitmapTest, RankBoundary) {
    bm.add(0);
    bm.add(UINT32_MAX);
    EXPECT_EQ(bm.rank(0), 1u);
    EXPECT_EQ(bm.rank(UINT32_MAX - 1), 1u);
    EXPECT_EQ(bm.rank(UINT32_MAX), 2u);
}


TEST_F(RoaringBitmapTest, RankBitmapContainer) {
    for (uint32_t i = 0; i < 5000; ++i) bm.add(i);
    EXPECT_EQ(bm.rank(0), 1u);
    EXPECT_EQ(bm.rank(4999), 5000u);
    EXPECT_EQ(bm.rank(5000), 5000u);
}


TEST_F(RoaringBitmapTest, RankRunContainer) {
    bm.addRange(100, 200);
    bm.optimize();
    EXPECT_EQ(bm.rank(99), 0u);
    EXPECT_EQ(bm.rank(100), 1u);
    EXPECT_EQ(bm.rank(150), 51u);
    EXPECT_EQ(bm.rank(199), 100u);
    EXPECT_EQ(bm.rank(200), 100u);
}


TEST_F(RoaringBitmapTest, RankSelectConsistency) {
    // rank and select should be inverses.
    for (uint32_t i = 0; i < 200; ++i) bm.add(i * 7);
    for (uint32_t r = 0; r < 200; ++r) {
        auto val = bm.select(r);
        ASSERT_TRUE(val.has_value());
        EXPECT_EQ(bm.rank(*val), r + 1);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// containsRange() tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, ContainsRangeBasic) {
    bm.addRange(10, 20);
    EXPECT_TRUE(bm.containsRange(10, 20));
    EXPECT_TRUE(bm.containsRange(10, 15));
    EXPECT_TRUE(bm.containsRange(15, 20));
    EXPECT_FALSE(bm.containsRange(9, 20));
    EXPECT_FALSE(bm.containsRange(10, 21));
}


TEST_F(RoaringBitmapTest, ContainsRangeEmpty) {
    EXPECT_TRUE(bm.containsRange(10, 10));   // empty range = true
    EXPECT_TRUE(bm.containsRange(5, 3));     // inverted = true
    EXPECT_FALSE(bm.containsRange(0, 1));    // non-empty range on empty bitmap
}


TEST_F(RoaringBitmapTest, ContainsRangeSingleChunk) {
    for (uint32_t i = 100; i < 200; ++i) bm.add(i);
    EXPECT_TRUE(bm.containsRange(100, 200));
    EXPECT_FALSE(bm.containsRange(99, 200));
    EXPECT_FALSE(bm.containsRange(100, 201));
}


TEST_F(RoaringBitmapTest, ContainsRangeCrossChunk) {
    bm.addRange(65530, 65542);  // spans chunks 0 and 1
    EXPECT_TRUE(bm.containsRange(65530, 65542));
    EXPECT_TRUE(bm.containsRange(65534, 65538));
    EXPECT_FALSE(bm.containsRange(65529, 65542));
}


TEST_F(RoaringBitmapTest, ContainsRangeBitmapContainer) {
    for (uint32_t i = 0; i < 5000; ++i) bm.add(i);
    EXPECT_TRUE(bm.containsRange(0, 5000));
    EXPECT_FALSE(bm.containsRange(0, 5001));
}


TEST_F(RoaringBitmapTest, ContainsRangeRunContainer) {
    bm.addRange(100, 200);
    bm.optimize();
    EXPECT_TRUE(bm.containsRange(100, 200));
    EXPECT_TRUE(bm.containsRange(150, 175));
    EXPECT_FALSE(bm.containsRange(99, 200));
}


TEST_F(RoaringBitmapTest, ContainsRangeWithGaps) {
    bm.addRange(10, 20);
    bm.addRange(30, 40);
    EXPECT_TRUE(bm.containsRange(10, 20));
    EXPECT_TRUE(bm.containsRange(30, 40));
    EXPECT_FALSE(bm.containsRange(10, 40));  // gap at 20-29
}


// ─────────────────────────────────────────────────────────────────────────────
// rangeCardinality() tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RangeCardinalityBasic) {
    bm.addRange(10, 20);
    EXPECT_EQ(bm.rangeCardinality(10, 20), 10u);
    EXPECT_EQ(bm.rangeCardinality(10, 15), 5u);
    EXPECT_EQ(bm.rangeCardinality(15, 20), 5u);
    EXPECT_EQ(bm.rangeCardinality(0, 100), 10u);
}


TEST_F(RoaringBitmapTest, RangeCardinalityEmpty) {
    EXPECT_EQ(bm.rangeCardinality(0, 100), 0u);
    EXPECT_EQ(bm.rangeCardinality(10, 10), 0u);
    EXPECT_EQ(bm.rangeCardinality(20, 10), 0u);
}


TEST_F(RoaringBitmapTest, RangeCardinalityCrossChunk) {
    bm.addRange(65530, 65542);
    EXPECT_EQ(bm.rangeCardinality(65530, 65542), 12u);
    EXPECT_EQ(bm.rangeCardinality(65530, 65536), 6u);
    EXPECT_EQ(bm.rangeCardinality(65536, 65542), 6u);
}


TEST_F(RoaringBitmapTest, RangeCardinalityBitmapContainer) {
    for (uint32_t i = 0; i < 5000; ++i) bm.add(i);
    EXPECT_EQ(bm.rangeCardinality(0, 5000), 5000u);
    EXPECT_EQ(bm.rangeCardinality(1000, 2000), 1000u);
    EXPECT_EQ(bm.rangeCardinality(4999, 5000), 1u);
    EXPECT_EQ(bm.rangeCardinality(5000, 6000), 0u);
}


TEST_F(RoaringBitmapTest, RangeCardinalityRunContainer) {
    bm.addRange(100, 200);
    bm.addRange(300, 400);
    bm.optimize();
    EXPECT_EQ(bm.rangeCardinality(100, 400), 200u);
    EXPECT_EQ(bm.rangeCardinality(150, 350), 100u);
    EXPECT_EQ(bm.rangeCardinality(200, 300), 0u);
}


TEST_F(RoaringBitmapTest, RangeCardinalityEqualsCardinalityForFullRange) {
    for (uint32_t i = 0; i < 1000; i += 3) bm.add(i);
    uint32_t card = bm.cardinality();
    // A range covering everything should match cardinality.
    EXPECT_EQ(bm.rangeCardinality(0, 1000), card);
}


// ─────────────────────────────────────────────────────────────────────────────
// flip() tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, FlipBasic) {
    bm.add(10);
    bm.add(12);
    bm.flip(10, 15);
    // 10 was present -> removed, 11 absent -> added, 12 present -> removed,
    // 13 absent -> added, 14 absent -> added
    EXPECT_FALSE(bm.contains(10));
    EXPECT_TRUE(bm.contains(11));
    EXPECT_FALSE(bm.contains(12));
    EXPECT_TRUE(bm.contains(13));
    EXPECT_TRUE(bm.contains(14));
    EXPECT_EQ(bm.cardinality(), 3u);
}


TEST_F(RoaringBitmapTest, FlipEmptyRange) {
    bm.flip(10, 10);  // empty range
    EXPECT_TRUE(bm.empty());
}


TEST_F(RoaringBitmapTest, FlipOnEmpty) {
    bm.flip(10, 20);
    EXPECT_EQ(bm.cardinality(), 10u);
    for (uint32_t i = 10; i < 20; ++i)
        EXPECT_TRUE(bm.contains(i));
}


TEST_F(RoaringBitmapTest, FlipSelfInverse) {
    bm.add(5);
    bm.add(15);
    bm.add(25);
    bm.flip(0, 30);
    bm.flip(0, 30);
    // Double flip = identity.
    EXPECT_EQ(bm.cardinality(), 3u);
    EXPECT_TRUE(bm.contains(5));
    EXPECT_TRUE(bm.contains(15));
    EXPECT_TRUE(bm.contains(25));
}


TEST_F(RoaringBitmapTest, FlipFullRange) {
    bm.addRange(0, 100);
    bm.flip(0, 100);
    EXPECT_TRUE(bm.empty());
}


TEST_F(RoaringBitmapTest, FlipCrossChunk) {
    bm.flip(65530, 65542);  // spans chunk boundary
    EXPECT_EQ(bm.cardinality(), 12u);
    for (uint32_t i = 65530; i < 65542; ++i)
        EXPECT_TRUE(bm.contains(i));
}


TEST_F(RoaringBitmapTest, FlipPartial) {
    bm.addRange(10, 30);
    bm.flip(20, 40);
    // 10..19 remain, 20..29 removed, 30..39 added.
    EXPECT_EQ(bm.cardinality(), 20u);
    for (uint32_t i = 10; i < 20; ++i) EXPECT_TRUE(bm.contains(i));
    for (uint32_t i = 20; i < 30; ++i) EXPECT_FALSE(bm.contains(i));
    for (uint32_t i = 30; i < 40; ++i) EXPECT_TRUE(bm.contains(i));
}


TEST_F(RoaringBitmapTest, FlipBitmapContainer) {
    for (uint32_t i = 0; i < 5000; ++i) bm.add(i);
    bm.flip(1000, 2000);
    EXPECT_EQ(bm.cardinality(), 4000u);
    EXPECT_FALSE(bm.contains(1500));
    EXPECT_TRUE(bm.contains(500));
    EXPECT_TRUE(bm.contains(3000));
}


TEST_F(RoaringBitmapTest, FlipContainerTypeTransitions) {
    // Start with array container, flip large range to force bitmap.
    bm.add(0);
    bm.flip(0, 10000);
    // 0 was present -> removed, 1..9999 added = 9999 elements.
    EXPECT_EQ(bm.cardinality(), 9999u);
    EXPECT_FALSE(bm.contains(0));
    EXPECT_TRUE(bm.contains(1));
    EXPECT_TRUE(bm.contains(9999));
}


// ─────────────────────────────────────────────────────────────────────────────
// removeRange() tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RemoveRangeBasic) {
    bm.addRange(10, 30);
    bm.removeRange(15, 25);
    EXPECT_EQ(bm.cardinality(), 10u);
    for (uint32_t i = 10; i < 15; ++i) EXPECT_TRUE(bm.contains(i));
    for (uint32_t i = 15; i < 25; ++i) EXPECT_FALSE(bm.contains(i));
    for (uint32_t i = 25; i < 30; ++i) EXPECT_TRUE(bm.contains(i));
}


TEST_F(RoaringBitmapTest, RemoveRangeEmpty) {
    bm.removeRange(0, 100);  // no-op on empty
    EXPECT_TRUE(bm.empty());
}


TEST_F(RoaringBitmapTest, RemoveRangeNoOp) {
    bm.addRange(10, 20);
    bm.removeRange(20, 30);  // nothing to remove
    EXPECT_EQ(bm.cardinality(), 10u);
}


TEST_F(RoaringBitmapTest, RemoveRangeInvertedRange) {
    bm.addRange(10, 20);
    bm.removeRange(20, 10);  // inverted = no-op
    EXPECT_EQ(bm.cardinality(), 10u);
}


TEST_F(RoaringBitmapTest, RemoveRangeWiderThanContentV2) {
    bm.addRange(10, 20);
    bm.removeRange(0, 100);
    EXPECT_TRUE(bm.empty());
}


TEST_F(RoaringBitmapTest, RemoveRangeLeftPartialV2) {
    bm.addRange(10, 20);
    bm.removeRange(10, 15);
    EXPECT_EQ(bm.cardinality(), 5u);
    for (uint32_t i = 15; i < 20; ++i) EXPECT_TRUE(bm.contains(i));
}


TEST_F(RoaringBitmapTest, RemoveRangeRightPartialV2) {
    bm.addRange(10, 20);
    bm.removeRange(15, 20);
    EXPECT_EQ(bm.cardinality(), 5u);
    for (uint32_t i = 10; i < 15; ++i) EXPECT_TRUE(bm.contains(i));
}


TEST_F(RoaringBitmapTest, RemoveRangeInteriorV2) {
    bm.addRange(10, 30);
    bm.removeRange(15, 25);
    EXPECT_EQ(bm.cardinality(), 10u);
}


TEST_F(RoaringBitmapTest, RemoveRangeCrossChunk) {
    bm.addRange(65530, 65542);
    bm.removeRange(65534, 65538);
    EXPECT_EQ(bm.cardinality(), 8u);
    EXPECT_TRUE(bm.contains(65530));
    EXPECT_FALSE(bm.contains(65535));
    EXPECT_FALSE(bm.contains(65536));
    EXPECT_TRUE(bm.contains(65541));
}


TEST_F(RoaringBitmapTest, RemoveRangeBitmapContainer) {
    for (uint32_t i = 0; i < 5000; ++i) bm.add(i);
    bm.removeRange(1000, 4000);
    EXPECT_EQ(bm.cardinality(), 2000u);
    EXPECT_TRUE(bm.contains(999));
    EXPECT_FALSE(bm.contains(1000));
    EXPECT_FALSE(bm.contains(3999));
    EXPECT_TRUE(bm.contains(4000));
}


TEST_F(RoaringBitmapTest, RemoveRangeRunContainer) {
    bm.addRange(100, 200);
    bm.optimize();
    bm.removeRange(140, 160);
    EXPECT_EQ(bm.cardinality(), 80u);
    EXPECT_TRUE(bm.contains(139));
    EXPECT_FALSE(bm.contains(140));
    EXPECT_FALSE(bm.contains(159));
    EXPECT_TRUE(bm.contains(160));
}


TEST_F(RoaringBitmapTest, RemoveRangeEntireContainer) {
    bm.addRange(0, 100);
    bm.removeRange(0, 100);
    EXPECT_TRUE(bm.empty());
}


// ═════════════════════════════════════════════════════════════════════════════
// Gap 23: rank/containerRank SIMD
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, RankBasicArray) {
    bm.add(10); bm.add(20); bm.add(30); bm.add(40);
    EXPECT_EQ(bm.rank(0), 0u);
    EXPECT_EQ(bm.rank(10), 1u);
    EXPECT_EQ(bm.rank(15), 1u);
    EXPECT_EQ(bm.rank(20), 2u);
    EXPECT_EQ(bm.rank(30), 3u);
    EXPECT_EQ(bm.rank(40), 4u);
    EXPECT_EQ(bm.rank(50), 4u);
}


TEST_F(RoaringBitmapTest, RankBitmapContainerSIMD) {
    // Fill enough to force bitmap container, then test rank at various positions
    // Exercises SIMD bitmap_popcount_n for prefix popcount
    for (uint32_t i = 0; i < 8000; ++i) bm.add(i * 2);  // evens 0..15998

    EXPECT_EQ(bm.rank(0), 1u);    // 0 is present
    EXPECT_EQ(bm.rank(1), 1u);    // 1 not present, 0 is below
    EXPECT_EQ(bm.rank(100), 51u); // evens 0..100 = 51 values
    EXPECT_EQ(bm.rank(999), 500u); // evens 0..998 = 500 values
    EXPECT_EQ(bm.rank(15998), 8000u); // all 8000 values <= 15998
}


TEST_F(RoaringBitmapTest, RankRunContainerOptimized) {
    // Use addRange to create RunContainer-friendly data
    bm.addRange(100, 200);
    bm.optimize();  // convert to run if beneficial

    EXPECT_EQ(bm.rank(99), 0u);
    EXPECT_EQ(bm.rank(100), 1u);
    EXPECT_EQ(bm.rank(150), 51u);
    EXPECT_EQ(bm.rank(199), 100u);
    EXPECT_EQ(bm.rank(200), 100u);
}


TEST_F(RoaringBitmapTest, RankMultiChunk) {
    bm.addRange(0, 100);
    bm.addRange(65536, 65636);

    EXPECT_EQ(bm.rank(50), 51u);
    EXPECT_EQ(bm.rank(65535), 100u);
    EXPECT_EQ(bm.rank(65536), 101u);
    EXPECT_EQ(bm.rank(65600), 165u);
}


TEST_F(RoaringBitmapTest, RankLargeBitmapEdge) {
    // Test rank near word boundaries (multiples of 64) in bitmap
    for (uint32_t i = 0; i < 8000; ++i) bm.add(i);
    EXPECT_EQ(bm.rank(63), 64u);
    EXPECT_EQ(bm.rank(64), 65u);
    EXPECT_EQ(bm.rank(127), 128u);
    EXPECT_EQ(bm.rank(7999), 8000u);
}
} // namespace arrow
