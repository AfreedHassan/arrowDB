#include "roaring_test_fixture.h"

namespace arrow {

// ─────────────────────────────────────────────────────────────────────────────
// AND — intersection (CRoaring: and_or_test, array_bitset_and_or tests)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, AndOverlapping) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(50, 150);
    auto result = a & b;
    EXPECT_EQ(result.cardinality(), 50u);
    EXPECT_TRUE(result.contains(50));
    EXPECT_TRUE(result.contains(99));
    EXPECT_FALSE(result.contains(49));
    EXPECT_FALSE(result.contains(100));
}


TEST_F(RoaringBitmapTest, AndDisjoint) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(200, 300);
    auto result = a & b;
    EXPECT_TRUE(result.empty());
}


TEST_F(RoaringBitmapTest, AndIdentical) {
    RoaringBitmap a;
    a.addRange(0, 100);
    auto result = a & a;
    EXPECT_EQ(result, a);
}


// CRoaring: and_or_test with coprime step patterns.
TEST_F(RoaringBitmapTest, AndCoprimePatterns) {
    RoaringBitmap a, b;
    // a: every 3rd value, b: every 62nd value (coprime with 3)
    for (uint32_t x = 0; x < 60000; x += 3) a.add(x);
    for (uint32_t x = 0; x < 60000; x += 62) b.add(x);

    auto result = a & b;
    // Intersection: multiples of lcm(3,62)=186
    uint32_t expected = 0;
    for (uint32_t x = 0; x < 60000; x += 186) ++expected;
    EXPECT_EQ(result.cardinality(), expected);
}


// ─────────────────────────────────────────────────────────────────────────────
// OR — union (CRoaring: and_or_test)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, OrOverlapping) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(50, 150);
    auto result = a | b;
    EXPECT_EQ(result.cardinality(), 150u);
    EXPECT_TRUE(result.contains(0));
    EXPECT_TRUE(result.contains(149));
}


TEST_F(RoaringBitmapTest, OrDisjoint) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(200, 300);
    auto result = a | b;
    EXPECT_EQ(result.cardinality(), 200u);
}


TEST_F(RoaringBitmapTest, OrIdentical) {
    RoaringBitmap a;
    a.addRange(0, 100);
    auto result = a | a;
    EXPECT_EQ(result, a);
}


TEST_F(RoaringBitmapTest, OrCoprimePatterns) {
    RoaringBitmap a, b;
    for (uint32_t x = 0; x < 60000; x += 3) a.add(x);
    for (uint32_t x = 0; x < 60000; x += 62) b.add(x);

    auto result = a | b;
    // Union: all multiples of 3 + all multiples of 62 - multiples of 186.
    std::set<uint32_t> expected;
    for (uint32_t x = 0; x < 60000; x += 3) expected.insert(x);
    for (uint32_t x = 0; x < 60000; x += 62) expected.insert(x);
    EXPECT_EQ(result.cardinality(), static_cast<uint32_t>(expected.size()));
}


// ─────────────────────────────────────────────────────────────────────────────
// ANDNOT — set difference (CRoaring: andnot_test, bitset_bitset_container_andnot)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, AndNotOverlapping) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(50, 150);
    auto result = a.andNot(b);
    EXPECT_EQ(result.cardinality(), 50u);
    EXPECT_TRUE(result.contains(0));
    EXPECT_TRUE(result.contains(49));
    EXPECT_FALSE(result.contains(50));
}


TEST_F(RoaringBitmapTest, AndNotDisjoint) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(200, 300);
    auto result = a.andNot(b);
    EXPECT_EQ(result, a);
}


TEST_F(RoaringBitmapTest, AndNotIdentical) {
    RoaringBitmap a;
    a.addRange(0, 100);
    auto result = a.andNot(a);
    EXPECT_TRUE(result.empty());
}


TEST_F(RoaringBitmapTest, AndNotCoprimePatterns) {
    RoaringBitmap a, b;
    for (uint32_t x = 0; x < 60000; x += 3) a.add(x);
    for (uint32_t x = 0; x < 60000; x += 62) b.add(x);

    auto result = a.andNot(b);
    // a minus b: multiples of 3 that are NOT multiples of 62.
    uint32_t expected = 0;
    for (uint32_t x = 0; x < 60000; x += 3) {
        if (x % 62 != 0) ++expected;
    }
    EXPECT_EQ(result.cardinality(), expected);
}


// ─────────────────────────────────────────────────────────────────────────────
// XOR — symmetric difference (CRoaring: xor_test)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, XorOverlapping) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(50, 150);
    auto result = a ^ b;
    EXPECT_EQ(result.cardinality(), 100u);  // [0,50) + [100,150)
    EXPECT_TRUE(result.contains(0));
    EXPECT_TRUE(result.contains(49));
    EXPECT_FALSE(result.contains(50));
    EXPECT_FALSE(result.contains(99));
    EXPECT_TRUE(result.contains(100));
    EXPECT_TRUE(result.contains(149));
}


TEST_F(RoaringBitmapTest, XorIdentical) {
    RoaringBitmap a;
    a.addRange(0, 100);
    auto result = a ^ a;
    EXPECT_TRUE(result.empty());
}


TEST_F(RoaringBitmapTest, XorDisjoint) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(200, 300);
    auto result = a ^ b;
    EXPECT_EQ(result.cardinality(), 200u);
}


TEST_F(RoaringBitmapTest, XorCoprimePatterns) {
    RoaringBitmap a, b;
    for (uint32_t x = 0; x < 60000; x += 3) a.add(x);
    for (uint32_t x = 0; x < 60000; x += 62) b.add(x);

    auto result = a ^ b;
    std::set<uint32_t> sA, sB;
    for (uint32_t x = 0; x < 60000; x += 3) sA.insert(x);
    for (uint32_t x = 0; x < 60000; x += 62) sB.insert(x);
    std::set<uint32_t> expected;
    std::set_symmetric_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                  std::inserter(expected, expected.begin()));
    EXPECT_EQ(result.cardinality(), static_cast<uint32_t>(expected.size()));
}


// ─────────────────────────────────────────────────────────────────────────────
// Cross-container operations (CRoaring: mixed_container_unit.c)
// Array × Bitmap, Array × Run, Bitmap × Run
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, CrossContainerAndArrayBitmap) {
    RoaringBitmap a, b;
    a.addRange(0, 100);       // array (100 elements)
    b.addRange(0, 5000);      // bitmap (>4096)
    auto result = a & b;
    EXPECT_EQ(result.cardinality(), 100u);
}


TEST_F(RoaringBitmapTest, CrossContainerOrArrayBitmap) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(0, 5000);
    auto result = a | b;
    EXPECT_EQ(result.cardinality(), 5000u);
}


TEST_F(RoaringBitmapTest, CrossContainerAndNotBitmapArray) {
    RoaringBitmap a, b;
    a.addRange(0, 5000);
    b.addRange(0, 100);
    auto result = a.andNot(b);
    EXPECT_EQ(result.cardinality(), 4900u);
    EXPECT_FALSE(result.contains(0));
    EXPECT_TRUE(result.contains(100));
}


TEST_F(RoaringBitmapTest, CrossContainerXorArrayBitmap) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(50, 5000);
    auto result = a ^ b;
    // [0,50) + [100,5000)
    EXPECT_EQ(result.cardinality(), 50u + 4900u);
}


// Array × Run (after optimize)
TEST_F(RoaringBitmapTest, CrossContainerAndArrayRun) {
    RoaringBitmap a, b;
    a.addRange(0, 100);       // array
    b.addRange(50, 200);
    b.optimize();             // convert to run container
    auto result = a & b;
    EXPECT_EQ(result.cardinality(), 50u);
}


TEST_F(RoaringBitmapTest, CrossContainerOrArrayRun) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(50, 200);
    b.optimize();
    auto result = a | b;
    EXPECT_EQ(result.cardinality(), 200u);
}


// Bitmap × Run
TEST_F(RoaringBitmapTest, CrossContainerAndBitmapRun) {
    RoaringBitmap a, b;
    a.addRange(0, 5000);      // bitmap
    b.addRange(100, 200);
    b.optimize();             // run
    auto result = a & b;
    EXPECT_EQ(result.cardinality(), 100u);
}


TEST_F(RoaringBitmapTest, CrossContainerOrBitmapRun) {
    RoaringBitmap a, b;
    a.addRange(0, 5000);
    b.addRange(4900, 6000);
    b.optimize();
    auto result = a | b;
    EXPECT_EQ(result.cardinality(), 6000u);
}


// Run × Run
TEST_F(RoaringBitmapTest, CrossContainerAndRunRun) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    a.addRange(200, 300);
    a.optimize();
    b.addRange(50, 250);
    b.optimize();
    auto result = a & b;
    EXPECT_EQ(result.cardinality(), 100u);  // [50,100) + [200,250)
}


TEST_F(RoaringBitmapTest, CrossContainerOrRunRun) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    a.optimize();
    b.addRange(50, 150);
    b.optimize();
    auto result = a | b;
    EXPECT_EQ(result.cardinality(), 150u);
}


// ─────────────────────────────────────────────────────────────────────────────
// In-place operations (CRoaring: ixor_test, iandnot_test)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, InPlaceAnd) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(50, 150);
    a &= b;
    EXPECT_EQ(a.cardinality(), 50u);
}


TEST_F(RoaringBitmapTest, InPlaceOr) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(50, 150);
    a |= b;
    EXPECT_EQ(a.cardinality(), 150u);
}


TEST_F(RoaringBitmapTest, InPlaceAndNot) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(50, 150);
    a -= b;
    EXPECT_EQ(a.cardinality(), 50u);
    EXPECT_TRUE(a.contains(0));
    EXPECT_FALSE(a.contains(50));
}


TEST_F(RoaringBitmapTest, InPlaceXor) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(50, 150);
    a ^= b;
    EXPECT_EQ(a.cardinality(), 100u);
}


// ─────────────────────────────────────────────────────────────────────────────
// Galloping intersection (CRoaring: skewed size ratio > 64:1)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, GallopingIntersection) {
    RoaringBitmap small, large;
    // small: 10 elements, large: 10000 elements.
    small.add(50);
    small.add(500);
    small.add(5000);
    small.add(9999);

    for (uint32_t i = 0; i < 10000; ++i) large.add(i);

    auto result = small & large;
    EXPECT_EQ(result.cardinality(), 4u);
    EXPECT_TRUE(result.contains(50));
    EXPECT_TRUE(result.contains(500));
    EXPECT_TRUE(result.contains(5000));
    EXPECT_TRUE(result.contains(9999));
}


// ─────────────────────────────────────────────────────────────────────────────
// Multi-chunk set operations (CRoaring: multi-container set operations)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, MultiChunkSetOperations) {
    RoaringBitmap a, b;
    // a: chunks 0, 1, 2
    a.addRange(0, 100);
    a.addRange(65536, 65636);
    a.addRange(131072, 131172);

    // b: chunks 1, 2, 3
    b.addRange(65536, 65636);
    b.addRange(131072, 131172);
    b.addRange(196608, 196708);

    // AND: only chunks 1 and 2.
    auto andResult = a & b;
    EXPECT_EQ(andResult.cardinality(), 200u);
    EXPECT_FALSE(andResult.contains(0));
    EXPECT_TRUE(andResult.contains(65536));

    // OR: chunks 0, 1, 2, 3.
    auto orResult = a | b;
    EXPECT_EQ(orResult.cardinality(), 400u);

    // ANDNOT: only chunk 0.
    auto diffResult = a.andNot(b);
    EXPECT_EQ(diffResult.cardinality(), 100u);
    EXPECT_TRUE(diffResult.contains(0));
    EXPECT_FALSE(diffResult.contains(65536));
}


// ─────────────────────────────────────────────────────────────────────────────
// fastunion tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, FastunionEmpty) {
    std::vector<const RoaringBitmap*> bitmaps;
    auto result = RoaringBitmap::fastunion(bitmaps);
    EXPECT_TRUE(result.empty());
}


TEST_F(RoaringBitmapTest, FastunionSingle) {
    RoaringBitmap a;
    a.addRange(0, 1000);
    std::vector<const RoaringBitmap*> bitmaps = {&a};
    auto result = RoaringBitmap::fastunion(bitmaps);
    EXPECT_EQ(result, a);
}


TEST_F(RoaringBitmapTest, FastunionThreeBitmaps) {
    RoaringBitmap a, b, c;
    a.addRange(0, 5000);
    b.addRange(3000, 8000);
    c.addRange(6000, 10000);

    std::vector<const RoaringBitmap*> bitmaps = {&a, &b, &c};
    auto fast = RoaringBitmap::fastunion(bitmaps);
    auto expected = (a | b) | c;
    EXPECT_EQ(fast, expected);
    EXPECT_EQ(fast.cardinality(), 10000u);
}


TEST_F(RoaringBitmapTest, FastunionManyBitmaps) {
    std::vector<RoaringBitmap> bms(12);
    for (int i = 0; i < 12; ++i) {
        bms[i].addRange(i * 1000, (i + 1) * 1000 + 500);
    }
    std::vector<const RoaringBitmap*> ptrs;
    for (auto& b : bms) ptrs.push_back(&b);

    auto fast = RoaringBitmap::fastunion(ptrs);

    // Compute expected via pairwise OR.
    RoaringBitmap expected = bms[0];
    for (size_t i = 1; i < bms.size(); ++i) expected |= bms[i];

    EXPECT_EQ(fast, expected);
}


// ─────────────────────────────────────────────────────────────────────────────
// lazyOr + repairCardinality tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, LazyOrMatchesRegularOr) {
    RoaringBitmap a, b;
    a.addRange(0, 10000);   // bitmap container
    b.addRange(5000, 15000);

    auto lazy = a.lazyOr(b);
    lazy.repairCardinality();
    auto regular = a | b;
    EXPECT_EQ(lazy, regular);
}


TEST_F(RoaringBitmapTest, LazyOrDeferredCardinality) {
    // Create two bitmaps that will use bitmap containers.
    RoaringBitmap a, b;
    a.addRange(0, 10000);
    b.addRange(5000, 15000);

    auto lazy = a.lazyOr(b);
    // The result should still work after repair.
    lazy.repairCardinality();
    EXPECT_EQ(lazy.cardinality(), 15000u);
}


// ─────────────────────────────────────────────────────────────────────────────
// andCardinality tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, AndCardinalityMatchesMaterialized) {
    RoaringBitmap a, b;
    a.addRange(0, 10000);
    b.addRange(5000, 15000);
    EXPECT_EQ(a.andCardinality(b), (a & b).cardinality());
}


TEST_F(RoaringBitmapTest, AndCardinalityArrayContainers) {
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 100; ++i) a.add(i * 2);
    for (uint32_t i = 0; i < 100; ++i) b.add(i * 3);
    EXPECT_EQ(a.andCardinality(b), (a & b).cardinality());
}


// ─────────────────────────────────────────────────────────────────────────────
// orCardinality tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, OrCardinalityMatchesMaterialized) {
    RoaringBitmap a, b;
    a.addRange(0, 10000);
    b.addRange(5000, 15000);
    EXPECT_EQ(a.orCardinality(b), (a | b).cardinality());
}


// ─────────────────────────────────────────────────────────────────────────────
// xorCardinality tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, XorCardinalityMatchesMaterialized) {
    RoaringBitmap a, b;
    a.addRange(0, 10000);
    b.addRange(5000, 15000);
    EXPECT_EQ(a.xorCardinality(b), (a ^ b).cardinality());
}


// ─────────────────────────────────────────────────────────────────────────────
// andNotCardinality tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, AndNotCardinalityMatchesMaterialized) {
    RoaringBitmap a, b;
    a.addRange(0, 10000);
    b.addRange(5000, 15000);
    EXPECT_EQ(a.andNotCardinality(b), (a - b).cardinality());
}


// ─────────────────────────────────────────────────────────────────────────────
// jaccardIndex tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, JaccardIndexIdentical) {
    RoaringBitmap a;
    a.addRange(0, 1000);
    EXPECT_DOUBLE_EQ(a.jaccardIndex(a), 1.0);
}


TEST_F(RoaringBitmapTest, JaccardIndexDisjoint) {
    RoaringBitmap a, b;
    a.addRange(0, 1000);
    b.addRange(1000, 2000);
    EXPECT_DOUBLE_EQ(a.jaccardIndex(b), 0.0);
}


TEST_F(RoaringBitmapTest, JaccardIndexPartialOverlap) {
    RoaringBitmap a, b;
    a.addRange(0, 1000);
    b.addRange(500, 1500);
    // |A&B| = 500, |A|B| = 1500
    double expected = 500.0 / 1500.0;
    EXPECT_NEAR(a.jaccardIndex(b), expected, 1e-9);
}


TEST_F(RoaringBitmapTest, JaccardIndexBothEmpty) {
    RoaringBitmap a, b;
    EXPECT_DOUBLE_EQ(a.jaccardIndex(b), 0.0);
}


// ─────────────────────────────────────────────────────────────────────────────
// intersects tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, IntersectsTrue) {
    RoaringBitmap a, b;
    a.addRange(0, 1000);
    b.addRange(999, 2000);
    EXPECT_TRUE(a.intersects(b));
}


TEST_F(RoaringBitmapTest, IntersectsFalse) {
    RoaringBitmap a, b;
    a.addRange(0, 1000);
    b.addRange(1000, 2000);
    EXPECT_FALSE(a.intersects(b));
}


TEST_F(RoaringBitmapTest, IntersectsEmptyBitmaps) {
    RoaringBitmap a, b;
    EXPECT_FALSE(a.intersects(b));
}


TEST_F(RoaringBitmapTest, IntersectsBitmapContainers) {
    // Force bitmap containers with >4096 elements.
    RoaringBitmap a, b;
    a.addRange(0, 10000);
    b.addRange(9999, 20000);
    EXPECT_TRUE(a.intersects(b));

    RoaringBitmap c;
    c.addRange(10000, 20000);
    EXPECT_FALSE(a.intersects(c));
}


TEST_F(RoaringBitmapTest, APIGapOperatorEqualNativeComparison) {
    // Two bitmap containers should use bitmap_equal, not vector materialization
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 10000; ++i) { a.add(i); b.add(i); }
    EXPECT_EQ(a, b);
    b.add(10001);
    EXPECT_NE(a, b);
}


TEST_F(RoaringBitmapTest, APIGapIsStrictSubset) {
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 10; ++i) { a.add(i); b.add(i); }
    EXPECT_FALSE(a.isStrictSubset(b));  // equal, not strict
    b.add(100);
    EXPECT_TRUE(a.isStrictSubset(b));   // a ⊂ b
    EXPECT_FALSE(b.isStrictSubset(a));  // b ⊄ a
}


TEST_F(RoaringBitmapTest, APIGapFlippedNonMutating) {
    for (uint32_t i = 0; i < 10; ++i) bm.add(i);
    auto flipped = bm.flipped(0, 20);
    // Original unchanged
    EXPECT_EQ(bm.cardinality(), 10u);
    // Flipped has values [10, 20)
    EXPECT_EQ(flipped.cardinality(), 10u);
    for (uint32_t i = 10; i < 20; ++i) EXPECT_TRUE(flipped.contains(i));
    for (uint32_t i = 0; i < 10; ++i) EXPECT_FALSE(flipped.contains(i));
}


TEST_F(RoaringBitmapTest, InfraGapHeapFastunionBasic) {
    RoaringBitmap a, b, c;
    for (uint32_t i = 0; i < 100; ++i) a.add(i);
    for (uint32_t i = 50; i < 150; ++i) b.add(i);
    for (uint32_t i = 100; i < 200; ++i) c.add(i);

    std::vector<const RoaringBitmap*> bitmaps = {&a, &b, &c};
    auto result = RoaringBitmap::fastunion(bitmaps);

    EXPECT_EQ(result.cardinality(), 200u);
    for (uint32_t i = 0; i < 200; ++i) {
        EXPECT_TRUE(result.contains(i)) << "Missing " << i;
    }
}


TEST_F(RoaringBitmapTest, InfraGapHeapFastunionMultiChunk) {
    RoaringBitmap a, b, c;
    // Different chunks
    for (uint32_t i = 0; i < 100; ++i) a.add(i);
    for (uint32_t i = 0x10000; i < 0x10000 + 100; ++i) b.add(i);
    for (uint32_t i = 0x20000; i < 0x20000 + 100; ++i) c.add(i);

    std::vector<const RoaringBitmap*> bitmaps = {&a, &b, &c};
    auto result = RoaringBitmap::fastunion(bitmaps);

    EXPECT_EQ(result.cardinality(), 300u);
}


TEST_F(RoaringBitmapTest, InfraGapHeapFastunionManyBitmaps) {
    // Test with many bitmaps to exercise the heap
    std::vector<RoaringBitmap> bitmaps(50);
    for (size_t i = 0; i < 50; ++i) {
        for (uint32_t j = 0; j < 100; ++j) {
            bitmaps[i].add(static_cast<uint32_t>(i * 50 + j));
        }
    }

    std::vector<const RoaringBitmap*> ptrs;
    for (auto& bm : bitmaps) ptrs.push_back(&bm);

    auto result = RoaringBitmap::fastunion(ptrs);

    // Verify via naive union
    RoaringBitmap expected;
    for (auto& bm : bitmaps) expected |= bm;
    EXPECT_EQ(result, expected);
}


// ─────────────────────────────────────────────────────────────────────────────
// In-place operations: |=, &=, -=, ^= equivalence
// ─────────────────────────────────────────────────────────────────────────────

// Helper to build bitmaps that produce specific container types in chunk 0.
static RoaringBitmap makeArrayBitmap(uint32_t base, uint32_t count, uint32_t step = 1) {
    RoaringBitmap bm;
    for (uint32_t i = 0; i < count; ++i) bm.add(base + i * step);
    return bm;
}

static RoaringBitmap makeBitmapContainer(uint32_t base) {
    // >4096 values → BitmapContainer
    RoaringBitmap bm;
    for (uint32_t i = 0; i < 5000; ++i) bm.add(base + i * 3);
    return bm;
}

static RoaringBitmap makeRunBitmap(uint32_t base, uint32_t runStart, uint32_t runLen) {
    RoaringBitmap bm;
    bm.addRange(base + runStart, base + runStart + runLen);
    bm.optimize();
    return bm;
}


TEST_F(RoaringBitmapTest, InPlaceOrEquivalence) {
    // Build diverse bitmaps.
    RoaringBitmap a;
    for (uint32_t i = 0; i < 8000; ++i) a.add(i * 7);
    a.addRange(100000, 102000);

    RoaringBitmap b;
    for (uint32_t i = 0; i < 6000; ++i) b.add(i * 11);
    b.addRange(101000, 103000);

    RoaringBitmap expected = a | b;
    RoaringBitmap inplace = a;
    inplace |= b;
    EXPECT_EQ(inplace, expected);
}


TEST_F(RoaringBitmapTest, InPlaceAndEquivalence) {
    RoaringBitmap a;
    for (uint32_t i = 0; i < 8000; ++i) a.add(i * 7);
    a.addRange(100000, 102000);

    RoaringBitmap b;
    for (uint32_t i = 0; i < 6000; ++i) b.add(i * 11);
    b.addRange(101000, 103000);

    RoaringBitmap expected = a & b;
    RoaringBitmap inplace = a;
    inplace &= b;
    EXPECT_EQ(inplace, expected);
}


TEST_F(RoaringBitmapTest, InPlaceAndNotEquivalence) {
    RoaringBitmap a;
    for (uint32_t i = 0; i < 8000; ++i) a.add(i * 7);
    a.addRange(100000, 102000);

    RoaringBitmap b;
    for (uint32_t i = 0; i < 6000; ++i) b.add(i * 11);
    b.addRange(101000, 103000);

    RoaringBitmap expected = a - b;
    RoaringBitmap inplace = a;
    inplace -= b;
    EXPECT_EQ(inplace, expected);
}


TEST_F(RoaringBitmapTest, InPlaceXorEquivalence) {
    RoaringBitmap a;
    for (uint32_t i = 0; i < 8000; ++i) a.add(i * 7);
    a.addRange(100000, 102000);

    RoaringBitmap b;
    for (uint32_t i = 0; i < 6000; ++i) b.add(i * 11);
    b.addRange(101000, 103000);

    RoaringBitmap expected = a ^ b;
    RoaringBitmap inplace = a;
    inplace ^= b;
    EXPECT_EQ(inplace, expected);
}


TEST_F(RoaringBitmapTest, InPlaceSelfOps) {
    RoaringBitmap a;
    for (uint32_t i = 0; i < 5000; ++i) a.add(i);
    RoaringBitmap orig = a;

    // a |= a → no change
    a |= a;
    EXPECT_EQ(a, orig);

    // a &= a → no change
    a &= a;
    EXPECT_EQ(a, orig);

    // a -= a → empty
    RoaringBitmap c = orig;
    c -= c;
    EXPECT_TRUE(c.empty());

    // a ^= a → empty
    RoaringBitmap d = orig;
    d ^= d;
    EXPECT_TRUE(d.empty());
}


// ─────────────────────────────────────────────────────────────────────────────
// Mixed container type in-place operations
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, InPlaceOrMixedContainers) {
    // Array × Array
    auto a1 = makeArrayBitmap(0, 100);
    auto b1 = makeArrayBitmap(50, 100);
    auto exp1 = a1 | b1;
    a1 |= b1;
    EXPECT_EQ(a1, exp1);

    // Bitmap × Bitmap
    auto a2 = makeBitmapContainer(0);
    auto b2 = makeBitmapContainer(1);
    auto exp2 = a2 | b2;
    a2 |= b2;
    EXPECT_EQ(a2, exp2);

    // Bitmap × Array
    auto a3 = makeBitmapContainer(0);
    auto b3 = makeArrayBitmap(0, 100);
    auto exp3 = a3 | b3;
    a3 |= b3;
    EXPECT_EQ(a3, exp3);

    // Bitmap × Run
    auto a4 = makeBitmapContainer(0);
    auto b4 = makeRunBitmap(0, 100, 2000);
    auto exp4 = a4 | b4;
    a4 |= b4;
    EXPECT_EQ(a4, exp4);

    // Array × Bitmap
    auto a5 = makeArrayBitmap(0, 100);
    auto b5 = makeBitmapContainer(0);
    auto exp5 = a5 | b5;
    a5 |= b5;
    EXPECT_EQ(a5, exp5);

    // Array × Run
    auto a6 = makeArrayBitmap(0, 100);
    auto b6 = makeRunBitmap(0, 50, 200);
    auto exp6 = a6 | b6;
    a6 |= b6;
    EXPECT_EQ(a6, exp6);

    // Run × Bitmap
    auto a7 = makeRunBitmap(0, 0, 5000);
    auto b7 = makeBitmapContainer(0);
    auto exp7 = a7 | b7;
    a7 |= b7;
    EXPECT_EQ(a7, exp7);

    // Run × Array
    auto a8 = makeRunBitmap(0, 0, 5000);
    auto b8 = makeArrayBitmap(0, 100, 7);
    auto exp8 = a8 | b8;
    a8 |= b8;
    EXPECT_EQ(a8, exp8);

    // Run × Run
    auto a9 = makeRunBitmap(0, 0, 3000);
    auto b9 = makeRunBitmap(0, 2000, 3000);
    auto exp9 = a9 | b9;
    a9 |= b9;
    EXPECT_EQ(a9, exp9);
}


TEST_F(RoaringBitmapTest, InPlaceAndMixedContainers) {
    auto a1 = makeArrayBitmap(0, 200);   auto b1 = makeArrayBitmap(100, 200);
    auto exp1 = a1 & b1; a1 &= b1; EXPECT_EQ(a1, exp1);

    auto a2 = makeBitmapContainer(0);    auto b2 = makeBitmapContainer(1);
    auto exp2 = a2 & b2; a2 &= b2; EXPECT_EQ(a2, exp2);

    auto a3 = makeBitmapContainer(0);    auto b3 = makeArrayBitmap(0, 100);
    auto exp3 = a3 & b3; a3 &= b3; EXPECT_EQ(a3, exp3);

    auto a4 = makeArrayBitmap(0, 200);   auto b4 = makeBitmapContainer(0);
    auto exp4 = a4 & b4; a4 &= b4; EXPECT_EQ(a4, exp4);

    auto a5 = makeArrayBitmap(0, 200);   auto b5 = makeRunBitmap(0, 50, 100);
    auto exp5 = a5 & b5; a5 &= b5; EXPECT_EQ(a5, exp5);

    auto a6 = makeRunBitmap(0, 0, 5000); auto b6 = makeBitmapContainer(0);
    auto exp6 = a6 & b6; a6 &= b6; EXPECT_EQ(a6, exp6);

    auto a7 = makeRunBitmap(0, 0, 5000); auto b7 = makeRunBitmap(0, 2000, 3000);
    auto exp7 = a7 & b7; a7 &= b7; EXPECT_EQ(a7, exp7);
}


TEST_F(RoaringBitmapTest, InPlaceAndNotMixedContainers) {
    auto a1 = makeArrayBitmap(0, 200);   auto b1 = makeArrayBitmap(100, 200);
    auto exp1 = a1 - b1; a1 -= b1; EXPECT_EQ(a1, exp1);

    auto a2 = makeBitmapContainer(0);    auto b2 = makeBitmapContainer(1);
    auto exp2 = a2 - b2; a2 -= b2; EXPECT_EQ(a2, exp2);

    auto a3 = makeBitmapContainer(0);    auto b3 = makeArrayBitmap(0, 100);
    auto exp3 = a3 - b3; a3 -= b3; EXPECT_EQ(a3, exp3);

    auto a4 = makeBitmapContainer(0);    auto b4 = makeRunBitmap(0, 100, 2000);
    auto exp4 = a4 - b4; a4 -= b4; EXPECT_EQ(a4, exp4);

    auto a5 = makeArrayBitmap(0, 200);   auto b5 = makeBitmapContainer(0);
    auto exp5 = a5 - b5; a5 -= b5; EXPECT_EQ(a5, exp5);

    auto a6 = makeArrayBitmap(0, 200);   auto b6 = makeRunBitmap(0, 50, 100);
    auto exp6 = a6 - b6; a6 -= b6; EXPECT_EQ(a6, exp6);

    auto a7 = makeRunBitmap(0, 0, 5000); auto b7 = makeBitmapContainer(0);
    auto exp7 = a7 - b7; a7 -= b7; EXPECT_EQ(a7, exp7);

    auto a8 = makeRunBitmap(0, 0, 5000); auto b8 = makeRunBitmap(0, 2000, 3000);
    auto exp8 = a8 - b8; a8 -= b8; EXPECT_EQ(a8, exp8);
}


TEST_F(RoaringBitmapTest, InPlaceXorMixedContainers) {
    auto a1 = makeArrayBitmap(0, 200);   auto b1 = makeArrayBitmap(100, 200);
    auto exp1 = a1 ^ b1; a1 ^= b1; EXPECT_EQ(a1, exp1);

    auto a2 = makeBitmapContainer(0);    auto b2 = makeBitmapContainer(1);
    auto exp2 = a2 ^ b2; a2 ^= b2; EXPECT_EQ(a2, exp2);

    auto a3 = makeBitmapContainer(0);    auto b3 = makeArrayBitmap(0, 100);
    auto exp3 = a3 ^ b3; a3 ^= b3; EXPECT_EQ(a3, exp3);

    auto a4 = makeArrayBitmap(0, 200);   auto b4 = makeBitmapContainer(0);
    auto exp4 = a4 ^ b4; a4 ^= b4; EXPECT_EQ(a4, exp4);

    auto a5 = makeRunBitmap(0, 0, 5000); auto b5 = makeBitmapContainer(0);
    auto exp5 = a5 ^ b5; a5 ^= b5; EXPECT_EQ(a5, exp5);

    auto a6 = makeRunBitmap(0, 0, 5000); auto b6 = makeArrayBitmap(0, 100, 7);
    auto exp6 = a6 ^ b6; a6 ^= b6; EXPECT_EQ(a6, exp6);

    auto a7 = makeRunBitmap(0, 0, 3000); auto b7 = makeRunBitmap(0, 2000, 3000);
    auto exp7 = a7 ^ b7; a7 ^= b7; EXPECT_EQ(a7, exp7);
}


// ─────────────────────────────────────────────────────────────────────────────
// Run×Run ANDNOT edge cases
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RunRunAndNotDisjoint) {
    // a = [0,100), b = [200,300) → result = a unchanged
    auto a = makeRunBitmap(0, 0, 100);
    auto b = makeRunBitmap(0, 200, 100);
    auto result = a - b;
    EXPECT_EQ(result, a);
}


TEST_F(RoaringBitmapTest, RunRunAndNotSubset) {
    // b fully covers a → empty
    auto a = makeRunBitmap(0, 100, 200);
    auto b = makeRunBitmap(0, 0, 500);
    auto result = a - b;
    EXPECT_TRUE(result.empty());
}


TEST_F(RoaringBitmapTest, RunRunAndNotPartialOverlap) {
    // a = [100,300), b = [200,400) → result = [100,200)
    auto a = makeRunBitmap(0, 100, 200);
    auto b = makeRunBitmap(0, 200, 200);
    auto result = a - b;
    RoaringBitmap expected;
    expected.addRange(100, 200);
    EXPECT_EQ(result, expected);
}


TEST_F(RoaringBitmapTest, RunRunAndNotSplit) {
    // a = [0,1000), b = [300,500) → result = [0,300) ∪ [500,1000)
    auto a = makeRunBitmap(0, 0, 1000);
    auto b = makeRunBitmap(0, 300, 200);
    auto result = a - b;
    RoaringBitmap expected;
    expected.addRange(0, 300);
    expected.addRange(500, 1000);
    EXPECT_EQ(result, expected);
}


TEST_F(RoaringBitmapTest, RunRunAndNotInterleaved) {
    // a = multiple runs, b = multiple runs that partially overlap
    RoaringBitmap a;
    a.addRange(0, 100);
    a.addRange(200, 300);
    a.addRange(400, 500);
    a.optimize();

    RoaringBitmap b;
    b.addRange(50, 150);
    b.addRange(250, 350);
    b.addRange(450, 550);
    b.optimize();

    auto result = a - b;
    RoaringBitmap expected;
    expected.addRange(0, 50);
    expected.addRange(200, 250);
    expected.addRange(400, 450);
    EXPECT_EQ(result, expected);
}


TEST_F(RoaringBitmapTest, RunRunAndNotMultipleSplits) {
    // a = one big run, b = multiple small holes
    RoaringBitmap a;
    a.addRange(0, 10000);
    a.optimize();

    RoaringBitmap b;
    b.addRange(1000, 2000);
    b.addRange(3000, 4000);
    b.addRange(5000, 6000);
    b.optimize();

    auto result = a - b;
    RoaringBitmap expected;
    expected.addRange(0, 1000);
    expected.addRange(2000, 3000);
    expected.addRange(4000, 5000);
    expected.addRange(6000, 10000);
    EXPECT_EQ(result, expected);
}


// ─────────────────────────────────────────────────────────────────────────────
// Run XOR operations
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RunBitmapXor) {
    RoaringBitmap a;
    a.addRange(0, 5000);
    a.optimize();  // Run container

    RoaringBitmap b;
    for (uint32_t i = 0; i < 5000; ++i) b.add(i * 3);  // Bitmap container

    auto result = a ^ b;
    // Verify against element-by-element check.
    for (uint32_t i = 0; i < 15000; ++i) {
        bool inA = a.contains(i);
        bool inB = b.contains(i);
        EXPECT_EQ(result.contains(i), inA != inB) << "i=" << i;
    }
}


TEST_F(RoaringBitmapTest, RunArrayXor) {
    RoaringBitmap a;
    a.addRange(0, 500);
    a.optimize();  // Run container

    RoaringBitmap b;
    for (uint32_t i = 0; i < 100; ++i) b.add(i * 10);  // Array container

    auto result = a ^ b;
    for (uint32_t i = 0; i < 1000; ++i) {
        bool inA = a.contains(i);
        bool inB = b.contains(i);
        EXPECT_EQ(result.contains(i), inA != inB) << "i=" << i;
    }
}


TEST_F(RoaringBitmapTest, RunRunXor) {
    RoaringBitmap a;
    a.addRange(0, 5000);
    a.optimize();

    RoaringBitmap b;
    b.addRange(3000, 8000);
    b.optimize();

    auto result = a ^ b;
    // XOR of [0,5000) and [3000,8000) = [0,3000) ∪ [5000,8000)
    RoaringBitmap expected;
    expected.addRange(0, 3000);
    expected.addRange(5000, 8000);
    EXPECT_EQ(result, expected);
}


// ─────────────────────────────────────────────────────────────────────────────
// Empty / single-element edge cases for in-place ops
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, InPlaceOpsWithEmpty) {
    RoaringBitmap a;
    for (uint32_t i = 0; i < 100; ++i) a.add(i);
    RoaringBitmap empty;

    // a |= empty → no change
    RoaringBitmap t1 = a;
    t1 |= empty;
    EXPECT_EQ(t1, a);

    // empty |= a → equals a
    RoaringBitmap t2;
    t2 |= a;
    EXPECT_EQ(t2, a);

    // a &= empty → empty
    RoaringBitmap t3 = a;
    t3 &= empty;
    EXPECT_TRUE(t3.empty());

    // a -= empty → no change
    RoaringBitmap t4 = a;
    t4 -= empty;
    EXPECT_EQ(t4, a);

    // empty -= a → empty
    RoaringBitmap t5;
    t5 -= a;
    EXPECT_TRUE(t5.empty());

    // a ^= empty → no change
    RoaringBitmap t6 = a;
    t6 ^= empty;
    EXPECT_EQ(t6, a);

    // empty ^= a → equals a
    RoaringBitmap t7;
    t7 ^= a;
    EXPECT_EQ(t7, a);
}


TEST_F(RoaringBitmapTest, InPlaceOpsMultiChunk) {
    // Test with multiple chunks to ensure two-pointer merge works.
    RoaringBitmap a;
    a.addRange(0, 100);           // chunk 0
    a.addRange(65536, 65636);     // chunk 1
    a.addRange(131072, 131172);   // chunk 2

    RoaringBitmap b;
    b.addRange(50, 150);          // chunk 0
    b.addRange(65586, 65700);     // chunk 1
    b.addRange(196608, 196700);   // chunk 3 (only in b)

    auto expOr = a | b;
    auto expAnd = a & b;
    auto expSub = a - b;
    auto expXor = a ^ b;

    RoaringBitmap t1 = a; t1 |= b; EXPECT_EQ(t1, expOr);
    RoaringBitmap t2 = a; t2 &= b; EXPECT_EQ(t2, expAnd);
    RoaringBitmap t3 = a; t3 -= b; EXPECT_EQ(t3, expSub);
    RoaringBitmap t4 = a; t4 ^= b; EXPECT_EQ(t4, expXor);
}


// ═════════════════════════════════════════════════════════════════════════════
// Gap 4: Branchless parallel binary search (galloping intersection)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, GallopingIntersectionHighRatio) {
    // Create two arrays in the same chunk with >64:1 size ratio to trigger galloping
    RoaringBitmap large, small;
    // Large: 5000 elements (will be a bitmap after 4096)
    for (uint32_t i = 0; i < 5000; ++i)
        large.add(i * 2);  // evens 0..9998
    // Small: 10 elements scattered in the range
    std::vector<uint32_t> smallVals = {4, 100, 500, 1000, 2000, 4000, 6000, 8000, 9000, 9998};
    for (auto v : smallVals) small.add(v);

    auto result = large & small;
    // All smallVals are even, so all should be in large
    EXPECT_EQ(result.cardinality(), static_cast<uint32_t>(smallVals.size()));
    for (auto v : smallVals) EXPECT_TRUE(result.contains(v));
}


TEST_F(RoaringBitmapTest, GallopingIntersectionNoOverlap) {
    // Galloping path with no overlap
    RoaringBitmap large, small;
    for (uint32_t i = 0; i < 4000; ++i)
        large.add(i * 2);  // evens
    for (uint32_t i = 0; i < 10; ++i)
        small.add(i * 2 + 1);  // odds

    auto result = large & small;
    EXPECT_EQ(result.cardinality(), 0u);
}


TEST_F(RoaringBitmapTest, GallopingIntersectionPartialOverlap) {
    // Some elements match, some don't
    RoaringBitmap large, small;
    for (uint32_t i = 0; i < 4000; ++i)
        large.add(i * 3);  // multiples of 3
    small.add(0); small.add(3); small.add(5); small.add(6);
    small.add(7); small.add(9); small.add(10); small.add(12);

    auto result = large & small;
    // Expected: 0, 3, 6, 9, 12 are multiples of 3
    EXPECT_EQ(result.cardinality(), 5u);
    EXPECT_TRUE(result.contains(0));
    EXPECT_TRUE(result.contains(3));
    EXPECT_TRUE(result.contains(6));
    EXPECT_TRUE(result.contains(9));
    EXPECT_TRUE(result.contains(12));
}


// ═════════════════════════════════════════════════════════════════════════════
// Gap 7: Fused Harley-Seal CSA — bitmap op+popcount correctness
// (tested indirectly through bitmap-bitmap AND/OR/XOR/ANDNOT operations)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, BitmapBitmapAndPopcount) {
    // Create two dense bitmaps (>4096 elements each) and verify AND cardinality
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 10000; ++i) a.add(i);
    for (uint32_t i = 5000; i < 15000; ++i) b.add(i);

    auto result = a & b;
    EXPECT_EQ(result.cardinality(), 5000u);  // overlap: [5000, 10000)

    auto orResult = a | b;
    EXPECT_EQ(orResult.cardinality(), 15000u);

    auto xorResult = a ^ b;
    EXPECT_EQ(xorResult.cardinality(), 10000u);  // symmetric difference

    auto diffResult = a - b;
    EXPECT_EQ(diffResult.cardinality(), 5000u);  // [0, 5000)
}


TEST_F(RoaringBitmapTest, BitmapBitmapOpsIdentities) {
    // Property: A & A == A, A | A == A, A ^ A == empty, A - A == empty
    RoaringBitmap a;
    for (uint32_t i = 0; i < 8000; ++i) a.add(i * 2);

    EXPECT_EQ(a & a, a);
    EXPECT_EQ(a | a, a);
    EXPECT_EQ((a ^ a).cardinality(), 0u);
    EXPECT_EQ((a - a).cardinality(), 0u);
}


// ═════════════════════════════════════════════════════════════════════════════
// Gap 17: Full-run early exit
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, FullRunOrShortcut) {
    RoaringBitmap full;
    full.addRange(0, 65536);  // fill entire first chunk
    full.optimize();

    RoaringBitmap partial;
    for (uint32_t i = 0; i < 100; ++i) partial.add(i);

    // OR with full run should produce full run's cardinality
    auto result = full | partial;
    EXPECT_EQ(result.cardinality(), 65536u);
}


TEST_F(RoaringBitmapTest, FullRunAndShortcut) {
    RoaringBitmap full;
    full.addRange(0, 65536);
    full.optimize();

    RoaringBitmap partial;
    for (uint32_t i = 0; i < 100; ++i) partial.add(i * 2);

    auto result = full & partial;
    EXPECT_EQ(result.cardinality(), 100u);
}


TEST_F(RoaringBitmapTest, FullRunAndNotShortcut) {
    RoaringBitmap full;
    full.addRange(0, 65536);
    full.optimize();

    RoaringBitmap partial;
    for (uint32_t i = 0; i < 100; ++i) partial.add(i);

    auto result = partial - full;
    EXPECT_EQ(result.cardinality(), 0u);
}


// ═════════════════════════════════════════════════════════════════════════════
// Gap 25: fastunion small N
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, FastUnionTwoBitmaps) {
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 100; ++i) a.add(i);
    for (uint32_t i = 50; i < 150; ++i) b.add(i);

    std::vector<const RoaringBitmap*> bitmaps = {&a, &b};
    auto result = RoaringBitmap::fastunion(bitmaps);
    EXPECT_EQ(result.cardinality(), 150u);
    EXPECT_EQ(result, a | b);
}


TEST_F(RoaringBitmapTest, FastUnionFourBitmaps) {
    RoaringBitmap a, b, c, d;
    for (uint32_t i = 0; i < 100; ++i) a.add(i);
    for (uint32_t i = 100; i < 200; ++i) b.add(i);
    for (uint32_t i = 200; i < 300; ++i) c.add(i);
    for (uint32_t i = 50; i < 250; ++i) d.add(i);

    std::vector<const RoaringBitmap*> bitmaps = {&a, &b, &c, &d};
    auto result = RoaringBitmap::fastunion(bitmaps);

    auto expected = a | b | c | d;
    EXPECT_EQ(result, expected);
}


// ═════════════════════════════════════════════════════════════════════════════
// Commutativity / associativity properties
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, IntersectionCommutativity) {
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 5000; ++i) a.add(i * 2);
    for (uint32_t i = 0; i < 3000; ++i) b.add(i * 3);
    EXPECT_EQ(a & b, b & a);
}


TEST_F(RoaringBitmapTest, UnionCommutativity) {
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 5000; ++i) a.add(i * 2);
    for (uint32_t i = 0; i < 3000; ++i) b.add(i * 3);
    EXPECT_EQ(a | b, b | a);
}


TEST_F(RoaringBitmapTest, XorCommutativity) {
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 5000; ++i) a.add(i * 2);
    for (uint32_t i = 0; i < 3000; ++i) b.add(i * 3);
    EXPECT_EQ(a ^ b, b ^ a);
}
} // namespace arrow
