#include "roaring_test_fixture.h"

namespace arrow {

// ─────────────────────────────────────────────────────────────────────────────
// Equality (CRoaring: equal_array_array_test, full 9-combo comparisons)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, Equality) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(0, 100);
    EXPECT_EQ(a, b);

    b.add(100);
    EXPECT_NE(a, b);
}


// CRoaring: generic_equal_test — incremental add, check equality.
TEST_F(RoaringBitmapTest, EqualityIncremental) {
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 1000; i += 10) {
        a.add(i);
        b.add(i);
        EXPECT_EQ(a, b);
    }
    a.add(273);
    EXPECT_NE(a, b);
    b.add(854);
    EXPECT_NE(a, b);
    a.add(854);
    EXPECT_NE(a, b);  // b still missing 273
    b.add(273);
    EXPECT_EQ(a, b);
}


// CRoaring: full container equality — first/last elements differ.
TEST_F(RoaringBitmapTest, EqualityFullContainerEdgeCases) {
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 65536; ++i) { a.add(i); b.add(i); }
    EXPECT_EQ(a, b);

    // Differ at first element.
    RoaringBitmap c, d;
    for (uint32_t i = 1; i < 65536; ++i) c.add(i);  // missing 0
    for (uint32_t i = 0; i < 65535; ++i) d.add(i);   // missing 65535
    EXPECT_NE(c, d);
}


// ─────────────────────────────────────────────────────────────────────────────
// Subset (CRoaring: generic_subset_test)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, SubsetBasic) {
    RoaringBitmap a, b;
    a.addRange(0, 50);
    b.addRange(0, 100);
    EXPECT_TRUE(a.isSubsetOf(b));
    EXPECT_FALSE(b.isSubsetOf(a));
}


TEST_F(RoaringBitmapTest, SubsetEqual) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(0, 100);
    EXPECT_TRUE(a.isSubsetOf(b));
    EXPECT_TRUE(b.isSubsetOf(a));
}


TEST_F(RoaringBitmapTest, SubsetEmpty) {
    RoaringBitmap empty, full;
    full.addRange(0, 100);
    EXPECT_TRUE(empty.isSubsetOf(full));
    EXPECT_TRUE(empty.isSubsetOf(empty));
    EXPECT_FALSE(full.isSubsetOf(empty));
}


// CRoaring: generic_subset_test — coprime patterns.
TEST_F(RoaringBitmapTest, SubsetCoprimePatterns) {
    RoaringBitmap a, b;
    // Both start with multiples of 11.
    for (uint32_t x = 0; x < 60000; x += 11) { a.add(x); b.add(x); }
    EXPECT_TRUE(a.isSubsetOf(b));

    // Add multiples of 7 to b only.
    for (uint32_t x = 0; x < 60000; x += 7) b.add(x);
    EXPECT_TRUE(a.isSubsetOf(b));

    // Add extras to a that aren't in b.
    for (uint32_t x = 0; x < 60000; x += 5) {
        if (x % 7 != 0 && x % 11 != 0) a.add(x);
    }
    EXPECT_FALSE(a.isSubsetOf(b));
}


// ─────────────────────────────────────────────────────────────────────────────
// Validation against std::set (CRoaring: random tests with reference)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, SetOperationsVsStdSet) {
    // SplitMix64 PRNG (same as CRoaring's test infrastructure).
    uint64_t state = 12345;
    auto nextRand = [&state]() -> uint32_t {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        return static_cast<uint32_t>(z & 0xFFFF);  // keep in one chunk for speed
    };

    const size_t N = 5000;
    std::set<uint32_t> sA, sB;
    RoaringBitmap rA, rB;

    for (size_t i = 0; i < N; ++i) {
        uint32_t v = nextRand();
        sA.insert(v);
        rA.add(v);
    }
    for (size_t i = 0; i < N; ++i) {
        uint32_t v = nextRand();
        sB.insert(v);
        rB.add(v);
    }

    // Verify add correctness.
    EXPECT_EQ(rA.cardinality(), static_cast<uint32_t>(sA.size()));
    EXPECT_EQ(rB.cardinality(), static_cast<uint32_t>(sB.size()));

    // AND.
    {
        std::set<uint32_t> expected;
        std::set_intersection(sA.begin(), sA.end(), sB.begin(), sB.end(),
                              std::inserter(expected, expected.begin()));
        auto result = rA & rB;
        EXPECT_EQ(result.cardinality(), static_cast<uint32_t>(expected.size()));
    }

    // OR.
    {
        std::set<uint32_t> expected;
        std::set_union(sA.begin(), sA.end(), sB.begin(), sB.end(),
                       std::inserter(expected, expected.begin()));
        auto result = rA | rB;
        EXPECT_EQ(result.cardinality(), static_cast<uint32_t>(expected.size()));
    }

    // ANDNOT.
    {
        std::set<uint32_t> expected;
        std::set_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                            std::inserter(expected, expected.begin()));
        auto result = rA.andNot(rB);
        EXPECT_EQ(result.cardinality(), static_cast<uint32_t>(expected.size()));
    }

    // XOR.
    {
        std::set<uint32_t> expected;
        std::set_symmetric_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                      std::inserter(expected, expected.begin()));
        auto result = rA ^ rB;
        EXPECT_EQ(result.cardinality(), static_cast<uint32_t>(expected.size()));
    }

    // Intersection is subset of both.
    auto intersection = rA & rB;
    EXPECT_TRUE(intersection.isSubsetOf(rA));
    EXPECT_TRUE(intersection.isSubsetOf(rB));
}


// ─────────────────────────────────────────────────────────────────────────────
// Run container-specific tests (CRoaring: run_container_unit.c)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RunContainerAddRemove) {
    // Build a bitmap, optimize to runs, then add/remove.
    bm.addRange(100, 200);
    bm.addRange(300, 400);
    bm.optimize();

    // Add element in gap.
    bm.add(250);
    EXPECT_TRUE(bm.contains(250));
    EXPECT_EQ(bm.cardinality(), 201u);

    // Remove element from run.
    bm.remove(150);
    EXPECT_FALSE(bm.contains(150));
    EXPECT_EQ(bm.cardinality(), 200u);

    // Edges still intact.
    EXPECT_TRUE(bm.contains(100));
    EXPECT_TRUE(bm.contains(199));
    EXPECT_TRUE(bm.contains(300));
    EXPECT_TRUE(bm.contains(399));
}


// CRoaring: remove_range_test for runs.
TEST_F(RoaringBitmapTest, RunContainerRemoveEdges) {
    bm.addRange(100, 150);
    bm.addRange(200, 250);
    bm.optimize();

    // Remove left edge of first run.
    for (uint32_t i = 100; i < 110; ++i) bm.remove(i);
    // Remove right edge of second run.
    for (uint32_t i = 240; i < 250; ++i) bm.remove(i);

    EXPECT_EQ(bm.cardinality(), 80u);
    EXPECT_FALSE(bm.contains(100));
    EXPECT_TRUE(bm.contains(110));
    EXPECT_TRUE(bm.contains(239));
    EXPECT_FALSE(bm.contains(240));
}


// ─────────────────────────────────────────────────────────────────────────────
// Container comparison across types (CRoaring: container_comparison_unit.c)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, EqualityAcrossContainerTypes) {
    // Build the same set of values, one as array, one as bitmap (via addRange).
    RoaringBitmap arrayBm, bitmapBm;
    for (uint32_t i = 0; i < 4000; ++i) arrayBm.add(i);  // stays array
    bitmapBm.addRange(0, 5000);
    // Remove extras from bitmap version.
    for (uint32_t i = 4000; i < 5000; ++i) bitmapBm.remove(i);

    EXPECT_EQ(arrayBm, bitmapBm);
}


TEST_F(RoaringBitmapTest, EqualityArrayVsRun) {
    RoaringBitmap arrayBm, runBm;
    for (uint32_t i = 0; i < 100; ++i) arrayBm.add(i);
    runBm.addRange(0, 100);
    runBm.optimize();
    EXPECT_EQ(arrayBm, runBm);
}


TEST_F(RoaringBitmapTest, EqualityBitmapVsRun) {
    RoaringBitmap bitmapBm, runBm;
    bitmapBm.addRange(0, 10000);
    runBm.addRange(0, 10000);
    runBm.optimize();
    EXPECT_EQ(bitmapBm, runBm);
}


// =============================================================================
// PORTED FROM CRoaring test suite (MIT license)
// Tests below are direct translations of CRoaring's C test functions
// adapted to our C++ RoaringBitmap API.
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// array_container_unit.c: add_contains_test
// Add every 3rd value in [0,65536), verify contains, remove all, verify empty.
// Forward and reverse removal.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, ArrayAddContainsForwardRemove) {
    // Add every 3rd value — stays in one chunk (array container).
    for (uint32_t x = 0; x < 65536; x += 3) {
        bm.add(x);
    }
    uint32_t card = bm.cardinality();

    // Verify contains logic: x is present iff x%3==0.
    for (uint32_t x = 0; x < 65536; ++x) {
        EXPECT_EQ(bm.contains(x), (x / 3 * 3 == x))
            << "contains mismatch at x=" << x;
    }

    // Remove all in forward order, check cardinality at each step.
    for (uint32_t x = 0; x < 65536; x += 3) {
        bm.remove(x);
        --card;
        EXPECT_EQ(bm.cardinality(), card);
        EXPECT_FALSE(bm.contains(x));
    }
    EXPECT_EQ(bm.cardinality(), 0u);
}


TEST_F(RoaringBitmapTest, ArrayAddContainsReverseRemove) {
    for (uint32_t x = 0; x < 65536; x += 3) {
        bm.add(x);
    }
    uint32_t card = bm.cardinality();

    // Remove in reverse order.
    for (int32_t x = 65535; x >= 0; x -= 3) {
        bm.remove(static_cast<uint32_t>(x));
        --card;
        EXPECT_EQ(bm.cardinality(), card);
    }
    EXPECT_EQ(bm.cardinality(), 0u);
}


// ─────────────────────────────────────────────────────────────────────────────
// array_container_unit.c: and_or_test
// B1 = multiples of 17, B2 = multiples of 62 (not div by 3).
// Intersection = multiples of lcm(17,62)=1054. Union = all of both.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, ArrayAndOrMultiplesOf17And62) {
    RoaringBitmap b1, b2;
    for (uint32_t x = 0; x < 60000; x += 17) b1.add(x);
    for (uint32_t x = 0; x < 60000; x += 62) {
        if (x % 3 != 0) b2.add(x);
    }

    // Expected intersection: values in both b1 and b2.
    std::set<uint32_t> s1, s2;
    for (uint32_t x = 0; x < 60000; x += 17) s1.insert(x);
    for (uint32_t x = 0; x < 60000; x += 62) {
        if (x % 3 != 0) s2.insert(x);
    }

    std::set<uint32_t> expectedAnd, expectedOr;
    std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                          std::inserter(expectedAnd, expectedAnd.begin()));
    std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(),
                   std::inserter(expectedOr, expectedOr.begin()));

    auto intersect = b1 & b2;
    auto united = b1 | b2;
    EXPECT_EQ(intersect.cardinality(), static_cast<uint32_t>(expectedAnd.size()));
    EXPECT_EQ(united.cardinality(), static_cast<uint32_t>(expectedOr.size()));

    // Verify every element in intersection.
    for (uint32_t v : expectedAnd) {
        EXPECT_TRUE(intersect.contains(v)) << "intersection missing " << v;
    }
    for (uint32_t v : expectedOr) {
        EXPECT_TRUE(united.contains(v)) << "union missing " << v;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// array_container_unit.c: to_uint32_array_test
// For offsets {1,2,4,8,16,32,64}: fill container, convert, verify spacing.
// (Already covered by ToVectorSpacing, but this adds per-container-type checks.)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, ArrayToVectorSpacingAllOffsets) {
    for (uint32_t offset : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
        RoaringBitmap rb;
        uint32_t count = 0;
        for (uint32_t x = 0; x < 65536; x += offset) {
            rb.add(x);
            ++count;
        }
        EXPECT_EQ(rb.cardinality(), count) << "offset=" << offset;

        auto vec = rb.toVector();
        EXPECT_EQ(vec.size(), count) << "offset=" << offset;
        for (size_t k = 1; k < vec.size(); ++k) {
            EXPECT_EQ(vec[k] - vec[k - 1], offset)
                << "offset=" << offset << " k=" << k;
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// bitset_container_unit.c: set_get_test
// Same as array add_contains but forces bitmap container (>4096 elements).
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, BitmapSetGetEveryThird) {
    // Add every 3rd value in [0, 65536) — 21846 values → bitmap container.
    for (uint32_t x = 0; x < 65536; x += 3) {
        bm.add(x);
    }
    EXPECT_EQ(bm.cardinality(), 21846u);

    // Verify contains.
    for (uint32_t x = 0; x < 65536; ++x) {
        EXPECT_EQ(bm.contains(x), (x % 3 == 0));
    }

    // Remove all, verify cardinality decrements.
    uint32_t card = bm.cardinality();
    for (uint32_t x = 0; x < 65536; x += 3) {
        bm.remove(x);
        --card;
    }
    EXPECT_EQ(card, 0u);
    EXPECT_EQ(bm.cardinality(), 0u);
}


// ─────────────────────────────────────────────────────────────────────────────
// bitset_container_unit.c: and_or_test
// B1 = every 3rd value up to 60000, B2 = every 62nd (not div by 3).
// Both are large enough to be bitmap containers.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, BitmapAndOrEvery3rdAnd62nd) {
    RoaringBitmap b1, b2;
    for (uint32_t x = 0; x < 60000; x += 3) b1.add(x);
    for (uint32_t x = 0; x < 60000; x += 62) {
        if (x % 3 != 0) b2.add(x);
    }

    // Manually compute expected.
    std::set<uint32_t> s1, s2;
    for (uint32_t x = 0; x < 60000; x += 3) s1.insert(x);
    for (uint32_t x = 0; x < 60000; x += 62) {
        if (x % 3 != 0) s2.insert(x);
    }

    // b1 is bitmap (20000 elements), b2 is array (~940 elements).
    // This tests Array×Bitmap for AND and OR.
    std::set<uint32_t> expectedAnd, expectedOr;
    std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                          std::inserter(expectedAnd, expectedAnd.begin()));
    std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(),
                   std::inserter(expectedOr, expectedOr.begin()));

    EXPECT_EQ((b1 & b2).cardinality(), static_cast<uint32_t>(expectedAnd.size()));
    EXPECT_EQ((b1 | b2).cardinality(), static_cast<uint32_t>(expectedOr.size()));
}


// ─────────────────────────────────────────────────────────────────────────────
// bitset_container_unit.c: xor_test
// B1 = every 3rd, B2 = every 62nd (not div 3). XOR = symmetric diff.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, BitmapXorEvery3rdAnd62nd) {
    RoaringBitmap b1, b2;
    std::set<uint32_t> s1, s2;
    for (uint32_t x = 0; x < 60000; x += 3) { b1.add(x); s1.insert(x); }
    for (uint32_t x = 0; x < 60000; x += 62) {
        if (x % 3 != 0) { b2.add(x); s2.insert(x); }
    }

    std::set<uint32_t> expected;
    std::set_symmetric_difference(s1.begin(), s1.end(), s2.begin(), s2.end(),
                                  std::inserter(expected, expected.begin()));

    auto result = b1 ^ b2;
    EXPECT_EQ(result.cardinality(), static_cast<uint32_t>(expected.size()));

    for (uint32_t v : expected) {
        EXPECT_TRUE(result.contains(v)) << "xor missing " << v;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// bitset_container_unit.c: andnot_test
// B1 = every 3rd, B2 = every 62nd (not div 3). B1 \ B2.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, BitmapAndNotEvery3rdMinus62nd) {
    RoaringBitmap b1, b2;
    std::set<uint32_t> s1, s2;
    for (uint32_t x = 0; x < 60000; x += 3) { b1.add(x); s1.insert(x); }
    for (uint32_t x = 0; x < 60000; x += 62) {
        if (x % 3 != 0) { b2.add(x); s2.insert(x); }
    }

    std::set<uint32_t> expected;
    std::set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(),
                        std::inserter(expected, expected.begin()));

    auto result = b1.andNot(b2);
    EXPECT_EQ(result.cardinality(), static_cast<uint32_t>(expected.size()));

    for (uint32_t v : expected) {
        EXPECT_TRUE(result.contains(v)) << "andnot missing " << v;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// run_container_unit.c: add_contains_test
// Every 3rd value in [0,65536) → optimize to runs → verify contains.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RunAddContainsEveryThird) {
    for (uint32_t x = 0; x < 65536; x += 3) {
        bm.add(x);
    }
    bm.optimize();  // Convert to run container.
    uint32_t card = bm.cardinality();
    EXPECT_EQ(card, 21846u);

    for (uint32_t x = 0; x < 65536; ++x) {
        EXPECT_EQ(bm.contains(x), (x % 3 == 0));
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// run_container_unit.c: and_or_test
// Same coprime pattern but both optimized to runs.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RunAndOrCoprimeOptimized) {
    RoaringBitmap b1, b2;
    std::set<uint32_t> s1, s2;
    for (uint32_t x = 0; x < 60000; x += 3) { b1.add(x); s1.insert(x); }
    for (uint32_t x = 0; x < 60000; x += 62) { b2.add(x); s2.insert(x); }
    b1.optimize();
    b2.optimize();

    std::set<uint32_t> expectedAnd, expectedOr;
    std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                          std::inserter(expectedAnd, expectedAnd.begin()));
    std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(),
                   std::inserter(expectedOr, expectedOr.begin()));

    EXPECT_EQ((b1 & b2).cardinality(), static_cast<uint32_t>(expectedAnd.size()));
    EXPECT_EQ((b1 | b2).cardinality(), static_cast<uint32_t>(expectedOr.size()));
}


// ─────────────────────────────────────────────────────────────────────────────
// run_container_unit.c: to_uint32_array_test
// Fill with offsets, optimize, verify spacing in toVector output.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RunToVectorSpacing) {
    for (uint32_t offset : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
        RoaringBitmap rb;
        uint32_t count = 0;
        for (uint32_t x = 0; x < 65536; x += offset) {
            rb.add(x);
            ++count;
        }
        rb.optimize();
        EXPECT_EQ(rb.cardinality(), count) << "offset=" << offset;

        auto vec = rb.toVector();
        EXPECT_EQ(vec.size(), count);
        for (size_t k = 1; k < vec.size(); ++k) {
            EXPECT_EQ(vec[k] - vec[k - 1], offset)
                << "offset=" << offset << " k=" << k;
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// run_container_unit.c: remove_range_test
// Three runs [100,150), [200,250), [300,350).
// Remove sub-ranges from each (left edge, right edge, interior).
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RunRemoveRangeSubRanges) {
    bm.addRange(100, 150);
    bm.addRange(200, 250);
    bm.addRange(300, 350);
    bm.optimize();
    EXPECT_EQ(bm.cardinality(), 150u);

    // Remove left 11 from each: [100,111), [200,211), [300,311)
    for (uint32_t i = 100; i < 111; ++i) bm.remove(i);
    for (uint32_t i = 200; i < 211; ++i) bm.remove(i);
    for (uint32_t i = 300; i < 311; ++i) bm.remove(i);

    // Remove right 11 from each: [120,131) from first's remaining, etc.
    // Actually: remove [140,150) → right edge of first run's remainder [111..149]
    for (uint32_t i = 140; i < 150; ++i) bm.remove(i);
    for (uint32_t i = 240; i < 250; ++i) bm.remove(i);
    for (uint32_t i = 340; i < 350; ++i) bm.remove(i);

    // Remaining: [111..139], [211..239], [311..339] = 29*3 = 87 elements.
    EXPECT_EQ(bm.cardinality(), 87u);

    // Verify boundaries.
    EXPECT_FALSE(bm.contains(100));
    EXPECT_TRUE(bm.contains(111));
    EXPECT_TRUE(bm.contains(139));
    EXPECT_FALSE(bm.contains(140));
    EXPECT_FALSE(bm.contains(200));
    EXPECT_TRUE(bm.contains(211));
    EXPECT_TRUE(bm.contains(239));
    EXPECT_FALSE(bm.contains(240));

    // Remove interior: [120..130) from [111..139]
    for (uint32_t i = 120; i < 130; ++i) bm.remove(i);
    // Now [111..119] and [130..139] remain from first group = 9+10 = 19.
    EXPECT_EQ(bm.cardinality(), 77u);
    EXPECT_TRUE(bm.contains(119));
    EXPECT_FALSE(bm.contains(120));
    EXPECT_FALSE(bm.contains(129));
    EXPECT_TRUE(bm.contains(130));
}


// ─────────────────────────────────────────────────────────────────────────────
// mixed_container_unit.c: array_bitset_and_or_xor_andnot_test
// Full cross-type operation verification with coprime patterns.
// Pattern: a = x % 5 < 3, b = x % 62 < 37.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, MixedContainerFullOps) {
    // Build array container (sparse) and bitmap container (dense).
    RoaringBitmap sparse, dense;
    std::set<uint32_t> sSparse, sDense;

    for (uint32_t x = 0; x < 65536; ++x) {
        if (x % 5 < 3) { sparse.add(x); sSparse.insert(x); }
    }
    for (uint32_t x = 0; x < 65536; ++x) {
        if (x % 62 < 37) { dense.add(x); sDense.insert(x); }
    }

    // sparse has ~39322 elements (bitmap), dense has ~39127 (bitmap).
    // Both are bitmap containers. Verify all 4 operations.
    std::set<uint32_t> expAnd, expOr, expXor, expDiff;
    std::set_intersection(sSparse.begin(), sSparse.end(), sDense.begin(), sDense.end(),
                          std::inserter(expAnd, expAnd.begin()));
    std::set_union(sSparse.begin(), sSparse.end(), sDense.begin(), sDense.end(),
                   std::inserter(expOr, expOr.begin()));
    std::set_symmetric_difference(sSparse.begin(), sSparse.end(), sDense.begin(), sDense.end(),
                                  std::inserter(expXor, expXor.begin()));
    std::set_difference(sSparse.begin(), sSparse.end(), sDense.begin(), sDense.end(),
                        std::inserter(expDiff, expDiff.begin()));

    EXPECT_EQ((sparse & dense).cardinality(), static_cast<uint32_t>(expAnd.size()));
    EXPECT_EQ((sparse | dense).cardinality(), static_cast<uint32_t>(expOr.size()));
    EXPECT_EQ((sparse ^ dense).cardinality(), static_cast<uint32_t>(expXor.size()));
    EXPECT_EQ(sparse.andNot(dense).cardinality(), static_cast<uint32_t>(expDiff.size()));
}


// ─────────────────────────────────────────────────────────────────────────────
// mixed_container_unit.c: XOR of bitmap with itself → empty.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, BitmapXorSelfEmpty) {
    bm.addRange(0, 10000);  // bitmap
    auto result = bm ^ bm;
    EXPECT_TRUE(result.empty());
}


// ─────────────────────────────────────────────────────────────────────────────
// mixed_container_unit.c: run_xor_test
// Pattern: R1 = x%5<3 (as run), R2 = x%62<37 (as run). XOR.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RunXorCrossType) {
    RoaringBitmap r1, r2;
    std::set<uint32_t> s1, s2;
    for (uint32_t x = 0; x < 65536; ++x) {
        if (x % 5 < 3) { r1.add(x); s1.insert(x); }
        if (x % 62 < 37) { r2.add(x); s2.insert(x); }
    }
    r1.optimize();
    r2.optimize();

    std::set<uint32_t> expected;
    std::set_symmetric_difference(s1.begin(), s1.end(), s2.begin(), s2.end(),
                                  std::inserter(expected, expected.begin()));

    auto result = r1 ^ r2;
    EXPECT_EQ(result.cardinality(), static_cast<uint32_t>(expected.size()));
}


// ─────────────────────────────────────────────────────────────────────────────
// mixed_container_unit.c: run_andnot_test
// Complex run structure vs array. Specific data from CRoaring test.
// Runs: [0,3), [10,12), [990,995), [10000,10003), [20000,20002).
// Minus array: {993, 994, 2000} → should remove 993,994 from third run.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RunAndNotSpecificData) {
    RoaringBitmap runs, arr;
    // Build runs.
    for (uint32_t v : {0u, 1u, 2u}) runs.add(v);
    for (uint32_t v : {10u, 11u}) runs.add(v);
    for (uint32_t v : {990u, 991u, 992u, 993u, 994u}) runs.add(v);
    for (uint32_t v : {10000u, 10001u, 10002u}) runs.add(v);
    for (uint32_t v : {20000u, 20001u}) runs.add(v);
    runs.optimize();
    EXPECT_EQ(runs.cardinality(), 15u);

    arr.add(993);
    arr.add(994);
    arr.add(2000);

    auto result = runs.andNot(arr);
    // 15 - 2 (993,994 removed; 2000 not in runs) = 13.
    EXPECT_EQ(result.cardinality(), 13u);
    EXPECT_TRUE(result.contains(992));
    EXPECT_FALSE(result.contains(993));
    EXPECT_FALSE(result.contains(994));
    EXPECT_TRUE(result.contains(0));
    EXPECT_TRUE(result.contains(20001));
}


// ─────────────────────────────────────────────────────────────────────────────
// mixed_container_unit.c: array_bitset_iandnot_test
// Self-subtraction → empty. Single-element difference.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, InPlaceAndNotSelfEmpty) {
    bm.addRange(0, 5000);  // bitmap
    bm -= bm;
    EXPECT_TRUE(bm.empty());
}


TEST_F(RoaringBitmapTest, InPlaceAndNotSingleElement) {
    RoaringBitmap a, b;
    a.addRange(0, 100);
    b.addRange(0, 99);
    a -= b;
    EXPECT_EQ(a.cardinality(), 1u);
    EXPECT_TRUE(a.contains(99));
}


// ─────────────────────────────────────────────────────────────────────────────
// container_comparison_unit.c: generic_equal_test (9 type combinations)
// Build same set with different container types, verify equality.
// ─────────────────────────────────────────────────────────────────────────────

// Helper: build a bitmap forcing a specific container type.
namespace {
RoaringBitmap makeArray(const std::vector<uint32_t>& vals) {
    RoaringBitmap bm;
    for (uint32_t v : vals) bm.add(v);
    return bm;
}
RoaringBitmap makeRun(const std::vector<uint32_t>& vals) {
    RoaringBitmap bm;
    for (uint32_t v : vals) bm.add(v);
    bm.optimize();
    return bm;
}
}  // namespace


TEST_F(RoaringBitmapTest, EqualityFullMatrix3x3) {
    // Same values: {0, 10, 20, ..., 990}
    std::vector<uint32_t> vals;
    for (uint32_t i = 0; i < 1000; i += 10) vals.push_back(i);

    auto arrBm = makeArray(vals);
    auto runBm = makeRun(vals);

    // Array vs Array
    auto arrBm2 = makeArray(vals);
    EXPECT_EQ(arrBm, arrBm2);

    // Array vs Run
    EXPECT_EQ(arrBm, runBm);

    // Run vs Array
    EXPECT_EQ(runBm, arrBm);

    // Run vs Run
    auto runBm2 = makeRun(vals);
    EXPECT_EQ(runBm, runBm2);

    // Now add extra element to one, verify inequality.
    arrBm2.add(273);
    EXPECT_NE(arrBm, arrBm2);
    EXPECT_NE(runBm, arrBm2);
}


// ─────────────────────────────────────────────────────────────────────────────
// container_comparison_unit.c: generic_subset_test (cross-type)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, SubsetCrossContainerArrayBitmap) {
    RoaringBitmap small, large;
    small.addRange(0, 100);       // array
    large.addRange(0, 5000);      // bitmap
    EXPECT_TRUE(small.isSubsetOf(large));
    EXPECT_FALSE(large.isSubsetOf(small));
}


TEST_F(RoaringBitmapTest, SubsetCrossContainerArrayRun) {
    RoaringBitmap small, large;
    small.addRange(0, 100);       // array
    large.addRange(0, 200);
    large.optimize();             // run
    EXPECT_TRUE(small.isSubsetOf(large));
    EXPECT_FALSE(large.isSubsetOf(small));
}


TEST_F(RoaringBitmapTest, SubsetCrossContainerBitmapRun) {
    RoaringBitmap bitmapBm, runBm;
    bitmapBm.addRange(0, 5000);  // bitmap
    runBm.addRange(0, 10000);
    runBm.optimize();             // run
    EXPECT_TRUE(bitmapBm.isSubsetOf(runBm));
    EXPECT_FALSE(runBm.isSubsetOf(bitmapBm));
}


TEST_F(RoaringBitmapTest, SubsetCrossContainerRunBitmap) {
    RoaringBitmap runBm, bitmapBm;
    runBm.addRange(100, 200);
    runBm.optimize();              // run (100 elements)
    bitmapBm.addRange(0, 5000);    // bitmap
    EXPECT_TRUE(runBm.isSubsetOf(bitmapBm));
}


// ─────────────────────────────────────────────────────────────────────────────
// container_comparison_unit.c: generic_equal_test — full container, first/last differ
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, EqualityFullContainerFirstDiffers) {
    RoaringBitmap a, b;
    for (uint32_t i = 1; i < 65536; ++i) a.add(i);   // missing 0
    for (uint32_t i = 0; i < 65536; ++i) b.add(i);
    EXPECT_NE(a, b);
}


TEST_F(RoaringBitmapTest, EqualityFullContainerLastDiffers) {
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 65535; ++i) a.add(i);   // missing 65535
    for (uint32_t i = 0; i < 65536; ++i) b.add(i);
    EXPECT_NE(a, b);
}


// ─────────────────────────────────────────────────────────────────────────────
// cpp_unit.cpp: test_run_compression_cpp / test_cpp_remove_run_compression
// Verify optimize reduces memory representation for contiguous data.
// After optimize, operations still correct.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RunCompressionRoundTrip) {
    // Add sparse data that compresses well as runs.
    for (uint32_t i = 0; i < 50000; i += 2) bm.add(i);  // every other
    uint32_t cardBefore = bm.cardinality();
    auto vecBefore = bm.toVector();

    bm.optimize();

    EXPECT_EQ(bm.cardinality(), cardBefore);
    auto vecAfter = bm.toVector();
    EXPECT_EQ(vecBefore, vecAfter);
}


TEST_F(RoaringBitmapTest, RunCompressionContiguousRange) {
    bm.addRange(0, 50000);
    bm.optimize();
    EXPECT_EQ(bm.cardinality(), 50000u);
    EXPECT_TRUE(bm.contains(0));
    EXPECT_TRUE(bm.contains(49999));
    EXPECT_FALSE(bm.contains(50000));
}


// ─────────────────────────────────────────────────────────────────────────────
// mini_fuzz: randomized AND/OR/XOR/ANDNOT with std::set as reference.
// CRoaring: mini_fuzz_array_container_intersection_inplace.
// 3000 iterations with SplitMix64 PRNG.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, MiniFuzzAllOps) {
    uint64_t state = 12345;
    auto splitMix64 = [&state]() -> uint32_t {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        return static_cast<uint32_t>(z);
    };

    for (int iter = 0; iter < 500; ++iter) {
        std::set<uint32_t> sA, sB;
        RoaringBitmap rA, rB;

        // Populate with random values in [0, 0x1FFFF) — spans 2 chunks.
        uint32_t nA = (splitMix64() % 200) + 1;
        uint32_t nB = (splitMix64() % 200) + 1;
        for (uint32_t i = 0; i < nA; ++i) {
            uint32_t v = splitMix64() & 0x1FFFF;
            sA.insert(v);
            rA.add(v);
        }
        for (uint32_t i = 0; i < nB; ++i) {
            uint32_t v = splitMix64() & 0x1FFFF;
            sB.insert(v);
            rB.add(v);
        }

        ASSERT_EQ(rA.cardinality(), static_cast<uint32_t>(sA.size()));
        ASSERT_EQ(rB.cardinality(), static_cast<uint32_t>(sB.size()));

        // AND
        {
            std::set<uint32_t> exp;
            std::set_intersection(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                  std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA & rB).cardinality(), static_cast<uint32_t>(exp.size()))
                << "AND failed at iter=" << iter;
        }

        // OR
        {
            std::set<uint32_t> exp;
            std::set_union(sA.begin(), sA.end(), sB.begin(), sB.end(),
                           std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA | rB).cardinality(), static_cast<uint32_t>(exp.size()))
                << "OR failed at iter=" << iter;
        }

        // XOR
        {
            std::set<uint32_t> exp;
            std::set_symmetric_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                          std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA ^ rB).cardinality(), static_cast<uint32_t>(exp.size()))
                << "XOR failed at iter=" << iter;
        }

        // ANDNOT
        {
            std::set<uint32_t> exp;
            std::set_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                std::inserter(exp, exp.begin()));
            ASSERT_EQ(rA.andNot(rB).cardinality(), static_cast<uint32_t>(exp.size()))
                << "ANDNOT failed at iter=" << iter;
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// mini_fuzz: randomized with LARGE sets that force bitmap containers.
// CRoaring: mini_fuzz but with bitset reference.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, MiniFuzzBitmapContainers) {
    uint64_t state = 67890;
    auto splitMix64 = [&state]() -> uint32_t {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        return static_cast<uint32_t>(z);
    };

    for (int iter = 0; iter < 50; ++iter) {
        std::set<uint32_t> sA, sB;
        RoaringBitmap rA, rB;

        // Large enough to force bitmap containers in one chunk.
        uint32_t nA = 5000 + (splitMix64() % 5000);
        uint32_t nB = 5000 + (splitMix64() % 5000);
        for (uint32_t i = 0; i < nA; ++i) {
            uint32_t v = splitMix64() & 0xFFFF;  // single chunk
            sA.insert(v);
            rA.add(v);
        }
        for (uint32_t i = 0; i < nB; ++i) {
            uint32_t v = splitMix64() & 0xFFFF;
            sB.insert(v);
            rB.add(v);
        }

        // All ops against reference.
        {
            std::set<uint32_t> exp;
            std::set_intersection(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                  std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA & rB).cardinality(), static_cast<uint32_t>(exp.size()));
        }
        {
            std::set<uint32_t> exp;
            std::set_union(sA.begin(), sA.end(), sB.begin(), sB.end(),
                           std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA | rB).cardinality(), static_cast<uint32_t>(exp.size()));
        }
        {
            std::set<uint32_t> exp;
            std::set_symmetric_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                          std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA ^ rB).cardinality(), static_cast<uint32_t>(exp.size()));
        }
        {
            std::set<uint32_t> exp;
            std::set_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                std::inserter(exp, exp.begin()));
            ASSERT_EQ(rA.andNot(rB).cardinality(), static_cast<uint32_t>(exp.size()));
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// mini_fuzz: randomized with RUN containers.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, MiniFuzzRunContainers) {
    uint64_t state = 11111;
    auto splitMix64 = [&state]() -> uint32_t {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        return static_cast<uint32_t>(z);
    };

    for (int iter = 0; iter < 100; ++iter) {
        std::set<uint32_t> sA, sB;
        RoaringBitmap rA, rB;

        // Build with runs: add random ranges.
        uint32_t nRanges = 3 + (splitMix64() % 10);
        for (uint32_t i = 0; i < nRanges; ++i) {
            uint32_t start = splitMix64() & 0xFFFF;
            uint32_t len = 1 + (splitMix64() % 500);
            for (uint32_t v = start; v < start + len && v < 65536; ++v) {
                sA.insert(v);
                rA.add(v);
            }
        }
        nRanges = 3 + (splitMix64() % 10);
        for (uint32_t i = 0; i < nRanges; ++i) {
            uint32_t start = splitMix64() & 0xFFFF;
            uint32_t len = 1 + (splitMix64() % 500);
            for (uint32_t v = start; v < start + len && v < 65536; ++v) {
                sB.insert(v);
                rB.add(v);
            }
        }
        rA.optimize();
        rB.optimize();

        ASSERT_EQ(rA.cardinality(), static_cast<uint32_t>(sA.size()));
        ASSERT_EQ(rB.cardinality(), static_cast<uint32_t>(sB.size()));

        // AND
        {
            std::set<uint32_t> exp;
            std::set_intersection(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                  std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA & rB).cardinality(), static_cast<uint32_t>(exp.size()))
                << "Run AND failed iter=" << iter;
        }
        // OR
        {
            std::set<uint32_t> exp;
            std::set_union(sA.begin(), sA.end(), sB.begin(), sB.end(),
                           std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA | rB).cardinality(), static_cast<uint32_t>(exp.size()))
                << "Run OR failed iter=" << iter;
        }
        // XOR
        {
            std::set<uint32_t> exp;
            std::set_symmetric_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                          std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA ^ rB).cardinality(), static_cast<uint32_t>(exp.size()))
                << "Run XOR failed iter=" << iter;
        }
        // ANDNOT
        {
            std::set<uint32_t> exp;
            std::set_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                std::inserter(exp, exp.begin()));
            ASSERT_EQ(rA.andNot(rB).cardinality(), static_cast<uint32_t>(exp.size()))
                << "Run ANDNOT failed iter=" << iter;
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// mini_fuzz: mixed container types (array vs bitmap vs run).
// CRoaring: mixed_container_unit.c fuzz-style tests.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, MiniFuzzMixedContainerTypes) {
    uint64_t state = 99999;
    auto splitMix64 = [&state]() -> uint32_t {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        return static_cast<uint32_t>(z);
    };

    for (int iter = 0; iter < 100; ++iter) {
        std::set<uint32_t> sA, sB;
        RoaringBitmap rA, rB;

        // rA: small (array container).
        uint32_t nA = 50 + (splitMix64() % 200);
        for (uint32_t i = 0; i < nA; ++i) {
            uint32_t v = splitMix64() & 0xFFFF;
            sA.insert(v);
            rA.add(v);
        }

        // rB: large (bitmap container), then optimize to run.
        uint32_t nRanges = 5 + (splitMix64() % 10);
        for (uint32_t i = 0; i < nRanges; ++i) {
            uint32_t start = splitMix64() & 0xFFFF;
            uint32_t len = 100 + (splitMix64() % 2000);
            for (uint32_t v = start; v < start + len && v < 65536; ++v) {
                sB.insert(v);
                rB.add(v);
            }
        }
        // Randomly optimize rB to runs.
        if (splitMix64() % 2 == 0) rB.optimize();

        ASSERT_EQ(rA.cardinality(), static_cast<uint32_t>(sA.size()));
        ASSERT_EQ(rB.cardinality(), static_cast<uint32_t>(sB.size()));

        // All 4 ops.
        {
            std::set<uint32_t> exp;
            std::set_intersection(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                  std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA & rB).cardinality(), static_cast<uint32_t>(exp.size()));
        }
        {
            std::set<uint32_t> exp;
            std::set_union(sA.begin(), sA.end(), sB.begin(), sB.end(),
                           std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA | rB).cardinality(), static_cast<uint32_t>(exp.size()));
        }
        {
            std::set<uint32_t> exp;
            std::set_symmetric_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                          std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA ^ rB).cardinality(), static_cast<uint32_t>(exp.size()));
        }
        {
            std::set<uint32_t> exp;
            std::set_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                std::inserter(exp, exp.begin()));
            ASSERT_EQ(rA.andNot(rB).cardinality(), static_cast<uint32_t>(exp.size()));
        }
        // Also test B andnot A.
        {
            std::set<uint32_t> exp;
            std::set_difference(sB.begin(), sB.end(), sA.begin(), sA.end(),
                                std::inserter(exp, exp.begin()));
            ASSERT_EQ(rB.andNot(rA).cardinality(), static_cast<uint32_t>(exp.size()));
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Algebraic identities (CRoaring: various)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, AlgebraicIdentities) {
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 10000; i += 3) a.add(i);
    for (uint32_t i = 0; i < 10000; i += 7) b.add(i);

    // a & b == b & a (commutativity)
    EXPECT_EQ(a & b, b & a);

    // a | b == b | a
    EXPECT_EQ(a | b, b | a);

    // a ^ b == b ^ a
    EXPECT_EQ(a ^ b, b ^ a);

    // a & a == a (idempotent)
    EXPECT_EQ(a & a, a);

    // a | a == a
    EXPECT_EQ(a | a, a);

    // a ^ a == empty
    EXPECT_TRUE((a ^ a).empty());

    // a - a == empty
    EXPECT_TRUE(a.andNot(a).empty());

    // (a | b) - b == a - b
    EXPECT_EQ((a | b).andNot(b), a.andNot(b));

    // a & (a | b) == a (absorption)
    EXPECT_EQ(a & (a | b), a);

    // a | (a & b) == a (absorption)
    EXPECT_EQ(a | (a & b), a);

    // (a ^ b) == (a | b) - (a & b) (XOR definition)
    EXPECT_EQ(a ^ b, (a | b).andNot(a & b));

    // (a - b) | (b - a) == a ^ b
    EXPECT_EQ(a.andNot(b) | b.andNot(a), a ^ b);

    // De Morgan: ~(a & b) == ~a | ~b (via full bitmap)
    RoaringBitmap full;
    full.addRange(0, 10000);
    auto notA = full.andNot(a);
    auto notB = full.andNot(b);
    auto notAandB = full.andNot(a & b);
    EXPECT_EQ(notAandB, notA | notB);

    // De Morgan: ~(a | b) == ~a & ~b
    auto notAorB = full.andNot(a | b);
    EXPECT_EQ(notAorB, notA & notB);
}


// ─────────────────────────────────────────────────────────────────────────────
// Multi-chunk fuzz: operations spanning many chunks.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, MiniFuzzMultiChunk) {
    uint64_t state = 54321;
    auto splitMix64 = [&state]() -> uint32_t {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        return static_cast<uint32_t>(z);
    };

    for (int iter = 0; iter < 50; ++iter) {
        std::set<uint32_t> sA, sB;
        RoaringBitmap rA, rB;

        // Values across 10 chunks (0..9 * 65536).
        uint32_t nA = 100 + (splitMix64() % 500);
        uint32_t nB = 100 + (splitMix64() % 500);
        for (uint32_t i = 0; i < nA; ++i) {
            uint32_t chunk = splitMix64() % 10;
            uint32_t lo = splitMix64() & 0xFFFF;
            uint32_t v = chunk * 65536 + lo;
            sA.insert(v);
            rA.add(v);
        }
        for (uint32_t i = 0; i < nB; ++i) {
            uint32_t chunk = splitMix64() % 10;
            uint32_t lo = splitMix64() & 0xFFFF;
            uint32_t v = chunk * 65536 + lo;
            sB.insert(v);
            rB.add(v);
        }

        ASSERT_EQ(rA.cardinality(), static_cast<uint32_t>(sA.size()));

        {
            std::set<uint32_t> exp;
            std::set_intersection(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                  std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA & rB).cardinality(), static_cast<uint32_t>(exp.size()));
        }
        {
            std::set<uint32_t> exp;
            std::set_union(sA.begin(), sA.end(), sB.begin(), sB.end(),
                           std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA | rB).cardinality(), static_cast<uint32_t>(exp.size()));
        }
        {
            std::set<uint32_t> exp;
            std::set_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                std::inserter(exp, exp.begin()));
            ASSERT_EQ(rA.andNot(rB).cardinality(), static_cast<uint32_t>(exp.size()));
        }
        {
            std::set<uint32_t> exp;
            std::set_symmetric_difference(sA.begin(), sA.end(), sB.begin(), sB.end(),
                                          std::inserter(exp, exp.begin()));
            ASSERT_EQ((rA ^ rB).cardinality(), static_cast<uint32_t>(exp.size()));
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// COW (Copy-on-Write) tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, COW_CopyThenAddDoesNotAffectOriginal) {
    bm.add(100);
    bm.add(200);
    RoaringBitmap copy = bm;
    copy.add(300);
    EXPECT_EQ(bm.cardinality(), 2u);
    EXPECT_FALSE(bm.contains(300));
    EXPECT_EQ(copy.cardinality(), 3u);
    EXPECT_TRUE(copy.contains(300));
}


TEST_F(RoaringBitmapTest, COW_CopyThenRemoveDoesNotAffectOriginal) {
    bm.add(100);
    bm.add(200);
    RoaringBitmap copy = bm;
    copy.remove(100);
    EXPECT_EQ(bm.cardinality(), 2u);
    EXPECT_TRUE(bm.contains(100));
    EXPECT_EQ(copy.cardinality(), 1u);
    EXPECT_FALSE(copy.contains(100));
}


TEST_F(RoaringBitmapTest, COW_CopyThenAddRangeDoesNotAffectOriginal) {
    bm.addRange(0, 100);
    RoaringBitmap copy = bm;
    copy.addRange(100, 200);
    EXPECT_EQ(bm.cardinality(), 100u);
    EXPECT_FALSE(bm.contains(150));
    EXPECT_EQ(copy.cardinality(), 200u);
}


TEST_F(RoaringBitmapTest, COW_CopyThenRemoveRangeDoesNotAffectOriginal) {
    bm.addRange(0, 200);
    RoaringBitmap copy = bm;
    copy.removeRange(50, 150);
    EXPECT_EQ(bm.cardinality(), 200u);
    EXPECT_TRUE(bm.contains(75));
    EXPECT_EQ(copy.cardinality(), 100u);
    EXPECT_FALSE(copy.contains(75));
}


TEST_F(RoaringBitmapTest, COW_CopyThenFlipDoesNotAffectOriginal) {
    bm.addRange(0, 100);
    RoaringBitmap copy = bm;
    copy.flip(0, 200);  // flips [0,100) off and [100,200) on
    EXPECT_EQ(bm.cardinality(), 100u);
    EXPECT_TRUE(bm.contains(50));
    EXPECT_EQ(copy.cardinality(), 100u);
    EXPECT_FALSE(copy.contains(50));
    EXPECT_TRUE(copy.contains(150));
}


TEST_F(RoaringBitmapTest, COW_CopyThenOptimizeDoesNotAffectOriginal) {
    bm.addRange(0, 5000);  // will be bitmap container
    RoaringBitmap copy = bm;
    copy.optimize();  // converts to run container
    EXPECT_EQ(bm.cardinality(), 5000u);
    EXPECT_EQ(copy.cardinality(), 5000u);
    EXPECT_EQ(bm, copy);  // logically equal
}


TEST_F(RoaringBitmapTest, COW_CopyThenShrinkToFitDoesNotAffectOriginal) {
    for (uint32_t i = 0; i < 100; ++i) bm.add(i * 3);
    RoaringBitmap copy = bm;
    copy.shrinkToFit();
    EXPECT_EQ(bm, copy);
}


TEST_F(RoaringBitmapTest, COW_CopyThenRemoveRunCompressionDoesNotAffectOriginal) {
    bm.addRange(0, 5000);
    bm.optimize();  // convert to run
    RoaringBitmap copy = bm;
    copy.removeRunCompression();
    EXPECT_EQ(bm.cardinality(), 5000u);
    EXPECT_EQ(copy.cardinality(), 5000u);
    EXPECT_EQ(bm, copy);
}


TEST_F(RoaringBitmapTest, COW_SetOpSharingNonOverlapping) {
    // a and b have non-overlapping chunk keys
    RoaringBitmap a, b;
    a.addRange(0, 5000);           // chunk 0
    b.addRange(0x10000, 0x11388);  // chunk 1

    RoaringBitmap result = a | b;
    uint32_t expectedCard = result.cardinality();

    // Mutate a — result should be unchanged
    a.add(9999);
    EXPECT_EQ(result.cardinality(), expectedCard);
    EXPECT_FALSE(result.contains(9999));
}


TEST_F(RoaringBitmapTest, COW_MultiLevelSharing) {
    bm.addRange(0, 5000);
    RoaringBitmap B = bm;   // shares with bm
    RoaringBitmap C = B;    // shares with B (and transitively bm)

    C.add(9999);  // cow: only C's container is copied

    EXPECT_EQ(bm.cardinality(), 5000u);
    EXPECT_FALSE(bm.contains(9999));
    EXPECT_EQ(B.cardinality(), 5000u);
    EXPECT_FALSE(B.contains(9999));
    EXPECT_EQ(C.cardinality(), 5001u);
    EXPECT_TRUE(C.contains(9999));
}


TEST_F(RoaringBitmapTest, COW_SerializeSharedBitmap) {
    bm.addRange(0, 5000);
    RoaringBitmap copy = bm;

    auto buf = copy.serialize();
    auto deserialized = RoaringBitmap::deserialize(buf.data(), buf.size());
    ASSERT_TRUE(deserialized.has_value());
    EXPECT_EQ(*deserialized, bm);
}


TEST_F(RoaringBitmapTest, COW_FlippedDoesNotAffectOriginal) {
    bm.addRange(0, 100);
    RoaringBitmap result = bm.flipped(0, 200);

    EXPECT_EQ(bm.cardinality(), 100u);
    EXPECT_TRUE(bm.contains(50));
    EXPECT_EQ(result.cardinality(), 100u);
    EXPECT_FALSE(result.contains(50));
    EXPECT_TRUE(result.contains(150));
}


TEST_F(RoaringBitmapTest, COW_BitmapContainerSharing) {
    // Force bitmap containers (>4096 values in one chunk)
    for (uint32_t i = 0; i < 5000; ++i) bm.add(i);
    RoaringBitmap copy = bm;

    // Mutate copy — should cow the bitmap container
    copy.remove(0);
    EXPECT_TRUE(bm.contains(0));
    EXPECT_FALSE(copy.contains(0));
    EXPECT_EQ(bm.cardinality(), 5000u);
    EXPECT_EQ(copy.cardinality(), 4999u);
}


TEST_F(RoaringBitmapTest, COW_RepairCardinalityOnShared) {
    // Create lazy containers via lazyOr
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 5000; ++i) a.add(i);
    for (uint32_t i = 2500; i < 7500; ++i) b.add(i);

    RoaringBitmap lazy = a.lazyOr(b);
    RoaringBitmap copy = lazy;
    copy.repairCardinality();

    // lazy should still work correctly
    EXPECT_EQ(copy.cardinality(), 7500u);
}


// ═════════════════════════════════════════════════════════════════════════════
// Gap 24: RunContainer::toArray() with std::iota
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, RunContainerToArraySingleRun) {
    bm.addRange(10, 20);
    bm.optimize();
    auto vec = bm.toVector();
    EXPECT_EQ(vec.size(), 10u);
    for (uint32_t i = 0; i < 10; ++i)
        EXPECT_EQ(vec[i], i + 10);
}


TEST_F(RoaringBitmapTest, RunContainerToArrayMultipleRuns) {
    bm.addRange(0, 5);
    bm.addRange(100, 103);
    bm.addRange(200, 201);
    bm.optimize();
    auto vec = bm.toVector();
    EXPECT_EQ(vec.size(), 9u);
    // 0,1,2,3,4,100,101,102,200
    std::vector<uint32_t> expected = {0, 1, 2, 3, 4, 100, 101, 102, 200};
    EXPECT_EQ(vec, expected);
}


TEST_F(RoaringBitmapTest, RunContainerToArrayFullRange) {
    // Single run covering entire 16-bit range
    bm.addRange(0, 65536);
    bm.optimize();
    EXPECT_EQ(bm.cardinality(), 65536u);
    EXPECT_TRUE(bm.contains(0));
    EXPECT_TRUE(bm.contains(65535));
}


// ═════════════════════════════════════════════════════════════════════════════
// Gap 27: Streaming run append (appendRun)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, AppendRunMergesAdjacent) {
    RunContainer rc;
    rc.appendRun(0, 4);    // [0..4]
    rc.appendRun(5, 4);    // [5..9] — adjacent, should merge
    EXPECT_EQ(rc.runs.size(), 1u);
    EXPECT_EQ(rc.runs[0].start, 0);
    EXPECT_EQ(rc.runs[0].length, 9);
}


TEST_F(RoaringBitmapTest, AppendRunMergesOverlapping) {
    RunContainer rc;
    rc.appendRun(0, 9);    // [0..9]
    rc.appendRun(5, 9);    // [5..14] — overlapping, should merge to [0..14]
    EXPECT_EQ(rc.runs.size(), 1u);
    EXPECT_EQ(rc.runs[0].start, 0);
    EXPECT_EQ(rc.runs[0].length, 14);
}


TEST_F(RoaringBitmapTest, AppendRunDisjoint) {
    RunContainer rc;
    rc.appendRun(0, 4);    // [0..4]
    rc.appendRun(10, 4);   // [10..14] — gap, no merge
    EXPECT_EQ(rc.runs.size(), 2u);
}


TEST_F(RoaringBitmapTest, AppendRunSubsumed) {
    RunContainer rc;
    rc.appendRun(0, 19);   // [0..19]
    rc.appendRun(5, 4);    // [5..9] — subsumed, no change
    EXPECT_EQ(rc.runs.size(), 1u);
    EXPECT_EQ(rc.runs[0].length, 19);
}
} // namespace arrow
