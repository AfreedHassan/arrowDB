#include <gtest/gtest.h>
#include <algorithm>
#include <set>
#include <vector>
#include "index/roaring_bitmap.h"
#include "index/roaring_simd.h"

namespace arrow {

class RoaringBitmapTest : public ::testing::Test {
protected:
    RoaringBitmap bm;
};

// ─────────────────────────────────────────────────────────────────────────────
// Basic add / contains / remove (CRoaring: test_example, add_contains_test)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, BasicAddContainsRemove) {
    EXPECT_TRUE(bm.empty());
    bm.add(42);
    EXPECT_TRUE(bm.contains(42));
    EXPECT_FALSE(bm.contains(43));
    EXPECT_EQ(bm.cardinality(), 1u);

    bm.remove(42);
    EXPECT_FALSE(bm.contains(42));
    EXPECT_TRUE(bm.empty());
}

TEST_F(RoaringBitmapTest, SingleElement) {
    bm.add(0);
    EXPECT_TRUE(bm.contains(0));
    EXPECT_EQ(bm.cardinality(), 1u);
    EXPECT_EQ(bm.maximum(), 0u);
    EXPECT_EQ(bm.minimum(), 0u);
}

TEST_F(RoaringBitmapTest, MaxUint32) {
    uint32_t maxVal = UINT32_MAX;
    bm.add(maxVal);
    EXPECT_TRUE(bm.contains(maxVal));
    EXPECT_EQ(bm.cardinality(), 1u);
    EXPECT_EQ(bm.maximum(), maxVal);
    EXPECT_EQ(bm.minimum(), maxVal);
}

TEST_F(RoaringBitmapTest, EmptyMinimumMaximum) {
    EXPECT_FALSE(bm.maximum().has_value());
    EXPECT_FALSE(bm.minimum().has_value());
}

// CRoaring: add_contains_test — every 3rd value, verify contains logic.
TEST_F(RoaringBitmapTest, AddEveryThirdValue) {
    for (uint32_t x = 0; x < 65536; x += 3) {
        bm.add(x);
    }
    uint32_t expected = (65536 + 2) / 3;  // ceil(65536/3)
    EXPECT_EQ(bm.cardinality(), expected);

    for (uint32_t x = 0; x < 65536; ++x) {
        EXPECT_EQ(bm.contains(x), (x % 3 == 0));
    }
}

// CRoaring: test_cpp_add_remove_checked — duplicate adds, nonexistent removes.
TEST_F(RoaringBitmapTest, CardinalityAfterDuplicateAdds) {
    bm.add(42);
    bm.add(42);
    bm.add(42);
    EXPECT_EQ(bm.cardinality(), 1u);
}

TEST_F(RoaringBitmapTest, RemoveNonexistent) {
    bm.add(1);
    bm.remove(2);  // no-op
    EXPECT_EQ(bm.cardinality(), 1u);
    EXPECT_TRUE(bm.contains(1));
}

// ─────────────────────────────────────────────────────────────────────────────
// Multiple chunks (CRoaring: toplevel multi-container tests)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, MultipleChunks) {
    bm.add(0);          // chunk 0
    bm.add(65536);      // chunk 1
    bm.add(131072);     // chunk 2
    EXPECT_EQ(bm.cardinality(), 3u);
    EXPECT_TRUE(bm.contains(0));
    EXPECT_TRUE(bm.contains(65536));
    EXPECT_TRUE(bm.contains(131072));
    EXPECT_EQ(bm.minimum(), 0u);
    EXPECT_EQ(bm.maximum(), 131072u);
}

TEST_F(RoaringBitmapTest, LargeCardinalityMultiChunk) {
    for (uint32_t i = 0; i < 10; ++i) {
        bm.addRange(i * 65536, i * 65536 + 1000);
    }
    EXPECT_EQ(bm.cardinality(), 10000u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Promotion / demotion (CRoaring: capacity_test, container type transitions)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, PromotionAt4096) {
    for (uint32_t i = 0; i <= 4096; ++i) {
        bm.add(i);
    }
    EXPECT_EQ(bm.cardinality(), 4097u);
    for (uint32_t i = 0; i <= 4096; ++i) {
        EXPECT_TRUE(bm.contains(i));
    }
    EXPECT_FALSE(bm.contains(4097));
}

TEST_F(RoaringBitmapTest, DemotionBelow4096) {
    for (uint32_t i = 0; i <= 4096; ++i) {
        bm.add(i);
    }
    EXPECT_EQ(bm.cardinality(), 4097u);

    for (uint32_t i = 4000; i <= 4096; ++i) {
        bm.remove(i);
    }
    EXPECT_EQ(bm.cardinality(), 4000u);
    EXPECT_TRUE(bm.contains(0));
    EXPECT_TRUE(bm.contains(3999));
    EXPECT_FALSE(bm.contains(4000));
}

// CRoaring: capacity_test — add all 65536 values.
TEST_F(RoaringBitmapTest, FullContainer65536) {
    for (uint32_t i = 0; i < 65536; ++i) {
        bm.add(i);
    }
    EXPECT_EQ(bm.cardinality(), 65536u);
    for (uint32_t i = 0; i < 65536; ++i) {
        EXPECT_TRUE(bm.contains(i));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// addRange (CRoaring: test_cpp_add_range, roaring_bitmap_from_range)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, AddRangeSmall) {
    bm.addRange(100, 200);
    EXPECT_EQ(bm.cardinality(), 100u);
    EXPECT_TRUE(bm.contains(100));
    EXPECT_TRUE(bm.contains(199));
    EXPECT_FALSE(bm.contains(99));
    EXPECT_FALSE(bm.contains(200));
}

TEST_F(RoaringBitmapTest, AddRangeEmpty) {
    bm.addRange(100, 100);
    EXPECT_TRUE(bm.empty());
}

// CRoaring: fuzz_001 — addRange with min > max is no-op.
TEST_F(RoaringBitmapTest, AddRangeInverted) {
    bm.addRange(173, 0);
    EXPECT_EQ(bm.cardinality(), 0u);
}

TEST_F(RoaringBitmapTest, AddRangePromotesToBitmap) {
    bm.addRange(0, 5000);  // > 4096, should promote
    EXPECT_EQ(bm.cardinality(), 5000u);
    for (uint32_t i = 0; i < 5000; ++i) {
        EXPECT_TRUE(bm.contains(i));
    }
    EXPECT_FALSE(bm.contains(5000));
}

TEST_F(RoaringBitmapTest, AddRangeCrossChunk) {
    // Range spanning two 16-bit chunks.
    bm.addRange(65530, 65542);  // chunk 0 [65530..65535] + chunk 1 [0..5]
    EXPECT_EQ(bm.cardinality(), 12u);
    for (uint32_t v = 65530; v < 65542; ++v) {
        EXPECT_TRUE(bm.contains(v));
    }
    EXPECT_FALSE(bm.contains(65529));
    EXPECT_FALSE(bm.contains(65542));
}

TEST_F(RoaringBitmapTest, AddRangeLargeCrossChunk) {
    // Span 3 full chunks.
    bm.addRange(0, 3 * 65536);
    EXPECT_EQ(bm.cardinality(), 3u * 65536);
}

// CRoaring: test_cpp_add_range_closed — single element range.
TEST_F(RoaringBitmapTest, AddRangeSingleElement) {
    bm.addRange(5, 6);
    EXPECT_EQ(bm.cardinality(), 1u);
    EXPECT_TRUE(bm.contains(5));
}

// ─────────────────────────────────────────────────────────────────────────────
// addMany (CRoaring: test_cpp_add_many)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, AddMany) {
    std::vector<uint32_t> vals = {5, 2, 3, 4, 1, 100, 65536, 200000};
    bm.addMany(vals.data(), vals.size());
    EXPECT_EQ(bm.cardinality(), 8u);
    for (auto v : vals) {
        EXPECT_TRUE(bm.contains(v));
    }
}

TEST_F(RoaringBitmapTest, AddManyDuplicates) {
    std::vector<uint32_t> vals = {5, 5, 5, 3, 3};
    bm.addMany(vals.data(), vals.size());
    EXPECT_EQ(bm.cardinality(), 2u);
}

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
// Empty bitmap operations (CRoaring: boundary tests)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, EmptyBitmapOps) {
    RoaringBitmap empty1, empty2;
    RoaringBitmap full;
    full.addRange(0, 100);

    EXPECT_TRUE((empty1 & empty2).empty());
    EXPECT_TRUE((empty1 | empty2).empty());
    EXPECT_TRUE(empty1.andNot(empty2).empty());
    EXPECT_TRUE((empty1 ^ empty2).empty());

    EXPECT_TRUE((empty1 & full).empty());
    EXPECT_EQ(empty1 | full, full);
    EXPECT_TRUE(empty1.andNot(full).empty());
    EXPECT_EQ(full.andNot(empty1), full);
    EXPECT_EQ(empty1 ^ full, full);
}

// ─────────────────────────────────────────────────────────────────────────────
// optimize() — run-length compression (CRoaring: test_run_compression_cpp)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, OptimizeCreatesRunContainers) {
    bm.addRange(0, 1000);
    bm.addRange(2000, 3000);
    uint32_t cardBefore = bm.cardinality();
    bm.optimize();
    EXPECT_EQ(bm.cardinality(), cardBefore);  // cardinality preserved

    // Verify all values still present.
    for (uint32_t i = 0; i < 1000; ++i) EXPECT_TRUE(bm.contains(i));
    for (uint32_t i = 1000; i < 2000; ++i) EXPECT_FALSE(bm.contains(i));
    for (uint32_t i = 2000; i < 3000; ++i) EXPECT_TRUE(bm.contains(i));
}

TEST_F(RoaringBitmapTest, OptimizeBitmapToRun) {
    // Bitmap container (>4096 elements) with contiguous range → run is more compact.
    bm.addRange(0, 10000);
    EXPECT_EQ(bm.cardinality(), 10000u);
    bm.optimize();
    EXPECT_EQ(bm.cardinality(), 10000u);
    EXPECT_TRUE(bm.contains(0));
    EXPECT_TRUE(bm.contains(9999));
    EXPECT_FALSE(bm.contains(10000));
}

TEST_F(RoaringBitmapTest, OptimizePreservesOpsCorrectness) {
    // After optimize, set operations still work.
    RoaringBitmap a, b;
    a.addRange(0, 500);
    a.addRange(1000, 1500);
    a.optimize();

    b.addRange(250, 1250);
    b.optimize();

    auto intersection = a & b;
    EXPECT_EQ(intersection.cardinality(), 500u);  // [250,500) + [1000,1250)

    auto united = a | b;
    EXPECT_EQ(united.cardinality(), 1500u);  // [0,1500)
}

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
// Remove + cardinality correctness (CRoaring: add_contains_test reverse remove)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RemoveAllForward) {
    for (uint32_t x = 0; x < 1000; x += 3) bm.add(x);
    uint32_t expected = bm.cardinality();
    for (uint32_t x = 0; x < 1000; x += 3) {
        bm.remove(x);
        --expected;
        EXPECT_EQ(bm.cardinality(), expected);
    }
    EXPECT_TRUE(bm.empty());
}

TEST_F(RoaringBitmapTest, RemoveAllReverse) {
    for (uint32_t x = 0; x < 1000; x += 3) bm.add(x);
    uint32_t expected = bm.cardinality();
    for (int32_t x = 999; x >= 0; x -= 3) {
        bm.remove(static_cast<uint32_t>(x));
        --expected;
        EXPECT_EQ(bm.cardinality(), expected);
    }
    EXPECT_TRUE(bm.empty());
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
// Edge cases near UINT32_MAX (CRoaring: test_cpp_remove_run_compression)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, NearMaxUint32) {
    uint32_t base = UINT32_MAX - 100;
    for (uint32_t i = base; i != 0; ++i) {  // wraps around after UINT32_MAX
        bm.add(i);
        if (i == UINT32_MAX) break;
    }
    EXPECT_EQ(bm.cardinality(), 101u);
    EXPECT_TRUE(bm.contains(UINT32_MAX));
    EXPECT_TRUE(bm.contains(UINT32_MAX - 100));
    EXPECT_EQ(bm.maximum(), UINT32_MAX);
    EXPECT_EQ(bm.minimum(), UINT32_MAX - 100);
}

// ─────────────────────────────────────────────────────────────────────────────
// Large-scale validation (CRoaring: sbs_t framework style)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, LargeScaleAddRemoveContains) {
    // Add 20K values, remove half, check all.
    std::vector<uint32_t> vals;
    for (uint32_t i = 0; i < 20000; ++i) vals.push_back(i * 7);

    for (auto v : vals) bm.add(v);
    EXPECT_EQ(bm.cardinality(), 20000u);

    // Remove even-indexed ones.
    for (size_t i = 0; i < vals.size(); i += 2) bm.remove(vals[i]);
    EXPECT_EQ(bm.cardinality(), 10000u);

    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_EQ(bm.contains(vals[i]), (i % 2 == 1));
    }
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
// toplevel_unit.c: roaring_bitmap_from_range with step
// Build ranges with step sizes, verify cardinality.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, FromRangeWithStep) {
    for (uint32_t step : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 16u, 32u}) {
        RoaringBitmap rb;
        uint32_t count = 0;
        for (uint32_t x = 0; x < 100000; x += step) {
            rb.add(x);
            ++count;
        }
        EXPECT_EQ(rb.cardinality(), count) << "step=" << step;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// toplevel_unit.c: statistics validation (min, max, cardinality)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, StatisticsMinMaxCard) {
    bm.add(5);
    bm.add(2);
    bm.add(3);
    bm.add(4);
    bm.add(1);
    EXPECT_EQ(bm.cardinality(), 5u);
    EXPECT_EQ(bm.minimum(), 1u);
    EXPECT_EQ(bm.maximum(), 5u);

    bm.add(100000);
    EXPECT_EQ(bm.maximum(), 100000u);
    EXPECT_EQ(bm.minimum(), 1u);
}

// ─────────────────────────────────────────────────────────────────────────────
// toplevel_unit.c: issue288 — operations on empty bitmaps
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, EmptyBitmapCardinality) {
    EXPECT_EQ(bm.cardinality(), 0u);
    EXPECT_TRUE(bm.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// cpp_unit.cpp: test_cpp_add_remove_checked — add returns idempotent.
// We test via cardinality stability.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, AddIdempotent) {
    bm.add(100);
    EXPECT_EQ(bm.cardinality(), 1u);
    bm.add(100);
    EXPECT_EQ(bm.cardinality(), 1u);
    bm.add(200);
    EXPECT_EQ(bm.cardinality(), 2u);
    bm.remove(100);
    EXPECT_EQ(bm.cardinality(), 1u);
    bm.remove(100);  // already removed
    EXPECT_EQ(bm.cardinality(), 1u);
    bm.remove(999);  // never existed
    EXPECT_EQ(bm.cardinality(), 1u);
}

// ─────────────────────────────────────────────────────────────────────────────
// cpp_unit.cpp: test_cpp_to_string equivalent — toVector round-trip.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, ToVectorRoundTrip) {
    std::vector<uint32_t> input = {1, 2, UINT32_MAX};
    for (uint32_t v : input) bm.add(v);
    auto output = bm.toVector();
    EXPECT_EQ(output, input);
}

// ─────────────────────────────────────────────────────────────────────────────
// cpp_unit.cpp: serial_test equivalent — add 5 values, verify toVector order.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, InsertOrderIndependence) {
    // Add out of order.
    bm.add(5);
    bm.add(2);
    bm.add(3);
    bm.add(4);
    bm.add(1);
    auto vec = bm.toVector();
    std::vector<uint32_t> expected = {1, 2, 3, 4, 5};
    EXPECT_EQ(vec, expected);
}

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
// toplevel_unit.c: convert_all_containers — verify adding to different
// container types preserves correctness after type transitions.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, ContainerTypeTransitions) {
    // Start as array.
    for (uint32_t i = 0; i < 4000; ++i) bm.add(i * 2);
    EXPECT_EQ(bm.cardinality(), 4000u);

    // Still array (4000 < 4096). Promote by adding more.
    for (uint32_t i = 4000; i < 4200; ++i) bm.add(i * 2);
    EXPECT_EQ(bm.cardinality(), 4200u);
    // Now bitmap (>4096).

    // Optimize to run.
    bm.optimize();
    EXPECT_EQ(bm.cardinality(), 4200u);

    // Verify all values.
    for (uint32_t i = 0; i < 4200; ++i) {
        EXPECT_TRUE(bm.contains(i * 2));
        EXPECT_FALSE(bm.contains(i * 2 + 1));
    }

    // Add single value to run container.
    bm.add(1);
    EXPECT_TRUE(bm.contains(1));
    EXPECT_EQ(bm.cardinality(), 4201u);
}

// ─────────────────────────────────────────────────────────────────────────────
// toplevel_unit.c: addRange on bitmap container — verify word-level setRange.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, AddRangeOnExistingBitmap) {
    // First, create a bitmap container.
    bm.addRange(0, 5000);
    EXPECT_EQ(bm.cardinality(), 5000u);

    // Add overlapping range.
    bm.addRange(4000, 10000);
    EXPECT_EQ(bm.cardinality(), 10000u);

    for (uint32_t i = 0; i < 10000; ++i) {
        EXPECT_TRUE(bm.contains(i));
    }
    EXPECT_FALSE(bm.contains(10000));
}

TEST_F(RoaringBitmapTest, AddRangeOnExistingArray) {
    bm.addRange(0, 100);
    bm.addRange(50, 200);
    EXPECT_EQ(bm.cardinality(), 200u);

    for (uint32_t i = 0; i < 200; ++i) {
        EXPECT_TRUE(bm.contains(i));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// toplevel_unit.c: issue660 — bitmap containing only 0.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, BitmapContainingOnlyZero) {
    bm.add(0);
    EXPECT_EQ(bm.cardinality(), 1u);
    EXPECT_TRUE(bm.contains(0));
    EXPECT_EQ(bm.minimum(), 0u);
    EXPECT_EQ(bm.maximum(), 0u);

    auto vec = bm.toVector();
    ASSERT_EQ(vec.size(), 1u);
    EXPECT_EQ(vec[0], 0u);
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

// ─────────────────────────────────────────────────────────────────────────────
// Copy independence test
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, CopyIndependence) {
    bm.addRange(10, 20);
    bm.add(65536);
    bm.add(131072);

    RoaringBitmap copy = bm;
    EXPECT_EQ(copy, bm);

    // Modify original.
    bm.add(999);
    bm.remove(65536);
    bm.removeRange(10, 15);

    // Copy should be unchanged.
    EXPECT_NE(copy, bm);
    EXPECT_EQ(copy.cardinality(), 12u);
    EXPECT_TRUE(copy.containsRange(10, 20));
    EXPECT_TRUE(copy.contains(65536));
    EXPECT_TRUE(copy.contains(131072));
    EXPECT_FALSE(copy.contains(999));
}

TEST_F(RoaringBitmapTest, CopyIndependenceBitmapContainer) {
    // Create bitmap container (> 4096 elements).
    for (uint32_t i = 0; i < 5000; ++i) bm.add(i);

    RoaringBitmap copy = bm;
    EXPECT_EQ(copy, bm);

    bm.removeRange(0, 5000);
    EXPECT_TRUE(bm.empty());
    EXPECT_EQ(copy.cardinality(), 5000u);
}

// ── SIMD Kernel Correctness Tests ────────────────────────────────────────────

TEST_F(RoaringBitmapTest, SIMDPopcount) {
    // Test empty, full, and patterned bitmaps
    uint64_t zeros[1024] = {};
    EXPECT_EQ(arrow::simd::bitmap_popcount(zeros), 0u);

    uint64_t ones[1024];
    memset(ones, 0xFF, sizeof(ones));
    EXPECT_EQ(arrow::simd::bitmap_popcount(ones), 65536u);

    // Every other bit
    uint64_t alt[1024];
    for (int i = 0; i < 1024; ++i) alt[i] = 0xAAAAAAAAAAAAAAAAULL;
    EXPECT_EQ(arrow::simd::bitmap_popcount(alt), 32768u);
}

TEST_F(RoaringBitmapTest, SIMDAndPopcount) {
    uint64_t a[1024], b[1024], out[1024];
    // a = every-other-bit, b = every-other-bit shifted
    for (int i = 0; i < 1024; ++i) {
        a[i] = 0xAAAAAAAAAAAAAAAAULL;
        b[i] = 0x5555555555555555ULL;
    }
    EXPECT_EQ(arrow::simd::bitmap_and_popcount(a, b, out), 0u); // no overlap

    // a AND a = a
    EXPECT_EQ(arrow::simd::bitmap_and_popcount(a, a, out), 32768u);
    for (int i = 0; i < 1024; ++i) EXPECT_EQ(out[i], a[i]);
}

TEST_F(RoaringBitmapTest, SIMDOrPopcount) {
    uint64_t a[1024], b[1024], out[1024];
    for (int i = 0; i < 1024; ++i) {
        a[i] = 0xAAAAAAAAAAAAAAAAULL;
        b[i] = 0x5555555555555555ULL;
    }
    EXPECT_EQ(arrow::simd::bitmap_or_popcount(a, b, out), 65536u); // all bits set
    for (int i = 0; i < 1024; ++i) EXPECT_EQ(out[i], ~0ULL);
}

TEST_F(RoaringBitmapTest, SIMDXorPopcount) {
    uint64_t a[1024], b[1024], out[1024];
    for (int i = 0; i < 1024; ++i) {
        a[i] = 0xFFFFFFFFFFFFFFFFULL;
        b[i] = 0xFFFFFFFFFFFFFFFFULL;
    }
    EXPECT_EQ(arrow::simd::bitmap_xor_popcount(a, b, out), 0u); // all cancel

    // XOR with complement = all ones
    for (int i = 0; i < 1024; ++i) {
        a[i] = 0xAAAAAAAAAAAAAAAAULL;
        b[i] = 0x5555555555555555ULL;
    }
    EXPECT_EQ(arrow::simd::bitmap_xor_popcount(a, b, out), 65536u);
}

TEST_F(RoaringBitmapTest, SIMDAndNotPopcount) {
    uint64_t a[1024], b[1024], out[1024];
    // a & ~b where a=all, b=half -> half remain
    for (int i = 0; i < 1024; ++i) {
        a[i] = 0xFFFFFFFFFFFFFFFFULL;
        b[i] = 0xAAAAAAAAAAAAAAAAULL;
    }
    EXPECT_EQ(arrow::simd::bitmap_andnot_popcount(a, b, out), 32768u);
    for (int i = 0; i < 1024; ++i) EXPECT_EQ(out[i], 0x5555555555555555ULL);
}

TEST_F(RoaringBitmapTest, SIMDBitmapEqual) {
    uint64_t a[1024], b[1024];
    memset(a, 0xAB, sizeof(a));
    memset(b, 0xAB, sizeof(b));
    EXPECT_TRUE(arrow::simd::bitmap_equal(a, b));

    b[512] ^= 1; // flip one bit
    EXPECT_FALSE(arrow::simd::bitmap_equal(a, b));

    b[512] ^= 1; // flip back
    b[0] ^= (1ULL << 63); // flip MSB of first word
    EXPECT_FALSE(arrow::simd::bitmap_equal(a, b));

    b[0] ^= (1ULL << 63);
    b[1023] ^= 1; // flip LSB of last word
    EXPECT_FALSE(arrow::simd::bitmap_equal(a, b));
}

TEST_F(RoaringBitmapTest, SIMDAllZeros) {
    uint64_t z[1024] = {};
    uint64_t out[1024];
    EXPECT_EQ(arrow::simd::bitmap_and_popcount(z, z, out), 0u);
    EXPECT_EQ(arrow::simd::bitmap_or_popcount(z, z, out), 0u);
    EXPECT_EQ(arrow::simd::bitmap_xor_popcount(z, z, out), 0u);
    EXPECT_EQ(arrow::simd::bitmap_andnot_popcount(z, z, out), 0u);
    EXPECT_TRUE(arrow::simd::bitmap_equal(z, out));
}

TEST_F(RoaringBitmapTest, SIMDAllOnes) {
    uint64_t ones[1024];
    memset(ones, 0xFF, sizeof(ones));
    uint64_t out[1024];
    EXPECT_EQ(arrow::simd::bitmap_and_popcount(ones, ones, out), 65536u);
    EXPECT_EQ(arrow::simd::bitmap_or_popcount(ones, ones, out), 65536u);
    EXPECT_EQ(arrow::simd::bitmap_xor_popcount(ones, ones, out), 0u);
    EXPECT_EQ(arrow::simd::bitmap_andnot_popcount(ones, ones, out), 0u);
}

TEST_F(RoaringBitmapTest, SIMDRandomVerification) {
    // Generate random bitmaps, verify SIMD matches scalar
    uint64_t a[1024], b[1024], simd_out[1024], scalar_out[1024];
    uint64_t seed = 0xDEADBEEF12345678ULL;
    auto splitmix = [&seed]() {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (int i = 0; i < 1024; ++i) { a[i] = splitmix(); b[i] = splitmix(); }

    // Scalar reference
    uint32_t scalar_and = 0, scalar_or = 0, scalar_xor = 0, scalar_andnot = 0;
    for (int i = 0; i < 1024; ++i) {
        scalar_and += __builtin_popcountll(a[i] & b[i]);
        scalar_or += __builtin_popcountll(a[i] | b[i]);
        scalar_xor += __builtin_popcountll(a[i] ^ b[i]);
        scalar_andnot += __builtin_popcountll(a[i] & ~b[i]);
    }

    EXPECT_EQ(arrow::simd::bitmap_and_popcount(a, b, simd_out), scalar_and);
    EXPECT_EQ(arrow::simd::bitmap_or_popcount(a, b, simd_out), scalar_or);
    EXPECT_EQ(arrow::simd::bitmap_xor_popcount(a, b, simd_out), scalar_xor);
    EXPECT_EQ(arrow::simd::bitmap_andnot_popcount(a, b, simd_out), scalar_andnot);

    // Verify output arrays match scalar
    for (int i = 0; i < 1024; ++i) scalar_out[i] = a[i] & b[i];
    arrow::simd::bitmap_and_popcount(a, b, simd_out);
    EXPECT_EQ(memcmp(simd_out, scalar_out, sizeof(simd_out)), 0);
}

// This verifies that the SIMD integration into RoaringBitmap produces correct results
TEST_F(RoaringBitmapTest, SIMDConsistencyWithTopLevel) {
    // Create two bitmaps that will use bitmap containers (>4096 elements in one chunk)
    RoaringBitmap a, b;
    a.addRange(0, 10000);  // bitmap container
    b.addRange(5000, 15000); // bitmap container

    auto rAnd = a & b;
    auto rOr = a | b;
    auto rXor = a ^ b;
    auto rAndNot = a - b;

    EXPECT_EQ(rAnd.cardinality(), 5000u);  // [5000,10000)
    EXPECT_EQ(rOr.cardinality(), 15000u);  // [0,15000)
    EXPECT_EQ(rXor.cardinality(), 10000u); // [0,5000) + [10000,15000)
    EXPECT_EQ(rAndNot.cardinality(), 5000u); // [0,5000)

    // Verify specific elements
    EXPECT_TRUE(rAnd.contains(5000));
    EXPECT_TRUE(rAnd.contains(9999));
    EXPECT_FALSE(rAnd.contains(4999));
    EXPECT_FALSE(rAnd.contains(10000));
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

// ─────────────────────────────────────────────────────────────────────────────
// addOffset tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, AddOffsetPositiveWithinChunk) {
    bm.add(10); bm.add(20); bm.add(30);
    auto shifted = bm.addOffset(5);
    EXPECT_EQ(shifted.cardinality(), 3u);
    EXPECT_TRUE(shifted.contains(15));
    EXPECT_TRUE(shifted.contains(25));
    EXPECT_TRUE(shifted.contains(35));
}

TEST_F(RoaringBitmapTest, AddOffsetCrossingChunks) {
    bm.add(65530);  // near end of chunk 0
    auto shifted = bm.addOffset(10);
    EXPECT_EQ(shifted.cardinality(), 1u);
    EXPECT_TRUE(shifted.contains(65540));  // now in chunk 1
}

TEST_F(RoaringBitmapTest, AddOffsetNegative) {
    bm.add(100); bm.add(200);
    auto shifted = bm.addOffset(-50);
    EXPECT_TRUE(shifted.contains(50));
    EXPECT_TRUE(shifted.contains(150));
}

TEST_F(RoaringBitmapTest, AddOffsetOverflowDrops) {
    bm.add(UINT32_MAX - 5);
    bm.add(100);
    auto shifted = bm.addOffset(10);
    // UINT32_MAX-5+10 overflows, should be dropped
    EXPECT_EQ(shifted.cardinality(), 1u);
    EXPECT_TRUE(shifted.contains(110));
}

TEST_F(RoaringBitmapTest, AddOffsetUnderflowDrops) {
    bm.add(5); bm.add(100);
    auto shifted = bm.addOffset(-10);
    // 5 - 10 = -5, underflows, should be dropped
    EXPECT_EQ(shifted.cardinality(), 1u);
    EXPECT_TRUE(shifted.contains(90));
}

TEST_F(RoaringBitmapTest, AddOffsetZeroIsIdentity) {
    bm.add(10); bm.add(20); bm.add(70000);
    auto shifted = bm.addOffset(0);
    EXPECT_EQ(shifted.toVector(), bm.toVector());
}

// ─────────────────────────────────────────────────────────────────────────────
// statistics tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, StatisticsEmpty) {
    auto s = bm.statistics();
    EXPECT_EQ(s.numContainers, 0u);
    EXPECT_EQ(s.numValues, 0u);
}

TEST_F(RoaringBitmapTest, StatisticsMixedContainers) {
    // Array container: few elements.
    bm.add(1); bm.add(2); bm.add(3);
    // Bitmap container: >4096 elements in one chunk.
    bm.addRange(65536, 65536 + 5000);
    // Run container via optimize.
    bm.addRange(131072, 131072 + 1000);  // chunk 2
    bm.optimize();

    auto s = bm.statistics();
    EXPECT_EQ(s.numContainers, 3u);
    EXPECT_EQ(s.numValues, 3u + 5000u + 1000u);
    EXPECT_EQ(s.minValue, 1u);
    EXPECT_EQ(s.maxValue, 131072u + 999u);
    // At least one run container from optimize.
    EXPECT_GE(s.numRunContainers, 1u);
    EXPECT_GE(s.numArrayContainers + s.numBitmapContainers + s.numRunContainers, 3u);
}

// ─────────────────────────────────────────────────────────────────────────────
// removeRunCompression tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, RemoveRunCompressionNoRunContainers) {
    bm.addRange(0, 1000);
    bm.optimize();
    auto before = bm.toVector();
    auto sBefore = bm.statistics();
    EXPECT_GE(sBefore.numRunContainers, 1u);

    bm.removeRunCompression();
    auto sAfter = bm.statistics();
    EXPECT_EQ(sAfter.numRunContainers, 0u);

    // Data should be unchanged.
    EXPECT_EQ(bm.toVector(), before);
}

TEST_F(RoaringBitmapTest, RemoveRunCompressionDataPreserved) {
    bm.addRange(0, 10000);
    bm.optimize();
    auto before = bm.toVector();

    bm.removeRunCompression();
    EXPECT_EQ(bm.toVector(), before);
    EXPECT_EQ(bm.cardinality(), 10000u);
    auto s = bm.statistics();
    EXPECT_EQ(s.numRunContainers, 0u);
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

// ─────────────────────────────────────────────────────────────────────────────
// SIMD bitmap_to_array test
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, SIMDBitmapToArray) {
    BitmapContainer bc;
    bc.add(0); bc.add(1); bc.add(63); bc.add(64);
    bc.add(1000); bc.add(65535);
    bc.computeCardinality();

    uint16_t out[65536];
    uint32_t count = simd::bitmap_to_array(bc.words, out);

    auto arr = bc.toArray();
    EXPECT_EQ(count, arr.cardinality());
    for (uint32_t i = 0; i < count; ++i) {
        EXPECT_EQ(out[i], arr.values[i]);
    }
}

TEST_F(RoaringBitmapTest, SIMDBitmapToArrayDense) {
    BitmapContainer bc;
    bc.setRange(0, 10000);
    bc.computeCardinality();

    std::vector<uint16_t> out(bc.cardinality());
    uint32_t count = simd::bitmap_to_array(bc.words, out.data());

    EXPECT_EQ(count, 10000u);
    for (uint32_t i = 0; i < count; ++i) {
        EXPECT_EQ(out[i], static_cast<uint16_t>(i));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD bitmap_not_popcount test
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, SIMDBitmapNotPopcount) {
    BitmapContainer bc;
    bc.setRange(0, 10000);
    bc.computeCardinality();

    uint64_t out[kBitmapWords];
    uint32_t notCount = simd::bitmap_not_popcount(bc.words, out);

    // NOT of 10000 set bits out of 65536 = 55536.
    EXPECT_EQ(notCount, 65536u - 10000u);

    // Verify against scalar.
    uint32_t scalarCount = 0;
    for (uint32_t i = 0; i < kBitmapWords; ++i) {
        EXPECT_EQ(out[i], ~bc.words[i]);
        scalarCount += static_cast<uint32_t>(__builtin_popcountll(~bc.words[i]));
    }
    EXPECT_EQ(notCount, scalarCount);
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, SerializeDeserializeEmpty) {
    auto data = bm.serialize();
    auto restored = RoaringBitmap::deserialize(data.data(), data.size());
    ASSERT_TRUE(restored.has_value());
    EXPECT_TRUE(restored->empty());
    EXPECT_EQ(restored->cardinality(), 0u);
}

TEST_F(RoaringBitmapTest, SerializeDeserializeArrayOnly) {
    for (uint32_t i = 0; i < 100; ++i) bm.add(i * 7);
    auto data = bm.serialize();
    auto restored = RoaringBitmap::deserialize(data.data(), data.size());
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(*restored, bm);
    EXPECT_EQ(restored->cardinality(), bm.cardinality());
}

TEST_F(RoaringBitmapTest, SerializeDeserializeBitmapOnly) {
    bm.addRange(0, 10000);  // > 4096 → bitmap container
    auto data = bm.serialize();
    auto restored = RoaringBitmap::deserialize(data.data(), data.size());
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(*restored, bm);
}

TEST_F(RoaringBitmapTest, SerializeDeserializeRunContainers) {
    bm.addRange(0, 5000);
    bm.optimize();  // should convert to run container
    auto data = bm.serialize();
    auto restored = RoaringBitmap::deserialize(data.data(), data.size());
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(restored->cardinality(), bm.cardinality());
    EXPECT_EQ(restored->toVector(), bm.toVector());
}

TEST_F(RoaringBitmapTest, SerializeDeserializeMixedContainers) {
    // Array container (chunk 0): sparse elements.
    for (uint32_t i = 0; i < 100; ++i) bm.add(i * 3);
    // Bitmap container (chunk 1): dense elements.
    bm.addRange(65536, 75536);
    // Run container (chunk 2): range, then optimize.
    bm.addRange(131072, 136072);
    bm.optimize();

    auto data = bm.serialize();
    auto restored = RoaringBitmap::deserialize(data.data(), data.size());
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(restored->cardinality(), bm.cardinality());
    EXPECT_EQ(restored->toVector(), bm.toVector());
}

TEST_F(RoaringBitmapTest, SerializeDeserializeMultiChunk) {
    // Span many different high-16 keys.
    for (uint32_t chunk = 0; chunk < 10; ++chunk) {
        uint32_t base = chunk * 65536;
        for (uint32_t i = 0; i < 50; ++i) bm.add(base + i * 100);
    }
    auto data = bm.serialize();
    auto restored = RoaringBitmap::deserialize(data.data(), data.size());
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(*restored, bm);
}

TEST_F(RoaringBitmapTest, DeserializeRejectsTruncatedCookie) {
    uint8_t data[2] = {0, 0};
    EXPECT_FALSE(RoaringBitmap::deserialize(data, 2).has_value());
}

TEST_F(RoaringBitmapTest, DeserializeRejectsInvalidCookie) {
    uint8_t data[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0};
    EXPECT_FALSE(RoaringBitmap::deserialize(data, 8).has_value());
}

TEST_F(RoaringBitmapTest, DeserializeRejectsTruncatedData) {
    bm.addRange(0, 10000);
    auto data = bm.serialize();
    // Truncate at various points.
    for (size_t cutAt : {size_t(4), size_t(8), size_t(12), data.size() / 2, data.size() - 1}) {
        if (cutAt < data.size()) {
            EXPECT_FALSE(RoaringBitmap::deserialize(data.data(), cutAt).has_value())
                << "Should reject truncation at byte " << cutAt;
        }
    }
}

TEST_F(RoaringBitmapTest, DeserializeRejectsInvalidContainerCount) {
    // Cookie=12346, then an absurdly large container count.
    uint8_t data[8];
    data[0] = 0x2A; data[1] = 0x30; data[2] = 0; data[3] = 0;  // 12346
    data[4] = 0xFF; data[5] = 0xFF; data[6] = 0xFF; data[7] = 0xFF;  // ~4 billion
    EXPECT_FALSE(RoaringBitmap::deserialize(data, 8).has_value());
}

TEST_F(RoaringBitmapTest, DeserializeNullptr) {
    EXPECT_FALSE(RoaringBitmap::deserialize(nullptr, 100).has_value());
}

TEST_F(RoaringBitmapTest, FrozenSerializeDeserializeRoundTrip) {
    bm.addRange(0, 10000);
    for (uint32_t i = 0; i < 50; ++i) bm.add(65536 + i * 100);
    bm.addRange(131072, 136072);
    bm.optimize();

    auto data = bm.serializeFrozen();
    auto restored = RoaringBitmap::deserializeFrozen(data.data(), data.size());
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(restored->cardinality(), bm.cardinality());
    EXPECT_EQ(restored->toVector(), bm.toVector());
}

TEST_F(RoaringBitmapTest, FrozenDeserializeRejectsInvalidMagic) {
    uint8_t data[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_FALSE(RoaringBitmap::deserializeFrozen(data, 9).has_value());
}

TEST_F(RoaringBitmapTest, FrozenDeserializeRejectsTruncated) {
    bm.addRange(0, 100);
    auto data = bm.serializeFrozen();
    EXPECT_FALSE(RoaringBitmap::deserializeFrozen(data.data(), 4).has_value());
}

TEST_F(RoaringBitmapTest, FrozenSerializeDeserializeEmpty) {
    auto data = bm.serializeFrozen();
    auto restored = RoaringBitmap::deserializeFrozen(data.data(), data.size());
    ASSERT_TRUE(restored.has_value());
    EXPECT_TRUE(restored->empty());
}

TEST_F(RoaringBitmapTest, SizeInBytesMatchesSerialize) {
    // Array container.
    for (uint32_t i = 0; i < 100; ++i) bm.add(i * 7);
    EXPECT_EQ(bm.sizeInBytes(), bm.serialize().size());

    // Add bitmap container.
    bm.addRange(65536, 75536);
    EXPECT_EQ(bm.sizeInBytes(), bm.serialize().size());

    // Add run container.
    bm.addRange(131072, 136072);
    bm.optimize();
    EXPECT_EQ(bm.sizeInBytes(), bm.serialize().size());
}

TEST_F(RoaringBitmapTest, SizeInBytesEmptyBitmap) {
    EXPECT_EQ(bm.sizeInBytes(), bm.serialize().size());
}

TEST_F(RoaringBitmapTest, SerializeDeserializeThenModifyReserialize) {
    // Serialize with array container.
    for (uint32_t i = 0; i < 100; ++i) bm.add(i);
    auto data1 = bm.serialize();
    auto r1 = RoaringBitmap::deserialize(data1.data(), data1.size());
    ASSERT_TRUE(r1.has_value());

    // Promote to bitmap by adding more elements.
    for (uint32_t i = 100; i < 10000; ++i) r1->add(i);

    // Re-serialize and round-trip again.
    auto data2 = r1->serialize();
    auto r2 = RoaringBitmap::deserialize(data2.data(), data2.size());
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2->cardinality(), 10000u);
    EXPECT_EQ(*r1, *r2);
}

TEST_F(RoaringBitmapTest, SIMDKernelHarleySealPopcount) {
    // Fill a bitmap container and verify harley_seal matches regular popcount
    uint64_t words[1024] = {};
    // Set various patterns
    for (int i = 0; i < 1024; ++i) words[i] = 0xAAAAAAAAAAAAAAAAULL;
    EXPECT_EQ(simd::bitmap_popcount(words), simd::bitmap_popcount_harley_seal(words));

    // All ones
    for (int i = 0; i < 1024; ++i) words[i] = ~0ULL;
    EXPECT_EQ(simd::bitmap_popcount(words), simd::bitmap_popcount_harley_seal(words));
    EXPECT_EQ(simd::bitmap_popcount_harley_seal(words), 65536u);

    // All zeros
    for (int i = 0; i < 1024; ++i) words[i] = 0;
    EXPECT_EQ(simd::bitmap_popcount_harley_seal(words), 0u);

    // Random-ish pattern
    for (int i = 0; i < 1024; ++i) words[i] = static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL;
    EXPECT_EQ(simd::bitmap_popcount(words), simd::bitmap_popcount_harley_seal(words));
}

TEST_F(RoaringBitmapTest, SIMDKernelBitmapNotNocard) {
    uint64_t input[1024], output[1024];
    for (int i = 0; i < 1024; ++i) input[i] = static_cast<uint64_t>(i);
    simd::bitmap_not_nocard(input, output);
    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(output[i], ~input[i]);
    }

    // All zeros -> all ones
    std::memset(input, 0, sizeof(input));
    simd::bitmap_not_nocard(input, output);
    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(output[i], ~0ULL);
    }
}

TEST_F(RoaringBitmapTest, SIMDKernelBitmapContainerAlignment) {
    // Verify BitmapContainer words are 64-byte aligned
    BitmapContainer bc;
    auto addr = reinterpret_cast<uintptr_t>(static_cast<const uint64_t*>(bc.words));
    EXPECT_EQ(addr % 64, 0u) << "BitmapContainer::words should be 64-byte aligned";
}

TEST_F(RoaringBitmapTest, APIGapOperatorEqualNativeComparison) {
    // Two bitmap containers should use bitmap_equal, not vector materialization
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 10000; ++i) { a.add(i); b.add(i); }
    EXPECT_EQ(a, b);
    b.add(10001);
    EXPECT_NE(a, b);
}

TEST_F(RoaringBitmapTest, APIGapOptimizeReturnsBool) {
    // Consecutive values should be compressible to runs
    for (uint32_t i = 0; i < 5000; ++i) bm.add(i);
    EXPECT_TRUE(bm.optimize());
    // Already optimized, should return false
    EXPECT_FALSE(bm.optimize());
}

TEST_F(RoaringBitmapTest, APIGapShrinkToFit) {
    for (uint32_t i = 0; i < 100; ++i) bm.add(i);
    for (uint32_t i = 50; i < 100; ++i) bm.remove(i);
    bm.shrinkToFit();  // Should not crash, should reclaim capacity
    EXPECT_EQ(bm.cardinality(), 50u);
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

TEST_F(RoaringBitmapTest, APIGapToString) {
    bm.add(1); bm.add(3); bm.add(5);
    std::string s = bm.toString();
    EXPECT_EQ(s, "{1, 3, 5}");

    RoaringBitmap empty;
    EXPECT_EQ(empty.toString(), "{}");
}

TEST_F(RoaringBitmapTest, APIGapAddRangeFullDomain) {
    // Test that uint64_t max allows full domain
    bm.addRange(0, 0x10000ULL);  // full first chunk
    EXPECT_EQ(bm.cardinality(), 65536u);
    EXPECT_TRUE(bm.contains(0));
    EXPECT_TRUE(bm.contains(65535));
}

TEST_F(RoaringBitmapTest, APIGapRemoveRangeUint64) {
    bm.addRange(0, 0x10000ULL);
    bm.removeRange(0, 0x10000ULL);
    EXPECT_TRUE(bm.empty());
}

TEST_F(RoaringBitmapTest, APIGapStatisticsExpanded) {
    // Array container
    for (uint32_t i = 0; i < 100; ++i) bm.add(i);
    // Bitmap container (different chunk)
    for (uint32_t i = 0x10000; i < 0x10000 + 5000; ++i) bm.add(i);

    auto stats = bm.statistics();
    EXPECT_EQ(stats.numContainers, 2u);
    EXPECT_EQ(stats.numArrayContainers, 1u);
    EXPECT_EQ(stats.numBitmapContainers, 1u);
    EXPECT_EQ(stats.numValuesArrayContainers, 100u);
    EXPECT_EQ(stats.numValuesBitmapContainers, 5000u);
    EXPECT_GT(stats.numBytesArrayContainers, 0u);
    EXPECT_GT(stats.numBytesBitmapContainers, 0u);
    EXPECT_GT(stats.sumValue, 0u);
    EXPECT_EQ(stats.minValue, 0u);
    EXPECT_EQ(stats.maxValue, 0x10000u + 4999);
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

TEST_F(RoaringBitmapTest, InfraGapFrozenV2RoundTrip) {
    // Test round-trip with new v2 format
    for (uint32_t i = 0; i < 100; ++i) bm.add(i);
    for (uint32_t i = 0x10000; i < 0x10000 + 5000; ++i) bm.add(i);
    bm.optimize();  // Creates a run container

    auto frozen = bm.serializeFrozen();
    auto restored = RoaringBitmap::deserializeFrozen(frozen.data(), frozen.size());
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(*restored, bm);
}

TEST_F(RoaringBitmapTest, InfraGapFrozenV2AllContainerTypes) {
    // Array container
    for (uint32_t i = 0; i < 100; ++i) bm.add(i * 3);
    // Bitmap container
    for (uint32_t i = 0x10000; i < 0x10000 + 5000; ++i) bm.add(i);
    // Run container (optimize after adding consecutive range)
    for (uint32_t i = 0x20000; i < 0x20000 + 1000; ++i) bm.add(i);
    bm.optimize();

    auto frozen = bm.serializeFrozen();
    auto restored = RoaringBitmap::deserializeFrozen(frozen.data(), frozen.size());
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(restored->cardinality(), bm.cardinality());
    EXPECT_EQ(*restored, bm);
}

TEST_F(RoaringBitmapTest, InfraGapFrozenV2RejectsInvalid) {
    // Wrong magic
    uint8_t bad[] = {0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00};
    EXPECT_FALSE(RoaringBitmap::deserializeFrozen(bad, sizeof(bad)).has_value());

    // Truncated
    auto frozen = bm.serializeFrozen();
    EXPECT_FALSE(RoaringBitmap::deserializeFrozen(frozen.data(), 4).has_value());

    // nullptr
    EXPECT_FALSE(RoaringBitmap::deserializeFrozen(nullptr, 100).has_value());
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
// Gap 11: SIMD bitmap_set/clear/flip_list — batch bit manipulation
// (tested through array-bitmap conversion and mixed container ops)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, BitmapSetListViaBulkAdd) {
    // bitmap_set_list is called during arrayToBitmap promotion
    // Add exactly 4097 elements to force promotion
    for (uint32_t i = 0; i < 4097; ++i)
        bm.add(i);
    EXPECT_EQ(bm.cardinality(), 4097u);
    for (uint32_t i = 0; i < 4097; ++i)
        EXPECT_TRUE(bm.contains(i));
    EXPECT_FALSE(bm.contains(4097));
}

TEST_F(RoaringBitmapTest, BitmapClearListViaRemoval) {
    // Add enough to promote to bitmap, then remove some
    for (uint32_t i = 0; i < 5000; ++i) bm.add(i);
    for (uint32_t i = 0; i < 100; ++i) bm.remove(i);
    EXPECT_EQ(bm.cardinality(), 4900u);
    for (uint32_t i = 0; i < 100; ++i)
        EXPECT_FALSE(bm.contains(i));
    for (uint32_t i = 100; i < 5000; ++i)
        EXPECT_TRUE(bm.contains(i));
}

TEST_F(RoaringBitmapTest, BitmapFlipListViaXor) {
    // XOR with an array container exercises bitmap_flip_list
    RoaringBitmap dense;
    for (uint32_t i = 0; i < 8000; ++i) dense.add(i);

    RoaringBitmap sparse;
    for (uint32_t i = 0; i < 100; ++i) sparse.add(i);

    auto result = dense ^ sparse;
    // XOR: elements in dense but not sparse (100..7999) + elements in sparse but not dense (none)
    EXPECT_EQ(result.cardinality(), 7900u);
    EXPECT_FALSE(result.contains(0));
    EXPECT_FALSE(result.contains(99));
    EXPECT_TRUE(result.contains(100));
    EXPECT_TRUE(result.contains(7999));
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

// ═════════════════════════════════════════════════════════════════════════════
// Gap 15: Frozen serialization / zero-copy view
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, FrozenRoundTrip) {
    for (uint32_t i = 0; i < 100; ++i) bm.add(i * 3);
    bm.addRange(65536, 70000);  // multi-chunk, dense

    auto frozen = bm.serializeFrozen();
    auto restored = RoaringBitmap::deserializeFrozen(frozen.data(), frozen.size());
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(*restored, bm);
}

TEST_F(RoaringBitmapTest, FrozenViewContains) {
    for (uint32_t i = 0; i < 200; ++i) bm.add(i * 7);

    auto frozen = bm.serializeFrozen();
    auto view = RoaringBitmapView::fromFrozen(frozen.data(), frozen.size());
    ASSERT_TRUE(view.has_value());

    EXPECT_EQ(view->cardinality(), bm.cardinality());
    for (uint32_t i = 0; i < 1500; ++i)
        EXPECT_EQ(view->contains(i), bm.contains(i));
}

TEST_F(RoaringBitmapTest, FrozenViewMaterialize) {
    bm.addRange(0, 5000);
    bm.addRange(65536, 66000);

    auto frozen = bm.serializeFrozen();
    auto view = RoaringBitmapView::fromFrozen(frozen.data(), frozen.size());
    ASSERT_TRUE(view.has_value());

    auto materialized = view->toRoaringBitmap();
    EXPECT_EQ(materialized, bm);
}

TEST_F(RoaringBitmapTest, FrozenViewInvalidData) {
    std::vector<uint8_t> garbage = {0, 1, 2, 3};
    auto view = RoaringBitmapView::fromFrozen(garbage.data(), garbage.size());
    EXPECT_FALSE(view.has_value());
}

// ═════════════════════════════════════════════════════════════════════════════
// Additional coverage: serialize round-trip (all container types)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, SerializeRoundTripAllTypes) {
    // Array container
    for (uint32_t i = 0; i < 100; ++i) bm.add(i * 5);
    // Bitmap container (different chunk)
    for (uint32_t i = 0; i < 5000; ++i) bm.add(65536 + i);
    // Run container (another chunk)
    bm.addRange(131072, 132000);
    bm.optimize();

    auto data = bm.serialize();
    auto restored = RoaringBitmap::deserialize(data.data(), data.size());
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(*restored, bm);
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

// ═════════════════════════════════════════════════════════════════════════════
// Gap 20: BitmapContainer::empty() fast path
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, EmptyAfterBitmapMutations) {
    // Build a bitmap container, then remove all elements
    for (uint32_t i = 0; i < 5000; ++i) bm.add(i);
    EXPECT_FALSE(bm.empty());
    for (uint32_t i = 0; i < 5000; ++i) bm.remove(i);
    EXPECT_TRUE(bm.empty());
    EXPECT_EQ(bm.cardinality(), 0u);
}

// ═════════════════════════════════════════════════════════════════════════════
// Gap 16: Run optimization heuristic
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(RoaringBitmapTest, OptimizeConvertsToRun) {
    // Dense consecutive range should become RunContainer
    bm.addRange(0, 10000);
    bool changed = bm.optimize();
    EXPECT_TRUE(changed);
    EXPECT_EQ(bm.cardinality(), 10000u);
    for (uint32_t i = 0; i < 10000; ++i) EXPECT_TRUE(bm.contains(i));
}

TEST_F(RoaringBitmapTest, OptimizePreservesData) {
    // Random-ish data shouldn't lose anything
    for (uint32_t i = 0; i < 2000; ++i) bm.add(i * 7);
    uint32_t cardBefore = bm.cardinality();
    bm.optimize();
    EXPECT_EQ(bm.cardinality(), cardBefore);
    for (uint32_t i = 0; i < 2000; ++i) EXPECT_TRUE(bm.contains(i * 7));
}

}  // namespace arrow
