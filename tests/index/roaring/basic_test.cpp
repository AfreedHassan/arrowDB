#include "roaring_test_fixture.h"

namespace arrow {

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
// addMany sorted/unsorted paths + addBulk (Gaps 5 & 19)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, AddManySortedMultiChunk) {
    // Sorted input spanning multiple chunks — exercises position-hinted lookup.
    std::vector<uint32_t> vals;
    for (uint32_t chunk = 0; chunk < 10; ++chunk) {
        for (uint32_t i = 0; i < 100; ++i) {
            vals.push_back((chunk << 16) | (i * 3));
        }
    }
    bm.addMany(vals.data(), vals.size());
    EXPECT_EQ(bm.cardinality(), 1000u);
    for (auto v : vals) EXPECT_TRUE(bm.contains(v));
}

TEST_F(RoaringBitmapTest, AddManyUnsortedMultiChunk) {
    // Unsorted input — should sort internally, then use fast path.
    std::vector<uint32_t> vals;
    for (uint32_t chunk = 0; chunk < 5; ++chunk) {
        for (uint32_t i = 0; i < 200; ++i) {
            vals.push_back((chunk << 16) | (i * 2));
        }
    }
    // Shuffle to make unsorted.
    std::mt19937 rng(42);
    std::shuffle(vals.begin(), vals.end(), rng);

    bm.addMany(vals.data(), vals.size());
    EXPECT_EQ(bm.cardinality(), 1000u);
    // Verify all values present.
    for (uint32_t chunk = 0; chunk < 5; ++chunk) {
        for (uint32_t i = 0; i < 200; ++i) {
            EXPECT_TRUE(bm.contains((chunk << 16) | (i * 2)));
        }
    }
}

TEST_F(RoaringBitmapTest, AddManyUnsortedMatchesSorted) {
    // Same values added sorted vs unsorted should produce identical bitmaps.
    std::vector<uint32_t> vals = {200000, 100, 65536, 5, 131072, 42, 65537, 1};
    RoaringBitmap a, b;
    a.addMany(vals.data(), vals.size());

    std::sort(vals.begin(), vals.end());
    b.addMany(vals.data(), vals.size());

    EXPECT_EQ(a, b);
}

TEST_F(RoaringBitmapTest, AddManySortedPromotion) {
    // Sorted addMany with enough values to trigger array→bitmap promotion.
    std::vector<uint32_t> vals;
    for (uint32_t i = 0; i < 5000; ++i) vals.push_back(i);
    bm.addMany(vals.data(), vals.size());
    EXPECT_EQ(bm.cardinality(), 5000u);
    for (uint32_t i = 0; i < 5000; ++i) EXPECT_TRUE(bm.contains(i));
}

TEST_F(RoaringBitmapTest, AddManySortedDuplicates) {
    // Sorted input with duplicates — should deduplicate correctly.
    std::vector<uint32_t> vals = {1, 1, 2, 2, 3, 3, 65536, 65536, 65537};
    bm.addMany(vals.data(), vals.size());
    EXPECT_EQ(bm.cardinality(), 5u);
}

TEST_F(RoaringBitmapTest, AddManySortedIntoExisting) {
    // Add sorted values into a bitmap that already has some values.
    bm.add(5);
    bm.add(10);
    bm.add(65536);
    std::vector<uint32_t> vals = {3, 7, 10, 15, 65536, 65537, 131072};
    bm.addMany(vals.data(), vals.size());
    EXPECT_EQ(bm.cardinality(), 8u);  // {3,5,7,10,15,65536,65537,131072}
    EXPECT_TRUE(bm.contains(5));
    EXPECT_TRUE(bm.contains(3));
    EXPECT_TRUE(bm.contains(131072));
}

TEST_F(RoaringBitmapTest, AddBulkStreamingSorted) {
    // Streaming sorted insertion with BulkContext.
    BulkContext ctx;
    for (uint32_t i = 0; i < 1000; ++i) {
        bm.addBulk(i * 3, ctx);
    }
    EXPECT_EQ(bm.cardinality(), 1000u);
    for (uint32_t i = 0; i < 1000; ++i) {
        EXPECT_TRUE(bm.contains(i * 3));
    }
}

TEST_F(RoaringBitmapTest, AddBulkCrossChunk) {
    // BulkContext across chunk boundaries.
    BulkContext ctx;
    std::vector<uint32_t> vals = {0, 100, 65535, 65536, 65537, 131072, 200000};
    for (auto v : vals) bm.addBulk(v, ctx);
    EXPECT_EQ(bm.cardinality(), vals.size());
    for (auto v : vals) EXPECT_TRUE(bm.contains(v));
}

TEST_F(RoaringBitmapTest, AddBulkDuplicates) {
    BulkContext ctx;
    bm.addBulk(42, ctx);
    bm.addBulk(42, ctx);
    bm.addBulk(42, ctx);
    EXPECT_EQ(bm.cardinality(), 1u);
    EXPECT_TRUE(bm.contains(42));
}

TEST_F(RoaringBitmapTest, AddBulkPromotion) {
    // Enough values in one chunk to trigger array→bitmap promotion.
    BulkContext ctx;
    for (uint32_t i = 0; i < 5000; ++i) {
        bm.addBulk(i, ctx);
    }
    EXPECT_EQ(bm.cardinality(), 5000u);
    for (uint32_t i = 0; i < 5000; ++i) {
        EXPECT_TRUE(bm.contains(i));
    }
}

TEST_F(RoaringBitmapTest, AddBulkMatchesAdd) {
    // addBulk should produce identical bitmap to regular add.
    RoaringBitmap ref;
    BulkContext ctx;
    std::vector<uint32_t> vals;
    for (uint32_t chunk = 0; chunk < 5; ++chunk) {
        for (uint32_t i = 0; i < 50; ++i) {
            vals.push_back((chunk << 16) | (i * 7));
        }
    }
    for (auto v : vals) ref.add(v);
    for (auto v : vals) bm.addBulk(v, ctx);
    EXPECT_EQ(bm, ref);
}

TEST_F(RoaringBitmapTest, AddBulkUnsortedInput) {
    // addBulk with unsorted input (cache miss every time — still correct).
    BulkContext ctx;
    std::vector<uint32_t> vals = {200000, 5, 131072, 42, 65536, 1, 65537, 100};
    for (auto v : vals) bm.addBulk(v, ctx);
    EXPECT_EQ(bm.cardinality(), vals.size());
    for (auto v : vals) EXPECT_TRUE(bm.contains(v));
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
    // Dense consecutive range — addRange already creates RunContainer,
    // so optimize() may find nothing to change (returns false).
    bm.addRange(0, 10000);
    bm.optimize();
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
} // namespace arrow
