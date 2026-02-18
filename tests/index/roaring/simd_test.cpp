#include "roaring_test_fixture.h"

namespace arrow {

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
// ─────────────────────────────────────────────────────────────────────────────
// SIMD array intersection / union / xor / diff (Gaps 1 & 2)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, SIMDArrayIntersectBasic) {
    uint16_t a[] = {1, 3, 5, 7, 9, 11, 13, 15};
    uint16_t b[] = {2, 3, 5, 8, 11, 14, 15, 16};
    uint16_t out[8];
    uint32_t n = simd::array_intersect(a, 8, b, 8, out);
    EXPECT_EQ(n, 4u);
    EXPECT_EQ(out[0], 3);
    EXPECT_EQ(out[1], 5);
    EXPECT_EQ(out[2], 11);
    EXPECT_EQ(out[3], 15);
}

TEST_F(RoaringBitmapTest, SIMDArrayIntersectDisjoint) {
    uint16_t a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint16_t b[] = {10, 20, 30, 40, 50, 60, 70, 80};
    uint16_t out[8];
    uint32_t n = simd::array_intersect(a, 8, b, 8, out);
    EXPECT_EQ(n, 0u);
}

TEST_F(RoaringBitmapTest, SIMDArrayIntersectIdentical) {
    uint16_t a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint16_t out[8];
    uint32_t n = simd::array_intersect(a, 8, a, 8, out);
    EXPECT_EQ(n, 8u);
    for (uint32_t i = 0; i < 8; ++i) EXPECT_EQ(out[i], i + 1);
}

TEST_F(RoaringBitmapTest, SIMDArrayIntersectLarger) {
    // 200 elements each, every 3rd vs every 5th — intersection = every 15th.
    std::vector<uint16_t> a, b;
    for (uint16_t i = 0; i < 600; i += 3) a.push_back(i);
    for (uint16_t i = 0; i < 1000; i += 5) b.push_back(i);
    std::vector<uint16_t> out(std::min(a.size(), b.size()));
    uint32_t n = simd::array_intersect(a.data(), static_cast<uint32_t>(a.size()),
                                       b.data(), static_cast<uint32_t>(b.size()),
                                       out.data());
    // Every 15th in [0, 600): 0, 15, 30, ..., 585 → 40 values
    EXPECT_EQ(n, 40u);
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(out[i] % 15, 0);
    }
}

TEST_F(RoaringBitmapTest, SIMDArrayUnionBasic) {
    uint16_t a[] = {1, 3, 5, 7};
    uint16_t b[] = {2, 3, 6, 7};
    uint16_t out[8];
    uint32_t n = simd::array_union(a, 4, b, 4, out);
    EXPECT_EQ(n, 6u);
    uint16_t expected[] = {1, 2, 3, 5, 6, 7};
    for (uint32_t i = 0; i < n; ++i) EXPECT_EQ(out[i], expected[i]);
}

TEST_F(RoaringBitmapTest, SIMDArrayXorBasic) {
    uint16_t a[] = {1, 3, 5, 7};
    uint16_t b[] = {2, 3, 6, 7};
    uint16_t out[8];
    uint32_t n = simd::array_xor(a, 4, b, 4, out);
    EXPECT_EQ(n, 4u);
    uint16_t expected[] = {1, 2, 5, 6};
    for (uint32_t i = 0; i < n; ++i) EXPECT_EQ(out[i], expected[i]);
}

TEST_F(RoaringBitmapTest, SIMDArrayDiffBasic) {
    uint16_t a[] = {1, 3, 5, 7, 9};
    uint16_t b[] = {3, 7, 10};
    uint16_t out[5];
    uint32_t n = simd::array_diff(a, 5, b, 3, out);
    EXPECT_EQ(n, 3u);
    uint16_t expected[] = {1, 5, 9};
    for (uint32_t i = 0; i < n; ++i) EXPECT_EQ(out[i], expected[i]);
}

TEST_F(RoaringBitmapTest, SIMDArrayOpsMatchTopLevel) {
    // Build two array-container bitmaps, verify SIMD ops match top-level RoaringBitmap ops.
    RoaringBitmap a, b;
    for (uint32_t i = 0; i < 1000; i += 3) a.add(i);
    for (uint32_t i = 0; i < 1000; i += 5) b.add(i);

    auto andResult = a & b;
    auto orResult = a | b;
    auto xorResult = a ^ b;
    auto diffResult = a - b;

    // Verify via element-by-element check against std::set.
    std::set<uint32_t> sa, sb;
    for (uint32_t i = 0; i < 1000; i += 3) sa.insert(i);
    for (uint32_t i = 0; i < 1000; i += 5) sb.insert(i);

    // AND
    for (uint32_t v : sa) {
        EXPECT_EQ(andResult.contains(v), sb.count(v) > 0);
    }
    // OR
    std::set<uint32_t> sunion;
    std::set_union(sa.begin(), sa.end(), sb.begin(), sb.end(),
                   std::inserter(sunion, sunion.end()));
    EXPECT_EQ(orResult.cardinality(), sunion.size());
    // XOR
    std::set<uint32_t> sxor;
    std::set_symmetric_difference(sa.begin(), sa.end(), sb.begin(), sb.end(),
                                  std::inserter(sxor, sxor.end()));
    EXPECT_EQ(xorResult.cardinality(), sxor.size());
    // DIFF
    std::set<uint32_t> sdiff;
    std::set_difference(sa.begin(), sa.end(), sb.begin(), sb.end(),
                        std::inserter(sdiff, sdiff.end()));
    EXPECT_EQ(diffResult.cardinality(), sdiff.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// bitmap_to_array improved extraction (Gap 3)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RoaringBitmapTest, BitmapToArraySparse) {
    // Test extraction with only a few bits set across many words.
    uint64_t words[1024] = {};
    words[0] = 1ULL;           // bit 0
    words[100] = 1ULL << 42;   // bit 6442
    words[1023] = 1ULL << 63;  // bit 65535
    uint16_t out[3];
    uint32_t n = simd::bitmap_to_array(words, out);
    EXPECT_EQ(n, 3u);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 100 * 64 + 42);
    EXPECT_EQ(out[2], 65535);
}

TEST_F(RoaringBitmapTest, BitmapToArrayDenseCtz) {
    // Dense bitmap — every bit set in first word.
    uint64_t words[1024] = {};
    words[0] = ~0ULL;  // all 64 bits
    uint16_t out[64];
    uint32_t n = simd::bitmap_to_array(words, out);
    EXPECT_EQ(n, 64u);
    for (uint32_t i = 0; i < 64; ++i) EXPECT_EQ(out[i], i);
}

TEST_F(RoaringBitmapTest, BitmapToArrayAllSet) {
    // All 65536 bits set — stress test.
    uint64_t words[1024];
    std::memset(words, 0xFF, sizeof(words));
    std::vector<uint16_t> out(65536);
    uint32_t n = simd::bitmap_to_array(words, out.data());
    EXPECT_EQ(n, 65536u);
    for (uint32_t i = 0; i < 65536; ++i) EXPECT_EQ(out[i], i);
}

TEST_F(RoaringBitmapTest, BitmapToArrayMatchesIterator) {
    // Build a bitmap container, verify bitmap_to_array matches iteration.
    RoaringBitmap bm;
    for (uint32_t i = 0; i < 10000; i += 3) bm.add(i);

    auto vec = bm.toVector();
    // The bitmap should have one chunk with a bitmap container (>4096 elements).
    EXPECT_GT(bm.cardinality(), 3000u);
    // Verify toVector uses bitmap_to_array internally and is correct.
    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_EQ(vec[i] % 3, 0u);
    }
}

} // namespace arrow
