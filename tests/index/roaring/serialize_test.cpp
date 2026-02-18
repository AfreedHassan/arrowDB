#include "roaring_test_fixture.h"

namespace arrow {

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
} // namespace arrow
