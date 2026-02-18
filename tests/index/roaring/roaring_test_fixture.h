#pragma once
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#include "index/roaring_bitmap.h"
#include "index/roaring_simd.h"

namespace arrow {

class RoaringBitmapTest : public ::testing::Test {
protected:
    RoaringBitmap bm;
};

} // namespace arrow
