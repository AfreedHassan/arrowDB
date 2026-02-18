#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "index/roaring_simd.h"

namespace arrow {

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr uint32_t kArrayMaxSize = 4096;  // promotion threshold
static constexpr uint32_t kBitmapWords = 1024;    // 64K bits = 8KB

// ─────────────────────────────────────────────────────────────────────────────
// Forward declarations
// ─────────────────────────────────────────────────────────────────────────────

struct ArrayContainer;
struct BitmapContainer;
struct RunContainer;

using Container = std::variant<ArrayContainer, BitmapContainer, RunContainer>;
class ContainerPtr;  // forward declaration — defined after container types

// ─────────────────────────────────────────────────────────────────────────────
// ArrayContainer — sorted uint16_t vector, binary search
// ─────────────────────────────────────────────────────────────────────────────

struct ArrayContainer {
    std::vector<uint16_t> values;

    uint32_t cardinality() const { return static_cast<uint32_t>(values.size()); }
    bool empty() const { return values.empty(); }

    bool contains(uint16_t v) const {
        return std::binary_search(values.begin(), values.end(), v);
    }

    // Returns true if newly added (was not present).
    bool add(uint16_t v) {
        auto it = std::lower_bound(values.begin(), values.end(), v);
        if (it != values.end() && *it == v) return false;
        values.insert(it, v);
        return true;
    }

    // Returns true if removed (was present).
    bool remove(uint16_t v) {
        auto it = std::lower_bound(values.begin(), values.end(), v);
        if (it == values.end() || *it != v) return false;
        values.erase(it);
        return true;
    }

    bool operator==(const ArrayContainer& o) const { return values == o.values; }
};

// ─────────────────────────────────────────────────────────────────────────────
// BitmapContainer — fixed 8KB (1024 × uint64_t)
// ─────────────────────────────────────────────────────────────────────────────

struct BitmapContainer {
    // Heap-allocated 8KB words array (Gap 10: reduces sizeof(BitmapContainer) from ~8KB to ~16B,
    // shrinking the Container variant so ArrayContainer/RunContainer don't waste 8KB each).
    struct Words {
        uint64_t* ptr;

        Words() : ptr(static_cast<uint64_t*>(
            ::operator new(kBitmapWords * sizeof(uint64_t), std::align_val_t{64}))) {
            std::memset(ptr, 0, kBitmapWords * sizeof(uint64_t));
        }
        Words(const Words& o) : ptr(static_cast<uint64_t*>(
            ::operator new(kBitmapWords * sizeof(uint64_t), std::align_val_t{64}))) {
            std::memcpy(ptr, o.ptr, kBitmapWords * sizeof(uint64_t));
        }
        Words(Words&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
        ~Words() {
            if (ptr) ::operator delete(ptr, std::align_val_t{64});
        }
        Words& operator=(const Words& o) {
            if (this != &o) std::memcpy(ptr, o.ptr, kBitmapWords * sizeof(uint64_t));
            return *this;
        }
        Words& operator=(Words&& o) noexcept {
            if (this != &o) {
                if (ptr) ::operator delete(ptr, std::align_val_t{64});
                ptr = o.ptr; o.ptr = nullptr;
            }
            return *this;
        }
        uint64_t& operator[](size_t i) { return ptr[i]; }
        const uint64_t& operator[](size_t i) const { return ptr[i]; }
        // Implicit conversion to pointer for SIMD functions.
        operator uint64_t*() { return ptr; }
        operator const uint64_t*() const { return ptr; }
        bool operator==(const Words&) const = delete;  // use bitmap_equal
    };

    Words words;
    mutable int32_t card = 0;  // -1 = lazy (unknown)

    uint32_t cardinality() const {
        if (card >= 0) return static_cast<uint32_t>(card);
        return computeCardinality();
    }

    uint32_t computeCardinality() const {
        uint32_t c = simd::bitmap_popcount_harley_seal(words);
        card = static_cast<int32_t>(c);
        return c;
    }

    bool empty() const {
        if (card > 0) return false;
        if (card == 0) return true;
        // card == -1 (lazy): SIMD early-exit scan instead of full popcount
        return simd::bitmap_is_empty(words);
    }

    bool contains(uint16_t v) const {
        return (words[v >> 6] & (1ULL << (v & 63))) != 0;
    }

    // Returns true if newly added.
    bool add(uint16_t v) {
        uint64_t& w = words[v >> 6];
        uint64_t bit = 1ULL << (v & 63);
        if (w & bit) return false;
        w |= bit;
        if (card >= 0) ++card;
        return true;
    }

    // Returns true if removed.
    bool remove(uint16_t v) {
        uint64_t& w = words[v >> 6];
        uint64_t bit = 1ULL << (v & 63);
        if (!(w & bit)) return false;
        w &= ~bit;
        if (card >= 0) --card;
        return true;
    }

    // Set bits in [start, end) — half-open, end can be up to 65536.
    void setRange(uint32_t start, uint32_t end) {
        if (start >= end || start >= 65536) return;
        if (end > 65536) end = 65536;
        uint32_t firstWord = start >> 6;
        uint32_t lastWord = (end - 1) >> 6;
        card = -1;  // lazy recompute

        if (firstWord == lastWord) {
            // Same word: set bits [start & 63, (end-1) & 63]
            uint64_t mask = (~0ULL << (start & 63));
            // Clear bits above (end-1) & 63
            if ((end & 63) != 0)
                mask &= (1ULL << (end & 63)) - 1;
            words[firstWord] |= mask;
        } else {
            // Partial first word
            words[firstWord] |= (~0ULL << (start & 63));
            // Full middle words
            for (uint32_t w = firstWord + 1; w < lastWord; ++w)
                words[w] = ~0ULL;
            // Partial last word
            if ((end & 63) != 0)
                words[lastWord] |= (1ULL << (end & 63)) - 1;
            else
                words[lastWord] = ~0ULL;
        }
    }

    // Clear bits in [start, end) — end can be up to 65536.
    void clearRange(uint32_t start, uint32_t end) {
        if (start >= end || start >= 65536) return;
        if (end > 65536) end = 65536;
        uint32_t firstWord = start >> 6;
        uint32_t lastWord = (end - 1) >> 6;
        card = -1;

        if (firstWord == lastWord) {
            uint64_t mask = (~0ULL << (start & 63));
            if ((end & 63) != 0)
                mask &= (1ULL << (end & 63)) - 1;
            words[firstWord] &= ~mask;
        } else {
            words[firstWord] &= ~(~0ULL << (start & 63));
            for (uint32_t w = firstWord + 1; w < lastWord; ++w)
                words[w] = 0;
            if ((end & 63) != 0)
                words[lastWord] &= ~((1ULL << (end & 63)) - 1);
            else
                words[lastWord] = 0;
        }
    }

    // Convert to array of set bit positions.
    ArrayContainer toArray() const {
        ArrayContainer a;
        uint32_t card = cardinality();
        a.values.resize(card);
        simd::bitmap_to_array(words, a.values.data());
        return a;
    }

    bool operator==(const BitmapContainer& o) const {
        return simd::bitmap_equal(words, o.words);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// RunContainer — sorted vector of {start, length} (inclusive end = start+length)
// ─────────────────────────────────────────────────────────────────────────────

struct RunContainer {
    struct Run {
        uint16_t start;
        uint16_t length;  // inclusive end = start + length
        bool operator==(const Run& o) const { return start == o.start && length == o.length; }
    };

    std::vector<Run> runs;

    uint32_t cardinality() const {
        uint32_t c = 0;
        for (auto& r : runs) c += r.length + 1;
        return c;
    }

    bool empty() const { return runs.empty(); }

    bool contains(uint16_t v) const {
        // Binary search for the run that could contain v.
        auto it = std::upper_bound(runs.begin(), runs.end(), v,
            [](uint16_t val, const Run& r) { return val < r.start; });
        if (it == runs.begin()) return false;
        --it;
        return v <= it->start + it->length;
    }

    bool add(uint16_t v) {
        if (runs.empty()) {
            runs.push_back({v, 0});
            return true;
        }

        auto it = std::upper_bound(runs.begin(), runs.end(), v,
            [](uint16_t val, const Run& r) { return val < r.start; });

        // Check if contained in previous run.
        if (it != runs.begin()) {
            auto prev = it - 1;
            if (v <= prev->start + prev->length) return false;  // already in run
        }

        // Check if we can extend previous run.
        if (it != runs.begin()) {
            auto prev = it - 1;
            if (prev->start + prev->length + 1 == v) {
                prev->length++;
                // Try merging with next.
                if (it != runs.end() && prev->start + prev->length + 1 == it->start) {
                    prev->length = it->start + it->length - prev->start;
                    runs.erase(it);
                }
                return true;
            }
        }

        // Check if we can extend next run.
        if (it != runs.end() && v + 1 == it->start) {
            it->start = v;
            it->length++;
            return true;
        }

        // Insert new singleton run.
        runs.insert(it, {v, 0});
        return true;
    }

    bool remove(uint16_t v) {
        auto it = std::upper_bound(runs.begin(), runs.end(), v,
            [](uint16_t val, const Run& r) { return val < r.start; });
        if (it == runs.begin()) return false;
        --it;
        uint16_t end = it->start + it->length;
        if (v > end) return false;

        if (it->start == end) {
            // Singleton run — remove it.
            runs.erase(it);
        } else if (v == it->start) {
            it->start++;
            it->length--;
        } else if (v == end) {
            it->length--;
        } else {
            // Split the run.
            uint16_t origStart = it->start;
            uint16_t origLen = it->length;
            it->length = v - origStart - 1;
            Run newRun{static_cast<uint16_t>(v + 1),
                       static_cast<uint16_t>(origStart + origLen - v - 1)};
            runs.insert(it + 1, newRun);
        }
        return true;
    }

    // Convert to ArrayContainer.
    ArrayContainer toArray() const {
        ArrayContainer a;
        uint32_t card = cardinality();
        a.values.resize(card);
        uint16_t* out = a.values.data();
        for (auto& r : runs) {
            std::iota(out, out + r.length + 1, r.start);
            out += r.length + 1;
        }
        return a;
    }

    // Convert to BitmapContainer.
    BitmapContainer toBitmap() const {
        BitmapContainer b;
        for (auto& r : runs)
            b.setRange(r.start, static_cast<uint32_t>(r.start) + r.length + 1);
        b.computeCardinality();
        return b;
    }

    // Set range [start, end) — merge into existing runs.
    void addRange(uint16_t start, uint16_t end) {
        if (start >= end) return;
        uint16_t runEnd = end - 1;  // inclusive
        Run newRun{start, static_cast<uint16_t>(runEnd - start)};

        // Find overlapping/adjacent runs and merge.
        auto first = runs.begin();
        while (first != runs.end() && first->start + first->length + 1 < start)
            ++first;

        auto last = first;
        while (last != runs.end() && last->start <= runEnd + 1u)
            ++last;

        if (first == last) {
            // No overlap — insert.
            runs.insert(first, newRun);
            return;
        }

        // Merge: take min start, max end.
        uint16_t mergedStart = std::min(start, first->start);
        auto back = last - 1;
        uint16_t mergedEnd = std::max(runEnd,
            static_cast<uint16_t>(back->start + back->length));
        first->start = mergedStart;
        first->length = mergedEnd - mergedStart;
        runs.erase(first + 1, last);
    }

    // Streaming append: O(1) merge with last run if adjacent/overlapping.
    // Caller must ensure start >= last run's start (monotonic append).
    void appendRun(uint16_t start, uint16_t length) {
        if (!runs.empty()) {
            auto& last = runs.back();
            uint32_t lastEnd = static_cast<uint32_t>(last.start) + last.length;
            uint32_t newEnd = static_cast<uint32_t>(start) + length;
            // Adjacent or overlapping: merge.
            if (start <= lastEnd + 1) {
                if (newEnd > lastEnd) {
                    last.length = static_cast<uint16_t>(newEnd - last.start);
                }
                return;
            }
        }
        runs.push_back({start, length});
    }

    bool operator==(const RunContainer& o) const { return runs == o.runs; }
};

// ── Intrusive refcounted container node ──────────────────────────────────────
// Non-atomic refcount: RoaringBitmap is not thread-safe, so atomic ops are
// pure overhead. Plain uint32_t eliminates memory barriers on every copy/destroy.
struct ContainerNode {
    mutable uint32_t refCount{1};
    Container data;

    explicit ContainerNode(Container c) : data(std::move(c)) {}
    ContainerNode(const ContainerNode& o) : data(o.data) {}  // refCount starts at 1
    ContainerNode& operator=(const ContainerNode&) = delete;

    // Pool allocator: recycle freed nodes to avoid malloc/free per container op.
    static void* operator new(size_t sz) {
        auto& pool = freeList();
        if (!pool.empty()) {
            void* p = pool.back();
            pool.pop_back();
            return p;
        }
        return ::operator new(sz);
    }
    static void operator delete(void* p, size_t) noexcept {
        freeList().push_back(p);
    }

private:
    static std::vector<void*>& freeList() {
        static thread_local std::vector<void*> pool;
        return pool;
    }
};

class ContainerPtr {
    ContainerNode* node_ = nullptr;

    void addRef() noexcept {
        if (node_) ++node_->refCount;
    }
    void release() noexcept {
        if (node_ && --node_->refCount == 0)
            delete node_;
    }

public:
    ContainerPtr() noexcept = default;
    explicit ContainerPtr(ContainerNode* p) noexcept : node_(p) {}

    ContainerPtr(const ContainerPtr& o) noexcept : node_(o.node_) { addRef(); }
    ContainerPtr(ContainerPtr&& o) noexcept : node_(o.node_) { o.node_ = nullptr; }

    ~ContainerPtr() { release(); }

    ContainerPtr& operator=(const ContainerPtr& o) noexcept {
        if (this != &o) {
            release();
            node_ = o.node_;
            addRef();
        }
        return *this;
    }
    ContainerPtr& operator=(ContainerPtr&& o) noexcept {
        if (this != &o) {
            release();
            node_ = o.node_;
            o.node_ = nullptr;
        }
        return *this;
    }

    const Container& operator*() const { return node_->data; }
    const Container* operator->() const { return &node_->data; }

    uint32_t use_count() const noexcept {
        return node_ ? node_->refCount : 0;
    }
    explicit operator bool() const noexcept { return node_ != nullptr; }
};

inline ContainerPtr makeContainer(Container c) {
    return ContainerPtr(new ContainerNode(std::move(c)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Container utilities
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

// Check if a RunContainer covers the entire 16-bit range [0, 65535].
inline bool isFullRun(const RunContainer& r) {
    return r.runs.size() == 1 && r.runs[0].start == 0 && r.runs[0].length == 0xFFFF;
}

inline uint32_t containerCardinality(const Container& c) {
    return std::visit([](const auto& x) -> uint32_t { return x.cardinality(); }, c);
}

inline bool containerContains(const Container& c, uint16_t v) {
    return std::visit([v](const auto& x) { return x.contains(v); }, c);
}

// ── Conversion helpers ──────────────────────────────────────────────────────

inline BitmapContainer arrayToBitmap(const ArrayContainer& a) {
    BitmapContainer b;
    simd::bitmap_set_list(b.words, a.values.data(),
                          static_cast<uint32_t>(a.values.size()));
    b.card = static_cast<int32_t>(a.values.size());
    return b;
}

inline ArrayContainer runToArray(const RunContainer& r) { return r.toArray(); }

inline BitmapContainer runToBitmap(const RunContainer& r) { return r.toBitmap(); }

// ── Promotion / demotion ────────────────────────────────────────────────────

// After an operation, check if a bitmap should demote.
inline Container maybedemote(BitmapContainer&& b) {
    uint32_t c = b.cardinality();
    if (c <= kArrayMaxSize) return b.toArray();
    return std::move(b);
}

// Branchless parallel binary search: search for 4 targets simultaneously.
// All 4 searches follow the same halving schedule over the same array,
// using conditional moves instead of branches.
// Returns found[i] = true if targets[i] was found.
inline void binarySearch4(const uint16_t* arr, int32_t n,
                          uint16_t t0, uint16_t t1, uint16_t t2, uint16_t t3,
                          bool& f0, bool& f1, bool& f2, bool& f3) {
    if (n == 0) { f0 = f1 = f2 = f3 = false; return; }
    int32_t lo0 = 0, lo1 = 0, lo2 = 0, lo3 = 0;
    int32_t len = n;
    while (len > 1) {
        int32_t h = len >> 1;
        lo0 += (arr[lo0 + h] < t0) ? h : 0;
        lo1 += (arr[lo1 + h] < t1) ? h : 0;
        lo2 += (arr[lo2 + h] < t2) ? h : 0;
        lo3 += (arr[lo3 + h] < t3) ? h : 0;
        len -= h;
    }
    // Final adjustment: branchless lower_bound converges to the element
    // before the target when arr[lo] < target. Advance by 1 in that case.
    lo0 += (arr[lo0] < t0) ? 1 : 0;
    lo1 += (arr[lo1] < t1) ? 1 : 0;
    lo2 += (arr[lo2] < t2) ? 1 : 0;
    lo3 += (arr[lo3] < t3) ? 1 : 0;
    f0 = (lo0 < n && arr[lo0] == t0);
    f1 = (lo1 < n && arr[lo1] == t1);
    f2 = (lo2 < n && arr[lo2] == t2);
    f3 = (lo3 < n && arr[lo3] == t3);
}

// ── AND (intersection) ──────────────────────────────────────────────────────

// Array × Array — two-pointer merge with galloping for skewed sizes.
inline Container andArrayArray(const ArrayContainer& a, const ArrayContainer& b) {
    ArrayContainer result;
    const auto& small = (a.values.size() <= b.values.size()) ? a : b;
    const auto& large = (a.values.size() <= b.values.size()) ? b : a;

    if (small.values.empty()) return result;

    // Galloping when size ratio > 64:1 — use branchless parallel binary search
    if (large.values.size() > 64 * small.values.size()) {
        result.values.reserve(small.values.size());
        const uint16_t* largeData = large.values.data();
        int32_t largeN = static_cast<int32_t>(large.values.size());
        uint32_t si = 0;
        uint32_t sn = static_cast<uint32_t>(small.values.size());

        // Process 4 targets at a time with parallel branchless binary search
        while (si + 4 <= sn) {
            bool f0, f1, f2, f3;
            binarySearch4(largeData, largeN,
                          small.values[si], small.values[si + 1],
                          small.values[si + 2], small.values[si + 3],
                          f0, f1, f2, f3);
            if (f0) result.values.push_back(small.values[si]);
            if (f1) result.values.push_back(small.values[si + 1]);
            if (f2) result.values.push_back(small.values[si + 2]);
            if (f3) result.values.push_back(small.values[si + 3]);
            si += 4;
        }
        // Scalar remainder
        for (; si < sn; ++si) {
            if (std::binary_search(large.values.begin(), large.values.end(), small.values[si]))
                result.values.push_back(small.values[si]);
        }
        return result;
    }

    // Use SIMD array intersection (NEON broadcast-compare or scalar with block-skip).
    size_t maxOut = std::min(a.values.size(), b.values.size());
    result.values.resize(maxOut);
    uint32_t count = simd::array_intersect(
        a.values.data(), static_cast<uint32_t>(a.values.size()),
        b.values.data(), static_cast<uint32_t>(b.values.size()),
        result.values.data());
    result.values.resize(count);
    return result;
}

// Array × Bitmap — iterate array, test each bit.
inline Container andArrayBitmap(const ArrayContainer& a, const BitmapContainer& b) {
    ArrayContainer result;
    result.values.reserve(a.values.size());
    for (uint16_t v : a.values) {
        if (b.contains(v)) result.values.push_back(v);
    }
    return result;
}

// Array × Run — iterate array, test each value against runs.
inline Container andArrayRun(const ArrayContainer& a, const RunContainer& r) {
    ArrayContainer result;
    result.values.reserve(a.values.size());
    for (uint16_t v : a.values) {
        if (r.contains(v)) result.values.push_back(v);
    }
    return result;
}

// Bitmap × Bitmap — word-wise AND.
// When result is sparse (<= 4096), use fused AND+extract to skip intermediate bitmap.
inline Container andBitmapBitmap(const BitmapContainer& a, const BitmapContainer& b) {
    BitmapContainer result;
    result.card = static_cast<int32_t>(simd::bitmap_and_popcount(a.words, b.words, result.words));
    if (static_cast<uint32_t>(result.card) <= kArrayMaxSize) {
        // Fused extract directly from the already-computed result bitmap.
        ArrayContainer arr;
        arr.values.resize(static_cast<uint32_t>(result.card));
        simd::bitmap_to_array(result.words, arr.values.data());
        return arr;
    }
    return result;
}

// Bitmap × Run — reset bitmap bits outside runs.
inline Container andBitmapRun(const BitmapContainer& bm, const RunContainer& r) {
    if (r.empty()) return ArrayContainer{};

    BitmapContainer result;
    // For each run, copy the bitmap bits within the run range.
    for (auto& run : r.runs) {
        uint32_t start = run.start;
        uint32_t end = static_cast<uint32_t>(run.start) + run.length;  // inclusive

        uint32_t startWord = start >> 6;
        uint32_t endWord = end >> 6;

        if (startWord == endWord) {
            uint64_t mask = (~0ULL << (start & 63));
            if (((end + 1) & 63) != 0)
                mask &= (1ULL << ((end + 1) & 63)) - 1;
            // else: end+1 == 65536, mask already covers from start to bit 63
            result.words[startWord] |= bm.words[startWord] & mask;
        } else {
            // First partial word
            result.words[startWord] |= bm.words[startWord] & (~0ULL << (start & 63));
            // Full middle words
            for (uint32_t w = startWord + 1; w < endWord; ++w)
                result.words[w] |= bm.words[w];
            // Last partial word
            if (((end + 1) & 63) != 0)
                result.words[endWord] |= bm.words[endWord] & ((1ULL << ((end + 1) & 63)) - 1);
            else
                result.words[endWord] |= bm.words[endWord];
        }
    }
    result.computeCardinality();
    return maybedemote(std::move(result));
}

// Run × Run — sweep-merge intervals, emit overlaps.
inline Container andRunRun(const RunContainer& a, const RunContainer& b) {
    RunContainer result;
    size_t i = 0, j = 0;
    while (i < a.runs.size() && j < b.runs.size()) {
        uint32_t aStart = a.runs[i].start;
        uint32_t aEnd = aStart + a.runs[i].length;
        uint32_t bStart = b.runs[j].start;
        uint32_t bEnd = bStart + b.runs[j].length;

        uint32_t overlapStart = std::max(aStart, bStart);
        uint32_t overlapEnd = std::min(aEnd, bEnd);

        if (overlapStart <= overlapEnd) {
            result.appendRun(static_cast<uint16_t>(overlapStart),
                             static_cast<uint16_t>(overlapEnd - overlapStart));
        }

        if (aEnd < bEnd) ++i;
        else ++j;
    }
    // Decide best container type for result.
    uint32_t c = result.cardinality();
    if (c == 0) return ArrayContainer{};
    if (c <= kArrayMaxSize) return result.toArray();
    return result;
}

// Dispatcher for AND.  Full-run shortcut: AND(x, full) = x.
inline Container containerAnd(const Container& a, const Container& b) {
    if (auto* r = std::get_if<RunContainer>(&b); r && isFullRun(*r)) return a;
    if (auto* r = std::get_if<RunContainer>(&a); r && isFullRun(*r)) return b;
    return std::visit([](const auto& x, const auto& y) -> Container {
        using X = std::decay_t<decltype(x)>;
        using Y = std::decay_t<decltype(y)>;

        if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, ArrayContainer>)
            return andArrayArray(x, y);
        else if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, BitmapContainer>)
            return andArrayBitmap(x, y);
        else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, ArrayContainer>)
            return andArrayBitmap(y, x);
        else if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, RunContainer>)
            return andArrayRun(x, y);
        else if constexpr (std::is_same_v<X, RunContainer> && std::is_same_v<Y, ArrayContainer>)
            return andArrayRun(y, x);
        else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, BitmapContainer>)
            return andBitmapBitmap(x, y);
        else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, RunContainer>)
            return andBitmapRun(x, y);
        else if constexpr (std::is_same_v<X, RunContainer> && std::is_same_v<Y, BitmapContainer>)
            return andBitmapRun(y, x);
        else // Run × Run
            return andRunRun(x, y);
    }, a, b);
}

// ── OR (union) ──────────────────────────────────────────────────────────────

// Array × Array — sorted merge with bulk copy for non-overlapping blocks.
inline Container orArrayArray(const ArrayContainer& a, const ArrayContainer& b) {
    ArrayContainer result;
    size_t maxOut = a.values.size() + b.values.size();
    result.values.resize(maxOut);
    uint32_t count = simd::array_union(
        a.values.data(), static_cast<uint32_t>(a.values.size()),
        b.values.data(), static_cast<uint32_t>(b.values.size()),
        result.values.data());
    result.values.resize(count);

    if (result.values.size() > kArrayMaxSize) {
        return arrayToBitmap(result);
    }
    return result;
}

// Array × Bitmap — copy bitmap, set array bits via batch set.
inline Container orArrayBitmap(const ArrayContainer& a, const BitmapContainer& b) {
    BitmapContainer result = b;
    simd::bitmap_set_list(result.words, a.values.data(),
                          static_cast<uint32_t>(a.values.size()));
    result.card = -1;  // invalidate — set_list doesn't track card
    result.computeCardinality();
    return result;
}

// Array × Run — add array values to a copy of the run container, or convert.
inline Container orArrayRun(const ArrayContainer& a, const RunContainer& r) {
    // Convert run to bitmap (always safe), batch set array bits.
    BitmapContainer bm = r.toBitmap();
    simd::bitmap_set_list(bm.words, a.values.data(),
                          static_cast<uint32_t>(a.values.size()));
    bm.card = -1;
    bm.computeCardinality();
    return maybedemote(std::move(bm));
}

// Bitmap × Bitmap — word-wise OR.
inline Container orBitmapBitmap(const BitmapContainer& a, const BitmapContainer& b) {
    BitmapContainer result;
    result.card = static_cast<int32_t>(simd::bitmap_or_popcount(a.words, b.words, result.words));
    return result;
}

// Bitmap × Run — copy bitmap, set run ranges.
inline Container orBitmapRun(const BitmapContainer& bm, const RunContainer& r) {
    BitmapContainer result = bm;
    for (auto& run : r.runs)
        result.setRange(run.start,
            static_cast<uint32_t>(run.start) + run.length + 1);
    result.computeCardinality();
    return result;
}

// Run × Run — merge runs using streaming appendRun.
inline Container orRunRun(const RunContainer& a, const RunContainer& b) {
    RunContainer result;
    size_t i = 0, j = 0;
    while (i < a.runs.size() && j < b.runs.size()) {
        if (a.runs[i].start <= b.runs[j].start) {
            result.appendRun(a.runs[i].start, a.runs[i].length); ++i;
        } else {
            result.appendRun(b.runs[j].start, b.runs[j].length); ++j;
        }
    }
    while (i < a.runs.size()) { result.appendRun(a.runs[i].start, a.runs[i].length); ++i; }
    while (j < b.runs.size()) { result.appendRun(b.runs[j].start, b.runs[j].length); ++j; }
    return result;
}

// Full-run shortcut: OR(x, full) = full.
inline Container containerOr(const Container& a, const Container& b) {
    if (auto* r = std::get_if<RunContainer>(&b); r && isFullRun(*r)) return b;
    if (auto* r = std::get_if<RunContainer>(&a); r && isFullRun(*r)) return a;
    return std::visit([](const auto& x, const auto& y) -> Container {
        using X = std::decay_t<decltype(x)>;
        using Y = std::decay_t<decltype(y)>;

        if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, ArrayContainer>)
            return orArrayArray(x, y);
        else if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, BitmapContainer>)
            return orArrayBitmap(x, y);
        else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, ArrayContainer>)
            return orArrayBitmap(y, x);
        else if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, RunContainer>)
            return orArrayRun(x, y);
        else if constexpr (std::is_same_v<X, RunContainer> && std::is_same_v<Y, ArrayContainer>)
            return orArrayRun(y, x);
        else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, BitmapContainer>)
            return orBitmapBitmap(x, y);
        else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, RunContainer>)
            return orBitmapRun(x, y);
        else if constexpr (std::is_same_v<X, RunContainer> && std::is_same_v<Y, BitmapContainer>)
            return orBitmapRun(y, x);
        else
            return orRunRun(x, y);
    }, a, b);
}

// ── ANDNOT (set difference) ─────────────────────────────────────────────────

// Array - Array: two-pointer.
inline Container andNotArrayArray(const ArrayContainer& a, const ArrayContainer& b) {
    ArrayContainer result;
    result.values.resize(a.values.size());
    uint32_t count = simd::array_diff(
        a.values.data(), static_cast<uint32_t>(a.values.size()),
        b.values.data(), static_cast<uint32_t>(b.values.size()),
        result.values.data());
    result.values.resize(count);
    return result;
}

// Array - Bitmap: iterate array, emit if bit NOT set.
inline Container andNotArrayBitmap(const ArrayContainer& a, const BitmapContainer& b) {
    ArrayContainer result;
    result.values.reserve(a.values.size());
    for (uint16_t v : a.values) {
        if (!b.contains(v)) result.values.push_back(v);
    }
    return result;
}

// Array - Run: iterate array, emit if not in any run.
inline Container andNotArrayRun(const ArrayContainer& a, const RunContainer& r) {
    ArrayContainer result;
    result.values.reserve(a.values.size());
    for (uint16_t v : a.values) {
        if (!r.contains(v)) result.values.push_back(v);
    }
    return result;
}

// Bitmap - Array: copy bitmap, batch clear array positions.
inline Container andNotBitmapArray(const BitmapContainer& bm, const ArrayContainer& a) {
    BitmapContainer result = bm;
    simd::bitmap_clear_list(result.words, a.values.data(),
                            static_cast<uint32_t>(a.values.size()));
    result.card = -1;
    result.computeCardinality();
    return maybedemote(std::move(result));
}

// Bitmap - Bitmap: word-wise & ~.
inline Container andNotBitmapBitmap(const BitmapContainer& a, const BitmapContainer& b) {
    BitmapContainer result;
    result.card = static_cast<int32_t>(simd::bitmap_andnot_popcount(a.words, b.words, result.words));
    if (static_cast<uint32_t>(result.card) <= kArrayMaxSize) {
        ArrayContainer arr;
        arr.values.resize(static_cast<uint32_t>(result.card));
        simd::bitmap_to_array(result.words, arr.values.data());
        return arr;
    }
    return result;
}

// Bitmap - Run: copy bitmap, clear run ranges.
inline Container andNotBitmapRun(const BitmapContainer& bm, const RunContainer& r) {
    BitmapContainer result = bm;
    for (auto& run : r.runs) {
        result.clearRange(run.start,
            static_cast<uint32_t>(run.start) + run.length + 1);
    }
    result.computeCardinality();
    return maybedemote(std::move(result));
}

// Run - Array: convert run to appropriate type, subtract.
inline Container andNotRunArray(const RunContainer& r, const ArrayContainer& a) {
    if (r.cardinality() <= kArrayMaxSize) {
        ArrayContainer ra = r.toArray();
        return andNotArrayArray(ra, a);
    }
    BitmapContainer bm = r.toBitmap();
    return andNotBitmapArray(bm, a);
}

// Run - Bitmap: convert run to bitmap, single SIMD pass.
inline Container andNotRunBitmap(const RunContainer& r, const BitmapContainer& b) {
    BitmapContainer bm = r.toBitmap();
    simd::bitmap_andnot_nocard(bm.words, b.words, bm.words);
    bm.card = -1;
    bm.computeCardinality();
    return maybedemote(std::move(bm));
}

// Run - Run: direct interval subtraction sweep.
inline Container andNotRunRun(const RunContainer& a, const RunContainer& b) {
    if (a.empty()) return ArrayContainer{};
    if (b.empty()) {
        if (a.cardinality() <= kArrayMaxSize) return a.toArray();
        return a;
    }

    RunContainer result;
    size_t j = 0;

    for (size_t i = 0; i < a.runs.size(); ++i) {
        uint32_t curStart = a.runs[i].start;
        uint32_t curEnd = static_cast<uint32_t>(a.runs[i].start) + a.runs[i].length;

        // Advance j past b runs that end before curStart.
        while (j < b.runs.size() &&
               static_cast<uint32_t>(b.runs[j].start) + b.runs[j].length < curStart)
            ++j;

        size_t k = j;
        while (k < b.runs.size() && b.runs[k].start <= curEnd && curStart <= curEnd) {
            uint32_t bStart = b.runs[k].start;
            uint32_t bEnd = static_cast<uint32_t>(b.runs[k].start) + b.runs[k].length;

            if (bStart <= curStart) {
                if (bEnd >= curEnd) {
                    // b fully covers current — nothing remains.
                    curStart = curEnd + 1;
                    break;
                }
                // b clips left: advance start past b's end.
                curStart = bEnd + 1;
            } else {
                // bStart > curStart — emit left fragment [curStart, bStart-1].
                if (bStart > curEnd) break;  // no overlap

                result.appendRun(
                    static_cast<uint16_t>(curStart),
                    static_cast<uint16_t>(bStart - 1 - curStart));

                if (bEnd >= curEnd) {
                    // b covers the rest.
                    curStart = curEnd + 1;
                    break;
                }
                // b splits current — continue with right part.
                curStart = bEnd + 1;
            }
            ++k;
        }

        // Emit whatever remains of current.
        if (curStart <= curEnd) {
            result.appendRun(
                static_cast<uint16_t>(curStart),
                static_cast<uint16_t>(curEnd - curStart));
        }
    }

    uint32_t c = result.cardinality();
    if (c == 0) return ArrayContainer{};
    if (c <= kArrayMaxSize) return result.toArray();
    return result;
}

// Full-run shortcut: ANDNOT(x, full) = empty, ANDNOT(full, x) = NOT(x).
inline Container containerAndNot(const Container& a, const Container& b) {
    if (auto* r = std::get_if<RunContainer>(&b); r && isFullRun(*r))
        return ArrayContainer{};  // x & ~full = empty
    return std::visit([](const auto& x, const auto& y) -> Container {
        using X = std::decay_t<decltype(x)>;
        using Y = std::decay_t<decltype(y)>;

        if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, ArrayContainer>)
            return andNotArrayArray(x, y);
        else if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, BitmapContainer>)
            return andNotArrayBitmap(x, y);
        else if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, RunContainer>)
            return andNotArrayRun(x, y);
        else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, ArrayContainer>)
            return andNotBitmapArray(x, y);
        else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, BitmapContainer>)
            return andNotBitmapBitmap(x, y);
        else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, RunContainer>)
            return andNotBitmapRun(x, y);
        else if constexpr (std::is_same_v<X, RunContainer> && std::is_same_v<Y, ArrayContainer>)
            return andNotRunArray(x, y);
        else if constexpr (std::is_same_v<X, RunContainer> && std::is_same_v<Y, BitmapContainer>)
            return andNotRunBitmap(x, y);
        else
            return andNotRunRun(x, y);
    }, a, b);
}

// ── XOR (symmetric difference) ──────────────────────────────────────────────

// Array ^ Array: sorted merge, skip common elements.
inline Container xorArrayArray(const ArrayContainer& a, const ArrayContainer& b) {
    ArrayContainer result;
    size_t maxOut = a.values.size() + b.values.size();
    result.values.resize(maxOut);
    uint32_t count = simd::array_xor(
        a.values.data(), static_cast<uint32_t>(a.values.size()),
        b.values.data(), static_cast<uint32_t>(b.values.size()),
        result.values.data());
    result.values.resize(count);

    if (result.values.size() > kArrayMaxSize) return arrayToBitmap(result);
    return result;
}

// Bitmap ^ Bitmap: word-wise XOR.
inline Container xorBitmapBitmap(const BitmapContainer& a, const BitmapContainer& b) {
    BitmapContainer result;
    result.card = static_cast<int32_t>(simd::bitmap_xor_popcount(a.words, b.words, result.words));
    if (static_cast<uint32_t>(result.card) <= kArrayMaxSize) {
        ArrayContainer arr;
        arr.values.resize(static_cast<uint32_t>(result.card));
        simd::bitmap_to_array(result.words, arr.values.data());
        return arr;
    }
    return result;
}

// Array ^ Bitmap: copy bitmap, batch flip array positions.
inline Container xorArrayBitmap(const ArrayContainer& a, const BitmapContainer& b) {
    BitmapContainer result = b;
    simd::bitmap_flip_list(result.words, a.values.data(),
                           static_cast<uint32_t>(a.values.size()));
    result.card = -1;
    result.computeCardinality();
    return maybedemote(std::move(result));
}

// Run × Bitmap XOR: single conversion + SIMD pass.
inline Container xorRunBitmap(const RunContainer& r, const BitmapContainer& b) {
    BitmapContainer bm = r.toBitmap();
    simd::bitmap_xor_nocard(bm.words, b.words, bm.words);
    bm.card = -1;
    bm.computeCardinality();
    return maybedemote(std::move(bm));
}

// Run × Array XOR: single conversion + batch flip.
inline Container xorRunArray(const RunContainer& r, const ArrayContainer& a) {
    BitmapContainer bm = r.toBitmap();
    simd::bitmap_flip_list(bm.words, a.values.data(),
                           static_cast<uint32_t>(a.values.size()));
    bm.card = -1;
    bm.computeCardinality();
    return maybedemote(std::move(bm));
}

// Run × Run XOR: 2 conversions + 1 SIMD pass (vs 3 full container ops).
inline Container xorRunRun(const RunContainer& a, const RunContainer& b) {
    BitmapContainer bmA = a.toBitmap();
    BitmapContainer bmB = b.toBitmap();
    simd::bitmap_xor_nocard(bmA.words, bmB.words, bmA.words);
    bmA.card = -1;
    bmA.computeCardinality();
    return maybedemote(std::move(bmA));
}

inline Container containerXor(const Container& a, const Container& b) {
    return std::visit([](const auto& x, const auto& y) -> Container {
        using X = std::decay_t<decltype(x)>;
        using Y = std::decay_t<decltype(y)>;

        if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, ArrayContainer>)
            return xorArrayArray(x, y);
        else if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, BitmapContainer>)
            return xorArrayBitmap(x, y);
        else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, ArrayContainer>)
            return xorArrayBitmap(y, x);
        else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, BitmapContainer>)
            return xorBitmapBitmap(x, y);
        else if constexpr (std::is_same_v<X, RunContainer> && std::is_same_v<Y, BitmapContainer>)
            return xorRunBitmap(x, y);
        else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, RunContainer>)
            return xorRunBitmap(y, x);
        else if constexpr (std::is_same_v<X, RunContainer> && std::is_same_v<Y, ArrayContainer>)
            return xorRunArray(x, y);
        else if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, RunContainer>)
            return xorRunArray(y, x);
        else
            return xorRunRun(x, y);
    }, a, b);
}

// ── Container-level cardinality-only operations ─────────────────────────────

inline uint32_t containerAndCardinality(const Container& a, const Container& b) {
    return std::visit([](const auto& x, const auto& y) -> uint32_t {
        using X = std::decay_t<decltype(x)>;
        using Y = std::decay_t<decltype(y)>;

        if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, BitmapContainer>) {
            return simd::bitmap_and_popcount_noout(x.words, y.words);
        } else if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, ArrayContainer>) {
            uint32_t count = 0;
            size_t i = 0, j = 0;
            while (i < x.values.size() && j < y.values.size()) {
                if (x.values[i] < y.values[j]) ++i;
                else if (x.values[i] > y.values[j]) ++j;
                else { ++count; ++i; ++j; }
            }
            return count;
        } else if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, BitmapContainer>) {
            uint32_t count = 0;
            for (uint16_t v : x.values)
                if (y.contains(v)) ++count;
            return count;
        } else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, ArrayContainer>) {
            uint32_t count = 0;
            for (uint16_t v : y.values)
                if (x.contains(v)) ++count;
            return count;
        } else {
            Container result = containerAnd(
                Container{x}, Container{y});
            return containerCardinality(result);
        }
    }, a, b);
}

inline uint32_t containerOrCardinality(const Container& a, const Container& b) {
    return std::visit([](const auto& x, const auto& y) -> uint32_t {
        using X = std::decay_t<decltype(x)>;
        using Y = std::decay_t<decltype(y)>;

        if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, BitmapContainer>) {
            return simd::bitmap_or_popcount_noout(x.words, y.words);
        } else {
            return x.cardinality() + y.cardinality() -
                containerAndCardinality(Container{x}, Container{y});
        }
    }, a, b);
}

inline uint32_t containerXorCardinality(const Container& a, const Container& b) {
    return std::visit([](const auto& x, const auto& y) -> uint32_t {
        using X = std::decay_t<decltype(x)>;
        using Y = std::decay_t<decltype(y)>;

        if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, BitmapContainer>) {
            return simd::bitmap_xor_popcount_noout(x.words, y.words);
        } else {
            uint32_t andCard = containerAndCardinality(Container{x}, Container{y});
            return x.cardinality() + y.cardinality() - 2 * andCard;
        }
    }, a, b);
}

inline uint32_t containerAndNotCardinality(const Container& a, const Container& b) {
    return std::visit([](const auto& x, const auto& y) -> uint32_t {
        using X = std::decay_t<decltype(x)>;
        using Y = std::decay_t<decltype(y)>;

        if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, BitmapContainer>) {
            return simd::bitmap_andnot_popcount_noout(x.words, y.words);
        } else {
            return x.cardinality() -
                containerAndCardinality(Container{x}, Container{y});
        }
    }, a, b);
}

// ── Container-level intersects (early exit) ─────────────────────────────────

inline bool containerIntersects(const Container& a, const Container& b) {
    return std::visit([](const auto& x, const auto& y) -> bool {
        using X = std::decay_t<decltype(x)>;
        using Y = std::decay_t<decltype(y)>;

        if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, BitmapContainer>) {
            return simd::bitmap_intersects_any(x.words, y.words);
        } else if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, ArrayContainer>) {
            size_t i = 0, j = 0;
            while (i < x.values.size() && j < y.values.size()) {
                if (x.values[i] < y.values[j]) ++i;
                else if (x.values[i] > y.values[j]) ++j;
                else return true;
            }
            return false;
        } else if constexpr (std::is_same_v<X, ArrayContainer> && std::is_same_v<Y, BitmapContainer>) {
            for (uint16_t v : x.values)
                if (y.contains(v)) return true;
            return false;
        } else if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, ArrayContainer>) {
            for (uint16_t v : y.values)
                if (x.contains(v)) return true;
            return false;
        } else {
            return containerAndCardinality(Container{x}, Container{y}) > 0;
        }
    }, a, b);
}

// ── Lazy OR (bitmap-bitmap case produces card=-1) ───────────────────────────

inline Container containerLazyOr(const Container& a, const Container& b) {
    return std::visit([](const auto& x, const auto& y) -> Container {
        using X = std::decay_t<decltype(x)>;
        using Y = std::decay_t<decltype(y)>;

        if constexpr (std::is_same_v<X, BitmapContainer> && std::is_same_v<Y, BitmapContainer>) {
            BitmapContainer result;
            simd::bitmap_or_nocard(x.words, y.words, result.words);
            result.card = -1;
            return result;
        } else {
            return containerOr(Container{x}, Container{y});
        }
    }, a, b);
}

// ── Container-level forEach (direct dispatch, no iterator) ───────────────

template <typename Fn>
inline void containerForEach(const Container& c, uint32_t base, Fn&& fn) {
    std::visit([base, &fn](const auto& cont) {
        using T = std::decay_t<decltype(cont)>;
        if constexpr (std::is_same_v<T, ArrayContainer>) {
            for (uint16_t v : cont.values) fn(base + v);
        } else if constexpr (std::is_same_v<T, BitmapContainer>) {
            for (uint32_t w = 0; w < kBitmapWords; ++w) {
                uint64_t bits = cont.words[w];
                if (bits == 0) continue;
                uint32_t wbase = base + (w << 6);
                while (bits) {
                    fn(wbase + static_cast<uint32_t>(__builtin_ctzll(bits)));
                    bits &= bits - 1;
                }
            }
        } else {
            for (auto& r : cont.runs) {
                for (uint32_t v = r.start;
                     v <= static_cast<uint32_t>(r.start) + r.length; ++v)
                    fn(base + v);
            }
        }
    }, c);
}

// ── Container to uint32_t array (direct write, no intermediate buffer) ───

inline uint32_t containerToUint32Array(const Container& c, uint32_t base,
                                        uint32_t* out) {
    return std::visit([base, out](const auto& cont) -> uint32_t {
        using T = std::decay_t<decltype(cont)>;
        if constexpr (std::is_same_v<T, ArrayContainer>) {
            for (size_t i = 0; i < cont.values.size(); ++i)
                out[i] = base + cont.values[i];
            return static_cast<uint32_t>(cont.values.size());
        } else if constexpr (std::is_same_v<T, BitmapContainer>) {
            return simd::bitmap_to_uint32_array(cont.words, base, out);
        } else {
            uint32_t pos = 0;
            for (auto& r : cont.runs) {
                for (uint32_t v = r.start;
                     v <= static_cast<uint32_t>(r.start) + r.length; ++v)
                    out[pos++] = base + v;
            }
            return pos;
        }
    }, c);
}

// ── Container to bitmap (force-convert any type) ─────────────────────────

inline BitmapContainer containerToBitmap(const Container& c) {
    return std::visit([](const auto& cont) -> BitmapContainer {
        using T = std::decay_t<decltype(cont)>;
        if constexpr (std::is_same_v<T, BitmapContainer>) {
            return cont;
        } else if constexpr (std::is_same_v<T, ArrayContainer>) {
            return arrayToBitmap(cont);
        } else {
            return cont.toBitmap();
        }
    }, c);
}

// ── Lazy OR in-place (force to bitmap, then word-OR) ─────────────────────

inline void containerLazyOrInPlace(Container& a, const Container& b) {
    if (!std::holds_alternative<BitmapContainer>(a)) {
        a = containerToBitmap(a);
    }
    auto& bm = std::get<BitmapContainer>(a);
    std::visit([&bm](const auto& cont) {
        using T = std::decay_t<decltype(cont)>;
        if constexpr (std::is_same_v<T, BitmapContainer>) {
            simd::bitmap_or_nocard(bm.words, cont.words, bm.words);
        } else if constexpr (std::is_same_v<T, ArrayContainer>) {
            simd::bitmap_set_list(bm.words, cont.values.data(),
                                  static_cast<uint32_t>(cont.values.size()));
        } else {
            for (auto& run : cont.runs)
                bm.setRange(run.start,
                    static_cast<uint32_t>(run.start) + run.length + 1);
        }
    }, b);
    bm.card = -1;
}

// ── In-place container operations ────────────────────────────────────────

inline void containerOrInPlace(Container& a, const Container& b) {
    if (auto* bm = std::get_if<BitmapContainer>(&a)) {
        if (auto* ob = std::get_if<BitmapContainer>(&b)) {
            simd::bitmap_or_nocard(bm->words, ob->words, bm->words);
            bm->card = -1;
        } else if (auto* oa = std::get_if<ArrayContainer>(&b)) {
            simd::bitmap_set_list(bm->words, oa->values.data(),
                                  static_cast<uint32_t>(oa->values.size()));
            bm->card = -1;
        } else {
            auto* or_ = std::get_if<RunContainer>(&b);
            for (auto& run : or_->runs)
                bm->setRange(run.start,
                    static_cast<uint32_t>(run.start) + run.length + 1);
        }
        return;
    }
    if (auto* aa = std::get_if<ArrayContainer>(&a)) {
        if (auto* ba = std::get_if<ArrayContainer>(&b)) {
            size_t maxOut = aa->values.size() + ba->values.size();
            std::vector<uint16_t> merged(maxOut);
            uint32_t count = simd::array_union(
                aa->values.data(), static_cast<uint32_t>(aa->values.size()),
                ba->values.data(), static_cast<uint32_t>(ba->values.size()),
                merged.data());
            merged.resize(count);
            if (count > kArrayMaxSize) {
                aa->values = std::move(merged);
                a = arrayToBitmap(*aa);
            } else {
                aa->values = std::move(merged);
            }
            return;
        }
    }
    a = containerOr(a, b);
}

inline void containerAndInPlace(Container& a, const Container& b) {
    if (auto* bm = std::get_if<BitmapContainer>(&a)) {
        if (auto* ob = std::get_if<BitmapContainer>(&b)) {
            simd::bitmap_and_nocard(bm->words, ob->words, bm->words);
            bm->card = -1;
            return;
        }
    }
    if (auto* arr = std::get_if<ArrayContainer>(&a)) {
        size_t out = 0;
        for (size_t i = 0; i < arr->values.size(); ++i) {
            if (containerContains(b, arr->values[i]))
                arr->values[out++] = arr->values[i];
        }
        arr->values.resize(out);
        return;
    }
    a = containerAnd(a, b);
}

inline void containerAndNotInPlace(Container& a, const Container& b) {
    if (auto* bm = std::get_if<BitmapContainer>(&a)) {
        if (auto* ob = std::get_if<BitmapContainer>(&b)) {
            simd::bitmap_andnot_nocard(bm->words, ob->words, bm->words);
            bm->card = -1;
            return;
        }
        if (auto* oa = std::get_if<ArrayContainer>(&b)) {
            simd::bitmap_clear_list(bm->words, oa->values.data(),
                                    static_cast<uint32_t>(oa->values.size()));
            bm->card = -1;
            return;
        }
        if (auto* or_ = std::get_if<RunContainer>(&b)) {
            for (auto& run : or_->runs)
                bm->clearRange(run.start,
                    static_cast<uint32_t>(run.start) + run.length + 1);
            return;
        }
    }
    if (auto* aa = std::get_if<ArrayContainer>(&a)) {
        if (auto* ba = std::get_if<ArrayContainer>(&b)) {
            std::vector<uint16_t> result(aa->values.size());
            uint32_t count = simd::array_diff(
                aa->values.data(), static_cast<uint32_t>(aa->values.size()),
                ba->values.data(), static_cast<uint32_t>(ba->values.size()),
                result.data());
            result.resize(count);
            aa->values = std::move(result);
            return;
        }
        size_t out = 0;
        for (size_t i = 0; i < aa->values.size(); ++i) {
            if (!containerContains(b, aa->values[i]))
                aa->values[out++] = aa->values[i];
        }
        aa->values.resize(out);
        return;
    }
    a = containerAndNot(a, b);
}

inline void containerXorInPlace(Container& a, const Container& b) {
    if (auto* bm = std::get_if<BitmapContainer>(&a)) {
        if (auto* ob = std::get_if<BitmapContainer>(&b)) {
            simd::bitmap_xor_nocard(bm->words, ob->words, bm->words);
            bm->card = -1;
            return;
        }
        if (auto* oa = std::get_if<ArrayContainer>(&b)) {
            simd::bitmap_flip_list(bm->words, oa->values.data(),
                                   static_cast<uint32_t>(oa->values.size()));
            bm->card = -1;
            return;
        }
    }
    if (auto* aa = std::get_if<ArrayContainer>(&a)) {
        if (auto* ba = std::get_if<ArrayContainer>(&b)) {
            size_t maxOut = aa->values.size() + ba->values.size();
            std::vector<uint16_t> result(maxOut);
            uint32_t count = simd::array_xor(
                aa->values.data(), static_cast<uint32_t>(aa->values.size()),
                ba->values.data(), static_cast<uint32_t>(ba->values.size()),
                result.data());
            result.resize(count);
            if (count > kArrayMaxSize) {
                aa->values = std::move(result);
                a = arrayToBitmap(*aa);
            } else {
                aa->values = std::move(result);
            }
            return;
        }
    }
    a = containerXor(a, b);
}

}  // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// RoaringStatistics — container type breakdown
// ─────────────────────────────────────────────────────────────────────────────

struct RoaringStatistics {
    uint32_t numArrayContainers = 0;
    uint32_t numBitmapContainers = 0;
    uint32_t numRunContainers = 0;
    uint32_t numContainers = 0;
    uint64_t numValues = 0;
    uint32_t minValue = 0;
    uint32_t maxValue = 0;
    // Per-type value counts
    uint64_t numValuesArrayContainers = 0;
    uint64_t numValuesBitmapContainers = 0;
    uint64_t numValuesRunContainers = 0;
    // Per-type byte counts
    uint64_t numBytesArrayContainers = 0;
    uint64_t numBytesBitmapContainers = 0;
    uint64_t numBytesRunContainers = 0;
    // Sum of all values
    uint64_t sumValue = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// RoaringBitmap — top-level class
// ─────────────────────────────────────────────────────────────────────────────

// Context for amortized O(1) bulk insertion. Caches the last-touched
// container so consecutive values with the same high-16 key skip binary search.
// Usage: create one BulkContext, then call addBulk() in a loop.
struct BulkContext {
    uint16_t key = 0;
    size_t   containerIdx = SIZE_MAX;  // SIZE_MAX = invalid/uninitialized
};

class RoaringBitmap {
public:
    // ── Modification ────────────────────────────────────────────────────────

    void add(uint32_t val) {
        uint16_t hi = static_cast<uint16_t>(val >> 16);
        uint16_t lo = static_cast<uint16_t>(val & 0xFFFF);

        auto& c = getOrCreateContainer(hi);
        std::visit([&](auto& cont) {
            using T = std::decay_t<decltype(cont)>;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                if (!cont.values.empty() && lo > cont.values.back()) {
                    cont.values.push_back(lo);
                } else {
                    cont.add(lo);
                }
                if (cont.cardinality() > kArrayMaxSize) {
                    c = detail::arrayToBitmap(cont);
                }
            } else {
                cont.add(lo);
            }
        }, c);
    }

    // Amortized O(1) insertion with cached container context.
    // Best for streaming sorted data. Create one BulkContext, call repeatedly.
    void addBulk(uint32_t val, BulkContext& ctx) {
        uint16_t hi = static_cast<uint16_t>(val >> 16);
        uint16_t lo = static_cast<uint16_t>(val & 0xFFFF);

        // Fast path: same key as last call — go directly to cached container.
        if (ctx.containerIdx != SIZE_MAX && ctx.key == hi &&
            ctx.containerIdx < keys_.size() && keys_[ctx.containerIdx] == hi) {
            auto& c = cow(containers_[ctx.containerIdx]);
            std::visit([&](auto& cont) {
                using T = std::decay_t<decltype(cont)>;
                if constexpr (std::is_same_v<T, ArrayContainer>) {
                    // Append fast path: if val > current max, O(1) push_back.
                    if (!cont.values.empty() && lo > cont.values.back()) {
                        cont.values.push_back(lo);
                    } else {
                        cont.add(lo);
                    }
                    if (cont.cardinality() > kArrayMaxSize) {
                        c = detail::arrayToBitmap(cont);
                    }
                } else {
                    cont.add(lo);
                }
            }, c);
            return;
        }

        // Slow path: different key — do lookup and update cache.
        auto& c = getOrCreateContainer(hi);
        // Update cache: find the index we just touched.
        auto it = std::lower_bound(keys_.begin(), keys_.end(), hi);
        ctx.key = hi;
        ctx.containerIdx = static_cast<size_t>(it - keys_.begin());

        std::visit([&](auto& cont) {
            using T = std::decay_t<decltype(cont)>;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                if (!cont.values.empty() && lo > cont.values.back()) {
                    cont.values.push_back(lo);
                } else {
                    cont.add(lo);
                }
                if (cont.cardinality() > kArrayMaxSize) {
                    c = detail::arrayToBitmap(cont);
                }
            } else {
                cont.add(lo);
            }
        }, c);
    }

    void remove(uint32_t val) {
        uint16_t hi = static_cast<uint16_t>(val >> 16);
        uint16_t lo = static_cast<uint16_t>(val & 0xFFFF);

        auto idx = findContainer(hi);
        if (idx == keys_.size()) return;

        auto& c = cow(containers_[idx]);
        std::visit([&](auto& cont) {
            using T = std::decay_t<decltype(cont)>;
            cont.remove(lo);
            if constexpr (std::is_same_v<T, BitmapContainer>) {
                if (cont.cardinality() <= kArrayMaxSize) {
                    c = cont.toArray();
                }
            }
        }, c);

        // Remove empty container.
        if (detail::containerCardinality(*containers_[idx]) == 0) {
            eraseChunkAt(idx);
        }
    }

    // Half-open range [min, max).
    // CRoaring-style: one bulk shift to make room, then right-to-left fill.
    // New chunks get RunContainers (trivially cheap for contiguous ranges).
    void addRange(uint32_t min, uint64_t max) {
        if (min >= max) return;
        if (max > 0x100000000ULL) max = 0x100000000ULL;

        uint16_t minKey = static_cast<uint16_t>(min >> 16);
        uint16_t maxKey = static_cast<uint16_t>((max - 1) >> 16);
        uint32_t numRequired = static_cast<uint32_t>(maxKey) - minKey + 1;

        // Count keys in [minKey..maxKey] range (common), before (prefix), after (suffix).
        auto prefixEnd = std::lower_bound(keys_.begin(), keys_.end(), minKey);
        size_t prefixLen = static_cast<size_t>(prefixEnd - keys_.begin());
        auto suffixBegin = std::upper_bound(prefixEnd, keys_.end(), maxKey);
        size_t suffixLen = static_cast<size_t>(keys_.end() - suffixBegin);
        size_t commonLen = keys_.size() - prefixLen - suffixLen;

        // Bulk expand: make room for new chunks if needed.
        if (numRequired > commonLen) {
            size_t growth = numRequired - commonLen;
            size_t oldSize = keys_.size();
            size_t newSize = oldSize + growth;
            keys_.resize(newSize);
            containers_.resize(newSize);
            // Shift suffix right to make room.
            if (suffixLen > 0) {
                for (size_t k = 0; k < suffixLen; ++k) {
                    size_t from = oldSize - 1 - k;
                    size_t to = newSize - 1 - k;
                    keys_[to] = keys_[from];
                    containers_[to] = std::move(containers_[from]);
                }
            }
        }

        // Fill right-to-left: existing containers get addRange, new ones get RunContainer.
        size_t src = prefixLen + commonLen;  // one past last existing in range
        size_t dst = prefixLen + numRequired; // one past last slot in range
        for (uint32_t key = maxKey; key != static_cast<uint32_t>(minKey) - 1; --key) {
            uint16_t loStart = (key == minKey) ? static_cast<uint16_t>(min & 0xFFFF) : 0;
            uint16_t loEndInc = (key == maxKey) ? static_cast<uint16_t>((max - 1) & 0xFFFF) : 0xFFFF;

            --dst;
            if (src > prefixLen && keys_[src - 1] == static_cast<uint16_t>(key)) {
                // Existing container — add range into it.
                --src;
                if (src != dst) {
                    keys_[dst] = keys_[src];
                    containers_[dst] = std::move(containers_[src]);
                }
                auto& c = cow(containers_[dst]);
                std::visit([&](auto& cont) {
                    using T = std::decay_t<decltype(cont)>;
                    if constexpr (std::is_same_v<T, ArrayContainer>) {
                        // For arrays: promote to bitmap, setRange, lazy card.
                        BitmapContainer bm = detail::arrayToBitmap(cont);
                        bm.setRange(loStart, static_cast<uint32_t>(loEndInc) + 1);
                        c = std::move(bm);
                    } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                        cont.setRange(loStart, static_cast<uint32_t>(loEndInc) + 1);
                    } else {
                        // RunContainer: merge range directly.
                        if (loStart == 0 && loEndInc == 0xFFFF) {
                            cont.runs.clear();
                            cont.runs.push_back({0, 0xFFFF});
                        } else {
                            cont.addRange(loStart, loEndInc + 1);
                        }
                    }
                }, c);
            } else {
                // New container — create RunContainer with single run.
                keys_[dst] = static_cast<uint16_t>(key);
                RunContainer rc;
                rc.runs.push_back({loStart, static_cast<uint16_t>(loEndInc - loStart)});
                containers_[dst] = makeContainer(Container{std::move(rc)});
            }
        }
        lastKeyIdx_ = 0;  // invalidate cache
    }

    void addMany(const uint32_t* vals, size_t n) {
        if (n == 0) return;

        // Check if input is sorted (common case for bulk insertion).
        bool sorted = true;
        for (size_t i = 1; i < n && sorted; ++i)
            sorted = (vals[i] >= vals[i - 1]);

        if (!sorted) {
            // Sort a copy, then use the sorted path. This avoids per-element
            // binary search on both keys_ and within containers.
            std::vector<uint32_t> buf(vals, vals + n);
            std::sort(buf.begin(), buf.end());
            addManySorted(buf.data(), n);
            return;
        }

        addManySorted(vals, n);
    }

    // ── Queries ─────────────────────────────────────────────────────────────

    bool contains(uint32_t val) const {
        uint16_t hi = static_cast<uint16_t>(val >> 16);
        uint16_t lo = static_cast<uint16_t>(val & 0xFFFF);
        auto idx = findContainer(hi);
        if (idx == keys_.size()) return false;
        return detail::containerContains(*containers_[idx], lo);
    }

    uint32_t cardinality() const {
        uint32_t total = 0;
        for (size_t _ci0 = 0; _ci0 < keys_.size(); ++_ci0)
        {
            auto& c = containers_[_ci0];
            total += detail::containerCardinality(*c);
        }
        return total;
    }

    bool empty() const { return keys_.empty(); }

    std::optional<uint32_t> minimum() const {
        if (keys_.empty()) return std::nullopt;
        auto key = keys_.front();
        const auto& c = containers_.front();
        uint32_t base = static_cast<uint32_t>(key) << 16;
        return std::visit([base](const auto& cont) -> std::optional<uint32_t> {
            using T = std::decay_t<decltype(cont)>;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                if (cont.values.empty()) return std::nullopt;
                return base + cont.values.front();
            } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                for (uint32_t w = 0; w < kBitmapWords; ++w) {
                    if (cont.words[w])
                        return base + (w << 6) + __builtin_ctzll(cont.words[w]);
                }
                return std::nullopt;
            } else {
                if (cont.runs.empty()) return std::nullopt;
                return base + cont.runs.front().start;
            }
        }, *c);
    }

    std::optional<uint32_t> maximum() const {
        if (keys_.empty()) return std::nullopt;
        auto key = keys_.back();
        const auto& c = containers_.back();
        uint32_t base = static_cast<uint32_t>(key) << 16;
        return std::visit([base](const auto& cont) -> std::optional<uint32_t> {
            using T = std::decay_t<decltype(cont)>;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                if (cont.values.empty()) return std::nullopt;
                return base + cont.values.back();
            } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                for (int w = kBitmapWords - 1; w >= 0; --w) {
                    if (cont.words[w])
                        return base + (w << 6) + 63 - __builtin_clzll(cont.words[w]);
                }
                return std::nullopt;
            } else {
                if (cont.runs.empty()) return std::nullopt;
                auto& last = cont.runs.back();
                return base + last.start + last.length;
            }
        }, *c);
    }

    // ── Set operations ──────────────────────────────────────────────────────

    RoaringBitmap operator&(const RoaringBitmap& o) const {
        RoaringBitmap result;
        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) ++i;
            else if (keys_[i] > o.keys_[j]) ++j;
            else {
                auto c = detail::containerAnd(*containers_[i], *o.containers_[j]);
                if (detail::containerCardinality(c) > 0)
                    result.pushChunk(keys_[i], makeContainer(std::move(c)));
                ++i; ++j;
            }
        }
        return result;
    }

    RoaringBitmap operator|(const RoaringBitmap& o) const {
        RoaringBitmap result;
        result.reserveChunks(keys_.size() + o.keys_.size());
        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) {
                result.pushChunk(keys_[i], containers_[i]); ++i;
            } else if (keys_[i] > o.keys_[j]) {
                result.pushChunk(o.keys_[j], o.containers_[j]); ++j;
            } else {
                result.pushChunk(keys_[i], makeContainer(detail::containerOr(*containers_[i], *o.containers_[j])));
                ++i; ++j;
            }
        }
        while (i < keys_.size()) { result.pushChunk(keys_[i], containers_[i]); ++i; }
        while (j < o.keys_.size()) { result.pushChunk(o.keys_[j], o.containers_[j]); ++j; }
        return result;
    }

    RoaringBitmap operator^(const RoaringBitmap& o) const {
        RoaringBitmap result;
        result.reserveChunks(keys_.size() + o.keys_.size());
        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) {
                result.pushChunk(keys_[i], containers_[i]); ++i;
            } else if (keys_[i] > o.keys_[j]) {
                result.pushChunk(o.keys_[j], o.containers_[j]); ++j;
            } else {
                auto c = detail::containerXor(*containers_[i], *o.containers_[j]);
                if (detail::containerCardinality(c) > 0)
                    result.pushChunk(keys_[i], makeContainer(std::move(c)));
                ++i; ++j;
            }
        }
        while (i < keys_.size()) { result.pushChunk(keys_[i], containers_[i]); ++i; }
        while (j < o.keys_.size()) { result.pushChunk(o.keys_[j], o.containers_[j]); ++j; }
        return result;
    }

    RoaringBitmap andNot(const RoaringBitmap& o) const {
        return *this - o;
    }

    RoaringBitmap operator-(const RoaringBitmap& o) const {
        RoaringBitmap result;
        result.reserveChunks(keys_.size());
        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) {
                result.pushChunk(keys_[i], containers_[i]); ++i;
            } else if (keys_[i] > o.keys_[j]) {
                ++j;
            } else {
                auto c = detail::containerAndNot(*containers_[i], *o.containers_[j]);
                if (detail::containerCardinality(c) > 0)
                    result.pushChunk(keys_[i], makeContainer(std::move(c)));
                ++i; ++j;
            }
        }
        while (i < keys_.size()) { result.pushChunk(keys_[i], containers_[i]); ++i; }
        return result;
    }

    // ── In-place operations ─────────────────────────────────────────────────

    RoaringBitmap& operator|=(const RoaringBitmap& o) {
        if (this == &o) return *this;

        std::vector<uint16_t> rk;
        std::vector<ContainerPtr> rc;
        rk.reserve(keys_.size() + o.keys_.size());
        rc.reserve(keys_.size() + o.keys_.size());

        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) {
                rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i])); ++i;
            } else if (keys_[i] > o.keys_[j]) {
                rk.push_back(o.keys_[j]); rc.push_back(o.containers_[j]); ++j;
            } else {
                auto& c = cow(containers_[i]);
                detail::containerOrInPlace(c, *o.containers_[j]);
                rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i]));
                ++i; ++j;
            }
        }
        while (i < keys_.size()) { rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i])); ++i; }
        while (j < o.keys_.size()) { rk.push_back(o.keys_[j]); rc.push_back(o.containers_[j]); ++j; }

        keys_ = std::move(rk);
        containers_ = std::move(rc);
        return *this;
    }

    RoaringBitmap& operator&=(const RoaringBitmap& o) {
        if (this == &o) return *this;

        std::vector<uint16_t> rk;
        std::vector<ContainerPtr> rc;

        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) {
                ++i;
            } else if (keys_[i] > o.keys_[j]) {
                ++j;
            } else {
                auto& c = cow(containers_[i]);
                detail::containerAndInPlace(c, *o.containers_[j]);
                uint32_t card = detail::containerCardinality(c);
                if (card > 0) {
                    if (auto* bm = std::get_if<BitmapContainer>(&c)) {
                        if (card <= kArrayMaxSize)
                            c = bm->toArray();
                    }
                    rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i]));
                }
                ++i; ++j;
            }
        }

        keys_ = std::move(rk);
        containers_ = std::move(rc);
        return *this;
    }

    RoaringBitmap& operator-=(const RoaringBitmap& o) {
        if (this == &o) { clearChunks(); return *this; }

        std::vector<uint16_t> rk;
        std::vector<ContainerPtr> rc;
        rk.reserve(keys_.size());
        rc.reserve(keys_.size());

        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) {
                rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i])); ++i;
            } else if (keys_[i] > o.keys_[j]) {
                ++j;
            } else {
                auto& c = cow(containers_[i]);
                detail::containerAndNotInPlace(c, *o.containers_[j]);
                uint32_t card = detail::containerCardinality(c);
                if (card > 0) {
                    if (auto* bm = std::get_if<BitmapContainer>(&c)) {
                        if (card <= kArrayMaxSize)
                            c = bm->toArray();
                    }
                    rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i]));
                }
                ++i; ++j;
            }
        }
        while (i < keys_.size()) { rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i])); ++i; }

        keys_ = std::move(rk);
        containers_ = std::move(rc);
        return *this;
    }

    RoaringBitmap& operator^=(const RoaringBitmap& o) {
        if (this == &o) { clearChunks(); return *this; }

        std::vector<uint16_t> rk;
        std::vector<ContainerPtr> rc;
        rk.reserve(keys_.size() + o.keys_.size());
        rc.reserve(keys_.size() + o.keys_.size());

        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) {
                rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i])); ++i;
            } else if (keys_[i] > o.keys_[j]) {
                rk.push_back(o.keys_[j]); rc.push_back(o.containers_[j]); ++j;
            } else {
                auto& c = cow(containers_[i]);
                detail::containerXorInPlace(c, *o.containers_[j]);
                uint32_t card = detail::containerCardinality(c);
                if (card > 0) {
                    if (auto* bm = std::get_if<BitmapContainer>(&c)) {
                        if (card <= kArrayMaxSize)
                            c = bm->toArray();
                    }
                    rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i]));
                }
                ++i; ++j;
            }
        }
        while (i < keys_.size()) { rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i])); ++i; }
        while (j < o.keys_.size()) { rk.push_back(o.keys_[j]); rc.push_back(o.containers_[j]); ++j; }

        keys_ = std::move(rk);
        containers_ = std::move(rc);
        return *this;
    }

    // ── Equality ────────────────────────────────────────────────────────────

    bool operator==(const RoaringBitmap& o) const {
        if (keys_.size() != o.keys_.size()) return false;
        for (size_t i = 0; i < keys_.size(); ++i) {
            if (keys_[i] != o.keys_[i]) return false;
            const auto& ca = *containers_[i];
            const auto& cb = *o.containers_[i];
            if (ca.index() != cb.index()) {
                // Different container types — compare cardinalities first (fast reject).
                if (detail::containerCardinality(ca) != detail::containerCardinality(cb))
                    return false;
                // Use XOR: if XOR cardinality is 0, they're equal.
                if (detail::containerXorCardinality(ca, cb) != 0)
                    return false;
            } else {
                // Same container type — use native comparison
                bool eq = std::visit([](const auto& a, const auto& b) -> bool {
                    using A = std::decay_t<decltype(a)>;
                    using B = std::decay_t<decltype(b)>;
                    if constexpr (std::is_same_v<A, B>) {
                        return a == b;
                    } else {
                        return false; // unreachable due to index check
                    }
                }, ca, cb);
                if (!eq) return false;
            }
        }
        return true;
    }

    bool operator!=(const RoaringBitmap& o) const { return !(*this == o); }

    // ── Optimize (run-length compression) ───────────────────────────────────

    bool optimize() {
        bool changed = false;
        for (size_t _ci1 = 0; _ci1 < keys_.size(); ++_ci1) {
            auto& c = containers_[_ci1];
            RunContainer rc = toRunContainer(*c);
            uint32_t card = detail::containerCardinality(*c);

            // Three-way comparison: pick smallest serialized representation.
            size_t runSize = 2 + rc.runs.size() * 4;   // uint16 numRuns + pairs
            size_t arraySize = static_cast<size_t>(card) * 2;
            size_t bitmapSize = 8192;

            size_t currentSize = std::visit([](const auto& cont) -> size_t {
                using T = std::decay_t<decltype(cont)>;
                if constexpr (std::is_same_v<T, ArrayContainer>)
                    return cont.values.size() * 2;
                else if constexpr (std::is_same_v<T, BitmapContainer>)
                    return 8192;
                else
                    return 2 + cont.runs.size() * 4;
            }, *c);

            // Find the best representation.
            size_t bestSize = std::min({runSize, arraySize, bitmapSize});
            if (bestSize >= currentSize) continue;  // already optimal

            if (bestSize == runSize) {
                c = makeContainer(std::move(rc));
                changed = true;
            } else if (bestSize == arraySize && !std::holds_alternative<ArrayContainer>(*c)) {
                // Convert to array if not already.
                if (auto* bm = std::get_if<BitmapContainer>(&*c)) {
                    c = makeContainer(bm->toArray());
                } else {
                    c = makeContainer(rc.toArray());
                }
                changed = true;
            } else if (bestSize == bitmapSize && !std::holds_alternative<BitmapContainer>(*c)) {
                if (auto* arr = std::get_if<ArrayContainer>(&*c)) {
                    c = makeContainer(detail::arrayToBitmap(*arr));
                } else {
                    c = makeContainer(rc.toBitmap());
                }
                changed = true;
            }
        }
        return changed;
    }

    // CRoaring-style runOptimize: only convert current→run if run is smaller.
    // Does NOT do 3-way (array→bitmap or bitmap→array) conversions.
    bool runOptimize() {
        bool changed = false;
        for (size_t i = 0; i < keys_.size(); ++i) {
            auto& c = containers_[i];
            RunContainer rc = toRunContainer(*c);
            size_t runSize = 2 + rc.runs.size() * 4;
            size_t currentSize = std::visit([](const auto& cont) -> size_t {
                using T = std::decay_t<decltype(cont)>;
                if constexpr (std::is_same_v<T, ArrayContainer>)
                    return cont.values.size() * 2;
                else if constexpr (std::is_same_v<T, BitmapContainer>)
                    return 8192;
                else
                    return 2 + cont.runs.size() * 4;
            }, *c);
            if (runSize < currentSize) {
                c = makeContainer(std::move(rc));
                changed = true;
            }
        }
        return changed;
    }

    void shrinkToFit() {
        for (size_t _ci2 = 0; _ci2 < keys_.size(); ++_ci2) {
            auto& c = containers_[_ci2];
            auto& mc = cow(c);
            std::visit([](auto& cont) {
                using T = std::decay_t<decltype(cont)>;
                if constexpr (std::is_same_v<T, ArrayContainer>) {
                    cont.values.shrink_to_fit();
                } else if constexpr (std::is_same_v<T, RunContainer>) {
                    cont.runs.shrink_to_fit();
                }
                // BitmapContainer is fixed-size array, nothing to shrink
            }, mc);
        }
        shrinkChunks();
    }

    // ── Iteration ───────────────────────────────────────────────────────────

    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using value_type = uint32_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const uint32_t*;
        using reference = uint32_t;

        Iterator() = default;

        Iterator(const RoaringBitmap* bm, size_t chunkIdx)
            : bm_(bm), chunkIdx_(chunkIdx) {
            if (chunkIdx_ < bm_->keys_.size()) {
                loadChunk();
            }
        }

        uint32_t operator*() const {
            uint32_t base = static_cast<uint32_t>(bm_->keys_[chunkIdx_]) << 16;
            return base + spanPtr_[posInChunk_];
        }

        Iterator& operator++() {
            ++posInChunk_;
            if (posInChunk_ >= spanSize_) {
                ++chunkIdx_;
                posInChunk_ = 0;
                if (chunkIdx_ < bm_->keys_.size())
                    loadChunk();
                else
                    spanSize_ = 0;
            }
            return *this;
        }

        Iterator operator++(int) { Iterator tmp = *this; ++*this; return tmp; }

        bool operator==(const Iterator& o) const {
            if (!bm_ && !o.bm_) return true;
            if (!bm_ || !o.bm_) return false;
            if (chunkIdx_ >= bm_->keys_.size() && o.chunkIdx_ >= o.bm_->keys_.size())
                return true;
            return chunkIdx_ == o.chunkIdx_ && posInChunk_ == o.posInChunk_;
        }
        bool operator!=(const Iterator& o) const { return !(*this == o); }

        // Advance to first element >= val. No-op if already there.
        void moveEqualOrLarger(uint32_t val) {
            if (!bm_ || chunkIdx_ >= bm_->keys_.size()) return;

            uint16_t targetHi = static_cast<uint16_t>(val >> 16);
            uint16_t targetLo = static_cast<uint16_t>(val & 0xFFFF);

            // Skip whole chunks whose key < targetHi.
            while (chunkIdx_ < bm_->keys_.size() &&
                   bm_->keys_[chunkIdx_] < targetHi) {
                ++chunkIdx_;
                posInChunk_ = 0;
                if (chunkIdx_ < bm_->keys_.size())
                    loadChunk();
                else {
                    spanSize_ = 0;
                    return;
                }
            }
            if (chunkIdx_ >= bm_->keys_.size()) {
                spanSize_ = 0;
                return;
            }

            if (bm_->keys_[chunkIdx_] == targetHi) {
                // Binary search within this chunk for targetLo.
                const uint16_t* begin = spanPtr_ + posInChunk_;
                const uint16_t* end = spanPtr_ + spanSize_;
                auto it = std::lower_bound(begin, end, targetLo);
                if (it != end) {
                    posInChunk_ = static_cast<size_t>(it - spanPtr_);
                } else {
                    // All values in this chunk < targetLo, advance to next chunk.
                    ++chunkIdx_;
                    posInChunk_ = 0;
                    if (chunkIdx_ < bm_->keys_.size())
                        loadChunk();
                    else
                        spanSize_ = 0;
                }
            }
            // If chunk key > targetHi, we're already past val — posInChunk_ = 0 is correct.
        }

        // Read up to maxCount values into buf. Returns actual count read.
        uint32_t readMany(uint32_t* buf, uint32_t maxCount) {
            uint32_t count = 0;
            while (count < maxCount && bm_ && chunkIdx_ < bm_->keys_.size()) {
                uint32_t base = static_cast<uint32_t>(bm_->keys_[chunkIdx_]) << 16;
                while (count < maxCount && posInChunk_ < spanSize_) {
                    buf[count++] = base + spanPtr_[posInChunk_++];
                }
                if (posInChunk_ >= spanSize_) {
                    ++chunkIdx_;
                    posInChunk_ = 0;
                    if (chunkIdx_ < bm_->keys_.size())
                        loadChunk();
                    else
                        spanSize_ = 0;
                }
            }
            return count;
        }

    private:
        void loadChunk() {
            const auto& c = *bm_->containers_[chunkIdx_];
            std::visit([this](const auto& cont) {
                using T = std::decay_t<decltype(cont)>;
                if constexpr (std::is_same_v<T, ArrayContainer>) {
                    // Zero-copy: point directly into the container's sorted array.
                    spanPtr_ = cont.values.data();
                    spanSize_ = cont.values.size();
                    ownedValues_.clear();
                } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                    uint32_t card = cont.cardinality();
                    ownedValues_.resize(card);
                    simd::bitmap_to_array(cont.words, ownedValues_.data());
                    spanPtr_ = ownedValues_.data();
                    spanSize_ = card;
                } else {
                    ownedValues_.clear();
                    ownedValues_.reserve(cont.cardinality());
                    for (auto& r : cont.runs)
                        for (uint32_t v = r.start;
                             v <= static_cast<uint32_t>(r.start) + r.length; ++v)
                            ownedValues_.push_back(static_cast<uint16_t>(v));
                    spanPtr_ = ownedValues_.data();
                    spanSize_ = ownedValues_.size();
                }
            }, c);
            posInChunk_ = 0;
        }

        const uint16_t* valuesData() const { return spanPtr_; }
        size_t valuesSize() const { return spanSize_; }

        const RoaringBitmap* bm_ = nullptr;
        size_t chunkIdx_ = 0;
        size_t posInChunk_ = 0;
        const uint16_t* spanPtr_ = nullptr;
        size_t spanSize_ = 0;
        std::vector<uint16_t> ownedValues_;  // only used for bitmap/run containers
    };

    Iterator begin() const { return Iterator(this, 0); }
    Iterator end() const { return Iterator(this, keys_.size()); }

    // ── Reverse Iteration ───────────────────────────────────────────────────

    struct ReverseIterator {
        using iterator_category = std::forward_iterator_tag;
        using value_type = uint32_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const uint32_t*;
        using reference = uint32_t;

        ReverseIterator() = default;

        // sentinel == true means "rend" (past-the-end in reverse).
        ReverseIterator(const RoaringBitmap* bm, bool sentinel)
            : bm_(bm), sentinel_(sentinel) {
            if (sentinel || bm_->keys_.empty()) {
                sentinel_ = true;
            } else {
                chunkIdx_ = bm_->keys_.size() - 1;
                loadChunkReverse();
                posInChunk_ = spanSize_ == 0 ? 0 : spanSize_ - 1;
            }
        }

        uint32_t operator*() const {
            uint32_t base = static_cast<uint32_t>(bm_->keys_[chunkIdx_]) << 16;
            return base + spanPtr_[posInChunk_];
        }

        ReverseIterator& operator++() {
            if (posInChunk_ == 0) {
                if (chunkIdx_ == 0) {
                    sentinel_ = true;
                    spanSize_ = 0;
                } else {
                    --chunkIdx_;
                    loadChunkReverse();
                    posInChunk_ = spanSize_ == 0 ? 0 : spanSize_ - 1;
                }
            } else {
                --posInChunk_;
            }
            return *this;
        }

        ReverseIterator operator++(int) { ReverseIterator tmp = *this; ++*this; return tmp; }

        bool operator==(const ReverseIterator& o) const {
            if (sentinel_ && o.sentinel_) return true;
            if (sentinel_ || o.sentinel_) return false;
            return chunkIdx_ == o.chunkIdx_ && posInChunk_ == o.posInChunk_;
        }
        bool operator!=(const ReverseIterator& o) const { return !(*this == o); }

    private:
        void loadChunkReverse() {
            const auto& c = *bm_->containers_[chunkIdx_];
            std::visit([this](const auto& cont) {
                using T = std::decay_t<decltype(cont)>;
                if constexpr (std::is_same_v<T, ArrayContainer>) {
                    spanPtr_ = cont.values.data();
                    spanSize_ = cont.values.size();
                    ownedValues_.clear();
                } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                    uint32_t card = cont.cardinality();
                    ownedValues_.resize(card);
                    simd::bitmap_to_array(cont.words, ownedValues_.data());
                    spanPtr_ = ownedValues_.data();
                    spanSize_ = card;
                } else {
                    ownedValues_.clear();
                    ownedValues_.reserve(cont.cardinality());
                    for (auto& r : cont.runs)
                        for (uint32_t v = r.start;
                             v <= static_cast<uint32_t>(r.start) + r.length; ++v)
                            ownedValues_.push_back(static_cast<uint16_t>(v));
                    spanPtr_ = ownedValues_.data();
                    spanSize_ = ownedValues_.size();
                }
            }, c);
        }

        const RoaringBitmap* bm_ = nullptr;
        size_t chunkIdx_ = 0;
        size_t posInChunk_ = 0;
        bool sentinel_ = false;
        const uint16_t* spanPtr_ = nullptr;
        size_t spanSize_ = 0;
        std::vector<uint16_t> ownedValues_;
    };

    ReverseIterator rbegin() const { return ReverseIterator(this, false); }
    ReverseIterator rend() const { return ReverseIterator(this, true); }

    // ── Serialization ────────────────────────────────────────────────────────

    // CRoaring-compatible portable binary serialization.
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buf;
        buf.reserve(sizeInBytes());

        uint32_t n = static_cast<uint32_t>(keys_.size());
        bool hasRuns = hasRunContainers();

        if (hasRuns) {
            uint32_t cookie = 12347u | ((n > 0 ? n - 1 : 0u) << 16);
            appendLE32(buf, cookie);
            uint32_t runBitsetBytes = (n + 7) / 8;
            for (uint32_t b = 0; b < runBitsetBytes; ++b) {
                uint8_t byte = 0;
                for (uint32_t bit = 0; bit < 8 && b * 8 + bit < n; ++bit) {
                    if (std::holds_alternative<RunContainer>(*containers_[b * 8 + bit]))
                        byte |= static_cast<uint8_t>(1u << bit);
                }
                buf.push_back(byte);
            }
        } else {
            appendLE32(buf, 12346u);
            appendLE32(buf, n);
        }

        for (uint32_t i = 0; i < n; ++i) {
            appendLE16(buf, keys_[i]);
            uint32_t card = detail::containerCardinality(*containers_[i]);
            appendLE16(buf, static_cast<uint16_t>(card - 1));
        }

        if (n >= 4) {
            uint32_t headerSize = static_cast<uint32_t>(buf.size()) + n * 4;
            uint32_t offset = headerSize;
            for (uint32_t i = 0; i < n; ++i) {
                appendLE32(buf, offset);
                offset += static_cast<uint32_t>(containerDataSize(*containers_[i]));
            }
        }

        for (uint32_t i = 0; i < n; ++i)
            serializeContainer(buf, *containers_[i]);

        return buf;
    }

    // CRoaring-compatible portable binary deserialization.
    static std::optional<RoaringBitmap> deserialize(const uint8_t* data, size_t len) {
        if (!data || len < 4) return std::nullopt;
        size_t pos = 0;
        uint32_t cookie = readLE32(data, pos); pos += 4;

        uint32_t n = 0;
        std::vector<bool> isRun;

        if ((cookie & 0xFFFF) == 12347u) {
            n = ((cookie >> 16) & 0xFFFF) + 1;
            uint32_t runBitsetBytes = (n + 7) / 8;
            if (pos + runBitsetBytes > len) return std::nullopt;
            isRun.resize(n, false);
            for (uint32_t i = 0; i < n; ++i) {
                if (data[pos + i / 8] & (1 << (i % 8)))
                    isRun[i] = true;
            }
            pos += runBitsetBytes;
        } else if (cookie == 12346u) {
            if (pos + 4 > len) return std::nullopt;
            n = readLE32(data, pos); pos += 4;
            isRun.resize(n, false);
        } else {
            return std::nullopt;
        }

        if (n > 65536) return std::nullopt;

        if (pos + static_cast<size_t>(n) * 4 > len) return std::nullopt;
        std::vector<uint16_t> keys(n);
        std::vector<uint32_t> cards(n);
        for (uint32_t i = 0; i < n; ++i) {
            keys[i] = readLE16(data, pos); pos += 2;
            cards[i] = static_cast<uint32_t>(readLE16(data, pos)) + 1; pos += 2;
        }

        if (n >= 4) {
            if (pos + static_cast<size_t>(n) * 4 > len) return std::nullopt;
            pos += static_cast<size_t>(n) * 4;
        }

        RoaringBitmap result;
        result.reserveChunks(n);
        for (uint32_t i = 0; i < n; ++i) {
            if (isRun[i]) {
                if (pos + 2 > len) return std::nullopt;
                uint16_t numRuns = readLE16(data, pos); pos += 2;
                size_t runBytes = static_cast<size_t>(numRuns) * 4;
                if (pos + runBytes > len) return std::nullopt;
                RunContainer rc;
                rc.runs.resize(numRuns);
                std::memcpy(rc.runs.data(), data + pos, runBytes);
                pos += runBytes;
                if (rc.cardinality() != cards[i]) return std::nullopt;
                result.pushChunk(keys[i], makeContainer(std::move(rc)));
            } else if (cards[i] > kArrayMaxSize) {
                if (pos + 8192 > len) return std::nullopt;
                BitmapContainer bm;
                std::memcpy(bm.words, data + pos, 8192);
                pos += 8192;
                bm.computeCardinality();
                if (bm.cardinality() != cards[i]) return std::nullopt;
                result.pushChunk(keys[i], makeContainer(std::move(bm)));
            } else {
                size_t dataBytes = static_cast<size_t>(cards[i]) * 2;
                if (pos + dataBytes > len) return std::nullopt;
                ArrayContainer ac;
                ac.values.resize(cards[i]);
                std::memcpy(ac.values.data(), data + pos, dataBytes);
                pos += dataBytes;
                result.pushChunk(keys[i], makeContainer(std::move(ac)));
            }
        }
        return result;
    }

    // Frozen serialization (v2 format with container type tags).
    std::vector<uint8_t> serializeFrozen() const {
        static constexpr uint32_t kFrozenMagic = 0x524F4152u;
        static constexpr uint8_t kFrozenVersion = 2;  // v2 with container types

        std::vector<uint8_t> buf;
        // Reserve approximate size
        buf.reserve(9 + keys_.size() * 32);

        auto writeU8 = [&](uint8_t v) { buf.push_back(v); };
        auto writeU16 = [&](uint16_t v) {
            buf.push_back(static_cast<uint8_t>(v & 0xFF));
            buf.push_back(static_cast<uint8_t>(v >> 8));
        };
        auto writeU32 = [&](uint32_t v) {
            for (int i = 0; i < 4; ++i)
                buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
        };
        writeU32(kFrozenMagic);
        writeU8(kFrozenVersion);
        writeU32(static_cast<uint32_t>(keys_.size()));

        for (size_t _ci3 = 0; _ci3 < keys_.size(); ++_ci3) {
            auto key = keys_[_ci3];
            const auto& c = containers_[_ci3];
            writeU16(key);
            std::visit([&](const auto& cont) {
                using T = std::decay_t<decltype(cont)>;
                if constexpr (std::is_same_v<T, ArrayContainer>) {
                    writeU8(0);  // array type
                    writeU32(static_cast<uint32_t>(cont.values.size()));
                    size_t bytes = cont.values.size() * 2;
                    size_t oldSize = buf.size();
                    buf.resize(oldSize + bytes);
                    std::memcpy(buf.data() + oldSize, cont.values.data(), bytes);
                } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                    writeU8(1);  // bitmap type
                    size_t oldSize = buf.size();
                    buf.resize(oldSize + 8192);
                    std::memcpy(buf.data() + oldSize, cont.words, 8192);
                } else {
                    writeU8(2);  // run type
                    writeU32(static_cast<uint32_t>(cont.runs.size()));
                    size_t bytes = cont.runs.size() * 4;
                    size_t oldSize = buf.size();
                    buf.resize(oldSize + bytes);
                    std::memcpy(buf.data() + oldSize, cont.runs.data(), bytes);
                }
            }, *c);
        }
        return buf;
    }

    // Frozen deserialization (v2 format).
    static std::optional<RoaringBitmap> deserializeFrozen(const uint8_t* data, size_t len) {
        static constexpr uint32_t kFrozenMagic = 0x524F4152u;

        if (!data || len < 9) return std::nullopt;
        size_t pos = 0;

        auto readU8 = [&]() -> std::optional<uint8_t> {
            if (pos + 1 > len) return std::nullopt;
            return data[pos++];
        };
        auto readU16 = [&]() -> std::optional<uint16_t> {
            if (pos + 2 > len) return std::nullopt;
            uint16_t v = static_cast<uint16_t>(data[pos]) |
                         (static_cast<uint16_t>(data[pos+1]) << 8);
            pos += 2;
            return v;
        };
        auto readU32 = [&]() -> std::optional<uint32_t> {
            if (pos + 4 > len) return std::nullopt;
            uint32_t v = 0;
            for (int i = 0; i < 4; ++i)
                v |= static_cast<uint32_t>(data[pos + i]) << (i * 8);
            pos += 4;
            return v;
        };
        auto magic = readU32();
        if (!magic || *magic != kFrozenMagic) return std::nullopt;

        auto version = readU8();
        if (!version) return std::nullopt;

        if (*version != 1 && *version != 2) return std::nullopt;

        auto numContainers = readU32();
        if (!numContainers) return std::nullopt;
        if (*numContainers > 65536) return std::nullopt;

        RoaringBitmap bm;
        for (uint32_t i = 0; i < *numContainers; ++i) {
            auto key = readU16();
            if (!key) return std::nullopt;

            auto type = readU8();
            if (!type) return std::nullopt;

            if (*type == 0) {  // ArrayContainer
                auto count = readU32();
                if (!count || *count > 65536) return std::nullopt;
                size_t bytes = static_cast<size_t>(*count) * 2;
                if (pos + bytes > len) return std::nullopt;
                ArrayContainer ac;
                ac.values.resize(*count);
                std::memcpy(ac.values.data(), data + pos, bytes);
                pos += bytes;
                bm.pushChunk(*key, makeContainer(std::move(ac)));
            } else if (*type == 1) {  // BitmapContainer
                if (pos + 8192 > len) return std::nullopt;
                BitmapContainer bc;
                std::memcpy(bc.words, data + pos, 8192);
                pos += 8192;
                bc.computeCardinality();
                bm.pushChunk(*key, makeContainer(std::move(bc)));
            } else if (*type == 2) {  // RunContainer
                auto numRuns = readU32();
                if (!numRuns || *numRuns > 32768) return std::nullopt;
                size_t bytes = static_cast<size_t>(*numRuns) * 4;
                if (pos + bytes > len) return std::nullopt;
                RunContainer rc;
                rc.runs.resize(*numRuns);
                std::memcpy(rc.runs.data(), data + pos, bytes);
                pos += bytes;
                bm.pushChunk(*key, makeContainer(std::move(rc)));
            } else {
                return std::nullopt;  // unknown container type
            }
        }
        return bm;
    }

    // Return serialized size (portable format) without allocating.
    size_t sizeInBytes() const {
        uint32_t n = static_cast<uint32_t>(keys_.size());
        bool hasRuns = hasRunContainers();
        size_t sz = 0;
        if (hasRuns) {
            sz += 4 + (n + 7) / 8;
        } else {
            sz += 8;
        }
        sz += static_cast<size_t>(n) * 4;
        if (n >= 4) sz += static_cast<size_t>(n) * 4;
        for (size_t _ci4 = 0; _ci4 < keys_.size(); ++_ci4)
        {
            auto& c = containers_[_ci4];
            sz += containerDataSize(*c);
        }
        return sz;
    }

    // ── Utility ─────────────────────────────────────────────────────────────

    std::vector<uint32_t> toVector() const {
        std::vector<uint32_t> result(cardinality());
        uint32_t* out = result.data();
        for (size_t ci = 0; ci < keys_.size(); ++ci) {
            uint32_t base = static_cast<uint32_t>(keys_[ci]) << 16;
            out += detail::containerToUint32Array(*containers_[ci], base, out);
        }
        result.resize(static_cast<size_t>(out - result.data()));
        return result;
    }

    template <typename Fn>
    void forEach(Fn&& fn) const {
        for (size_t ci = 0; ci < keys_.size(); ++ci) {
            uint32_t base = static_cast<uint32_t>(keys_[ci]) << 16;
            detail::containerForEach(*containers_[ci], base, fn);
        }
    }

    std::string toString() const {
        std::string s = "{";
        bool first = true;
        forEach([&](uint32_t val) {
            if (!first) s += ", ";
            s += std::to_string(val);
            first = false;
        });
        s += "}";
        return s;
    }

    // Check if this is a subset of other.
    bool isSubsetOf(const RoaringBitmap& o) const {
        return (*this - o).empty();
    }

    bool isStrictSubset(const RoaringBitmap& o) const {
        return isSubsetOf(o) && *this != o;
    }

    // ── select / rank / range operations ─────────────────────────────────

    // Return the element at 0-based rank, or nullopt if rank >= cardinality.
    std::optional<uint32_t> select(uint32_t rank) const {
        uint32_t remaining = rank;
        for (size_t _ci5 = 0; _ci5 < keys_.size(); ++_ci5) {
            auto key = keys_[_ci5];
            auto& c = containers_[_ci5];
            uint32_t card = detail::containerCardinality(*c);
            if (remaining < card) {
                uint32_t base = static_cast<uint32_t>(key) << 16;
                uint16_t lo = containerSelect(*c, remaining);
                return base + lo;
            }
            remaining -= card;
        }
        return std::nullopt;
    }

    // Count elements <= val.
    uint32_t rank(uint32_t val) const {
        uint16_t hi = static_cast<uint16_t>(val >> 16);
        uint16_t lo = static_cast<uint16_t>(val & 0xFFFF);
        uint32_t total = 0;

        for (size_t _ci6 = 0; _ci6 < keys_.size(); ++_ci6) {
            auto key = keys_[_ci6];
            auto& c = containers_[_ci6];
            if (key < hi) {
                total += detail::containerCardinality(*c);
            } else if (key == hi) {
                total += containerRank(*c, lo);
                break;
            } else {
                break;
            }
        }
        return total;
    }

    // Check if ALL values in [min, max) are present.
    bool containsRange(uint32_t min, uint32_t max) const {
        if (min >= max) return true;

        uint16_t hiStart = static_cast<uint16_t>(min >> 16);
        uint16_t hiEnd = static_cast<uint16_t>((max - 1) >> 16);

        for (uint32_t hi = hiStart; hi <= hiEnd; ++hi) {
            auto idx = findContainer(static_cast<uint16_t>(hi));
            if (idx == keys_.size()) return false;

            uint32_t lo_start = (hi == hiStart) ? (min & 0xFFFF) : 0;
            uint32_t lo_end = (hi == hiEnd) ? (((max - 1) & 0xFFFF) + 1u) : 65536u;

            if (!containerContainsRange(*containers_[idx],
                    static_cast<uint16_t>(lo_start), lo_end))
                return false;
        }
        return true;
    }

    // Count elements in [min, max).
    uint32_t rangeCardinality(uint32_t min, uint32_t max) const {
        if (min >= max) return 0;

        uint16_t hiStart = static_cast<uint16_t>(min >> 16);
        uint16_t hiEnd = static_cast<uint16_t>((max - 1) >> 16);
        uint32_t total = 0;

        for (size_t _ci7 = 0; _ci7 < keys_.size(); ++_ci7) {
            auto key = keys_[_ci7];
            auto& c = containers_[_ci7];
            if (key < hiStart || key > hiEnd) {
                if (key > hiEnd) break;
                continue;
            }
            uint32_t lo_start = (key == hiStart) ? (min & 0xFFFF) : 0;
            uint32_t lo_end = (key == hiEnd) ? (((max - 1) & 0xFFFF) + 1u) : 65536u;

            total += containerRangeCardinality(*c,
                static_cast<uint16_t>(lo_start), lo_end);
        }
        return total;
    }

    // In-place complement of [min, max).
    void flip(uint32_t min, uint32_t max) {
        if (min >= max) return;

        uint16_t hiStart = static_cast<uint16_t>(min >> 16);
        uint16_t hiEnd = static_cast<uint16_t>((max - 1) >> 16);

        for (uint32_t hi = hiStart; hi <= hiEnd; ++hi) {
            uint16_t chunkKey = static_cast<uint16_t>(hi);
            uint32_t lo_start = (hi == hiStart) ? (min & 0xFFFF) : 0;
            uint32_t lo_end = (hi == hiEnd) ? (((max - 1) & 0xFFFF) + 1u) : 65536u;

            auto idx = findContainer(chunkKey);
            if (idx == keys_.size()) {
                // No container exists — flipping means adding the whole range.
                auto& c = getOrCreateContainer(chunkKey);
                containerSetRange(c, static_cast<uint16_t>(lo_start), lo_end);
                // Check promotion.
                promoteIfNeeded(c);
            } else {
                auto& mc = cow(containers_[idx]);
                containerFlipRange(mc,
                    static_cast<uint16_t>(lo_start), lo_end);
                // Demote or clean up.
                demoteIfNeeded(mc);
                if (detail::containerCardinality(mc) == 0) {
                    eraseChunkAt(idx);
                }
            }
        }
    }

    RoaringBitmap flipped(uint32_t min, uint32_t max) const {
        RoaringBitmap result = *this;
        result.flip(min, max);
        return result;
    }

    // Remove all values in [min, max).
    void removeRange(uint32_t min, uint64_t max) {
        if (min >= max) return;
        if (max > 0x100000000ULL) max = 0x100000000ULL;
        if (min >= max) return;

        uint16_t hiStart = static_cast<uint16_t>(min >> 16);
        uint16_t hiEnd = static_cast<uint16_t>((max - 1) >> 16);

        // Iterate in reverse so erasing doesn't invalidate indices.
        for (int64_t hi = hiEnd; hi >= static_cast<int64_t>(hiStart); --hi) {
            auto idx = findContainer(static_cast<uint16_t>(hi));
            if (idx == keys_.size()) continue;

            uint32_t lo_start = (hi == hiStart) ? (min & 0xFFFF) : 0;
            uint32_t lo_end = (hi == hiEnd) ? (((max - 1) & 0xFFFF) + 1u) : 65536u;

            auto& mc = cow(containers_[idx]);
            containerClearRange(mc,
                static_cast<uint16_t>(lo_start), lo_end);
            demoteIfNeeded(mc);

            if (detail::containerCardinality(mc) == 0) {
                eraseChunkAt(idx);
            }
        }
    }

    // ── addOffset ────────────────────────────────────────────────────────────

    RoaringBitmap addOffset(int64_t offset) const {
        RoaringBitmap result;
        if (offset == 0) {
            result.keys_ = keys_;
            result.containers_ = containers_;
            return result;
        }

        for (size_t _ci8 = 0; _ci8 < keys_.size(); ++_ci8) {
            auto key = keys_[_ci8];
            const auto& c = containers_[_ci8];
            auto vals = containerToValues(*c);
            for (uint16_t lo : vals) {
                int64_t full = (static_cast<int64_t>(key) << 16) + lo + offset;
                if (full < 0 || full > static_cast<int64_t>(UINT32_MAX)) continue;
                result.add(static_cast<uint32_t>(full));
            }
        }
        return result;
    }

    // ── statistics ───────────────────────────────────────────────────────────

    RoaringStatistics statistics() const {
        RoaringStatistics s;
        s.numContainers = static_cast<uint32_t>(keys_.size());

        for (size_t _ci9 = 0; _ci9 < keys_.size(); ++_ci9) {
            auto key = keys_[_ci9];
            const auto& c = containers_[_ci9];
            uint32_t base = static_cast<uint32_t>(key) << 16;
            std::visit([&](const auto& cont) {
                using T = std::decay_t<decltype(cont)>;
                uint32_t card = cont.cardinality();
                if constexpr (std::is_same_v<T, ArrayContainer>) {
                    s.numArrayContainers++;
                    s.numValuesArrayContainers += card;
                    s.numBytesArrayContainers += cont.values.size() * sizeof(uint16_t);
                    for (uint16_t v : cont.values)
                        s.sumValue += base + v;
                } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                    s.numBitmapContainers++;
                    s.numValuesBitmapContainers += card;
                    s.numBytesBitmapContainers += sizeof(BitmapContainer::words);
                    for (uint32_t w = 0; w < kBitmapWords; ++w) {
                        uint64_t bits = cont.words[w];
                        while (bits) {
                            uint16_t lo = static_cast<uint16_t>((w << 6) | __builtin_ctzll(bits));
                            s.sumValue += base + lo;
                            bits &= bits - 1;
                        }
                    }
                } else {
                    s.numRunContainers++;
                    s.numValuesRunContainers += card;
                    s.numBytesRunContainers += cont.runs.size() * sizeof(typename T::Run);
                    for (const auto& r : cont.runs) {
                        for (uint32_t v = r.start; v <= static_cast<uint32_t>(r.start) + r.length; ++v)
                            s.sumValue += base + v;
                    }
                }
                s.numValues += card;
            }, *c);
        }

        if (!keys_.empty()) {
            s.minValue = minimum().value_or(0);
            s.maxValue = maximum().value_or(0);
        }
        return s;
    }

    // ── removeRunCompression ─────────────────────────────────────────────────

    void removeRunCompression() {
        for (size_t _ci10 = 0; _ci10 < keys_.size(); ++_ci10) {
            auto& c = containers_[_ci10];
            if (auto* rc = std::get_if<RunContainer>(&*c)) {
                uint32_t card = rc->cardinality();
                if (card <= kArrayMaxSize) {
                    ArrayContainer ac;
                    for (auto& r : rc->runs)
                        for (uint32_t v = r.start;
                             v <= static_cast<uint32_t>(r.start) + r.length; ++v)
                            ac.values.push_back(static_cast<uint16_t>(v));
                    c = makeContainer(std::move(ac));
                } else {
                    c = makeContainer(rc->toBitmap());
                }
            }
        }
    }

    // ── Lazy OR ─────────────────────────────────────────────────────────────

    RoaringBitmap lazyOr(const RoaringBitmap& o) const {
        RoaringBitmap result;
        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) {
                result.pushChunk(keys_[i], containers_[i]); ++i;
            } else if (keys_[i] > o.keys_[j]) {
                result.pushChunk(o.keys_[j], o.containers_[j]); ++j;
            } else {
                result.pushChunk(keys_[i], makeContainer(detail::containerLazyOr(*containers_[i], *o.containers_[j])));
                ++i; ++j;
            }
        }
        while (i < keys_.size()) { result.pushChunk(keys_[i], containers_[i]); ++i; }
        while (j < o.keys_.size()) { result.pushChunk(o.keys_[j], o.containers_[j]); ++j; }
        return result;
    }

    // ── Lazy OR in-place (force containers to bitmap for fast accumulation) ─

    void lazyOrInPlace(const RoaringBitmap& o) {
        if (this == &o) return;

        std::vector<uint16_t> rk;
        std::vector<ContainerPtr> rc;
        rk.reserve(keys_.size() + o.keys_.size());
        rc.reserve(keys_.size() + o.keys_.size());

        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) {
                rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i])); ++i;
            } else if (keys_[i] > o.keys_[j]) {
                rk.push_back(o.keys_[j]); rc.push_back(o.containers_[j]); ++j;
            } else {
                auto& c = cow(containers_[i]);
                detail::containerLazyOrInPlace(c, *o.containers_[j]);
                rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i]));
                ++i; ++j;
            }
        }
        while (i < keys_.size()) { rk.push_back(keys_[i]); rc.push_back(std::move(containers_[i])); ++i; }
        while (j < o.keys_.size()) { rk.push_back(o.keys_[j]); rc.push_back(o.containers_[j]); ++j; }

        keys_ = std::move(rk);
        containers_ = std::move(rc);
    }

    // ── Multi-way union (fastunion) ────────────────────────────────────────

    static RoaringBitmap fastunion(const std::vector<const RoaringBitmap*>& bitmaps) {
        if (bitmaps.empty()) return RoaringBitmap{};
        if (bitmaps.size() == 1) return *bitmaps[0];

        // Sequential lazy OR fold matching CRoaring's roaring_bitmap_or_many.
        //
        // Key design choices (matching CRoaring):
        // - Non-colliding containers are shared as-is (no bitmap promotion)
        // - Collisions promote to bitmap via containerLazyOrInPlace dispatch
        // - This avoids wasteful 8KB alloc+memset for containers that may
        //   never collide, and handles array+array natively when small

        // Step 1: lazy OR of first two bitmaps (containerLazyOr dispatch)
        RoaringBitmap result;
        {
            const auto& a = *bitmaps[0];
            const auto& b = *bitmaps[1];
            result.reserveChunks(a.keys_.size() + b.keys_.size());

            size_t i = 0, j = 0;
            while (i < a.keys_.size() && j < b.keys_.size()) {
                if (a.keys_[i] < b.keys_[j]) {
                    result.pushChunk(a.keys_[i], a.containers_[i]); ++i;
                } else if (a.keys_[i] > b.keys_[j]) {
                    result.pushChunk(b.keys_[j], b.containers_[j]); ++j;
                } else {
                    // Collision: lazy OR with bitset conversion.
                    // If neither is a bitmap, promote the first to bitmap,
                    // then OR the second into it (matches CRoaring's
                    // LAZY_OR_BITSET_CONVERSION behavior).
                    auto cptr = makeContainer(
                        detail::containerLazyOr(*a.containers_[i],
                                               *b.containers_[j]));
                    result.pushChunk(a.keys_[i], std::move(cptr));
                    ++i; ++j;
                }
            }
            while (i < a.keys_.size()) {
                result.pushChunk(a.keys_[i], a.containers_[i]); ++i;
            }
            while (j < b.keys_.size()) {
                result.pushChunk(b.keys_[j], b.containers_[j]); ++j;
            }
        }

        // Step 2: fold remaining bitmaps in-place
        for (size_t bi = 2; bi < bitmaps.size(); ++bi) {
            const auto& o = *bitmaps[bi];
            if (o.keys_.empty()) continue;

            size_t ri = 0, oi = 0;
            while (ri < result.keys_.size() && oi < o.keys_.size()) {
                if (result.keys_[ri] < o.keys_[oi]) {
                    ++ri;
                } else if (result.keys_[ri] > o.keys_[oi]) {
                    // New key — share container as-is (no bitmap promotion).
                    // It will be promoted on first collision, or stay
                    // compact if it never collides.
                    result.insertChunkAt(ri, o.keys_[oi], o.containers_[oi]);
                    ++ri; ++oi;
                } else {
                    // Matching key — lazy OR in-place with bitmap promotion.
                    auto& c = cow(result.containers_[ri]);
                    detail::containerLazyOrInPlace(c, *o.containers_[oi]);
                    ++ri; ++oi;
                }
            }
            while (oi < o.keys_.size()) {
                result.pushChunk(o.keys_[oi], o.containers_[oi]);
                ++oi;
            }
        }

        result.repairCardinality();
        return result;
    }

public:

    // ── Repair cardinality (fix lazy containers) ────────────────────────────

    void repairCardinality() {
        for (size_t i = 0; i < keys_.size(); ) {
            auto& c = containers_[i];
            // card is mutable, so computeCardinality() works through const.
            // Only cow if we need to demote.
            if (auto* bm = std::get_if<BitmapContainer>(&*c)) {
                if (bm->card == -1) {
                    bm->computeCardinality();
                    if (bm->cardinality() <= kArrayMaxSize) {
                        c = makeContainer(bm->toArray());
                    }
                }
            }
            if (detail::containerCardinality(*c) == 0) {
                eraseChunkAt(i);
            } else {
                ++i;
            }
        }
    }

    // ── Cardinality-only set operations ─────────────────────────────────────

    uint32_t andCardinality(const RoaringBitmap& o) const {
        uint32_t total = 0;
        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) ++i;
            else if (keys_[i] > o.keys_[j]) ++j;
            else {
                total += detail::containerAndCardinality(
                    *containers_[i], *o.containers_[j]);
                ++i; ++j;
            }
        }
        return total;
    }

    uint32_t orCardinality(const RoaringBitmap& o) const {
        uint32_t total = 0;
        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) {
                total += detail::containerCardinality(*containers_[i]);
                ++i;
            } else if (keys_[i] > o.keys_[j]) {
                total += detail::containerCardinality(*o.containers_[j]);
                ++j;
            } else {
                total += detail::containerOrCardinality(
                    *containers_[i], *o.containers_[j]);
                ++i; ++j;
            }
        }
        while (i < keys_.size()) {
            total += detail::containerCardinality(*containers_[i]);
            ++i;
        }
        while (j < o.keys_.size()) {
            total += detail::containerCardinality(*o.containers_[j]);
            ++j;
        }
        return total;
    }

    uint32_t xorCardinality(const RoaringBitmap& o) const {
        uint32_t total = 0;
        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) {
                total += detail::containerCardinality(*containers_[i]);
                ++i;
            } else if (keys_[i] > o.keys_[j]) {
                total += detail::containerCardinality(*o.containers_[j]);
                ++j;
            } else {
                total += detail::containerXorCardinality(
                    *containers_[i], *o.containers_[j]);
                ++i; ++j;
            }
        }
        while (i < keys_.size()) {
            total += detail::containerCardinality(*containers_[i]);
            ++i;
        }
        while (j < o.keys_.size()) {
            total += detail::containerCardinality(*o.containers_[j]);
            ++j;
        }
        return total;
    }

    uint32_t andNotCardinality(const RoaringBitmap& o) const {
        uint32_t total = 0;
        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) {
                total += detail::containerCardinality(*containers_[i]);
                ++i;
            } else if (keys_[i] > o.keys_[j]) {
                ++j;
            } else {
                total += detail::containerAndNotCardinality(
                    *containers_[i], *o.containers_[j]);
                ++i; ++j;
            }
        }
        while (i < keys_.size()) {
            total += detail::containerCardinality(*containers_[i]);
            ++i;
        }
        return total;
    }

    // ── Jaccard index ───────────────────────────────────────────────────────

    double jaccardIndex(const RoaringBitmap& o) const {
        uint32_t andCard = andCardinality(o);
        uint32_t orCard = orCardinality(o);
        if (orCard == 0) return 0.0;
        return static_cast<double>(andCard) / static_cast<double>(orCard);
    }

    // ── Intersects (early-exit intersection test) ───────────────────────────

    bool intersects(const RoaringBitmap& o) const {
        size_t i = 0, j = 0;
        while (i < keys_.size() && j < o.keys_.size()) {
            if (keys_[i] < o.keys_[j]) ++i;
            else if (keys_[i] > o.keys_[j]) ++j;
            else {
                if (detail::containerIntersects(
                        *containers_[i], *o.containers_[j]))
                    return true;
                ++i; ++j;
            }
        }
        return false;
    }

private:
    // Flat parallel arrays — keys_ is compact (2 bytes/entry) for cache-friendly
    // binary search. containers_ holds the COW pointers at the same indices.
    std::vector<uint16_t> keys_;
    std::vector<ContainerPtr> containers_;
    mutable size_t lastKeyIdx_ = 0;  // cached last-accessed key index

    // ── Chunk helpers ──────────────────────────────────────────────────
    size_t numChunks() const { return keys_.size(); }

    void pushChunk(uint16_t key, ContainerPtr c) {
        keys_.push_back(key);
        containers_.push_back(std::move(c));
    }

    void insertChunkAt(size_t pos, uint16_t key, ContainerPtr c) {
        keys_.insert(keys_.begin() + static_cast<ptrdiff_t>(pos), key);
        containers_.insert(containers_.begin() + static_cast<ptrdiff_t>(pos), std::move(c));
    }

    void eraseChunkAt(size_t pos) {
        keys_.erase(keys_.begin() + static_cast<ptrdiff_t>(pos));
        containers_.erase(containers_.begin() + static_cast<ptrdiff_t>(pos));
    }

    void reserveChunks(size_t n) {
        keys_.reserve(n);
        containers_.reserve(n);
    }

    void clearChunks() {
        keys_.clear();
        containers_.clear();
    }

    void shrinkChunks() {
        keys_.shrink_to_fit();
        containers_.shrink_to_fit();
    }

    // Copy-on-write: if shared, deep-copy before mutation.
    static Container& cow(ContainerPtr& ptr) {
        if (ptr.use_count() > 1) ptr = makeContainer(*ptr);
        return const_cast<Container&>(*ptr);
    }

    // Binary search for chunk index. Returns numChunks() if not found.
    // Checks cached position first for sequential access patterns.
    size_t findContainer(uint16_t key) const {
        if (lastKeyIdx_ < keys_.size() && keys_[lastKeyIdx_] == key)
            return lastKeyIdx_;
        auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
        if (it != keys_.end() && *it == key) {
            lastKeyIdx_ = static_cast<size_t>(it - keys_.begin());
            return lastKeyIdx_;
        }
        return keys_.size();
    }

    // Get or create container for chunk key. Returns mutable ref via cow().
    // Checks cached position first for sequential access patterns.
    Container& getOrCreateContainer(uint16_t key) {
        if (lastKeyIdx_ < keys_.size() && keys_[lastKeyIdx_] == key)
            return cow(containers_[lastKeyIdx_]);
        auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
        if (it != keys_.end() && *it == key) {
            size_t idx = static_cast<size_t>(it - keys_.begin());
            lastKeyIdx_ = idx;
            return cow(containers_[idx]);
        }
        size_t idx = static_cast<size_t>(it - keys_.begin());
        insertChunkAt(idx, key, makeContainer(ArrayContainer{}));
        lastKeyIdx_ = idx;
        return const_cast<Container&>(*containers_[idx]);
    }

    // Sorted addMany implementation with position-hinted container lookup.
    // Input MUST be sorted (non-decreasing). Groups by high-16 key and
    // batch-inserts per container. O(n) for sorted input into a sorted bitmap.
    void addManySorted(const uint32_t* vals, size_t n) {
        size_t hint = 0;  // Position hint for key lookup.
        size_t i = 0;
        while (i < n) {
            uint16_t hi = static_cast<uint16_t>(vals[i] >> 16);

            // Position-hinted lookup: since keys are monotonically non-decreasing,
            // start binary search from 'hint' instead of the beginning.
            auto startIt = keys_.begin() + static_cast<ptrdiff_t>(hint);
            auto it = std::lower_bound(startIt, keys_.end(), hi);
            size_t idx = static_cast<size_t>(it - keys_.begin());

            if (it == keys_.end() || *it != hi) {
                // Create new container at the insertion point.
                insertChunkAt(idx, hi, makeContainer(ArrayContainer{}));
            }
            hint = idx;  // Next group's key is >= hi, so start from here.

            auto& c = cow(containers_[idx]);

            // Collect all values with same high key.
            size_t groupStart = i;
            while (i < n && static_cast<uint16_t>(vals[i] >> 16) == hi) ++i;

            // Batch insert into this container.
            std::visit([&](auto& cont) {
                using T = std::decay_t<decltype(cont)>;
                if constexpr (std::is_same_v<T, ArrayContainer>) {
                    bool allAboveMax = cont.values.empty() ||
                        static_cast<uint16_t>(vals[groupStart] & 0xFFFF) > cont.values.back();

                    if (allAboveMax) {
                        // O(1) append per value — skip binary search entirely.
                        for (size_t j = groupStart; j < i; ++j) {
                            uint16_t lo = static_cast<uint16_t>(vals[j] & 0xFFFF);
                            if (!cont.values.empty() && cont.values.back() == lo)
                                continue;
                            cont.values.push_back(lo);
                        }
                    } else {
                        for (size_t j = groupStart; j < i; ++j) {
                            uint16_t lo = static_cast<uint16_t>(vals[j] & 0xFFFF);
                            cont.add(lo);
                        }
                    }
                    if (cont.cardinality() > kArrayMaxSize) {
                        c = detail::arrayToBitmap(cont);
                    }
                } else {
                    for (size_t j = groupStart; j < i; ++j) {
                        uint16_t lo = static_cast<uint16_t>(vals[j] & 0xFFFF);
                        cont.add(lo);
                    }
                }
            }, c);
        }
    }

    // Helper to convert any container to a sorted vector of 16-bit values.
    static std::vector<uint16_t> containerToValues(const Container& c) {
        return std::visit([](const auto& cont) -> std::vector<uint16_t> {
            using T = std::decay_t<decltype(cont)>;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                return cont.values;
            } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                return cont.toArray().values;
            } else {
                return cont.toArray().values;
            }
        }, c);
    }

    // Helper to convert container to full 32-bit vector.
    static std::vector<uint32_t> containerToVector(const Container& c, uint16_t key) {
        auto vals = containerToValues(c);
        uint32_t base = static_cast<uint32_t>(key) << 16;
        std::vector<uint32_t> result;
        result.reserve(vals.size());
        for (uint16_t v : vals) result.push_back(base + v);
        return result;
    }

    // ── Container-level helpers for select/rank/range ──────────────────

    static uint16_t containerSelect(const Container& c, uint32_t rank) {
        return std::visit([rank](const auto& cont) -> uint16_t {
            using T = std::decay_t<decltype(cont)>;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                return cont.values[rank];
            } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                uint32_t remaining = rank;
                for (uint32_t w = 0; w < kBitmapWords; ++w) {
                    uint32_t pop = static_cast<uint32_t>(__builtin_popcountll(cont.words[w]));
                    if (remaining < pop) {
                        uint64_t bits = cont.words[w];
                        for (uint32_t i = 0; i < remaining; ++i)
                            bits &= bits - 1;  // clear lowest set bit
                        return static_cast<uint16_t>((w << 6) | __builtin_ctzll(bits));
                    }
                    remaining -= pop;
                }
                return 0;  // shouldn't reach here
            } else {
                // RunContainer
                uint32_t remaining = rank;
                for (auto& r : cont.runs) {
                    uint32_t runSize = static_cast<uint32_t>(r.length) + 1;
                    if (remaining < runSize) {
                        return static_cast<uint16_t>(r.start + remaining);
                    }
                    remaining -= runSize;
                }
                return 0;  // shouldn't reach here
            }
        }, c);
    }

    static uint32_t containerRank(const Container& c, uint16_t val) {
        return std::visit([val](const auto& cont) -> uint32_t {
            using T = std::decay_t<decltype(cont)>;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                auto it = std::upper_bound(cont.values.begin(), cont.values.end(), val);
                return static_cast<uint32_t>(it - cont.values.begin());
            } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                uint32_t word = val >> 6;
                // SIMD popcount for full words prefix
                uint32_t total = simd::bitmap_popcount_n(cont.words, word);
                // Count bits in the target word up to and including val's bit.
                uint64_t mask = (val & 63) == 63 ? ~0ULL : ((1ULL << ((val & 63) + 1)) - 1);
                total += static_cast<uint32_t>(__builtin_popcountll(cont.words[word] & mask));
                return total;
            } else {
                // RunContainer
                uint32_t total = 0;
                for (auto& r : cont.runs) {
                    if (val < r.start) break;
                    uint16_t end = r.start + r.length;
                    if (val <= end) {
                        total += val - r.start + 1;
                        break;
                    }
                    total += static_cast<uint32_t>(r.length) + 1;
                }
                return total;
            }
        }, c);
    }

    // Check if container contains all values in [lo_start, lo_end).
    // lo_end can be up to 65536.
    static bool containerContainsRange(const Container& c,
            uint16_t lo_start, uint32_t lo_end) {
        return std::visit([lo_start, lo_end](const auto& cont) -> bool {
            using T = std::decay_t<decltype(cont)>;
            uint32_t rangeSize = lo_end - lo_start;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                auto it = std::lower_bound(cont.values.begin(), cont.values.end(), lo_start);
                if (static_cast<uint32_t>(cont.values.end() - it) < rangeSize) return false;
                // Check that we have a contiguous sequence.
                for (uint32_t i = 0; i < rangeSize; ++i) {
                    if (it[i] != static_cast<uint16_t>(lo_start + i)) return false;
                }
                return true;
            } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                uint32_t firstWord = lo_start >> 6;
                uint32_t lastWord = (lo_end - 1) >> 6;

                if (firstWord == lastWord) {
                    uint64_t mask = (~0ULL << (lo_start & 63));
                    if ((lo_end & 63) != 0)
                        mask &= (1ULL << (lo_end & 63)) - 1;
                    return (cont.words[firstWord] & mask) == mask;
                }

                // First partial word.
                uint64_t firstMask = ~0ULL << (lo_start & 63);
                if ((cont.words[firstWord] & firstMask) != firstMask) return false;

                // Full middle words.
                for (uint32_t w = firstWord + 1; w < lastWord; ++w) {
                    if (cont.words[w] != ~0ULL) return false;
                }

                // Last partial word.
                uint64_t lastMask = (lo_end & 63) != 0
                    ? (1ULL << (lo_end & 63)) - 1
                    : ~0ULL;
                return (cont.words[lastWord] & lastMask) == lastMask;
            } else {
                // RunContainer — check that runs cover [lo_start, lo_end).
                uint32_t pos = lo_start;
                for (auto& r : cont.runs) {
                    if (pos >= lo_end) return true;
                    uint32_t rEnd = static_cast<uint32_t>(r.start) + r.length;
                    if (r.start > pos) return false;  // gap
                    if (rEnd >= lo_end - 1) return true;  // fully covered
                    if (pos <= rEnd) pos = rEnd + 1;
                }
                return pos >= lo_end;
            }
        }, c);
    }

    // Count elements in [lo_start, lo_end) within a single container.
    static uint32_t containerRangeCardinality(const Container& c,
            uint16_t lo_start, uint32_t lo_end) {
        return std::visit([lo_start, lo_end](const auto& cont) -> uint32_t {
            using T = std::decay_t<decltype(cont)>;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                auto lo_it = std::lower_bound(cont.values.begin(), cont.values.end(), lo_start);
                auto hi_it = (lo_end <= 0xFFFF)
                    ? std::lower_bound(lo_it, cont.values.end(), static_cast<uint16_t>(lo_end))
                    : cont.values.end();
                return static_cast<uint32_t>(hi_it - lo_it);
            } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                if (lo_start >= lo_end) return 0;
                uint32_t firstWord = lo_start >> 6;
                uint32_t lastWord = (lo_end - 1) >> 6;
                uint32_t total = 0;

                if (firstWord == lastWord) {
                    uint64_t mask = (~0ULL << (lo_start & 63));
                    if ((lo_end & 63) != 0)
                        mask &= (1ULL << (lo_end & 63)) - 1;
                    return static_cast<uint32_t>(__builtin_popcountll(cont.words[firstWord] & mask));
                }

                // First partial word.
                uint64_t firstMask = ~0ULL << (lo_start & 63);
                total += static_cast<uint32_t>(__builtin_popcountll(cont.words[firstWord] & firstMask));

                // Full middle words.
                for (uint32_t w = firstWord + 1; w < lastWord; ++w)
                    total += static_cast<uint32_t>(__builtin_popcountll(cont.words[w]));

                // Last partial word.
                uint64_t lastMask = (lo_end & 63) != 0
                    ? (1ULL << (lo_end & 63)) - 1
                    : ~0ULL;
                total += static_cast<uint32_t>(__builtin_popcountll(cont.words[lastWord] & lastMask));
                return total;
            } else {
                // RunContainer
                uint32_t total = 0;
                for (auto& r : cont.runs) {
                    uint32_t rStart = r.start;
                    uint32_t rEnd = static_cast<uint32_t>(r.start) + r.length;
                    if (rStart >= lo_end) break;
                    if (rEnd < lo_start) continue;
                    uint32_t overlapStart = std::max(static_cast<uint32_t>(lo_start), rStart);
                    uint32_t overlapEnd = std::min(lo_end - 1, rEnd);
                    if (overlapStart <= overlapEnd)
                        total += overlapEnd - overlapStart + 1;
                }
                return total;
            }
        }, c);
    }

    // Set range [lo_start, lo_end) in a container (for flip on nonexistent chunk).
    static void containerSetRange(Container& c, uint16_t lo_start, uint32_t lo_end) {
        std::visit([lo_start, lo_end, &c](auto& cont) {
            using T = std::decay_t<decltype(cont)>;
            uint32_t rangeSize = lo_end - lo_start;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                if (rangeSize > kArrayMaxSize) {
                    BitmapContainer bm;
                    bm.setRange(lo_start, lo_end <= 65536 ? static_cast<uint32_t>(lo_end) : 65536);
                    bm.computeCardinality();
                    c = std::move(bm);
                } else {
                    for (uint32_t v = lo_start; v < lo_end; ++v)
                        cont.add(static_cast<uint16_t>(v));
                }
            } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                cont.setRange(lo_start, lo_end <= 65536 ? static_cast<uint32_t>(lo_end) : 65536);
                cont.computeCardinality();
            } else {
                cont.addRange(lo_start, static_cast<uint16_t>(lo_end > 65535 ? 65535 : lo_end));
            }
        }, c);
    }

    // Flip range [lo_start, lo_end) in an existing container using XOR semantics.
    static void containerFlipRange(Container& c, uint16_t lo_start, uint32_t lo_end) {
        // Convert to bitmap for the flip, then demote if needed.
        BitmapContainer bm = std::visit([](const auto& cont) -> BitmapContainer {
            using T = std::decay_t<decltype(cont)>;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                return detail::arrayToBitmap(cont);
            } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                return cont;
            } else {
                return cont.toBitmap();
            }
        }, c);

        // XOR the range.
        uint32_t end = lo_end <= 65536 ? static_cast<uint32_t>(lo_end) : 65536;
        uint32_t firstWord = lo_start >> 6;
        uint32_t lastWord = (end - 1) >> 6;
        bm.card = -1;

        if (firstWord == lastWord) {
            uint64_t mask = (~0ULL << (lo_start & 63));
            if ((end & 63) != 0)
                mask &= (1ULL << (end & 63)) - 1;
            bm.words[firstWord] ^= mask;
        } else {
            bm.words[firstWord] ^= (~0ULL << (lo_start & 63));
            for (uint32_t w = firstWord + 1; w < lastWord; ++w)
                bm.words[w] ^= ~0ULL;
            if ((end & 63) != 0)
                bm.words[lastWord] ^= (1ULL << (end & 63)) - 1;
            else
                bm.words[lastWord] ^= ~0ULL;
        }
        bm.computeCardinality();
        c = detail::maybedemote(std::move(bm));
    }

    // Clear range [lo_start, lo_end) in a container.
    static void containerClearRange(Container& c, uint16_t lo_start, uint32_t lo_end) {
        std::visit([lo_start, lo_end, &c](auto& cont) {
            using T = std::decay_t<decltype(cont)>;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                auto lo_it = std::lower_bound(cont.values.begin(), cont.values.end(), lo_start);
                auto hi_it = (lo_end <= 0xFFFF)
                    ? std::lower_bound(lo_it, cont.values.end(), static_cast<uint16_t>(lo_end))
                    : cont.values.end();
                cont.values.erase(lo_it, hi_it);
            } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                cont.clearRange(lo_start, lo_end <= 65536 ? static_cast<uint32_t>(lo_end) : 65536);
                cont.computeCardinality();
            } else {
                // RunContainer — convert to bitmap, clear, convert back.
                BitmapContainer bm = cont.toBitmap();
                bm.clearRange(lo_start, lo_end <= 65536 ? static_cast<uint32_t>(lo_end) : 65536);
                bm.computeCardinality();
                c = detail::maybedemote(std::move(bm));
            }
        }, c);
    }

    // Promote ArrayContainer to BitmapContainer if too large.
    static void promoteIfNeeded(Container& c) {
        if (auto* a = std::get_if<ArrayContainer>(&c)) {
            if (a->cardinality() > kArrayMaxSize) {
                c = detail::arrayToBitmap(*a);
            }
        }
    }

    // Demote BitmapContainer to ArrayContainer if small enough.
    static void demoteIfNeeded(Container& c) {
        if (auto* b = std::get_if<BitmapContainer>(&c)) {
            if (b->cardinality() <= kArrayMaxSize) {
                c = b->toArray();
            }
        }
    }

    // Convert any container to RunContainer.
    static RunContainer toRunContainer(const Container& c) {
        auto vals = containerToValues(c);
        RunContainer rc;
        if (vals.empty()) return rc;
        uint16_t start = vals[0];
        uint16_t prev = vals[0];
        for (size_t i = 1; i < vals.size(); ++i) {
            if (vals[i] == prev + 1) {
                prev = vals[i];
            } else {
                rc.appendRun(start, static_cast<uint16_t>(prev - start));
                start = vals[i];
                prev = vals[i];
            }
        }
        rc.appendRun(start, static_cast<uint16_t>(prev - start));
        return rc;
    }

    // ── Serialization helpers ────────────────────────────────────────────

    bool hasRunContainers() const {
        for (size_t _ci11 = 0; _ci11 < keys_.size(); ++_ci11) {
            auto& c = containers_[_ci11];
            if (std::holds_alternative<RunContainer>(*c)) return true;
        }
        return false;
    }

    static size_t containerDataSize(const Container& c) {
        return std::visit([](const auto& cont) -> size_t {
            using T = std::decay_t<decltype(cont)>;
            if constexpr (std::is_same_v<T, ArrayContainer>)
                return cont.values.size() * 2;
            else if constexpr (std::is_same_v<T, BitmapContainer>)
                return 8192;
            else
                return 2 + cont.runs.size() * 4;  // uint16 numRuns + pairs
        }, c);
    }

    static void serializeContainer(std::vector<uint8_t>& buf, const Container& c) {
        std::visit([&buf](const auto& cont) {
            using T = std::decay_t<decltype(cont)>;
            if constexpr (std::is_same_v<T, ArrayContainer>) {
                // Bulk memcpy: array of uint16_t is already LE on LE systems.
                size_t bytes = cont.values.size() * 2;
                size_t oldSize = buf.size();
                buf.resize(oldSize + bytes);
                std::memcpy(buf.data() + oldSize, cont.values.data(), bytes);
            } else if constexpr (std::is_same_v<T, BitmapContainer>) {
                // Bulk memcpy: 8KB bitmap words are already LE on LE systems.
                size_t oldSize = buf.size();
                buf.resize(oldSize + 8192);
                std::memcpy(buf.data() + oldSize, cont.words, 8192);
            } else {
                appendLE16(buf, static_cast<uint16_t>(cont.runs.size()));
                // Bulk memcpy: Run is {uint16_t, uint16_t} = 4 bytes, packed.
                size_t bytes = cont.runs.size() * 4;
                size_t oldSize = buf.size();
                buf.resize(oldSize + bytes);
                std::memcpy(buf.data() + oldSize, cont.runs.data(), bytes);
            }
        }, c);
    }

    static void appendLE16(std::vector<uint8_t>& buf, uint16_t val) {
        size_t oldSize = buf.size();
        buf.resize(oldSize + 2);
        std::memcpy(buf.data() + oldSize, &val, 2);
    }

    static void appendLE32(std::vector<uint8_t>& buf, uint32_t val) {
        size_t oldSize = buf.size();
        buf.resize(oldSize + 4);
        std::memcpy(buf.data() + oldSize, &val, 4);
    }

    static void appendLE64(std::vector<uint8_t>& buf, uint64_t val) {
        size_t oldSize = buf.size();
        buf.resize(oldSize + 8);
        std::memcpy(buf.data() + oldSize, &val, 8);
    }

    static uint16_t readLE16(const uint8_t* data, size_t pos) {
        uint16_t val;
        std::memcpy(&val, data + pos, 2);
        return val;
    }

    static uint32_t readLE32(const uint8_t* data, size_t pos) {
        uint32_t val;
        std::memcpy(&val, data + pos, 4);
        return val;
    }

    static uint64_t readLE64(const uint8_t* data, size_t pos) {
        uint64_t val;
        std::memcpy(&val, data + pos, 8);
        return val;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// RoaringBitmapView — zero-copy frozen deserialization (Gap 15)
//
// A read-only view over a frozen-serialized buffer. No heap allocation for
// container data: ArrayContainer values and BitmapContainer words point directly
// into the buffer. The buffer must outlive this object.
//
// Supports: contains, cardinality, empty, minimum, maximum, iteration,
//           toRoaringBitmap (materialization).
// Does NOT support: mutation, set operations (materialize first).
// ─────────────────────────────────────────────────────────────────────────────

class RoaringBitmapView {
public:
    RoaringBitmapView() = default;

    // Parse a frozen v2 buffer into a zero-copy view.
    // Returns nullopt on invalid data. The buffer must remain valid and
    // unmodified for the lifetime of this view.
    static std::optional<RoaringBitmapView> fromFrozen(const uint8_t* data, size_t len) {
        static constexpr uint32_t kFrozenMagic = 0x524F4152u;

        if (!data || len < 9) return std::nullopt;
        size_t pos = 0;

        auto readU8 = [&]() -> std::optional<uint8_t> {
            if (pos + 1 > len) return std::nullopt;
            return data[pos++];
        };
        auto readU16 = [&]() -> std::optional<uint16_t> {
            if (pos + 2 > len) return std::nullopt;
            uint16_t v;
            std::memcpy(&v, data + pos, 2);
            pos += 2;
            return v;
        };
        auto readU32 = [&]() -> std::optional<uint32_t> {
            if (pos + 4 > len) return std::nullopt;
            uint32_t v;
            std::memcpy(&v, data + pos, 4);
            pos += 4;
            return v;
        };

        auto magic = readU32();
        if (!magic || *magic != kFrozenMagic) return std::nullopt;

        auto version = readU8();
        if (!version || (*version != 1 && *version != 2)) return std::nullopt;

        auto numContainers = readU32();
        if (!numContainers || *numContainers > 65536) return std::nullopt;

        RoaringBitmapView view;
        view.data_ = data;
        view.len_ = len;
        view.chunks_.reserve(*numContainers);

        for (uint32_t i = 0; i < *numContainers; ++i) {
            auto key = readU16();
            if (!key) return std::nullopt;

            auto type = readU8();
            if (!type) return std::nullopt;

            if (*type == 0) {  // ArrayContainer
                auto count = readU32();
                if (!count || *count > 65536) return std::nullopt;
                size_t bytes = static_cast<size_t>(*count) * 2;
                if (pos + bytes > len) return std::nullopt;
                view.chunks_.push_back({*key, ViewChunk::Array,
                    data + pos, *count, *count});
                pos += bytes;
            } else if (*type == 1) {  // BitmapContainer
                if (pos + 8192 > len) return std::nullopt;
                const auto* words = reinterpret_cast<const uint64_t*>(data + pos);
                uint32_t card = simd::bitmap_popcount_harley_seal(words);
                view.chunks_.push_back({*key, ViewChunk::Bitmap,
                    data + pos, kBitmapWords, card});
                pos += 8192;
            } else if (*type == 2) {  // RunContainer
                auto numRuns = readU32();
                if (!numRuns || *numRuns > 32768) return std::nullopt;
                size_t bytes = static_cast<size_t>(*numRuns) * 4;
                if (pos + bytes > len) return std::nullopt;
                const auto* runs = reinterpret_cast<const RunContainer::Run*>(data + pos);
                uint32_t card = 0;
                for (uint32_t r = 0; r < *numRuns; ++r)
                    card += runs[r].length + 1;
                view.chunks_.push_back({*key, ViewChunk::Run,
                    data + pos, *numRuns, card});
                pos += bytes;
            } else {
                return std::nullopt;
            }
        }
        return view;
    }

    bool contains(uint32_t val) const {
        uint16_t hi = static_cast<uint16_t>(val >> 16);
        uint16_t lo = static_cast<uint16_t>(val & 0xFFFF);
        auto idx = findChunk(hi);
        if (idx == chunks_.size()) return false;
        return chunkContains(chunks_[idx], lo);
    }

    uint32_t cardinality() const {
        uint32_t total = 0;
        for (auto& ch : chunks_) total += ch.card;
        return total;
    }

    bool empty() const { return chunks_.empty(); }

    std::optional<uint32_t> minimum() const {
        if (chunks_.empty()) return std::nullopt;
        auto& ch = chunks_.front();
        uint32_t base = static_cast<uint32_t>(ch.key) << 16;
        switch (ch.type) {
            case ViewChunk::Array: {
                auto span = ch.arraySpan();
                return span.empty() ? std::nullopt : std::optional(base + span[0]);
            }
            case ViewChunk::Bitmap: {
                auto words = ch.bitmapWords();
                for (uint32_t w = 0; w < kBitmapWords; ++w) {
                    if (words[w])
                        return base + (w << 6) + __builtin_ctzll(words[w]);
                }
                return std::nullopt;
            }
            case ViewChunk::Run: {
                auto runs = ch.runSpan();
                return runs.empty() ? std::nullopt : std::optional(base + runs[0].start);
            }
        }
        return std::nullopt;
    }

    std::optional<uint32_t> maximum() const {
        if (chunks_.empty()) return std::nullopt;
        auto& ch = chunks_.back();
        uint32_t base = static_cast<uint32_t>(ch.key) << 16;
        switch (ch.type) {
            case ViewChunk::Array: {
                auto span = ch.arraySpan();
                return span.empty() ? std::nullopt : std::optional(base + span.back());
            }
            case ViewChunk::Bitmap: {
                auto words = ch.bitmapWords();
                for (int w = kBitmapWords - 1; w >= 0; --w) {
                    if (words[w])
                        return base + (w << 6) + 63 - __builtin_clzll(words[w]);
                }
                return std::nullopt;
            }
            case ViewChunk::Run: {
                auto runs = ch.runSpan();
                return runs.empty() ? std::nullopt
                    : std::optional(base + runs.back().start + runs.back().length);
            }
        }
        return std::nullopt;
    }

    // Materialize into a full (owning) RoaringBitmap by re-deserializing.
    std::optional<RoaringBitmap> toRoaringBitmap() const {
        if (!data_) return std::nullopt;
        return RoaringBitmap::deserializeFrozen(data_, len_);
    }

    // Forward iterator over all values in the view.
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using value_type = uint32_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const uint32_t*;
        using reference = uint32_t;

        Iterator() = default;

        Iterator(const RoaringBitmapView* view, size_t chunkIdx)
            : view_(view), chunkIdx_(chunkIdx) {
            if (chunkIdx_ < view_->chunks_.size())
                loadChunk();
        }

        uint32_t operator*() const {
            uint32_t base = static_cast<uint32_t>(view_->chunks_[chunkIdx_].key) << 16;
            return base + currentValues_[posInChunk_];
        }

        Iterator& operator++() {
            ++posInChunk_;
            if (posInChunk_ >= currentSize_) {
                ++chunkIdx_;
                posInChunk_ = 0;
                if (chunkIdx_ < view_->chunks_.size())
                    loadChunk();
                else
                    currentSize_ = 0;
            }
            return *this;
        }

        Iterator operator++(int) { Iterator tmp = *this; ++*this; return tmp; }

        bool operator==(const Iterator& o) const {
            if (!view_ && !o.view_) return true;
            if (!view_ || !o.view_) return false;
            if (chunkIdx_ >= view_->chunks_.size() && o.chunkIdx_ >= o.view_->chunks_.size())
                return true;
            return chunkIdx_ == o.chunkIdx_ && posInChunk_ == o.posInChunk_;
        }
        bool operator!=(const Iterator& o) const { return !(*this == o); }

    private:
        void loadChunk() {
            auto& ch = view_->chunks_[chunkIdx_];
            posInChunk_ = 0;
            switch (ch.type) {
                case ViewChunk::Array: {
                    // Zero-copy: point directly into the buffer.
                    auto span = ch.arraySpan();
                    currentValues_ = span.data();
                    currentSize_ = static_cast<uint32_t>(span.size());
                    break;
                }
                case ViewChunk::Bitmap: {
                    // Must materialize bit positions.
                    materialized_.resize(ch.card);
                    simd::bitmap_to_array(ch.bitmapWords(), materialized_.data());
                    currentValues_ = materialized_.data();
                    currentSize_ = ch.card;
                    break;
                }
                case ViewChunk::Run: {
                    // Must materialize run values.
                    materialized_.clear();
                    materialized_.reserve(ch.card);
                    auto runs = ch.runSpan();
                    for (auto& r : runs) {
                        for (uint32_t v = r.start;
                             v <= static_cast<uint32_t>(r.start) + r.length; ++v)
                            materialized_.push_back(static_cast<uint16_t>(v));
                    }
                    currentValues_ = materialized_.data();
                    currentSize_ = static_cast<uint32_t>(materialized_.size());
                    break;
                }
            }
        }

        const RoaringBitmapView* view_ = nullptr;
        size_t chunkIdx_ = 0;
        uint32_t posInChunk_ = 0;
        uint32_t currentSize_ = 0;
        const uint16_t* currentValues_ = nullptr;
        std::vector<uint16_t> materialized_;
    };

    Iterator begin() const { return Iterator(this, 0); }
    Iterator end() const { return Iterator(this, chunks_.size()); }

    std::vector<uint32_t> toVector() const {
        std::vector<uint32_t> result;
        result.reserve(cardinality());
        for (auto v : *this) result.push_back(v);
        return result;
    }

private:
    struct ViewChunk {
        enum Type : uint8_t { Array = 0, Bitmap = 1, Run = 2 };

        uint16_t key;
        Type type;
        const uint8_t* ptr;  // raw pointer into the frozen buffer
        uint32_t count;       // array: num values, bitmap: kBitmapWords, run: num runs
        uint32_t card;        // cached cardinality

        std::span<const uint16_t> arraySpan() const {
            return {reinterpret_cast<const uint16_t*>(ptr), count};
        }
        const uint64_t* bitmapWords() const {
            return reinterpret_cast<const uint64_t*>(ptr);
        }
        std::span<const RunContainer::Run> runSpan() const {
            return {reinterpret_cast<const RunContainer::Run*>(ptr), count};
        }
    };

    size_t findChunk(uint16_t key) const {
        auto it = std::lower_bound(chunks_.begin(), chunks_.end(), key,
            [](const ViewChunk& ch, uint16_t k) { return ch.key < k; });
        if (it != chunks_.end() && it->key == key)
            return static_cast<size_t>(it - chunks_.begin());
        return chunks_.size();
    }

    static bool chunkContains(const ViewChunk& ch, uint16_t lo) {
        switch (ch.type) {
            case ViewChunk::Array: {
                auto span = ch.arraySpan();
                return std::binary_search(span.begin(), span.end(), lo);
            }
            case ViewChunk::Bitmap: {
                auto words = ch.bitmapWords();
                return (words[lo >> 6] & (1ULL << (lo & 63))) != 0;
            }
            case ViewChunk::Run: {
                auto runs = ch.runSpan();
                auto it = std::upper_bound(runs.begin(), runs.end(), lo,
                    [](uint16_t val, const RunContainer::Run& r) { return val < r.start; });
                if (it == runs.begin()) return false;
                --it;
                return lo <= it->start + it->length;
            }
        }
        return false;
    }

    const uint8_t* data_ = nullptr;
    size_t len_ = 0;
    std::vector<ViewChunk> chunks_;
};

}  // namespace arrow
