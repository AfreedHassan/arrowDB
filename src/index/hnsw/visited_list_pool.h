#pragma once

#include <atomic>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>

namespace hnsw {

using epoch_t = uint16_t;

class VisitedList {
public:
    epoch_t curEpoch;
    epoch_t* visitedEpoch;
    size_t numElements;

    explicit VisitedList(size_t numElementsParam)
        : curEpoch(std::numeric_limits<epoch_t>::max()),
          numElements(numElementsParam) {
            visitedEpoch = new epoch_t[numElements];
          }

    ~VisitedList() {
        delete[] visitedEpoch;
    }

    VisitedList(const VisitedList&) = delete;
    VisitedList& operator=(const VisitedList&) = delete;

    VisitedList(VisitedList&& other) noexcept
        : curEpoch(other.curEpoch),
          visitedEpoch(other.visitedEpoch),
          numElements(other.numElements) {
        other.visitedEpoch = nullptr;
    }

    VisitedList& operator=(VisitedList&& other) noexcept {
        if (this != &other) {
            delete[] visitedEpoch;
            curEpoch = other.curEpoch;
            visitedEpoch = other.visitedEpoch;
            numElements = other.numElements;
            other.visitedEpoch = nullptr;
        }
        return *this;
    }

    void reset() noexcept {
        ++curEpoch;
        if (curEpoch == 0) {
            std::memset(visitedEpoch, 0, sizeof(epoch_t) * numElements);
            ++curEpoch;
        }
    }
};

class VisitedListPool {
    std::deque<VisitedList *> pool_;
    std::mutex poolGuard_;
    std::atomic<size_t> numElements_;

public:
    VisitedListPool(size_t initMaxPools, size_t numElementsParam)
        : numElements_(numElementsParam) {
        for (size_t i = 0; i < initMaxPools; ++i) {
            pool_.push_front(new VisitedList(numElements_));
        }
    }

    VisitedListPool(const VisitedListPool&) = delete;
    VisitedListPool& operator=(const VisitedListPool&) = delete;

    /// Update the expected number of elements. Existing pooled lists that are
    /// too small will be discarded on next acquire.
    void resize(size_t newNumElements) {
        numElements_.store(newNumElements, std::memory_order_release);
    }

    VisitedList* getFreeVisitedList() {
        VisitedList* vlist = nullptr;
        size_t requiredSize = numElements_.load(std::memory_order_acquire);
        {
            std::unique_lock lock(poolGuard_);
            while (!pool_.empty()) {
                vlist = pool_.front();
                pool_.pop_front();
                // Discard lists that are too small for the current index size
                if (vlist->numElements >= requiredSize) {
                    break;
                }
                delete vlist;
                vlist = nullptr;
            }
        }
        if (!vlist) {
            vlist = new VisitedList(requiredSize);
        }
        vlist->reset();
        return vlist;
    }

    void releaseVisitedList(VisitedList* vlist) {
        std::unique_lock lock(poolGuard_);
        pool_.push_front(vlist);
    }

    ~VisitedListPool() {
        while (pool_.size()) {
            VisitedList *rez = pool_.front();
            pool_.pop_front();
            delete rez;
        }
    }

};

}  // namespace hnsw
