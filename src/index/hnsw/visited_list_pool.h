#pragma once

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

    VisitedList(const VisitedList&) = delete;
    VisitedList& operator=(const VisitedList&) = delete;
    VisitedList(VisitedList&&) = default;
    VisitedList& operator=(VisitedList&&) = default;

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
    size_t numElements_;

public:
    VisitedListPool(size_t initMaxPools, size_t numElementsParam)
        : numElements_(numElementsParam) {
        for (size_t i = 0; i < initMaxPools; ++i) {
            pool_.push_front(new VisitedList(numElements_));
        }
    }

    VisitedListPool(const VisitedListPool&) = delete;
    VisitedListPool& operator=(const VisitedListPool&) = delete;

    VisitedList* getFreeVisitedList() {
        VisitedList* vlist;
        {
            std::unique_lock lock(poolGuard_);
            if (!pool_.empty()) {
                vlist = pool_.front();
                pool_.pop_front();
            } else {
                vlist = new VisitedList(numElements_);
            }
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
