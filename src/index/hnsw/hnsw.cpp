#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <random>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "lib.h"
#include "visited_list_pool.h"
#include "backend_registry.h"
#include "impl_kernels.h"

namespace hnsw {
  using tableidx_t = uint32_t;
  using linklistsize_t = uint32_t;

  template <typename dist_t>
  class HierarchicalNSW;

  template <typename dist_t>
  class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    private:
      static const tableidx_t MAX_LABEL_OPERATION_LOCKS = 65536;
      static const unsigned char DELETE_MARK = 0x01;

      size_t maxElements_{0};
      mutable std::atomic<size_t> elementCount_{0};
      mutable std::atomic<size_t> deletedElementCount_{0};

      size_t M_{0};
      size_t maxM_{0};
      size_t maxM0_{0};
      size_t efConstruction_{0};
      size_t efSearch_{0};

      // kElementSize_ = kLvl0AdjListSize_ + kElemVecSize_ + sizeof(label_t);
      size_t kElementSize_{0}; // size_data_per_element_
      size_t kElemVecSize_{0}; // data_size_
      size_t kAdjListSize_{0}; // size_links_level0_

      size_t kVectorOffset_{0};
      size_t kLvl0AdjListOffset_{0};
      size_t kLabelOffset_{0};
      size_t kLvl0AdjListSize_{0}; // size_links_level0_


      double levelMult_{0.0};
      double invLevelMult_{0.0};
      int maxLevel_{0};


      static constexpr tableidx_t INVALID_ID = std::numeric_limits<tableidx_t>::max();
      tableidx_t entryPoint_ = INVALID_ID;


      char *pElementsBlock_{nullptr};
      char **pAdjListsBlock_{nullptr};
      std::vector<int> elementLevels_{0};

      pdistfunc_t<dist_t> distFunc_{nullptr};
      pbatchdistfunc_t<dist_t> batchDistFunc_{nullptr};
      void* distFuncParam_{nullptr};
      std::size_t dim_{0};

      std::default_random_engine levelGen_;
      std::default_random_engine updateProbGen_;

      std::unique_ptr<VisitedListPool> visitedListPool_{nullptr};

      std::mutex globalMutex_;
      std::vector<std::mutex> adjacencyMutexes_;
      mutable std::vector<std::mutex> labelMutexes_;

      mutable std::mutex labelLookupMutex_;
      std::unordered_map<label_t, tableidx_t> labelLookup_;

      mutable std::atomic<uint64_t> metricDistanceCalls{0};
      mutable std::atomic<uint64_t> metricGraphHops{0};

      bool allowReuseDeleted_{false};
      std::mutex deletedElementMutex_;
      std::unordered_set<tableidx_t> deletedElementSet_;

    public:
      // Type definitions - must be before method declarations that use them
      using DistIdxPair = std::pair<dist_t, tableidx_t>;
      struct PairLess {
        constexpr bool operator()(std::pair<dist_t, tableidx_t> const& a,
            std::pair<dist_t, tableidx_t> const& b) const noexcept {
          return a.first < b.first;
        }
      };
      using CandidateQueue = std::priority_queue<DistIdxPair, std::vector<DistIdxPair>, PairLess>;

    public:
      explicit HierarchicalNSW(SpaceInterface<dist_t> *s) { }

      explicit HierarchicalNSW(
          SpaceInterface<dist_t> *s,
          const std::string &location,
          size_t maxElems = 0,
          bool allowReuseDeleted = false)
        : allowReuseDeleted_(allowReuseDeleted) {
          loadIndex(location, s, maxElems);
        }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t maxElems,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t randSeed = 100,
        bool allowReuseDeleted = false)
        : labelMutexes_(MAX_LABEL_OPERATION_LOCKS),
            adjacencyMutexes_(maxElems),
            elementLevels_(maxElems),
            allowReuseDeleted_(allowReuseDeleted) {
        maxElements_ = maxElems;
        deletedElementCount_ = 0;
        kElemVecSize_ = s->getDataSize();
        distFunc_ = s->getDistFunc();
        batchDistFunc_ = s->getBatchDistFunc();
        if (batchDistFunc_ == nullptr) {
            throw std::runtime_error("SpaceInterface must provide a batch distance function");
        }
        distFuncParam_ = s->getDistFuncParam();
        dim_ = *static_cast<std::size_t*>(distFuncParam_);
        if (M <= 10000) {
            M_ = M;
        } else {
          std::cerr << "warning: M parameter exceeds 10000. This may lead to adverse effects." << std::endl;
          std::cerr << "Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        efConstruction_ = std::max(ef_construction, M_);
        efSearch_ = 10;

        levelGen_.seed(randSeed);
        updateProbGen_.seed(randSeed + 1);

        kLvl0AdjListSize_ = maxM0_ * sizeof(tableidx_t) + sizeof(linklistsize_t);
        kLvl0AdjListOffset_ = 0;
        kVectorOffset_ = kLvl0AdjListSize_;
        kLabelOffset_ = kLvl0AdjListSize_ + kElemVecSize_;
        kElementSize_ = kLvl0AdjListSize_ + kElemVecSize_ + sizeof(label_t);

        pElementsBlock_ = (char *) malloc(maxElements_ * kElementSize_);
        if (pElementsBlock_ == nullptr)
            throw std::runtime_error("Not enough memory");

        elementCount_ = 0;

        visitedListPool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, maxElements_));

        // initializations for special treatment of the first node

        entryPoint_ = INVALID_ID;
        maxLevel_ = -1;

        pAdjListsBlock_ = (char **) malloc(sizeof(void *) * maxElements_);
        if (pAdjListsBlock_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        kAdjListSize_ = maxM_ * sizeof(tableidx_t) + sizeof(linklistsize_t);
        levelMult_ = 1 / log(1.0 * M_);
        invLevelMult_ = 1.0 / levelMult_;
    }

      // Method implementations are at the end of the class
      size_t size() const noexcept { return elementCount_; }
      size_t getDeletedCount() const noexcept { return deletedElementCount_; }

    ~HierarchicalNSW() {
        clear();
    }

    void saveIndex(const std::string &location) override {
        std::ofstream output(location, std::ios::binary);

        // WRITE HEADER 
        write(output, kLvl0AdjListOffset_);
        write(output, maxElements_);
        write(output, elementCount_);
        write(output, kElementSize_);
        write(output, kLabelOffset_);
        write(output, kVectorOffset_);
        write(output, maxLevel_);
        write(output, entryPoint_);
        write(output, maxM_);

        write(output, maxM0_);
        write(output, M_);
        write(output, levelMult_);
        write(output, efConstruction_);
        // END WRITE HEADER 

        // WRITE ELEMENTS BLOCK
        output.write(pElementsBlock_, elementCount_ * kElementSize_);
        // END WRITE ELEMENTS BLOCK

        // WRITE ADJACENCY LISTS
        for (size_t i = 0; i < elementCount_; i++) {
            unsigned int adjLSize = elementLevels_[i] > 0 ? kAdjListSize_ * elementLevels_[i] : 0;
            write(output, adjLSize);
            if (adjLSize)
                output.write(pAdjListsBlock_[i], adjLSize);
        }
        // END WRITE ADJACENCY LISTS
        output.close();
    }

      void loadIndex(const std::string &location, SpaceInterface<dist_t> *pSpace, size_t maxElemsParam = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
          throw std::runtime_error("HNSW index file could not be opened");

        clear();

        input.seekg(0, input.end);
        std::streampos totalFileSize = input.tellg();
        input.seekg(0, input.beg);

        // HEADER REGION
        read(input, kLvl0AdjListOffset_);
        read(input, maxElements_);
        read(input, elementCount_);

        maxElements_ = std::max(maxElemsParam, elementCount_.load());
        read(input, kElementSize_);
        read(input, kLabelOffset_);
        read(input, kVectorOffset_);
        read(input, maxLevel_);
        read(input, entryPoint_);

        read(input, maxM_);
        read(input, maxM0_);
        read(input, M_);
        read(input, levelMult_);
        read(input, efConstruction_);
        // END HEADER REGION

        kElemVecSize_ = pSpace->getDataSize();
        distFunc_ = pSpace->getDistFunc();
        distFuncParam_ = pSpace->getDistFuncParam();
        dim_ = *static_cast<std::size_t*>(distFuncParam_);

        // Get batch distance function from space (correctly selects L2 vs IP)
        batchDistFunc_ = pSpace->getBatchDistFunc();
        if (batchDistFunc_ == nullptr) {
            throw std::runtime_error("SpaceInterface must provide a batch distance function");
        }

        auto pos = input.tellg();
        validateIndexFileBody(input, totalFileSize);
        input.seekg(pos, input.beg);

        // Allocate memory block for the elements
        pElementsBlock_ = static_cast<char*>(malloc(maxElements_ * kElementSize_));
        if (pElementsBlock_ == nullptr)
          throw std::runtime_error("Not enough memory: loadIndex failed to allocate vector data");

        // Read into the memory block
        input.read(pElementsBlock_, elementCount_ * kElementSize_);

        kAdjListSize_ = maxM_ * sizeof(tableidx_t) + sizeof(linklistsize_t);

        kLvl0AdjListSize_ = maxM0_ * sizeof(tableidx_t) + sizeof(linklistsize_t);

        std::vector<std::mutex>(maxElements_).swap(adjacencyMutexes_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(labelMutexes_);

        visitedListPool_.reset(new VisitedListPool(1, maxElements_));

        pAdjListsBlock_ = static_cast<char**>(malloc(sizeof(void *) * maxElements_));
        if (pAdjListsBlock_ == nullptr)
          throw std::runtime_error("Not enough memory: loadIndex failed to allocate adjacency lists");

        elementLevels_ = std::vector<int>(maxElements_);
        invLevelMult_ = 1.0 / levelMult_;
        efSearch_ = 10;

        // Read the adjacency lists
        for (size_t i = 0; i < elementCount_; i++) {
          labelLookup_[getExternalLabel(i)] = i;
          unsigned int adjacencyListBytes;
          read(input, adjacencyListBytes);
          if (adjacencyListBytes == 0) {
            elementLevels_[i] = 0;
            pAdjListsBlock_[i] = nullptr;
          } else {
            elementLevels_[i] = adjacencyListBytes / kAdjListSize_;
            pAdjListsBlock_[i] = (char *) malloc(adjacencyListBytes);
            if (pAdjListsBlock_[i] == nullptr)
              throw std::runtime_error("Not enough memory: loadIndex failed to allocate adjacency list");
            input.read(pAdjListsBlock_[i], adjacencyListBytes);
          }
          if (isMarkedDeleted(i)) {
            deletedElementCount_ += 1;
            if (allowReuseDeleted_)
              deletedElementSet_.insert(i);
          }
        }

        /* MOVED INTO FOR LOOP ABOVE
        for (size_t i = 0; i < elementCount_; i++) {
          if (isMarkedDeleted(i)) {
            deletedElementCount_ += 1;
            if (allowReuseDeleted_)
              deletedElementSet_.insert(i);
          }
        }
        */

        input.close();
      }

    void resizeIndex(size_t newMaxElems) {
        if (newMaxElems < elementCount_)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visitedListPool_.reset(new VisitedListPool(1, newMaxElems));

        elementLevels_.resize(newMaxElems);

        std::vector<std::mutex>(newMaxElems).swap(adjacencyMutexes_);

        // Reallocate base layer
        char * newPElementsBlock = (char *) realloc(pElementsBlock_, newMaxElems * kElementSize_);
        if (newPElementsBlock == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        pElementsBlock_ = newPElementsBlock;

        // Reallocate all other layers
        char ** newPAdjListsBlock = (char **) realloc(pAdjListsBlock_, sizeof(void *) * newMaxElems);
        if (newPAdjListsBlock == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        pAdjListsBlock_ = newPAdjListsBlock;

        maxElements_ = newMaxElems;
    }

      void clear() {
        free(pElementsBlock_);
        pElementsBlock_ = nullptr;
        for (tableidx_t i = 0; i < elementCount_; i++) {
          if (elementLevels_[i] > 0)
            free(pAdjListsBlock_[i]);
        }
        free(pAdjListsBlock_);
        labelLookup_.clear();
        deletedElementSet_.clear();
        pAdjListsBlock_ = nullptr;
        elementCount_ = 0;
        visitedListPool_.reset(nullptr);
      }

      /*
       * Checks the first 16 bits of the memory to see if the element is marked deleted.
       */
      bool isMarkedDeleted(tableidx_t internalId) const {
        unsigned char *adjLCur = ((unsigned char*)getAdjListL0(internalId)) + 2;
        return *adjLCur & DELETE_MARK;
      }

      void getNeighborsByHeuristic(CandidateQueue &topCandidates, const size_t M) {
        // Fast path
        if (topCandidates.empty() || M == 0) return;

        thread_local std::vector<DistIdxPair> candidates;
        thread_local std::vector<DistIdxPair> kept;
        candidates.clear();
        kept.clear();

        // Move heap -> vector (we'll rebuild a smaller heap at the end)
        while (!topCandidates.empty()) {
          candidates.push_back(topCandidates.top());
          topCandidates.pop();
        }

        const size_t n = candidates.size();
        if (n == 0) return;

        auto cmpAsc = [](const DistIdxPair &a, const DistIdxPair &b) {
          return a.first < b.first;
        };

        // If we have more than M, reduce the set first.
        if (n > M) {
          // Heuristic: if M is very small compared to n, nth_element + sort M is best.
          // Otherwise partial_sort (which produces the first M sorted) is fine.
          if (M * 20 < n) {
            std::nth_element(candidates.begin(), candidates.begin() + M, candidates.end(), cmpAsc);
            candidates.resize(M);
            std::sort(candidates.begin(), candidates.end(), cmpAsc); // now near -> far
          } else {
            std::partial_sort(candidates.begin(), candidates.begin() + M, candidates.end(), cmpAsc);
            candidates.resize(M); // first M are sorted ascending
          }
        } else {
          // Few candidates: just sort them ascending
          std::sort(candidates.begin(), candidates.end(), cmpAsc);
        }

        // Apply HNSW diversification heuristic: keep candidate if it's not closer to any selected neighbor
        kept.reserve(std::min(M, candidates.size()));
        for (const DistIdxPair &cand : candidates) {
          bool good = true;

          // pointer to candidate data
          const dist_t* pCandData = reinterpret_cast<const dist_t*>(getDataByInternalId(cand.second));

          for (const DistIdxPair &sel : kept) {
            const dist_t* pSelData = reinterpret_cast<const dist_t*>(getDataByInternalId(sel.second));
            // distance between selected neighbor and candidate
            dist_t inter = distFunc_(pSelData, pCandData, dim_);
            if (inter < cand.first) { // if selected neighbor is closer to cand than cand->query
              good = false;
              break;
            }
          }

          if (good) {
            kept.push_back(cand);
            if (kept.size() == M) break;
          }
        }

        // Push selected neighbors back into the provided heap
        for (const auto &p : kept)
          topCandidates.push(p);
      }

    /// Search the base layer using batch distance computation for efficiency.
    /// @param entryId   Entry point node for search
    /// @param queryPoint Query vector
    /// @param layer     Layer to search (0 = base layer)
    /// @param ef        Search expansion factor (number of candidates to track)
    /// @return Priority queue of (distance, internal_id) pairs
    CandidateQueue searchBaseLayer(tableidx_t entryId, const void* queryPoint, int layer, size_t ef) {
        VisitedList* vl = visitedListPool_->getFreeVisitedList();
        epoch_t* visited = vl->visitedEpoch;
        const epoch_t tag = vl->curEpoch;

        const dist_t* query = static_cast<const dist_t*>(queryPoint);

        // Thread-local scratch buffers - sized dynamically based on max neighbors
        thread_local std::vector<const dist_t*> neighborPtrs;
        thread_local std::vector<tableidx_t> unvisitedIds;
        thread_local std::vector<dist_t> distances;

        // Ensure buffers have sufficient capacity for max neighbors at this layer
        const size_t maxNeighbors = (layer == 0) ? maxM0_ : maxM_;
        if (neighborPtrs.capacity() < maxNeighbors) {
            neighborPtrs.reserve(maxNeighbors);
            unvisitedIds.reserve(maxNeighbors);
            distances.reserve(maxNeighbors);
        }

        CandidateQueue topCandidates;  // max-heap: largest dist at top for efficient pruning
        CandidateQueue candidateSet;   // max-heap via negated dist: work queue (smallest first)

        dist_t lowerBound;

        // Initialize with entry point
        if (!isMarkedDeleted(entryId)) {
            dist_t dist = distFunc_(query,
                reinterpret_cast<const dist_t*>(getDataByInternalId(entryId)), dim_);
            topCandidates.emplace(dist, entryId);
            lowerBound = dist;
            candidateSet.emplace(-dist, entryId);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, entryId);
        }
        visited[entryId] = tag;

        while (!candidateSet.empty()) {
            auto [negDist, curNodeId] = candidateSet.top();
            dist_t curDist = -negDist;

            // Early termination: no better candidates possible
            if (curDist > lowerBound && topCandidates.size() >= ef) {
                break;
            }
            candidateSet.pop();

            size_t unvisitedCount = 0;

            // Scope for adjacency lock - release before batch distance computation
            {
                std::unique_lock<std::mutex> lock(adjacencyMutexes_[curNodeId]);

                linklistsize_t* adjList = getAdjListAtLevel(curNodeId, layer);
                size_t numNeighbors = getListCount(adjList);
                tableidx_t* neighbors = reinterpret_cast<tableidx_t*>(
                    reinterpret_cast<char*>(adjList) + sizeof(linklistsize_t));

                // Resize buffers for this iteration
                neighborPtrs.resize(numNeighbors);
                unvisitedIds.resize(numNeighbors);
                distances.resize(numNeighbors);

                // Phase 1: Collect unvisited neighbors with prefetching
                for (size_t j = 0; j < numNeighbors; ++j) {
                    tableidx_t nid = neighbors[j];

                    // Prefetch ahead for next iterations
                    if (j + 2 < numNeighbors) {
                        impl::prefetchL1(&visited[neighbors[j + 2]]);
                        impl::prefetchL1(getDataByInternalId(neighbors[j + 2]));
                    }

                    if (visited[nid] != tag) {
                        visited[nid] = tag;
                        unvisitedIds[unvisitedCount] = nid;
                        neighborPtrs[unvisitedCount] = reinterpret_cast<const dist_t*>(
                            getDataByInternalId(nid));
                        ++unvisitedCount;
                    }
                }
            }  // Lock released here before batch computation

            if (unvisitedCount == 0) continue;

            // Phase 2: Batch distance computation (lock-free, SIMD-optimized)
            batchDistFunc_(query, neighborPtrs.data(), unvisitedCount, dim_, distances.data());

            // Phase 3: Process computed distances
            for (size_t j = 0; j < unvisitedCount; ++j) {
                tableidx_t neighborId = unvisitedIds[j];
                dist_t dist = distances[j];

                // Check if worth exploring
                if (topCandidates.size() < ef || lowerBound > dist) {
                    candidateSet.emplace(-dist, neighborId);

                    if (!isMarkedDeleted(neighborId)) {
                        topCandidates.emplace(dist, neighborId);
                    }

                    // Prune to maintain ef size
                    if (topCandidates.size() > ef) {
                        topCandidates.pop();
                    }

                    if (!topCandidates.empty()) {
                        lowerBound = topCandidates.top().first;
                    }
                }
            }
        }

        visitedListPool_->releaseVisitedList(vl);
        return topCandidates;
    }

    /// Legacy interface for backward compatibility - uses efConstruction_ as ef
    CandidateQueue searchBaseLayer(tableidx_t entryId, const void* queryPoint, int layer) {
        return searchBaseLayer(entryId, queryPoint, layer, efConstruction_);
    }

    template <bool UseController = false, typename Controller = BaseSearchController<dist_t>>
      CandidateQueue searchBaseLayer(
          tableidx_t entryId,
          const void* queryPoint,
          int layer,
          size_t ef,
          Controller* controller = nullptr) 
      {
        VisitedList* vl = visitedListPool_->getFreeVisitedList();
        epoch_t* visited = vl->visitedEpoch;
        const epoch_t tag = vl->curEpoch;

        const dist_t* query = static_cast<const dist_t*>(queryPoint);

        thread_local std::vector<const dist_t*> neighborPtrs;
        thread_local std::vector<tableidx_t> unvisitedIds;
        thread_local std::vector<dist_t> distances;

        const size_t maxNeighbors = (layer == 0) ? maxM0_ : maxM_;
        if (neighborPtrs.capacity() < maxNeighbors) {
          neighborPtrs.reserve(maxNeighbors);
          unvisitedIds.reserve(maxNeighbors);
          distances.reserve(maxNeighbors);
        }

        CandidateQueue topCandidates;   // max-heap: largest dist at top for pruning
        CandidateQueue candidateSet;    // max-heap via negated dist: work queue
        dist_t lowerBound;

        if (!isMarkedDeleted(entryId)) {
          dist_t dist = distFunc_(query,
              reinterpret_cast<const dist_t*>(getDataByInternalId(entryId)), dim_);
          topCandidates.emplace(dist, entryId);
          lowerBound = dist;
          candidateSet.emplace(-dist, entryId);

          if constexpr (UseController) {
            controller->onCandidateFound(entryId, queryPoint, dist);
          }
        } else {
          lowerBound = std::numeric_limits<dist_t>::max();
          candidateSet.emplace(-lowerBound, entryId);
          if constexpr (UseController) {
            controller->onCandidateDiscarded(entryId, queryPoint, lowerBound);
          }
        }
        visited[entryId] = tag;

        while (!candidateSet.empty()) {
          auto [negDist, curNodeId] = candidateSet.top();
          dist_t curDist = -negDist;

          if (curDist > lowerBound && topCandidates.size() >= ef) break;

          candidateSet.pop();
          size_t unvisitedCount = 0;

          {
            std::unique_lock<std::mutex> lock(adjacencyMutexes_[curNodeId]);

            linklistsize_t* adjList = getAdjListAtLevel(curNodeId, layer);
            size_t numNeighbors = getListCount(adjList);
            tableidx_t* neighbors = reinterpret_cast<tableidx_t*>(
                reinterpret_cast<char*>(adjList) + sizeof(linklistsize_t));

            neighborPtrs.resize(numNeighbors);
            unvisitedIds.resize(numNeighbors);
            distances.resize(numNeighbors);

            for (size_t j = 0; j < numNeighbors; ++j) {
              tableidx_t nid = neighbors[j];
              if (j + 2 < numNeighbors) {
                impl::prefetchL1(&visited[neighbors[j + 2]]);
                impl::prefetchL1(getDataByInternalId(neighbors[j + 2]));
              }

              if (visited[nid] != tag) {
                visited[nid] = tag;
                unvisitedIds[unvisitedCount] = nid;
                neighborPtrs[unvisitedCount] = reinterpret_cast<const dist_t*>(
                    getDataByInternalId(nid));
                ++unvisitedCount;
              }
            }
          }

          if (unvisitedCount == 0) continue;

          batchDistFunc_(query, neighborPtrs.data(), unvisitedCount, dim_, distances.data());

          for (size_t j = 0; j < unvisitedCount; ++j) {
            tableidx_t neighborId = unvisitedIds[j];
            dist_t dist = distances[j];

            bool worthExploring = true;
            if constexpr (UseController) {
              worthExploring = controller->isWorthExploring(dist, lowerBound);
            }

            if (worthExploring) {
              candidateSet.emplace(-dist, neighborId);

              if (!isMarkedDeleted(neighborId)) {
                topCandidates.emplace(dist, neighborId);
                if constexpr (UseController) {
                  controller->onCandidateFound(neighborId,
                      getDataByInternalId(neighborId),
                      dist);
                }
              } else if constexpr (UseController) {
                controller->onCandidateDiscarded(neighborId,
                    getDataByInternalId(neighborId),
                    dist);
              }

              if (topCandidates.size() > ef) topCandidates.pop();
              if (!topCandidates.empty()) lowerBound = topCandidates.top().first;
            }
          }
        }

        if constexpr (UseController) {
          // Convert CandidateQueue to vector for filterResults interface
          std::vector<std::pair<dist_t, label_t>> resultsVec;
          resultsVec.reserve(topCandidates.size());
          while (!topCandidates.empty()) {
            auto [dist, internalId] = topCandidates.top();
            resultsVec.emplace_back(dist, getExternalLabel(internalId));
            topCandidates.pop();
          }
          controller->filterResults(resultsVec);
          // Rebuild queue from filtered results (convert back to internal IDs)
          // Note: filterResults may have modified the vector
          for (const auto& [dist, label] : resultsVec) {
            std::unique_lock<std::mutex> lock(labelLookupMutex_);
            auto it = labelLookup_.find(label);
            if (it != labelLookup_.end()) {
              topCandidates.emplace(dist, it->second);
            }
          }
        }

        visitedListPool_->releaseVisitedList(vl);
        return topCandidates;
      }


    template <bool bare_bone_search = true, bool collect_metrics = false>
    CandidateQueue searchBaseLayerST(
        tableidx_t ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchController<dist_t>* controller = nullptr) const {
        VisitedList *vl = visitedListPool_->getFreeVisitedList();
        epoch_t *visited = vl->visitedEpoch;
        epoch_t tag = vl->curEpoch;

        const dist_t* query = static_cast<const dist_t*>(data_point);

        CandidateQueue topCandidates;
        CandidateQueue candidateSet;

        dist_t lowerBound;
        if (bare_bone_search ||
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = distFunc_(query, reinterpret_cast<const dist_t*>(ep_data), dim_);
            lowerBound = dist;
            topCandidates.emplace(dist, ep_id);
            if (!bare_bone_search && controller) {
                controller->onCandidateFound(getExternalLabel(ep_id), ep_data, dist);
            }
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }

        visited[ep_id] = tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableidx_t> current_node_pair = candidateSet.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (controller) {
                    flag_stop_search = controller->isSearchComplete(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && topCandidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidateSet.pop();

            tableidx_t current_node_id = current_node_pair.second;
            linklistsize_t *data = getAdjListL0(current_node_id);
            size_t size = getListCount((linklistsize_t*)data);
            if (collect_metrics) {
                metricGraphHops++;
                metricDistanceCalls+=size;
            }

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                if (visited[candidate_id] != tag) {
                    visited[candidate_id] = tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = distFunc_(query, reinterpret_cast<const dist_t*>(currObj1), dim_);

                    bool flagConsiderCandidate;
                    if (!bare_bone_search && controller) {
                        flagConsiderCandidate = controller->isWorthExploring(dist, lowerBound);
                    } else {
                        flagConsiderCandidate = topCandidates.size() < ef || lowerBound > dist;
                    }

                    if (flagConsiderCandidate) {
                        candidateSet.emplace(-dist, candidate_id);

                        if (bare_bone_search ||
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            topCandidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && controller) {
                                controller->onCandidateFound(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && controller) {
                            flag_remove_extra = controller->requiresPruning();
                        } else {
                            flag_remove_extra = topCandidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableidx_t id = topCandidates.top().second;
                            topCandidates.pop();
                            if (!bare_bone_search && controller) {
                                controller->onCandidateDiscarded(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = controller->requiresPruning();
                            } else {
                                flag_remove_extra = topCandidates.size() > ef;
                            }
                        }

                        if (!topCandidates.empty())
                            lowerBound = topCandidates.top().first;
                    }
                }
            }
        }

        visitedListPool_->releaseVisitedList(vl);
        return topCandidates;
    }

      void setEF(size_t ef) { efSearch_ = ef; }
      size_t getEF() const noexcept { return efSearch_; }

      unsigned short getListCount(linklistsize_t * ptr) const {
          return *reinterpret_cast<const unsigned short*>(ptr);
      }

      void setListCount(linklistsize_t* ptr, unsigned short size) const {
          *reinterpret_cast<unsigned short*>(ptr) = size;
      }

      inline std::mutex& labelMutex(label_t label) const {
        size_t mutexId = label & (MAX_LABEL_OPERATION_LOCKS - 1); // hash
        return labelMutexes_[mutexId];
      }

      inline label_t getExternalLabel(tableidx_t internal_id) const {
        label_t retLabel;
        memcpy(
          &retLabel,
          (pElementsBlock_ + internal_id * kElementSize_ + kLabelOffset_),
          sizeof(label_t));
        return retLabel;
      }


      inline void setExternalLabel(tableidx_t internalId, label_t label) const {
        memcpy(
            (pElementsBlock_ + internalId * kElementSize_ + kLabelOffset_),
            &label, 
            sizeof(label_t)
        );
      }

    inline label_t *getExternalLabeLPtr(tableidx_t internalId) const {
        return reinterpret_cast<label_t *>(
          (pElementsBlock_ + (internalId * kElementSize_) + kLabelOffset_)
        );
    }

    inline char *getVectorByInternalId(tableidx_t internalId) const {
        return 
          (pElementsBlock_ + (internalId * kElementSize_) + kVectorOffset_);
    }

    inline char *getDataByInternalId(tableidx_t internalId) const {
        return (pElementsBlock_ + (internalId * kElementSize_) + kVectorOffset_);
    }

    int getRandomLevel(double revSize) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        return static_cast<int>(-log(std::max(distribution(levelGen_), 1e-10)) * revSize
        );
    }

    size_t getMaxElements() {
        return maxElements_;
    }

    size_t getElementCount() {
        return elementCount_;
    }

    size_t getDeletedCount() {
        return deletedElementCount_;
    }

      linklistsize_t *getAdjListL0(tableidx_t internalId) const {
        return reinterpret_cast<linklistsize_t*>(
            pElementsBlock_ + (internalId * kElementSize_) + kLvl0AdjListOffset_);
      }

      linklistsize_t *getAdjListL0(tableidx_t internalId, char *pElemsBlock) const {
        return reinterpret_cast<linklistsize_t*>(
            pElemsBlock + (internalId * kElementSize_) + kLvl0AdjListOffset_);
      }

      linklistsize_t *getAdjList(tableidx_t internalId, int level) const {
        assert(level >= 1 && "getAdjList requires level >= 1; use getAdjListL0 for level 0");
        return reinterpret_cast<linklistsize_t*>(
            pAdjListsBlock_[internalId] + (level - 1) * kAdjListSize_);
      }

      linklistsize_t *getAdjListAtLevel(tableidx_t internalId, int level) const {
        return level == 0 ? getAdjListL0(internalId) : getAdjList(internalId, level);
      }

      inline void validateIndexFileBody(std::ifstream& input, std::streampos totalFileSize) {
        input.seekg(elementCount_ * kElementSize_, input.cur);

        for (size_t i = 0; i < elementCount_; i++) {
          if (input.tellg() < 0 || input.tellg() >= totalFileSize) {
            throw std::runtime_error("Index seems to be corrupted or unsupported");
          }

          unsigned int adjacencyListBytes;
          read(input, adjacencyListBytes);

          if (adjacencyListBytes != 0) {
            input.seekg(adjacencyListBytes, input.cur);
          }
        }

        if (input.tellg() != totalFileSize)
          throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
      }

    // ============================================================
    // Method Implementations
    // ============================================================

    std::vector<tableidx_t> getConnectionsWithLock(tableidx_t internalId, int level) {
        std::unique_lock<std::mutex> lock(adjacencyMutexes_[internalId]);
        linklistsize_t* data = getAdjListAtLevel(internalId, level);
        int size = getListCount(data);
        std::vector<tableidx_t> result(size);
        tableidx_t* ll = reinterpret_cast<tableidx_t*>(data + 1);
        memcpy(result.data(), ll, size * sizeof(tableidx_t));
        return result;
    }

    tableidx_t mutuallyConnectNewElement(
        const void* dataPoint,
        tableidx_t curC,
        CandidateQueue& topCandidates,
        int level,
        bool isUpdate) {

        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic(topCandidates, Mcurmax); // was M_

        if (topCandidates.size() > Mcurmax)
            throw std::runtime_error("Should not be more than M_ candidates returned by heuristic");

        // Extract selected neighbors
        std::vector<tableidx_t> selectedNeighbors;
        selectedNeighbors.reserve(Mcurmax);
        while (!topCandidates.empty()) {
            selectedNeighbors.push_back(topCandidates.top().second);
            topCandidates.pop();
        }

        tableidx_t nextClosestEntryPoint = selectedNeighbors.back();

        // Write connections to new element
        {
            std::unique_lock<std::mutex> lock(adjacencyMutexes_[curC], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }

            linklistsize_t* llCur = getAdjListAtLevel(curC, level);

            if (*llCur && !isUpdate) {
                throw std::runtime_error("Newly inserted element should have blank link list");
            }

            setListCount(llCur, selectedNeighbors.size());
            tableidx_t* data = reinterpret_cast<tableidx_t*>(llCur + 1);

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > elementLevels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make link on non-existent level");
                data[idx] = selectedNeighbors[idx];
            }
        }

        // Add bidirectional connections to neighbors
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock<std::mutex> lock(adjacencyMutexes_[selectedNeighbors[idx]]);

            linklistsize_t* llOther = getAdjListAtLevel(selectedNeighbors[idx], level);
            size_t szLinkListOther = getListCount(llOther);

            if (szLinkListOther > Mcurmax)
                throw std::runtime_error("Bad value of szLinkListOther");
            if (selectedNeighbors[idx] == curC)
                throw std::runtime_error("Trying to connect element to itself");
            if (level > elementLevels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make link on non-existent level");

            tableidx_t* data = reinterpret_cast<tableidx_t*>(llOther + 1);

            // Check if curC already present (for updates)
            bool isCurCPresent = false;
            if (isUpdate) {
                for (size_t j = 0; j < szLinkListOther; j++) {
                    if (data[j] == curC) {
                        isCurCPresent = true;
                        break;
                    }
                }
            }

            if (!isCurCPresent) {
                if (szLinkListOther < Mcurmax) {
                    // Room available - just add
                    data[szLinkListOther] = curC;
                    setListCount(llOther, szLinkListOther + 1);
                } else {
                    // Full - need to prune using heuristic
                    dist_t dMax = distFunc_(
                        reinterpret_cast<const dist_t*>(getDataByInternalId(curC)),
                        reinterpret_cast<const dist_t*>(getDataByInternalId(selectedNeighbors[idx])),
                        dim_);

                    CandidateQueue candidates;
                    candidates.emplace(dMax, curC);

                    for (size_t j = 0; j < szLinkListOther; j++) {
                        candidates.emplace(
                            distFunc_(
                                reinterpret_cast<const dist_t*>(getDataByInternalId(data[j])),
                                reinterpret_cast<const dist_t*>(getDataByInternalId(selectedNeighbors[idx])),
                                dim_),
                            data[j]);
                    }

                    getNeighborsByHeuristic(candidates, Mcurmax);

                    int indx = 0;
                    while (!candidates.empty()) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }
                    setListCount(llOther, indx);
                }
            }
        }

        return nextClosestEntryPoint;
    }

    void addPoint(const void* vectorData, label_t label, bool replaceDeleted = false) override {
        if (!allowReuseDeleted_ && replaceDeleted) {
            throw std::runtime_error("Replacement of deleted elements is disabled");
        }

        std::unique_lock<std::mutex> lockLabel(labelMutex(label));

        // Handle replace_deleted path
        if (replaceDeleted) {
            std::unique_lock<std::mutex> lockDeleted(deletedElementMutex_);
            if (!deletedElementSet_.empty()) {
                tableidx_t replacedId = *deletedElementSet_.begin();
                deletedElementSet_.erase(replacedId);
                lockDeleted.unlock();

                label_t replacedLabel = getExternalLabel(replacedId);
                setExternalLabel(replacedId, label);

                std::unique_lock<std::mutex> lockTable(labelLookupMutex_);
                labelLookup_.erase(replacedLabel);
                labelLookup_[label] = replacedId;
                lockTable.unlock();

                unmarkDeletedInternal(replacedId);
                memset(getAdjListL0(replacedId), 0, kLvl0AdjListSize_);
                if (elementLevels_[replacedId] > 0) {
                  memset(pAdjListsBlock_[replacedId], 0, kAdjListSize_ * elementLevels_[replacedId]);
                }
                updatePoint(vectorData, replacedId, 1.0f);
                return;
            }
        }

        // Main insertion path
        tableidx_t curC = 0;
        {
            std::unique_lock<std::mutex> lockTable(labelLookupMutex_);
            auto search = labelLookup_.find(label);

            // Update existing element
            if (search != labelLookup_.end()) {
                tableidx_t existingId = search->second;
                if (allowReuseDeleted_ && isMarkedDeleted(existingId)) {
                    throw std::runtime_error("Can't use addPoint to update deleted elements if replacement enabled");
                }
                lockTable.unlock();

                if (isMarkedDeleted(existingId)) {
                    unmarkDeletedInternal(existingId);
                }
                updatePoint(vectorData, existingId, 1.0f);
                return;
            }

            // Check capacity
            if (elementCount_ >= maxElements_) {
                throw std::runtime_error("Number of elements exceeds specified limit");
            }

            curC = elementCount_;
            elementCount_++;
            labelLookup_[label] = curC;
            
            // Zero L0 adjacency list for the new element
            auto* l0 = getAdjListL0(curC);
            memset(l0, 0, kLvl0AdjListSize_);
        }

        // Lock element and generate level
        std::unique_lock<std::mutex> lockEl(adjacencyMutexes_[curC]);
        int curLevel = getRandomLevel(invLevelMult_);
        elementLevels_[curC] = curLevel;

        // Global lock for entry point update
        std::unique_lock<std::mutex> tempLock(globalMutex_);
        int maxLevelCopy = maxLevel_;
        if (curLevel <= maxLevelCopy)
            tempLock.unlock();

        tableidx_t currObj = entryPoint_;
        tableidx_t entryPointCopy = entryPoint_;

        // Initialize element memory (zero entire element including L0 adjacency list)
        memset(pElementsBlock_ + curC * kElementSize_, 0, kElementSize_);

        // Copy label and vector data
        memcpy(getExternalLabeLPtr(curC), &label, sizeof(label_t));
        memcpy(getDataByInternalId(curC), vectorData, kElemVecSize_);

        // Allocate higher level links
        if (curLevel > 0) {
            // TODO: rethink whether to have curLevel or curLevel+1 here
            //pAdjListsBlock_[curC] = static_cast<char*>(malloc(kAdjListSize_ * curLevel + 1)); 
            pAdjListsBlock_[curC] = static_cast<char*>(malloc(kAdjListSize_ * curLevel)); 
            if (pAdjListsBlock_[curC] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            //memset(pAdjListsBlock_[curC], 0, kAdjListSize_ * curLevel + 1);
            memset(pAdjListsBlock_[curC], 0, kAdjListSize_ * curLevel);
        }

        // Insert into graph
        const dist_t* vecData = static_cast<const dist_t*>(vectorData);
        if (currObj != INVALID_ID) {
            // Greedy search through upper levels
            if (curLevel < maxLevelCopy) {
                dist_t curDist = distFunc_(vecData, reinterpret_cast<const dist_t*>(getDataByInternalId(currObj)), dim_);
                for (int level = maxLevelCopy; level > curLevel; level--) {
                    if (level > elementLevels_[currObj]) continue;
                    bool changed  = true; 
                    while (changed) {
                        changed = false;
                        std::unique_lock<std::mutex> lock(adjacencyMutexes_[currObj]);
                        linklistsize_t* data = getAdjList(currObj, level);
                        int size = getListCount(data);
                        tableidx_t* datal = reinterpret_cast<tableidx_t*>(data + 1);

                        for (int i = 0; i < size; i++) {
                            tableidx_t cand = datal[i];
                            dist_t d = distFunc_(vecData, reinterpret_cast<const dist_t*>(getDataByInternalId(cand)), dim_);
                            if (d < curDist) {
                                curDist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            // Connect at each level
            bool epDeleted = isMarkedDeleted(entryPointCopy);
            for (int level = std::min(curLevel, maxLevelCopy); level >= 0; level--) {
                CandidateQueue topCandidates = searchBaseLayer(currObj, vectorData, level);

                if (epDeleted) {
                    topCandidates.emplace(
                        distFunc_(vecData, reinterpret_cast<const dist_t*>(getDataByInternalId(entryPointCopy)), dim_),
                        entryPointCopy);
                    if (topCandidates.size() > efConstruction_)
                        topCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(vectorData, curC, topCandidates, level, false);
            }
        } else {
            // First element
            entryPoint_ = 0;
            maxLevel_ = curLevel;
        }

        // Update entry point if new max level
        if (curLevel > maxLevelCopy) {
            entryPoint_ = curC;
            maxLevel_ = curLevel;
        }
    }

    std::priority_queue<std::pair<dist_t, label_t>>
    searchKnn(const void* queryData, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const override {
        std::priority_queue<std::pair<dist_t, label_t>> result;
        if (elementCount_ == 0) return result;

        const dist_t* qData = static_cast<const dist_t*>(queryData);
        tableidx_t currObj = entryPoint_;
        dist_t curDist = distFunc_(qData, reinterpret_cast<const dist_t*>(getDataByInternalId(entryPoint_)), dim_);

        // Greedy descent through upper layers
        for (int level = maxLevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                linklistsize_t* data = getAdjList(currObj, level);
                int size = getListCount(data);
                metricGraphHops++;
                metricDistanceCalls += size;

                tableidx_t* datal = reinterpret_cast<tableidx_t*>(data + 1);
                for (int i = 0; i < size; i++) {
                    tableidx_t cand = datal[i];
                    if (cand > maxElements_)
                        throw std::runtime_error("cand error");

                    dist_t d = distFunc_(qData, reinterpret_cast<const dist_t*>(getDataByInternalId(cand)), dim_);
                    if (d < curDist) {
                        curDist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        // Search base layer
        CandidateQueue topCandidates;
        bool bareBoneSearch = !deletedElementCount_ && !isIdAllowed;

        if (bareBoneSearch) {
            topCandidates = searchBaseLayerST<true>(currObj, queryData, std::max(efSearch_, k), isIdAllowed);
        } else {
            topCandidates = searchBaseLayerST<false>(currObj, queryData, std::max(efSearch_, k), isIdAllowed);
        }

        // Trim to k results
        while (topCandidates.size() > k) {
            topCandidates.pop();
        }

        // Convert to external labels
        while (!topCandidates.empty()) {
            auto [dist, internalId] = topCandidates.top();
            result.emplace(dist, getExternalLabel(internalId));
            topCandidates.pop();
        }

        return result;
    }

    void markDelete(label_t label) {
        std::unique_lock<std::mutex> lockLabel(labelMutex(label));

        std::unique_lock<std::mutex> lockTable(labelLookupMutex_);
        auto search = labelLookup_.find(label);
        if (search == labelLookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableidx_t internalId = search->second;
        lockTable.unlock();

        markDeletedInternal(internalId);
    }

    void markDeletedInternal(tableidx_t internalId) {
        assert(internalId < elementCount_);
        if (!isMarkedDeleted(internalId)) {
            unsigned char* llCur = reinterpret_cast<unsigned char*>(getAdjListL0(internalId)) + 2;
            *llCur |= DELETE_MARK;
            deletedElementCount_++;
            if (allowReuseDeleted_) {
                std::unique_lock<std::mutex> lockDeleted(deletedElementMutex_);
                deletedElementSet_.insert(internalId);
            }
        } else {
            throw std::runtime_error("Element is already deleted");
        }
    }

    void unmarkDelete(label_t label) {
        std::unique_lock<std::mutex> lockLabel(labelMutex(label));

        std::unique_lock<std::mutex> lockTable(labelLookupMutex_);
        auto search = labelLookup_.find(label);
        if (search == labelLookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableidx_t internalId = search->second;
        lockTable.unlock();

        unmarkDeletedInternal(internalId);
    }

    void unmarkDeletedInternal(tableidx_t internalId) {
        assert(internalId < elementCount_);
        if (isMarkedDeleted(internalId)) {
            unsigned char* llCur = reinterpret_cast<unsigned char*>(getAdjListL0(internalId)) + 2;
            *llCur &= ~DELETE_MARK;
            deletedElementCount_--;
            if (allowReuseDeleted_) {
                std::unique_lock<std::mutex> lockDeleted(deletedElementMutex_);
                deletedElementSet_.erase(internalId);
            }
        } else {
            throw std::runtime_error("Element is not deleted");
        }
    }

    void updatePoint(const void* dataPoint, tableidx_t internalId, float updateNeighborProbability) {
        // Update vector data
        memcpy(getDataByInternalId(internalId), dataPoint, kElemVecSize_);

        int maxLevelCopy = maxLevel_;
        tableidx_t entryPointCopy = entryPoint_;

        // If single element and it's the entry point, nothing more to do
        if (entryPointCopy == internalId && elementCount_ == 1)
            return;

        int elemLevel = elementLevels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableidx_t> sCand;
            std::unordered_set<tableidx_t> sNeigh;

            std::vector<tableidx_t> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.empty())
                continue;

            sCand.insert(internalId);

            for (auto& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(updateProbGen_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableidx_t> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto& neigh : sNeigh) {
                CandidateQueue candidates;
                size_t size = sCand.count(neigh) == 0 ? sCand.size() : sCand.size() - 1;
                size_t elementsToKeep = std::min(efConstruction_, size);

                for (auto& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = distFunc_(
                        reinterpret_cast<const dist_t*>(getDataByInternalId(neigh)),
                        reinterpret_cast<const dist_t*>(getDataByInternalId(cand)),
                        dim_);

                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else if (distance < candidates.top().first) {
                        candidates.pop();
                        candidates.emplace(distance, cand);
                    }
                }

                getNeighborsByHeuristic(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock<std::mutex> lock(adjacencyMutexes_[neigh]);
                    linklistsize_t* llCur = getAdjListAtLevel(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(llCur, candSize);
                    tableidx_t* data = reinterpret_cast<tableidx_t*>(llCur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }

    void repairConnectionsForUpdate(
        const void* dataPoint,
        tableidx_t entryPointInternalId,
        tableidx_t dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {

        tableidx_t currObj = entryPointInternalId;
        const dist_t* dp = static_cast<const dist_t*>(dataPoint);

        if (dataPointLevel < maxLevel) {
            dist_t curDist = distFunc_(dp, reinterpret_cast<const dist_t*>(getDataByInternalId(currObj)), dim_);

            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    std::unique_lock<std::mutex> lock(adjacencyMutexes_[currObj]);
                    linklistsize_t* data = getAdjListAtLevel(currObj, level);
                    int size = getListCount(data);
                    tableidx_t* datal = reinterpret_cast<tableidx_t*>(data + 1);

                    for (int i = 0; i < size; i++) {
                        tableidx_t cand = datal[i];
                        dist_t d = distFunc_(dp, reinterpret_cast<const dist_t*>(getDataByInternalId(cand)), dim_);
                        if (d < curDist) {
                            curDist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            CandidateQueue topCandidates = searchBaseLayer(currObj, dataPoint, level);

            // Filter out the point being updated
            CandidateQueue filteredCandidates;
            while (!topCandidates.empty()) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredCandidates.push(topCandidates.top());
                topCandidates.pop();
            }

            if (!filteredCandidates.empty()) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    const dist_t* dp = static_cast<const dist_t*>(dataPoint);
                    filteredCandidates.emplace(
                        distFunc_(dp, reinterpret_cast<const dist_t*>(getDataByInternalId(entryPointInternalId)), dim_),
                        entryPointInternalId);
                    if (filteredCandidates.size() > efConstruction_)
                        filteredCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredCandidates, level, true);
            }
        }
    }

    void checkIntegrity() {
        int connectionsChecked = 0;
        std::vector<int> inboundConnectionsNum(elementCount_, 0);

        for (size_t i = 0; i < elementCount_; i++) {
            for (int l = 0; l <= elementLevels_[i]; l++) {
                linklistsize_t* llCur = getAdjListAtLevel(i, l);
                int size = getListCount(llCur);
                tableidx_t* data = reinterpret_cast<tableidx_t*>(llCur + 1);
                std::unordered_set<tableidx_t> s;

                for (int j = 0; j < size; j++) {
                    assert(data[j] < elementCount_);  // Valid ID
                    assert(data[j] != i);              // No self-loops
                    inboundConnectionsNum[data[j]]++;
                    s.insert(data[j]);
                    connectionsChecked++;
                }

                assert(s.size() == static_cast<size_t>(size));  // No duplicates
            }
        }

        if (elementCount_ > 1) {
            int min1 = inboundConnectionsNum[0];
            int max1 = inboundConnectionsNum[0];

            for (size_t i = 0; i < elementCount_; i++) {
                assert(inboundConnectionsNum[i] > 0);  // All reachable
                min1 = std::min(inboundConnectionsNum[i], min1);
                max1 = std::max(inboundConnectionsNum[i], max1);
            }

            std::cout << "Min inbound: " << min1 << ", Max inbound: " << max1 << "\n";
        }

        std::cout << "Integrity ok, checked " << connectionsChecked << " connections\n";
    }

  };

} // namespace hnsw
