#pragma once

#include "lib.h"
#include "multi_vector_space.h"
#include <algorithm>
#include <cassert>
#include <queue>
#include <unordered_map>

namespace hnsw {

/// KNN (K-Nearest Neighbors) search controller.
/// Stops search when K neighbors are found and no closer candidates exist.
template<typename dist_t = float>
class KNNSearchController : public BaseSearchController<dist_t> {
private:
    std::size_t k_;
    std::size_t numCandidates_ = 0;
    dist_t worstDist_ = 0;

public:
    /// Construct KNN controller to find k nearest neighbors.
    explicit KNNSearchController(std::size_t k) : k_(k) {}

    void onCandidateFound(label_t label, const void* point, dist_t dist) override {
        ++numCandidates_;
        worstDist_ = dist;
    }

    void onCandidateDiscarded(label_t label, const void* point, dist_t dist) override {
        if (numCandidates_ > 0) --numCandidates_;
    }

    [[nodiscard]] bool isSearchComplete(dist_t candidateDist, dist_t lowerBound) override {
        return candidateDist > lowerBound && numCandidates_ >= k_;
    }

    [[nodiscard]] bool isWorthExploring(dist_t candidateDist, dist_t lowerBound) override {
        return numCandidates_ < k_ || lowerBound > candidateDist;
    }

    [[nodiscard]] bool requiresPruning() override {
        return numCandidates_ > k_;
    }

    void filterResults(std::vector<std::pair<dist_t, label_t>>& candidates) override {
        if (candidates.size() > k_) {
            candidates.erase(candidates.begin() + static_cast<std::ptrdiff_t>(k_), candidates.end());
        }
    }

    /// Reset controller state for reuse across multiple searches.
    void reset() {
        numCandidates_ = 0;
        worstDist_ = 0;
    }

    /// Get the configured K value.
    [[nodiscard]] std::size_t k() const { return k_; }

    /// Get number of candidates currently found.
    [[nodiscard]] std::size_t numCandidates() const { return numCandidates_; }

    ~KNNSearchController() = default;
};

/// Epsilon/Range search controller.
/// Returns all candidates within epsilon distance, subject to min/max bounds.
/// Stops when all remaining candidates are outside epsilon and minimum is met.
template<typename dist_t = float>
class EpsilonSearchController : public BaseSearchController<dist_t> {
private:
    dist_t epsilon_;
    std::size_t minCandidates_;
    std::size_t maxCandidates_;
    std::size_t numCandidates_ = 0;

public:
    /// Construct epsilon search controller.
    /// @param epsilon Maximum distance threshold for results
    /// @param minCandidates Minimum number of candidates before early termination
    /// @param maxCandidates Maximum number of candidates to return
    EpsilonSearchController(dist_t epsilon,
                            std::size_t minCandidates = 1,
                            std::size_t maxCandidates = 1000)
        : epsilon_(epsilon)
        , minCandidates_(minCandidates)
        , maxCandidates_(maxCandidates) {
        assert(minCandidates <= maxCandidates);
    }

    void onCandidateFound(label_t /*label*/, const void* /*point*/, dist_t /*dist*/) override {
        ++numCandidates_;
    }

    void onCandidateDiscarded(label_t /*label*/, const void* /*point*/, dist_t /*dist*/) override {
        if (numCandidates_ > 0) --numCandidates_;
    }

    [[nodiscard]] bool isSearchComplete(dist_t candidateDist, dist_t lowerBound) override {
        // Stop if we can't improve and have max candidates
        if (candidateDist > lowerBound && numCandidates_ >= maxCandidates_) {
            return true;
        }
        // Stop if candidate is outside epsilon and we have minimum
        if (candidateDist > epsilon_ && numCandidates_ >= minCandidates_) {
            return true;
        }
        return false;
    }

    [[nodiscard]] bool isWorthExploring(dist_t candidateDist, dist_t lowerBound) override {
        return numCandidates_ < maxCandidates_ || lowerBound > candidateDist;
    }

    [[nodiscard]] bool requiresPruning() override {
        return numCandidates_ > maxCandidates_;
    }

    void filterResults(std::vector<std::pair<dist_t, label_t>>& candidates) override {
        // Remove candidates outside epsilon
        auto it = std::remove_if(candidates.begin(), candidates.end(),
            [this](const auto& p) { return p.first > epsilon_; });
        candidates.erase(it, candidates.end());

        // Enforce max candidates limit
        if (candidates.size() > maxCandidates_) {
            candidates.erase(candidates.begin() + static_cast<std::ptrdiff_t>(maxCandidates_),
                           candidates.end());
        }
    }

    /// Reset controller state for reuse across multiple searches.
    void reset() {
        numCandidates_ = 0;
    }

    /// Get the configured epsilon threshold.
    [[nodiscard]] dist_t epsilon() const { return epsilon_; }

    /// Get number of candidates currently found.
    [[nodiscard]] std::size_t numCandidates() const { return numCandidates_; }

    ~EpsilonSearchController() = default;
};

/// Multi-vector search controller.
/// Groups results by document ID extracted from datapoints via space interface.
/// Finds K unique documents, using efCollection as search expansion factor.
template<typename DocIdType, typename dist_t = float>
class MultiVectorSearchController : public BaseSearchController<dist_t> {
private:
    std::size_t numDocsToFind_;
    std::size_t efCollection_;
    std::size_t currNumDocs_ = 0;
    std::unordered_map<DocIdType, std::size_t> docCounter_;
    std::priority_queue<std::pair<dist_t, DocIdType>> searchResults_;
    MultiVectorSpaceInterface<DocIdType>& space_;

public:
    /// Construct multi-vector search controller.
    /// @param space Reference to space interface for extracting doc IDs from datapoints
    /// @param numDocsToFind Number of unique documents to return
    /// @param efCollection Search expansion factor (explores more to find unique docs)
    MultiVectorSearchController(
        MultiVectorSpaceInterface<DocIdType>& space,
        std::size_t numDocsToFind,
        std::size_t efCollection = 10)
        : numDocsToFind_(numDocsToFind)
        , efCollection_(std::max(efCollection, numDocsToFind))
        , space_(space) {}

    void onCandidateFound(label_t /*label*/, const void* datapoint, dist_t dist) override {
        DocIdType docId = space_.getDocId(datapoint);
        if (docCounter_[docId] == 0) {
            ++currNumDocs_;
        }
        searchResults_.emplace(dist, docId);
        ++docCounter_[docId];
    }

    void onCandidateDiscarded(label_t /*label*/, const void* datapoint, dist_t /*dist*/) override {
        DocIdType docId = space_.getDocId(datapoint);
        if (docCounter_[docId] > 0) {
            --docCounter_[docId];
            if (docCounter_[docId] == 0) {
                --currNumDocs_;
            }
        }
        if (!searchResults_.empty()) {
            searchResults_.pop();
        }
    }

    [[nodiscard]] bool isSearchComplete(dist_t candidateDist, dist_t lowerBound) override {
        return candidateDist > lowerBound && currNumDocs_ >= efCollection_;
    }

    [[nodiscard]] bool isWorthExploring(dist_t candidateDist, dist_t lowerBound) override {
        return currNumDocs_ < efCollection_ || lowerBound > candidateDist;
    }

    [[nodiscard]] bool requiresPruning() override {
        return currNumDocs_ > efCollection_;
    }

    void filterResults(std::vector<std::pair<dist_t, label_t>>& candidates) override {
        // Remove excess documents from the back (worst distance)
        while (currNumDocs_ > numDocsToFind_ && !searchResults_.empty()) {
            DocIdType docId = searchResults_.top().second;
            --docCounter_[docId];
            if (docCounter_[docId] == 0) {
                --currNumDocs_;
            }
            searchResults_.pop();
            if (!candidates.empty()) {
                candidates.pop_back();
            }
        }
    }

    /// Reset controller state for reuse across multiple searches.
    void reset() {
        currNumDocs_ = 0;
        docCounter_.clear();
        while (!searchResults_.empty()) {
            searchResults_.pop();
        }
    }

    /// Get number of unique documents currently found.
    [[nodiscard]] std::size_t numUniqueDocuments() const { return currNumDocs_; }

    /// Get the configured number of documents to find.
    [[nodiscard]] std::size_t numDocsToFind() const { return numDocsToFind_; }

    /// Get the configured efCollection value.
    [[nodiscard]] std::size_t efCollection() const { return efCollection_; }

    ~MultiVectorSearchController() = default;
};

}  // namespace hnsw
