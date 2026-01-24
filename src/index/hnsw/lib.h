#pragma once

#include <iostream>
#include <queue>
#include <stdint.h>
#include <vector>

namespace hnsw {
using label_t = uint64_t;

class BaseFilterFunctor {
public:
  virtual bool operator()(label_t id) { return true; }
  virtual ~BaseFilterFunctor() {};
};

template<typename dist_t>
class BaseSearchStopCondition {
 public:
    virtual void add_point_to_result(label_t label, const void *datapoint, dist_t dist) = 0;

    virtual void remove_point_from_result(label_t label, const void *datapoint, dist_t dist) = 0;

    virtual bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) = 0;

    virtual bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) = 0;

    virtual bool should_remove_extra() = 0;

    virtual void filter_results(std::vector<std::pair<dist_t, label_t>> &candidates) = 0;

    virtual ~BaseSearchStopCondition() {}
};

template <typename dist_t> 
class AlgorithmInterface {
public:
  virtual void addPoint(const void *point, label_t label,
                        bool replaceDeleted = false) = 0;

  virtual std::priority_queue<std::pair<dist_t, label_t>>
  searchKnn(const void *, size_t,
            BaseFilterFunctor *isIdAllowed = nullptr) const = 0;

  // Return k nearest neighbor in the order of closer fist
  virtual std::vector<std::pair<dist_t, label_t>>
  searchKnnCloserFirst(const void *queryData, size_t k,
                       BaseFilterFunctor *isIdAllowed = nullptr) const;

  virtual void saveIndex(const std::string &location) = 0;
  virtual ~AlgorithmInterface() {}
};

template <typename dist_t>
std::vector<std::pair<dist_t, label_t>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(
    const void *queryData, size_t k, BaseFilterFunctor *isIdAllowed) const {
  std::vector<std::pair<dist_t, label_t>> result;

  auto ret = searchKnn(queryData, k, isIdAllowed);
  {
    size_t sz = ret.size();
    result.resize(sz);
    while (!ret.empty()) {
      result[--sz] = ret.top();
      ret.pop();
    }
  }

  return result;
}

template <typename dist_t> 
class BaseSearchController {
public:
  virtual void onCandidateFound(label_t label, const void *point,
                                dist_t dist) = 0;

  virtual void onCandidateDiscarded(label_t label, const void *point,
                                    dist_t dist) = 0;

  virtual bool isSearchComplete(dist_t candidate_dist, dist_t lowerBound) = 0;

  virtual bool isWorthExploring(dist_t candidate_dist, dist_t lowerBound) = 0;

  virtual bool requiresPruning() = 0;

  virtual void
  filterResults(std::vector<std::pair<dist_t, label_t>> &candidates) = 0;

  virtual ~BaseSearchController() {}
};

template<typename metric_t>
using pdistfunc_t = float (*)(const metric_t* a, const metric_t* b, std::size_t dim);

template<typename metric_t>
using pbatchdistfunc_t = void (*)(
    const metric_t* query,
    const metric_t* const* targets,
    std::size_t numTargets,
    std::size_t dim,
    metric_t* outDistances);

template <typename metric_t> class SpaceInterface {
public:
  // virtual void search(void *);
  virtual size_t getDataSize() = 0;

  virtual pdistfunc_t<metric_t> getDistFunc() = 0;

  virtual pbatchdistfunc_t<metric_t> getBatchDistFunc() = 0;

  virtual void *getDistFuncParam() = 0;

  virtual ~SpaceInterface() {}
};

template <typename T> static void read(std::istream &in, T &dest) {
  static_assert(std::is_trivially_copyable_v<T>);
  in.read(reinterpret_cast<char *>(&dest), sizeof(T));
}

template <typename T> static void write(std::ostream &out, const T &val) {
  static_assert(std::is_trivially_copyable_v<T>);
  out.write(reinterpret_cast<const char *>(&val), sizeof(T));
}


} // namespace hnsw
