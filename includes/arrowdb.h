#ifndef  ARROWDB_H
#define  ARROWDB_H
#include <variant>
#include <vector>
#include <string>

namespace arrow {
	enum class DistanceMetric { Cosine, L2, InnerProduct };

	enum class DataType { Int16, Float16 };

	enum IndexType { HNSW };

	class CollectionConfig {
	public:
		std::string name;
		size_t dimensions;
		DistanceMetric metric;
		DataType dtype;
		IndexType idxType;
	
	// constructor
	CollectionConfig(
        std::string name_,
        size_t dimensions_,
        DistanceMetric metric_,
        DataType dtype_
    )
        : name(std::move(name_)),
          dimensions(dimensions_),
          metric(metric_),
          dtype(dtype_) {

        if (dimensions <= 0) {
            throw std::invalid_argument("dimension must be > 0");
        }

        if (dtype != DataType::Float16 && dtype != DataType::Int16) {
            throw std::invalid_argument("only float16 and int16 supported");
        }
    }
	};

	class Collection {
		public:
			explicit Collection(CollectionConfig config) : config_(std::move(config)) {}
			// ----- Accessors -----

			const std::string& name() const {
				return config_.name;
			}

			size_t dimension() const {
				return config_.dimensions;
			}

			DistanceMetric metric() const {
				return config_.metric;
			}

			DataType dtype() const {
				return config_.dtype;
			}
		private:
			const CollectionConfig config_;

			
			// ----- Internal state (added later) -----
			// VectorStore vector_store_;
			// HNSWIndex hnsw_;
			// WAL wal_;
	};
} // namespace arrow

struct arrowRecord {
	uint64_t id;
	std::vector<float> embedding;
	//Metadata metadata;
};

class arrowDB {
	public : 
	std::vector<int> store; 
	arrowDB(int n) { 
		store.reserve(n);
	};
	arrowDB(const arrowDB &) = default;
	arrowDB(arrowDB &&) = default;
	arrowDB &operator=(const arrowDB &) = default;
	arrowDB &operator=(arrowDB &&) = default;

	void insert(int n) {
		this->store.push_back(n);
	}
};

#endif
