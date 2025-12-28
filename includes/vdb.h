#ifndef  ARROWDB_H
#define  ARROWDB_H
#include <vector>

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
};

#endif
