#include "arrow/utils/uuid.h"
#include <cassert>
#include <gtest/gtest.h>
#include <iostream>
#include <unordered_set>

using namespace arrow::uuid;

TEST(UUIDTest, StringFormat) {
  std::string id = uuidv4();

  assert(id.length() == 36);
  assert(id[8] == '-');
  assert(id[13] == '-');
  assert(id[18] == '-');
  assert(id[23] == '-');

  for (char c : id) {
    if (c == '-')
      continue;
    assert((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
  }
}

TEST(UUIDTest, Determinism) {
  std::string id1 = uuidv4(123);
  std::string id2 = uuidv4(123);
  std::string id3 = uuidv4(124);

  assert(id1 == id2);
  assert(id1 != id3);
}

TEST(UUIDTest, Uniqueness) {
  std::unordered_set<std::string> uuids;

  for (int i = 0; i < 100000; ++i) {
    std::string id = uuidv4();
    assert(uuids.find(id) == uuids.end());
    uuids.insert(id);
  }

  assert(uuids.size() == 100000);
}

TEST(UUIDTest, HashStability) {
  std::string id1 = uuidv4(42);
  std::string id2 = uuidv4(42);
  std::string id3 = uuidv4(43);

  std::hash<std::string> hasher;
  assert(hasher(id1) == hasher(id2));
  assert(hasher(id1) != hasher(id3));
}

TEST(UUIDTest, ComparisonOperators) {
  std::string id1 = uuidv4(1);
  std::string id2 = uuidv4(1);
  std::string id3 = uuidv4(2);

  assert(id1 == id2);
  assert(!(id1 != id2));
  assert(id1 != id3);

  assert((id1 < id3) || (id3 < id1));
  assert(!(id1 < id2) && !(id2 < id1));
}
