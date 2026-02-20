import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { Collection, Filter } from "../dist/index";

const DIM = 32;

function makeVector(seed: number): number[] {
  let s = seed;
  const vec: number[] = [];
  for (let i = 0; i < DIM; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    vec.push((s / 0x7fffffff) * 2 - 1);
  }
  return vec;
}

function createFilteredCollection(): Collection {
  const col = new Collection({ name: "filter_test", dimensions: DIM });
  for (let i = 0; i < 50; i++) {
    col.insert(`v${i}`, makeVector(i), {
      category: i % 2 === 0 ? "even" : "odd",
      value: i,
      active: i < 25,
    });
  }
  return col;
}

describe("MetadataFilter DSL", () => {
  it("should filter with eq", () => {
    const col = createFilteredCollection();
    const results = col.search(makeVector(0), 10, {
      filter: Filter.eq("category", "even"),
    });
    for (const r of results) {
      assert.equal(col.getMetadata(r.id).category, "even");
    }
  });

  it("should filter with gt", () => {
    const col = createFilteredCollection();
    const results = col.search(makeVector(0), 10, {
      filter: Filter.gt("value", 40),
    });
    for (const r of results) {
      assert.ok((col.getMetadata(r.id).value as number) > 40);
    }
  });

  it("should combine with and", () => {
    const col = createFilteredCollection();
    const results = col.search(makeVector(0), 10, {
      filter: Filter.and(
        Filter.eq("category", "even"),
        Filter.lt("value", 10)
      ),
    });
    for (const r of results) {
      const meta = col.getMetadata(r.id);
      assert.equal(meta.category, "even");
      assert.ok((meta.value as number) < 10);
    }
  });

  it("should combine with or", () => {
    const col = createFilteredCollection();
    const results = col.search(makeVector(0), 10, {
      filter: Filter.or(Filter.eq("value", 0), Filter.eq("value", 1)),
    });
    for (const r of results) {
      const v = col.getMetadata(r.id).value;
      assert.ok(v === 0 || v === 1);
    }
  });

  it("should negate with not", () => {
    const col = createFilteredCollection();
    const results = col.search(makeVector(0), 10, {
      filter: Filter.not(Filter.eq("category", "even")),
    });
    for (const r of results) {
      assert.notEqual(col.getMetadata(r.id).category, "even");
    }
  });

  it("should filter with in", () => {
    const col = createFilteredCollection();
    const results = col.search(makeVector(0), 10, {
      filter: Filter.in("value", [0, 1, 2]),
    });
    for (const r of results) {
      const v = col.getMetadata(r.id).value;
      assert.ok(v === 0 || v === 1 || v === 2);
    }
  });
});
