import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Collection, Space, Filter } from "../dist/index";
import type { CollectionConfig, Document } from "../dist/index";

const DIM = 32;

function makeVector(seed: number): number[] {
  // Simple seeded pseudo-random
  let s = seed;
  const vec: number[] = [];
  for (let i = 0; i < DIM; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    vec.push((s / 0x7fffffff) * 2 - 1);
  }
  return vec;
}

describe("Collection", () => {
  let col: Collection;

  before(() => {
    col = new Collection({ name: "test", dimensions: DIM });
  });

  describe("insert", () => {
    it("should insert with auto-generated id", () => {
      const id = col.insert(makeVector(1));
      assert.ok(typeof id === "string" && id.length > 0);
    });

    it("should insert with explicit id", () => {
      col.insert("v1", makeVector(2));
      assert.equal(col.size, 2); // auto-id + v1
    });

    it("should insert with metadata", () => {
      col.insert("v_meta", makeVector(3), { category: "test", score: 0.95 });
      const meta = col.getMetadata("v_meta");
      assert.equal(meta.category, "test");
      assert.ok(Math.abs((meta.score as number) - 0.95) < 0.001);
    });

    it("should reject wrong dimensions", () => {
      assert.throws(() => col.insert([1.0, 2.0]));
    });
  });

  describe("batch insert", () => {
    it("should batch insert documents", () => {
      const col2 = new Collection({ name: "batch", dimensions: DIM });
      const docs: Document[] = Array.from({ length: 10 }, (_, i) => ({
        id: `b${i}`,
        embedding: makeVector(100 + i),
      }));
      const result = col2.insertBatch(docs);
      assert.equal(result.successCount, 10);
      assert.equal(result.failureCount, 0);
      assert.equal(col2.size, 10);
    });
  });

  describe("search", () => {
    it("should find nearest neighbors", () => {
      const col2 = new Collection({ name: "search", dimensions: DIM });
      for (let i = 0; i < 20; i++) {
        col2.insert(`s${i}`, makeVector(i));
      }
      const results = col2.search(makeVector(0), 5);
      assert.equal(results.length, 5);
      assert.equal(results[0].id, "s0");
    });

    it("should query with metadata", () => {
      const col2 = new Collection({ name: "query", dimensions: DIM });
      for (let i = 0; i < 10; i++) {
        col2.insert(`q${i}`, makeVector(i), { idx: i });
      }
      const result = col2.query(makeVector(0), 3);
      assert.equal(result.hits.length, 3);
      assert.ok("metadata" in result.hits[0]);
    });

    it("should search with filter", () => {
      const col2 = new Collection({ name: "filtered", dimensions: DIM });
      for (let i = 0; i < 30; i++) {
        col2.insert(`f${i}`, makeVector(i), {
          category: i % 2 === 0 ? "even" : "odd",
          value: i,
        });
      }
      const results = col2.search(makeVector(0), 10, {
        filter: Filter.eq("category", "even"),
      });
      for (const r of results) {
        const meta = col2.getMetadata(r.id);
        assert.equal(meta.category, "even");
      }
    });
  });

  describe("get/update/remove", () => {
    it("should get a vector", () => {
      const col2 = new Collection({ name: "crud", dimensions: DIM });
      col2.insert("crud1", makeVector(42));
      const vec = col2.get("crud1");
      assert.ok(vec instanceof Float32Array);
      assert.equal(vec.length, DIM);
    });

    it("should update", () => {
      const col2 = new Collection({ name: "upd", dimensions: DIM });
      col2.insert("u1", makeVector(1));
      col2.update("u1", makeVector(2), { updated: true });
      const meta = col2.getMetadata("u1");
      assert.equal(meta.updated, true);
    });

    it("should upsert", () => {
      const col2 = new Collection({ name: "ups", dimensions: DIM });
      col2.upsert("ups1", makeVector(1));
      assert.equal(col2.size, 1);
      col2.upsert("ups1", makeVector(2));
      assert.equal(col2.size, 1);
    });

    it("should remove", () => {
      const col2 = new Collection({ name: "rm", dimensions: DIM });
      col2.insert("rm1", makeVector(1));
      col2.remove("rm1");
      assert.throws(() => col2.get("rm1"));
    });
  });

  describe("properties", () => {
    it("should return correct properties", () => {
      const c = new Collection({ name: "props", dimensions: DIM, space: Space.L2 });
      assert.equal(c.name, "props");
      assert.equal(c.dimension, DIM);
      assert.equal(c.space, Space.L2);
    });

    it("should return stats", () => {
      const c = new Collection({ name: "stats", dimensions: DIM });
      c.insert("s1", makeVector(1));
      const s = c.stats();
      assert.equal(s.vectorCount, 1);
      assert.equal(s.dimensions, DIM);
    });
  });
});
