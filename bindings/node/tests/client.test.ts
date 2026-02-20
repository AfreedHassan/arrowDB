import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client, Collection } from "../dist/index";
import type { CollectionConfig } from "../dist/index";

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

describe("Client", () => {
  let tmpDir: string;

  before(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "arrowdb_test_"));
  });

  after(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should create and get collections", () => {
    const client = new Client(tmpDir);
    const config: CollectionConfig = { dimensions: DIM };
    const col = client.createCollection("col1", config);
    assert.equal(col.name, "col1");

    const col2 = client.getCollection("col1");
    assert.equal(col2.name, "col1");
    client.close();
  });

  it("should list collections", () => {
    const dir = mkdtempSync(join(tmpdir(), "arrowdb_test_"));
    const client = new Client(dir);
    client.createCollection("a", { dimensions: DIM });
    client.createCollection("b", { dimensions: DIM });
    const names = client.listCollections();
    assert.ok(names.includes("a"));
    assert.ok(names.includes("b"));
    client.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it("should check collection existence", () => {
    const dir = mkdtempSync(join(tmpdir(), "arrowdb_test_"));
    const client = new Client(dir);
    assert.equal(client.hasCollection("x"), false);
    client.createCollection("x", { dimensions: DIM });
    assert.equal(client.hasCollection("x"), true);
    client.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it("should drop collection", () => {
    const dir = mkdtempSync(join(tmpdir(), "arrowdb_test_"));
    const client = new Client(dir);
    client.createCollection("drop_me", { dimensions: DIM });
    client.dropCollection("drop_me");
    assert.equal(client.hasCollection("drop_me"), false);
    client.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it("should get or create", () => {
    const dir = mkdtempSync(join(tmpdir(), "arrowdb_test_"));
    const client = new Client(dir);
    const col = client.getOrCreateCollection("goc", { dimensions: DIM });
    col.insert("v1", makeVector(1));
    const col2 = client.getOrCreateCollection("goc", { dimensions: DIM });
    assert.equal(col2.size, 1);
    client.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it("should search via collection ref", () => {
    const dir = mkdtempSync(join(tmpdir(), "arrowdb_test_"));
    const client = new Client(dir);
    const col = client.createCollection("search", { dimensions: DIM });
    for (let i = 0; i < 20; i++) {
      col.insert(`v${i}`, makeVector(i));
    }
    const results = col.search(makeVector(0), 5);
    assert.equal(results.length, 5);
    assert.equal(results[0].id, "v0");
    client.close();
    rmSync(dir, { recursive: true, force: true });
  });
});
