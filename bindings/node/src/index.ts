import * as path from "path";

// Load the native addon
const addon = require("../build/Release/arrowdb_addon.node");

// Re-export types
export {
  Space,
  IndexType,
  Quantization,
  Filter,
} from "./types";

export type {
  HNSWParams,
  IndexConfig,
  CollectionConfig,
  Document,
  Metadata,
  IndexSearchResult,
  ScoredDocument,
  SearchResult,
  InsertResult,
  BatchInsertResult,
  CollectionStats,
  SearchOptions,
  MetadataFilterDSL,
} from "./types";

import type {
  CollectionConfig,
  Document,
  Metadata,
  IndexSearchResult,
  SearchResult,
  BatchInsertResult,
  CollectionStats,
  SearchOptions,
} from "./types";

/** A collection of vectors for similarity search. */
export class Collection {
  private _native: any;

  constructor(config: CollectionConfig) {
    this._native = new addon.Collection(config);
  }

  /** @internal Used by Client to wrap native collection refs */
  static _fromNative(native: any): Collection {
    const c = Object.create(Collection.prototype);
    c._native = native;
    return c;
  }

  /** Create a persistent collection with WAL. */
  static create(config: CollectionConfig, persistencePath: string): Collection {
    const native = addon.createCollection(config, persistencePath);
    return Collection._fromNative(native);
  }

  /** Load a collection from disk. */
  static load(directoryPath: string): Collection {
    const native = addon.loadCollection(directoryPath);
    return Collection._fromNative(native);
  }

  get name(): string { return this._native.name; }
  get dimension(): number { return this._native.dimension; }
  get space(): number { return this._native.space; }
  get size(): number { return this._native.size; }

  /** Insert a vector. If id is provided, uses that ID; otherwise auto-generates. */
  insert(embedding: number[] | Float32Array, metadata?: Metadata): string;
  insert(id: string, embedding: number[] | Float32Array, metadata?: Metadata): void;
  insert(
    idOrEmbedding: string | number[] | Float32Array,
    embeddingOrMetadata?: number[] | Float32Array | Metadata,
    metadata?: Metadata
  ): string | void {
    return this._native.insert(idOrEmbedding, embeddingOrMetadata, metadata);
  }

  /** Batch insert documents. */
  insertBatch(documents: Document[]): BatchInsertResult {
    return this._native.insertBatch(documents);
  }

  /** Get a vector by ID. Returns Float32Array. */
  get(id: string): Float32Array {
    return this._native.get(id);
  }

  /** Update an existing vector. */
  update(id: string, embedding: number[] | Float32Array, metadata?: Metadata): void {
    this._native.update(id, embedding, metadata);
  }

  /** Insert or update a vector. */
  upsert(id: string, embedding: number[] | Float32Array, metadata?: Metadata): void {
    this._native.upsert(id, embedding, metadata);
  }

  /** Remove a vector by ID. */
  remove(id: string): void {
    this._native.remove(id);
  }

  /** Search for k nearest neighbors. */
  search(query: number[] | Float32Array, k: number, options?: SearchOptions): IndexSearchResult[] {
    return this._native.search(query, k, options);
  }

  /** Query with metadata in results. */
  query(query: number[] | Float32Array, k: number, options?: SearchOptions): SearchResult {
    return this._native.query(query, k, options);
  }

  /** Batch search multiple queries. */
  searchBatch(queries: (number[] | Float32Array)[], k: number, ef?: number): IndexSearchResult[][] {
    return this._native.searchBatch(queries, k, ef);
  }

  /** Set metadata for a vector. */
  setMetadata(id: string, metadata: Metadata): void {
    this._native.setMetadata(id, metadata);
  }

  /** Get metadata for a vector. */
  getMetadata(id: string): Metadata {
    return this._native.getMetadata(id);
  }

  /** Optimize the collection for search performance. */
  optimize(): void {
    this._native.optimize();
  }

  /** Save the collection to disk. */
  save(directoryPath: string): void {
    this._native.save(directoryPath);
  }

  /** Close the collection. */
  close(): void {
    this._native.close();
  }

  /** Get collection statistics. */
  stats(): CollectionStats {
    return this._native.stats();
  }

  [Symbol.dispose](): void {
    this.close();
  }
}

/** ArrowDB client for managing multiple collections. */
export class Client {
  private _native: any;

  constructor(dataDir: string) {
    this._native = new addon.Client(dataDir);
  }

  get dataDir(): string { return this._native.dataDir; }

  /** Create a new collection. */
  createCollection(name: string, config: CollectionConfig): Collection {
    const native = this._native.createCollection(name, config);
    return Collection._fromNative(native);
  }

  /** Get an existing collection. */
  getCollection(name: string): Collection {
    const native = this._native.getCollection(name);
    return Collection._fromNative(native);
  }

  /** Get or create a collection. */
  getOrCreateCollection(name: string, config?: CollectionConfig): Collection {
    const native = config
      ? this._native.getOrCreateCollection(name, config)
      : this._native.getOrCreateCollection(name);
    return Collection._fromNative(native);
  }

  /** Drop a collection. */
  dropCollection(name: string): void {
    this._native.dropCollection(name);
  }

  /** List all collection names. */
  listCollections(): string[] {
    return this._native.listCollections();
  }

  /** Check if a collection exists. */
  hasCollection(name: string): boolean {
    return this._native.hasCollection(name);
  }

  /** Close the client and all collections. */
  close(): void {
    this._native.close();
  }

  [Symbol.dispose](): void {
    this.close();
  }
}
