/** Distance space for vector similarity computation. */
export enum Space {
  COSINE = 0,
  L2 = 1,
  INNER_PRODUCT = 2,
}

/** Index type for vector search. */
export enum IndexType {
  HNSW = 0,
}

/** Vector quantization strategy. */
export enum Quantization {
  NONE = 0,
  INT8 = 1,
}

/** HNSW index parameters. */
export interface HNSWParams {
  M?: number;
  efConstruction?: number;
  efSearch?: number;
}

/** Index configuration. */
export interface IndexConfig {
  maxElements?: number;
  quantization?: Quantization;
  hnswParams?: HNSWParams;
}

/** Collection configuration. */
export interface CollectionConfig {
  name?: string;
  dimensions: number;
  space?: Space;
  indexConfig?: IndexConfig;
}

/** A document with embedding and metadata. */
export interface Document {
  id?: string;
  embedding: number[] | Float32Array;
  metadata?: Metadata;
}

/** Metadata key-value map. */
export type Metadata = Record<string, number | string | boolean>;

/** Result from index search (id + score only). */
export interface IndexSearchResult {
  id: string;
  score: number;
}

/** A scored document with metadata from a query. */
export interface ScoredDocument {
  id: string;
  score: number;
  metadata: Metadata;
}

/** Result of a search/query operation. */
export interface SearchResult {
  hits: ScoredDocument[];
}

/** Per-vector result in a batch insert. */
export interface InsertResult {
  id: string;
  ok: boolean;
  message: string;
}

/** Aggregate batch insert result. */
export interface BatchInsertResult {
  results: InsertResult[];
  successCount: number;
  failureCount: number;
}

/** Collection statistics. */
export interface CollectionStats {
  vectorCount: number;
  metadataCount: number;
  maxCapacity: number;
  dimensions: number;
}

/** Search/query options. */
export interface SearchOptions {
  ef?: number;
  filter?: MetadataFilterDSL;
}

// ─── MetadataFilter DSL ──────────────────────────────────

export type MetadataFilterDSL =
  | { op: "eq"; field: string; value: number | string | boolean }
  | { op: "neq"; field: string; value: number | string | boolean }
  | { op: "gt"; field: string; value: number }
  | { op: "gte"; field: string; value: number }
  | { op: "lt"; field: string; value: number }
  | { op: "lte"; field: string; value: number }
  | { op: "in"; field: string; values: (number | string | boolean)[] }
  | { op: "and"; filters: MetadataFilterDSL[] }
  | { op: "or"; filters: MetadataFilterDSL[] }
  | { op: "not"; filter: MetadataFilterDSL };

// Filter factory helpers
export const Filter = {
  eq: (field: string, value: number | string | boolean): MetadataFilterDSL => ({
    op: "eq",
    field,
    value,
  }),
  neq: (field: string, value: number | string | boolean): MetadataFilterDSL => ({
    op: "neq",
    field,
    value,
  }),
  gt: (field: string, value: number): MetadataFilterDSL => ({
    op: "gt",
    field,
    value,
  }),
  gte: (field: string, value: number): MetadataFilterDSL => ({
    op: "gte",
    field,
    value,
  }),
  lt: (field: string, value: number): MetadataFilterDSL => ({
    op: "lt",
    field,
    value,
  }),
  lte: (field: string, value: number): MetadataFilterDSL => ({
    op: "lte",
    field,
    value,
  }),
  in: (
    field: string,
    values: (number | string | boolean)[]
  ): MetadataFilterDSL => ({ op: "in", field, values }),
  and: (...filters: MetadataFilterDSL[]): MetadataFilterDSL => ({
    op: "and",
    filters,
  }),
  or: (...filters: MetadataFilterDSL[]): MetadataFilterDSL => ({
    op: "or",
    filters,
  }),
  not: (filter: MetadataFilterDSL): MetadataFilterDSL => ({
    op: "not",
    filter,
  }),
};
