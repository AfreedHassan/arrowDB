"""ArrowDB: A lightweight vector database for similarity search."""

from arrowdb._arrowdb import (
    # Enums
    Space,
    IndexType,
    Quantization,
    FieldType,
    StatusCode,

    # Config
    HNSWParams,
    IndexConfig,
    CollectionConfig,
    ClientOptions,
    FieldDef,
    MetadataSchema,

    # Types
    Document,
    InsertResult,
    BatchInsertResult,
    ScoredDocument,
    SearchResult,
    IndexSearchResult,
    CollectionStats,

    # Core
    Collection,
    CollectionRef,
    Client as _Client,

    # Filter
    MetadataFilter,

    # Errors
    ArrowDBError,
)


class Client(_Client):
    """ArrowDB client with context manager support."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


__all__ = [
    "Space",
    "IndexType",
    "Quantization",
    "FieldType",
    "StatusCode",
    "HNSWParams",
    "IndexConfig",
    "CollectionConfig",
    "ClientOptions",
    "FieldDef",
    "MetadataSchema",
    "Document",
    "InsertResult",
    "BatchInsertResult",
    "ScoredDocument",
    "SearchResult",
    "IndexSearchResult",
    "CollectionStats",
    "Collection",
    "CollectionRef",
    "Client",
    "MetadataFilter",
    "ArrowDBError",
]

__version__ = "0.1.0"
