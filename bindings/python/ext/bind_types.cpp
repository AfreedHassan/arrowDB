#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <arrow/types.h>
#include <arrow/collection.h>
#include "helpers.h"

namespace nb = nanobind;
using namespace arrow;

void bind_types(nb::module_& m) {
    // Enums
    nb::enum_<Space>(m, "Space")
        .value("COSINE", Space::Cosine)
        .value("L2", Space::L2)
        .value("INNER_PRODUCT", Space::InnerProduct);

    nb::enum_<IndexType>(m, "IndexType")
        .value("HNSW", IndexType::HNSW);

    nb::enum_<Quantization>(m, "Quantization")
        .value("NONE", Quantization::None)
        .value("INT8", Quantization::INT8);

    nb::enum_<FieldType>(m, "FieldType")
        .value("INT64", FieldType::Int64)
        .value("DOUBLE", FieldType::Double)
        .value("STRING", FieldType::String)
        .value("BOOL", FieldType::Bool);

    // FieldDef
    nb::class_<FieldDef>(m, "FieldDef")
        .def(nb::init<>())
        .def("__init__", [](FieldDef* fd, std::string name, FieldType type, bool required) {
            new (fd) FieldDef{std::move(name), type, required};
        }, nb::arg("name"), nb::arg("type"), nb::arg("required") = false)
        .def_rw("name", &FieldDef::name)
        .def_rw("type", &FieldDef::type)
        .def_rw("required", &FieldDef::required);

    // MetadataSchema
    nb::class_<MetadataSchema>(m, "MetadataSchema")
        .def(nb::init<>())
        .def("field", &MetadataSchema::field,
            nb::arg("name"), nb::arg("type"), nb::arg("required") = false,
            nb::rv_policy::reference)
        .def("empty", &MetadataSchema::empty);

    // Document
    nb::class_<Document>(m, "Document")
        .def(nb::init<>())
        .def("__init__", [](Document* doc, const nb::object& embedding,
                           nb::dict metadata, std::string id) {
            new (doc) Document{};
            doc->id = std::move(id);
            doc->embedding = to_float_vector(embedding);
            doc->metadata = dict_to_metadata(metadata);
        }, nb::arg("embedding"), nb::arg("metadata") = nb::dict(), nb::arg("id") = "")
        .def_rw("id", &Document::id)
        .def_rw("embedding", &Document::embedding)
        .def_prop_rw("metadata",
            [](const Document& d) { return metadata_to_dict(d.metadata); },
            [](Document& d, const nb::dict& dict) { d.metadata = dict_to_metadata(dict); });

    // InsertResult
    nb::class_<InsertResult>(m, "InsertResult")
        .def_ro("id", &InsertResult::id)
        .def_prop_ro("ok", [](const InsertResult& r) { return r.status.ok(); })
        .def_prop_ro("message", [](const InsertResult& r) { return r.status.message(); });

    // BatchInsertResult
    nb::class_<BatchInsertResult>(m, "BatchInsertResult")
        .def_ro("results", &BatchInsertResult::results)
        .def_ro("success_count", &BatchInsertResult::successCount)
        .def_ro("failure_count", &BatchInsertResult::failureCount);

    // ScoredDocument
    nb::class_<ScoredDocument>(m, "ScoredDocument")
        .def_ro("id", &ScoredDocument::id)
        .def_ro("score", &ScoredDocument::score)
        .def_prop_ro("metadata", [](const ScoredDocument& d) {
            return metadata_to_dict(d.metadata);
        });

    // SearchResult
    nb::class_<SearchResult>(m, "SearchResult")
        .def_ro("hits", &SearchResult::hits)
        .def("__len__", [](const SearchResult& r) { return r.hits.size(); })
        .def("__getitem__", [](const SearchResult& r, size_t i) -> const ScoredDocument& {
            if (i >= r.hits.size()) throw nb::index_error();
            return r.hits[i];
        }, nb::rv_policy::reference_internal);

    // IndexSearchResult
    nb::class_<IndexSearchResult>(m, "IndexSearchResult")
        .def_ro("id", &IndexSearchResult::id)
        .def_ro("score", &IndexSearchResult::score);

    // Collection::Stats
    nb::class_<Collection::Stats>(m, "CollectionStats")
        .def_ro("vector_count", &Collection::Stats::vectorCount)
        .def_ro("metadata_count", &Collection::Stats::metadataCount)
        .def_ro("max_capacity", &Collection::Stats::maxCapacity)
        .def_ro("dimensions", &Collection::Stats::dimensions);
}
