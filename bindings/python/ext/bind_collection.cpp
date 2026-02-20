#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <arrow/collection.h>
#include <arrow/filter.h>
#include "helpers.h"

namespace nb = nanobind;
using namespace arrow;

void bind_collection(nb::module_& m) {
    nb::class_<Collection>(m, "Collection")
        // In-memory constructor
        .def(nb::init<const CollectionConfig&>(), nb::arg("config"))

        // Static factory: persistent collection
        .def_static("create", [](const CollectionConfig& config, const std::string& path) {
            return unwrap(Collection::create(config, path));
        }, nb::arg("config"), nb::arg("path"))

        // Static factory: load from disk
        .def_static("load", [](const std::string& path) {
            nb::gil_scoped_release release;
            return unwrap(Collection::load(path));
        }, nb::arg("path"))

        // Properties
        .def_prop_ro("name", &Collection::name)
        .def_prop_ro("dimension", &Collection::dimension)
        .def_prop_ro("space", &Collection::space)
        .def("__len__", &Collection::size)

        // Insert with auto-generated ID
        .def("insert", [](Collection& c, const nb::object& vec, nb::dict metadata) -> std::string {
            auto v = to_float_vector(vec);
            auto meta = dict_to_metadata(metadata);
            nb::gil_scoped_release release;
            return unwrap(c.insert(v, std::move(meta)));
        }, nb::arg("embedding"), nb::arg("metadata") = nb::dict(),
           "Insert a vector with an auto-generated ID. Returns the generated ID.")

        // Insert with explicit ID
        .def("insert", [](Collection& c, const std::string& id,
                          const nb::object& vec, nb::dict metadata) {
            auto v = to_float_vector(vec);
            auto meta = dict_to_metadata(metadata);
            nb::gil_scoped_release release;
            throw_on_error(c.insert(id, v, std::move(meta)));
        }, nb::arg("id"), nb::arg("embedding"), nb::arg("metadata") = nb::dict(),
           "Insert a vector with a specific ID.")

        // Insert document
        .def("insert_doc", [](Collection& c, Document doc) -> std::string {
            nb::gil_scoped_release release;
            return unwrap(c.insert(std::move(doc)));
        }, nb::arg("doc"), "Insert a Document object. Returns the ID.")

        // Batch insert
        .def("insert_batch", [](Collection& c, nb::list docs_list) {
            std::vector<Document> docs;
            docs.reserve(nb::len(docs_list));
            for (auto item : docs_list) {
                docs.push_back(nb::cast<Document>(nb::borrow(item)));
            }
            nb::gil_scoped_release release;
            return unwrap(c.insertBatch(std::move(docs)));
        }, nb::arg("documents"), "Batch insert documents.")

        // Get vector by ID
        .def("get", [](const Collection& c, const std::string& id) {
            return unwrap(c.get(id));
        }, nb::arg("id"), "Retrieve a vector by ID.")

        // Update
        .def("update", [](Collection& c, const std::string& id,
                          const nb::object& vec, nb::dict metadata) {
            auto v = to_float_vector(vec);
            auto meta = dict_to_metadata(metadata);
            nb::gil_scoped_release release;
            throw_on_error(c.update(id, v, std::move(meta)));
        }, nb::arg("id"), nb::arg("embedding"), nb::arg("metadata") = nb::dict())

        // Upsert
        .def("upsert", [](Collection& c, const std::string& id,
                          const nb::object& vec, nb::dict metadata) {
            auto v = to_float_vector(vec);
            auto meta = dict_to_metadata(metadata);
            nb::gil_scoped_release release;
            throw_on_error(c.upsert(id, v, std::move(meta)));
        }, nb::arg("id"), nb::arg("embedding"), nb::arg("metadata") = nb::dict())

        // Remove
        .def("remove", [](Collection& c, const std::string& id) {
            throw_on_error(c.remove(id));
        }, nb::arg("id"))

        // Search (returns IndexSearchResult list)
        .def("search", [](const Collection& c, const nb::object& query,
                          uint32_t k, uint32_t ef) {
            auto q = to_float_vector(query);
            nb::gil_scoped_release release;
            return c.search(q, k, ef);
        }, nb::arg("query"), nb::arg("k"), nb::arg("ef") = 200,
           "Search for k nearest neighbors. Returns list of IndexSearchResult.")

        // Search with filter
        .def("search", [](const Collection& c, const nb::object& query,
                          uint32_t k, const MetadataFilter& filter, uint32_t ef) {
            auto q = to_float_vector(query);
            nb::gil_scoped_release release;
            return c.search(q, k, filter, ef);
        }, nb::arg("query"), nb::arg("k"), nb::arg("filter"), nb::arg("ef") = 200,
           "Filtered search for k nearest neighbors.")

        // Query (returns SearchResult with metadata)
        .def("query", [](const Collection& c, const nb::object& query,
                         uint32_t k, uint32_t ef) {
            auto q = to_float_vector(query);
            nb::gil_scoped_release release;
            return c.query(q, k, ef);
        }, nb::arg("query"), nb::arg("k"), nb::arg("ef") = 200,
           "Search with metadata in results.")

        // Query with filter
        .def("query", [](const Collection& c, const nb::object& query,
                         uint32_t k, const MetadataFilter& filter, uint32_t ef) {
            auto q = to_float_vector(query);
            nb::gil_scoped_release release;
            return c.query(q, k, filter, ef);
        }, nb::arg("query"), nb::arg("k"), nb::arg("filter"), nb::arg("ef") = 200,
           "Filtered query with metadata.")

        // Search batch
        .def("search_batch", [](const Collection& c,
                                const std::vector<std::vector<float>>& queries,
                                uint32_t k, uint32_t ef) {
            nb::gil_scoped_release release;
            return unwrap(c.searchBatch(queries, k, ef));
        }, nb::arg("queries"), nb::arg("k"), nb::arg("ef") = 200)

        // Metadata operations
        .def("set_metadata", [](Collection& c, const std::string& id, nb::dict metadata) {
            auto meta = dict_to_metadata(metadata);
            throw_on_error(c.setMetadata(id, meta));
        }, nb::arg("id"), nb::arg("metadata"))

        .def("get_metadata", [](Collection& c, const std::string& id) {
            auto meta = unwrap(c.getMetadata(id));
            return metadata_to_dict(meta);
        }, nb::arg("id"))

        // Optimize
        .def("optimize", [](Collection& c) {
            nb::gil_scoped_release release;
            throw_on_error(c.optimize());
        })

        // Save/load/close
        .def("save", [](Collection& c, const std::string& path) {
            nb::gil_scoped_release release;
            throw_on_error(c.save(path));
        }, nb::arg("path"))

        .def("close", [](Collection& c) {
            throw_on_error(c.close());
        })

        .def("recovered_from_wal", &Collection::recoveredFromWal)
        .def("stats", &Collection::stats);
}
