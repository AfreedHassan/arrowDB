#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <arrow/client.h>
#include <arrow/collection.h>
#include "helpers.h"

namespace nb = nanobind;
using namespace arrow;

// Prevent Python from GC-ing the Client while a CollectionRef is alive
struct PyCollectionRef {
    Collection* ptr;
    nb::object client_ref;  // prevent GC of parent Client

    PyCollectionRef(Collection* p, nb::object ref)
        : ptr(p), client_ref(std::move(ref)) {}
};

void bind_client(nb::module_& m) {
    // CollectionRef — a non-owning handle that keeps the client alive
    nb::class_<PyCollectionRef>(m, "CollectionRef")
        .def_prop_ro("name", [](const PyCollectionRef& r) { return r.ptr->name(); })
        .def_prop_ro("dimension", [](const PyCollectionRef& r) { return r.ptr->dimension(); })
        .def_prop_ro("space", [](const PyCollectionRef& r) { return r.ptr->space(); })
        .def("__len__", [](const PyCollectionRef& r) { return r.ptr->size(); })

        .def("insert", [](PyCollectionRef& r, const nb::object& vec, nb::dict metadata) -> std::string {
            auto v = to_float_vector(vec);
            auto meta = dict_to_metadata(metadata);
            nb::gil_scoped_release release;
            return unwrap(r.ptr->insert(v, std::move(meta)));
        }, nb::arg("embedding"), nb::arg("metadata") = nb::dict())

        .def("insert", [](PyCollectionRef& r, const std::string& id,
                          const nb::object& vec, nb::dict metadata) {
            auto v = to_float_vector(vec);
            auto meta = dict_to_metadata(metadata);
            nb::gil_scoped_release release;
            throw_on_error(r.ptr->insert(id, v, std::move(meta)));
        }, nb::arg("id"), nb::arg("embedding"), nb::arg("metadata") = nb::dict())

        .def("insert_batch", [](PyCollectionRef& r, nb::list docs_list) {
            std::vector<Document> docs;
            docs.reserve(nb::len(docs_list));
            for (auto item : docs_list) {
                docs.push_back(nb::cast<Document>(nb::borrow(item)));
            }
            nb::gil_scoped_release release;
            return unwrap(r.ptr->insertBatch(std::move(docs)));
        }, nb::arg("documents"))

        .def("get", [](const PyCollectionRef& r, const std::string& id) {
            return unwrap(r.ptr->get(id));
        }, nb::arg("id"))

        .def("update", [](PyCollectionRef& r, const std::string& id,
                          const nb::object& vec, nb::dict metadata) {
            auto v = to_float_vector(vec);
            auto meta = dict_to_metadata(metadata);
            nb::gil_scoped_release release;
            throw_on_error(r.ptr->update(id, v, std::move(meta)));
        }, nb::arg("id"), nb::arg("embedding"), nb::arg("metadata") = nb::dict())

        .def("upsert", [](PyCollectionRef& r, const std::string& id,
                          const nb::object& vec, nb::dict metadata) {
            auto v = to_float_vector(vec);
            auto meta = dict_to_metadata(metadata);
            nb::gil_scoped_release release;
            throw_on_error(r.ptr->upsert(id, v, std::move(meta)));
        }, nb::arg("id"), nb::arg("embedding"), nb::arg("metadata") = nb::dict())

        .def("remove", [](PyCollectionRef& r, const std::string& id) {
            throw_on_error(r.ptr->remove(id));
        }, nb::arg("id"))

        .def("search", [](const PyCollectionRef& r, const nb::object& query,
                          uint32_t k, uint32_t ef) {
            auto q = to_float_vector(query);
            nb::gil_scoped_release release;
            return r.ptr->search(q, k, ef);
        }, nb::arg("query"), nb::arg("k"), nb::arg("ef") = 200)

        .def("search", [](const PyCollectionRef& r, const nb::object& query,
                          uint32_t k, const MetadataFilter& filter, uint32_t ef) {
            auto q = to_float_vector(query);
            nb::gil_scoped_release release;
            return r.ptr->search(q, k, filter, ef);
        }, nb::arg("query"), nb::arg("k"), nb::arg("filter"), nb::arg("ef") = 200)

        .def("query", [](const PyCollectionRef& r, const nb::object& query,
                         uint32_t k, uint32_t ef) {
            auto q = to_float_vector(query);
            nb::gil_scoped_release release;
            return r.ptr->query(q, k, ef);
        }, nb::arg("query"), nb::arg("k"), nb::arg("ef") = 200)

        .def("query", [](const PyCollectionRef& r, const nb::object& query,
                         uint32_t k, const MetadataFilter& filter, uint32_t ef) {
            auto q = to_float_vector(query);
            nb::gil_scoped_release release;
            return r.ptr->query(q, k, filter, ef);
        }, nb::arg("query"), nb::arg("k"), nb::arg("filter"), nb::arg("ef") = 200)

        .def("search_batch", [](const PyCollectionRef& r,
                                const std::vector<std::vector<float>>& queries,
                                uint32_t k, uint32_t ef) {
            nb::gil_scoped_release release;
            return unwrap(r.ptr->searchBatch(queries, k, ef));
        }, nb::arg("queries"), nb::arg("k"), nb::arg("ef") = 200)

        .def("set_metadata", [](PyCollectionRef& r, const std::string& id, nb::dict metadata) {
            auto meta = dict_to_metadata(metadata);
            throw_on_error(r.ptr->setMetadata(id, meta));
        }, nb::arg("id"), nb::arg("metadata"))

        .def("get_metadata", [](PyCollectionRef& r, const std::string& id) {
            auto meta = unwrap(r.ptr->getMetadata(id));
            return metadata_to_dict(meta);
        }, nb::arg("id"))

        .def("optimize", [](PyCollectionRef& r) {
            nb::gil_scoped_release release;
            throw_on_error(r.ptr->optimize());
        })

        .def("save", [](PyCollectionRef& r, const std::string& path) {
            nb::gil_scoped_release release;
            throw_on_error(r.ptr->save(path));
        }, nb::arg("path"))

        .def("close", [](PyCollectionRef& r) {
            throw_on_error(r.ptr->close());
        })

        .def("stats", [](const PyCollectionRef& r) { return r.ptr->stats(); });

    // Client
    nb::class_<Client>(m, "Client")
        .def("__init__", [](Client* c, const std::string& data_dir) {
            new (c) Client(std::filesystem::path(data_dir));
        }, nb::arg("data_dir"))

        .def("__init__", [](Client* c, const ClientOptions& opts) {
            new (c) Client(opts);
        }, nb::arg("options"))

        // __enter__/__exit__ are defined in Python's __init__.py for clean None handling

        .def("create_collection", [](nb::object self, const std::string& name,
                                     const CollectionConfig& config) {
            Client& c = nb::cast<Client&>(self);
            Collection* ptr = unwrap(c.createCollection(name, config));
            return PyCollectionRef(ptr, self);
        }, nb::arg("name"), nb::arg("config"))

        .def("get_collection", [](nb::object self, const std::string& name) {
            Client& c = nb::cast<Client&>(self);
            Collection* ptr = unwrap(c.getCollection(name));
            return PyCollectionRef(ptr, self);
        }, nb::arg("name"))

        .def("get_or_create_collection", [](nb::object self, const std::string& name,
                                            const CollectionConfig& config) {
            Client& c = nb::cast<Client&>(self);
            Collection* ptr = unwrap(c.getOrCreateCollection(name, config));
            return PyCollectionRef(ptr, self);
        }, nb::arg("name"), nb::arg("config"))

        .def("get_or_create_collection", [](nb::object self, const std::string& name) {
            Client& c = nb::cast<Client&>(self);
            Collection* ptr = unwrap(c.getOrCreateCollection(name));
            return PyCollectionRef(ptr, self);
        }, nb::arg("name"))

        .def("drop_collection", [](Client& c, const std::string& name) {
            throw_on_error(c.dropCollection(name));
        }, nb::arg("name"))

        .def("list_collections", &Client::listCollections)
        .def("has_collection", &Client::hasCollection, nb::arg("name"))

        .def("close", [](Client& c) {
            throw_on_error(c.close());
        })

        .def_prop_ro("data_dir", [](const Client& c) { return c.dataDir().string(); });
}
