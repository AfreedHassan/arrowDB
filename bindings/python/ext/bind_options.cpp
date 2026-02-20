#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <arrow/options.h>

namespace nb = nanobind;
using namespace arrow;

void bind_options(nb::module_& m) {
    // HNSWParams
    nb::class_<HNSWParams>(m, "HNSWParams")
        .def(nb::init<>())
        .def("__init__", [](HNSWParams* p, size_t M, size_t ef_construction, size_t ef_search) {
            new (p) HNSWParams{M, ef_construction, ef_search};
        }, nb::arg("M") = 16, nb::arg("ef_construction") = 200, nb::arg("ef_search") = 200)
        .def_rw("M", &HNSWParams::M)
        .def_rw("ef_construction", &HNSWParams::ef_construction)
        .def_rw("ef_search", &HNSWParams::ef_search);

    // IndexConfig
    nb::class_<IndexConfig>(m, "IndexConfig")
        .def(nb::init<>())
        .def("__init__", [](IndexConfig* c, IndexType index_type, size_t max_elements,
                           Quantization quantization, HNSWParams hnsw_params) {
            new (c) IndexConfig{index_type, max_elements, quantization, hnsw_params};
        }, nb::arg("index_type") = IndexType::HNSW,
           nb::arg("max_elements") = 1000000,
           nb::arg("quantization") = Quantization::None,
           nb::arg("hnsw_params") = HNSWParams{})
        .def_rw("index_type", &IndexConfig::index_type)
        .def_rw("max_elements", &IndexConfig::max_elements)
        .def_rw("quantization", &IndexConfig::quantization)
        .def_rw("hnsw_params", &IndexConfig::hnsw_params);

    // CollectionConfig
    nb::class_<CollectionConfig>(m, "CollectionConfig")
        .def(nb::init<>())
        .def("__init__", [](CollectionConfig* c, std::string name, uint32_t dimensions,
                           Space space, IndexConfig index_config) {
            new (c) CollectionConfig{std::move(name), dimensions, space, index_config};
        }, nb::arg("name"), nb::arg("dimensions"),
           nb::arg("space") = Space::Cosine,
           nb::arg("index_config") = IndexConfig{})
        .def_rw("name", &CollectionConfig::name)
        .def_rw("dimensions", &CollectionConfig::dimensions)
        .def_rw("space", &CollectionConfig::space)
        .def_rw("index_config", &CollectionConfig::index_config)
        .def_rw("schema", &CollectionConfig::schema);

    // ClientOptions
    nb::class_<ClientOptions>(m, "ClientOptions")
        .def(nb::init<>())
        .def("__init__", [](ClientOptions* o, std::string data_dir) {
            new (o) ClientOptions{std::filesystem::path(data_dir)};
        }, nb::arg("data_dir"))
        .def_prop_rw("data_dir",
            [](const ClientOptions& o) { return o.dataDir.string(); },
            [](ClientOptions& o, const std::string& p) { o.dataDir = p; })
        .def_rw("default_index_config", &ClientOptions::defaultIndexConfig);
}
