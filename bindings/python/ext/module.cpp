#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Forward declarations for binding functions
void bind_status(nb::module_& m);
void bind_types(nb::module_& m);
void bind_options(nb::module_& m);
void bind_filter(nb::module_& m);
void bind_collection(nb::module_& m);
void bind_client(nb::module_& m);

NB_MODULE(_arrowdb, m) {
    m.doc() = "ArrowDB: A lightweight vector database for similarity search";

    bind_status(m);
    bind_types(m);
    bind_options(m);
    bind_filter(m);
    bind_collection(m);
    bind_client(m);
}
