#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <arrow/filter.h>
#include "helpers.h"

namespace nb = nanobind;
using namespace arrow;

void bind_filter(nb::module_& m) {
    nb::class_<MetadataFilter>(m, "MetadataFilter")
        // Construct from Python callable(dict) -> bool
        .def("__init__", [](MetadataFilter* f, nb::callable fn) {
            nb::object fn_obj = nb::borrow<nb::object>(fn);
            new (f) MetadataFilter([fn_obj](const Metadata& meta) -> bool {
                nb::gil_scoped_acquire gil;
                nb::dict d = metadata_to_dict(meta);
                nb::object result = fn_obj(d);
                return nb::cast<bool>(result);
            });
        }, nb::arg("predicate"))
        // DSL factory methods
        .def_static("eq", [](std::string field, nb::handle value) {
            return MetadataFilter::Eq(std::move(field), py_to_metadata_value(value));
        }, nb::arg("field"), nb::arg("value"))
        .def_static("neq", [](std::string field, nb::handle value) {
            return MetadataFilter::Neq(std::move(field), py_to_metadata_value(value));
        }, nb::arg("field"), nb::arg("value"))
        .def_static("gt", [](std::string field, nb::handle value) {
            return MetadataFilter::Gt(std::move(field), py_to_metadata_value(value));
        }, nb::arg("field"), nb::arg("value"))
        .def_static("gte", [](std::string field, nb::handle value) {
            return MetadataFilter::Gte(std::move(field), py_to_metadata_value(value));
        }, nb::arg("field"), nb::arg("value"))
        .def_static("lt", [](std::string field, nb::handle value) {
            return MetadataFilter::Lt(std::move(field), py_to_metadata_value(value));
        }, nb::arg("field"), nb::arg("value"))
        .def_static("lte", [](std::string field, nb::handle value) {
            return MetadataFilter::Lte(std::move(field), py_to_metadata_value(value));
        }, nb::arg("field"), nb::arg("value"))
        .def_static("in_", [](std::string field, nb::list values) {
            std::vector<MetadataValue> vals;
            vals.reserve(nb::len(values));
            for (size_t i = 0; i < nb::len(values); ++i) {
                vals.push_back(py_to_metadata_value(values[i]));
            }
            return MetadataFilter::In(std::move(field), std::move(vals));
        }, nb::arg("field"), nb::arg("values"))
        // Logical combinators
        .def_static("and_", [](MetadataFilter a, MetadataFilter b) {
            return MetadataFilter::And(std::move(a), std::move(b));
        }, nb::arg("a"), nb::arg("b"))
        .def_static("or_", [](MetadataFilter a, MetadataFilter b) {
            return MetadataFilter::Or(std::move(a), std::move(b));
        }, nb::arg("a"), nb::arg("b"))
        .def_static("not_", [](MetadataFilter inner) {
            return MetadataFilter::Not(std::move(inner));
        }, nb::arg("filter"))
        // Direct evaluation
        .def("__call__", [](const MetadataFilter& f, nb::dict d) {
            Metadata meta = dict_to_metadata(d);
            return f(meta);
        }, nb::arg("metadata"));
}
