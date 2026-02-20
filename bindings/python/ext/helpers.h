#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/pair.h>
#include <arrow/types.h>
#include <arrow/utils/status.h>
#include <arrow/utils/result.h>

namespace nb = nanobind;

// Declared in bind_status.cpp — throws C++ exception, safe without GIL
void throw_on_error(const arrow::utils::Status& status);

template <typename T>
T unwrap(arrow::utils::Result<T>&& result) {
    if (!result.ok()) {
        throw_on_error(result.status());
    }
    return std::move(*result);
}

// Convert a single Python object to MetadataValue
inline arrow::MetadataValue py_to_metadata_value(nb::handle obj) {
    if (nb::isinstance<nb::bool_>(obj)) {
        return nb::cast<bool>(obj);
    } else if (nb::isinstance<nb::int_>(obj)) {
        return nb::cast<int64_t>(obj);
    } else if (nb::isinstance<nb::float_>(obj)) {
        return nb::cast<double>(obj);
    } else if (nb::isinstance<nb::str>(obj)) {
        return nb::cast<std::string>(obj);
    }
    throw nb::type_error("Metadata values must be int, float, str, or bool");
}

// Convert Python dict to C++ Metadata
inline arrow::Metadata dict_to_metadata(const nb::dict& d) {
    arrow::Metadata meta;
    for (auto item : d) {
        std::string k = nb::cast<std::string>(item.first);
        meta[k] = py_to_metadata_value(item.second);
    }
    return meta;
}

// Convert C++ Metadata to Python dict
inline nb::dict metadata_to_dict(const arrow::Metadata& meta) {
    nb::dict d;
    for (const auto& [key, value] : meta) {
        std::visit([&](const auto& v) {
            d[nb::str(key.c_str())] = nb::cast(v);
        }, value);
    }
    return d;
}

// Convert Python list to std::vector<float>
inline std::vector<float> to_float_vector(const nb::handle& obj) {
    nb::list lst = nb::cast<nb::list>(obj);
    size_t n = nb::len(lst);
    std::vector<float> vec;
    vec.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        vec.push_back(nb::cast<float>(lst[i]));
    }
    return vec;
}
