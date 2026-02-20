#include "helpers.h"
#include <arrow/filter.h>

// Forward declaration
arrow::MetadataFilter js_to_filter(Napi::Value val);

arrow::MetadataFilter js_to_filter(Napi::Value val) {
    if (!val.IsObject()) {
        throw std::runtime_error("MetadataFilter must be an object with {op, field, value} or {op, filters}");
    }

    auto obj = val.As<Napi::Object>();
    std::string op = obj.Get("op").As<Napi::String>().Utf8Value();

    if (op == "eq" || op == "neq" || op == "gt" || op == "gte" || op == "lt" || op == "lte") {
        std::string field = obj.Get("field").As<Napi::String>().Utf8Value();
        auto value = js_to_metadata_value(obj.Get("value"));

        if (op == "eq")  return arrow::MetadataFilter::Eq(field, value);
        if (op == "neq") return arrow::MetadataFilter::Neq(field, value);
        if (op == "gt")  return arrow::MetadataFilter::Gt(field, value);
        if (op == "gte") return arrow::MetadataFilter::Gte(field, value);
        if (op == "lt")  return arrow::MetadataFilter::Lt(field, value);
        return arrow::MetadataFilter::Lte(field, value);
    }

    if (op == "in") {
        std::string field = obj.Get("field").As<Napi::String>().Utf8Value();
        auto jsVals = obj.Get("values").As<Napi::Array>();
        std::vector<arrow::MetadataValue> values;
        values.reserve(jsVals.Length());
        for (uint32_t i = 0; i < jsVals.Length(); ++i) {
            values.push_back(js_to_metadata_value(jsVals.Get(i)));
        }
        return arrow::MetadataFilter::In(field, std::move(values));
    }

    if (op == "and" || op == "or") {
        auto filters = obj.Get("filters").As<Napi::Array>();
        if (filters.Length() < 2) {
            throw std::runtime_error("and/or require at least 2 filters");
        }
        auto result = js_to_filter(filters.Get(uint32_t(0)));
        for (uint32_t i = 1; i < filters.Length(); ++i) {
            auto next = js_to_filter(filters.Get(i));
            result = (op == "and")
                ? arrow::MetadataFilter::And(std::move(result), std::move(next))
                : arrow::MetadataFilter::Or(std::move(result), std::move(next));
        }
        return result;
    }

    if (op == "not") {
        return arrow::MetadataFilter::Not(js_to_filter(obj.Get("filter")));
    }

    throw std::runtime_error("Unknown filter op: " + op);
}
