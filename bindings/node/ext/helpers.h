#pragma once

#include <napi.h>
#include <arrow/types.h>
#include <arrow/utils/status.h>
#include <arrow/utils/result.h>
#include <string>
#include <vector>

// Throw a JS Error from a Status
inline void throw_on_error(Napi::Env env, const arrow::utils::Status& status) {
    if (status.ok()) return;
    std::string msg = status.message().empty()
        ? "ArrowDB error (code " + std::to_string(static_cast<int>(status.code())) + ")"
        : status.message();
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
}

// Unwrap a Result<T>, throwing on error
template <typename T>
T unwrap(Napi::Env env, arrow::utils::Result<T>&& result) {
    if (!result.ok()) {
        throw_on_error(env, result.status());
        // Return default-constructed T (caller must check env for exception)
        if constexpr (std::is_default_constructible_v<T>) {
            return T{};
        } else {
            throw std::runtime_error("ArrowDB error");
        }
    }
    return std::move(*result);
}

// Convert JS value to MetadataValue
inline arrow::MetadataValue js_to_metadata_value(Napi::Value val) {
    if (val.IsBoolean()) {
        return val.As<Napi::Boolean>().Value();
    } else if (val.IsNumber()) {
        double d = val.As<Napi::Number>().DoubleValue();
        // If it's an integer value, store as int64_t
        if (d == static_cast<double>(static_cast<int64_t>(d)) && d >= -9007199254740992.0 && d <= 9007199254740992.0) {
            return static_cast<int64_t>(d);
        }
        return d;
    } else if (val.IsBigInt()) {
        bool lossless;
        return val.As<Napi::BigInt>().Int64Value(&lossless);
    } else if (val.IsString()) {
        return val.As<Napi::String>().Utf8Value();
    }
    throw std::runtime_error("Metadata values must be number, bigint, string, or boolean");
}

// Convert JS object to Metadata
inline arrow::Metadata js_to_metadata(Napi::Object obj) {
    arrow::Metadata meta;
    auto names = obj.GetPropertyNames();
    for (uint32_t i = 0; i < names.Length(); ++i) {
        std::string key = names.Get(i).As<Napi::String>().Utf8Value();
        meta[key] = js_to_metadata_value(obj.Get(key));
    }
    return meta;
}

// Convert Metadata to JS object
inline Napi::Object metadata_to_js(Napi::Env env, const arrow::Metadata& meta) {
    auto obj = Napi::Object::New(env);
    for (const auto& [key, value] : meta) {
        std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, int64_t>) {
                obj.Set(key, Napi::Number::New(env, static_cast<double>(v)));
            } else if constexpr (std::is_same_v<T, double>) {
                obj.Set(key, Napi::Number::New(env, v));
            } else if constexpr (std::is_same_v<T, std::string>) {
                obj.Set(key, Napi::String::New(env, v));
            } else if constexpr (std::is_same_v<T, bool>) {
                obj.Set(key, Napi::Boolean::New(env, v));
            }
        }, value);
    }
    return obj;
}

// Convert JS array or Float32Array to std::vector<float>
inline std::vector<float> js_to_float_vector(Napi::Value val) {
    if (val.IsTypedArray()) {
        auto arr = val.As<Napi::Float32Array>();
        return std::vector<float>(arr.Data(), arr.Data() + arr.ElementLength());
    }
    auto arr = val.As<Napi::Array>();
    std::vector<float> vec;
    vec.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i) {
        vec.push_back(arr.Get(i).As<Napi::Number>().FloatValue());
    }
    return vec;
}

// Convert std::vector<float> to JS Float32Array
inline Napi::Float32Array float_vector_to_js(Napi::Env env, const std::vector<float>& vec) {
    auto buf = Napi::ArrayBuffer::New(env, vec.size() * sizeof(float));
    std::memcpy(buf.Data(), vec.data(), vec.size() * sizeof(float));
    return Napi::Float32Array::New(env, vec.size(), buf, 0);
}

// Convert IndexSearchResult vector to JS array
inline Napi::Array search_results_to_js(Napi::Env env, const std::vector<arrow::IndexSearchResult>& results) {
    auto arr = Napi::Array::New(env, results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        auto obj = Napi::Object::New(env);
        obj.Set("id", Napi::String::New(env, results[i].id));
        obj.Set("score", Napi::Number::New(env, results[i].score));
        arr.Set(i, obj);
    }
    return arr;
}

// Convert SearchResult to JS object
inline Napi::Object search_result_to_js(Napi::Env env, const arrow::SearchResult& result) {
    auto obj = Napi::Object::New(env);
    auto hits = Napi::Array::New(env, result.hits.size());
    for (size_t i = 0; i < result.hits.size(); ++i) {
        auto hit = Napi::Object::New(env);
        hit.Set("id", Napi::String::New(env, result.hits[i].id));
        hit.Set("score", Napi::Number::New(env, result.hits[i].score));
        hit.Set("metadata", metadata_to_js(env, result.hits[i].metadata));
        hits.Set(i, hit);
    }
    obj.Set("hits", hits);
    return obj;
}
