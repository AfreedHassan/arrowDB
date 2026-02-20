#include "helpers.h"
#include <arrow/options.h>

arrow::HNSWParams js_to_hnsw_params(Napi::Object obj) {
    arrow::HNSWParams params;
    if (obj.Has("M")) params.M = obj.Get("M").As<Napi::Number>().Uint32Value();
    if (obj.Has("efConstruction")) params.ef_construction = obj.Get("efConstruction").As<Napi::Number>().Uint32Value();
    if (obj.Has("efSearch")) params.ef_search = obj.Get("efSearch").As<Napi::Number>().Uint32Value();
    return params;
}

arrow::IndexConfig js_to_index_config(Napi::Object obj) {
    arrow::IndexConfig config;
    if (obj.Has("maxElements")) config.max_elements = obj.Get("maxElements").As<Napi::Number>().Uint32Value();
    if (obj.Has("quantization")) config.quantization = static_cast<arrow::Quantization>(obj.Get("quantization").As<Napi::Number>().Uint32Value());
    if (obj.Has("hnswParams")) config.hnsw_params = js_to_hnsw_params(obj.Get("hnswParams").As<Napi::Object>());
    return config;
}

arrow::CollectionConfig js_to_collection_config(std::string name, Napi::Object obj) {
    arrow::CollectionConfig config;
    config.name = std::move(name);
    if (obj.Has("dimensions")) config.dimensions = obj.Get("dimensions").As<Napi::Number>().Uint32Value();
    if (obj.Has("space")) config.space = static_cast<arrow::Space>(obj.Get("space").As<Napi::Number>().Uint32Value());
    if (obj.Has("indexConfig")) config.index_config = js_to_index_config(obj.Get("indexConfig").As<Napi::Object>());
    return config;
}
