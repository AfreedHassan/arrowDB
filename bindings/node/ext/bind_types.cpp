#include "helpers.h"

Napi::Object InitTypes(Napi::Env env, Napi::Object exports) {
    // Space enum
    auto space = Napi::Object::New(env);
    space.Set("COSINE", Napi::Number::New(env, 0));
    space.Set("L2", Napi::Number::New(env, 1));
    space.Set("INNER_PRODUCT", Napi::Number::New(env, 2));
    exports.Set("Space", space);

    // IndexType enum
    auto indexType = Napi::Object::New(env);
    indexType.Set("HNSW", Napi::Number::New(env, 0));
    exports.Set("IndexType", indexType);

    // Quantization enum
    auto quant = Napi::Object::New(env);
    quant.Set("NONE", Napi::Number::New(env, 0));
    quant.Set("INT8", Napi::Number::New(env, 1));
    exports.Set("Quantization", quant);

    return exports;
}
