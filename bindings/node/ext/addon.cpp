#include "helpers.h"
#include "bind_collection.cpp"
#include "bind_client.cpp"

Napi::Object InitTypes(Napi::Env env, Napi::Object exports);

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Enums
    InitTypes(env, exports);

    // Collection class + static factories
    auto collClass = CollectionWrapper::GetClass(env);
    exports.Set("Collection", collClass);

    // Static methods on Collection
    exports.Set("createCollection", Napi::Function::New(env, CollectionWrapper::CreatePersistent));
    exports.Set("loadCollection", Napi::Function::New(env, CollectionWrapper::Load));

    // Client class
    exports.Set("Client", ClientWrapper::GetClass(env));

    return exports;
}

NODE_API_MODULE(arrowdb_addon, Init)
