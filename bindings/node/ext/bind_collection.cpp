#pragma once

#include "helpers.h"
#include <arrow/collection.h>
#include <arrow/filter.h>

// Forward declarations from other bind files
arrow::CollectionConfig js_to_collection_config(std::string name, Napi::Object obj);
arrow::MetadataFilter js_to_filter(Napi::Value val);

class CollectionWrapper : public Napi::ObjectWrap<CollectionWrapper> {
public:
    static Napi::Function GetClass(Napi::Env env) {
        return DefineClass(env, "Collection", {
            InstanceMethod("insert", &CollectionWrapper::Insert),
            InstanceMethod("insertBatch", &CollectionWrapper::InsertBatch),
            InstanceMethod("get", &CollectionWrapper::Get),
            InstanceMethod("update", &CollectionWrapper::Update),
            InstanceMethod("upsert", &CollectionWrapper::Upsert),
            InstanceMethod("remove", &CollectionWrapper::Remove),
            InstanceMethod("search", &CollectionWrapper::Search),
            InstanceMethod("query", &CollectionWrapper::Query),
            InstanceMethod("searchBatch", &CollectionWrapper::SearchBatch),
            InstanceMethod("setMetadata", &CollectionWrapper::SetMetadata),
            InstanceMethod("getMetadata", &CollectionWrapper::GetMetadata),
            InstanceMethod("optimize", &CollectionWrapper::Optimize),
            InstanceMethod("save", &CollectionWrapper::Save),
            InstanceMethod("close", &CollectionWrapper::Close),
            InstanceMethod("stats", &CollectionWrapper::GetStats),
            InstanceAccessor("name", &CollectionWrapper::GetName, nullptr),
            InstanceAccessor("dimension", &CollectionWrapper::GetDimension, nullptr),
            InstanceAccessor("space", &CollectionWrapper::GetSpace, nullptr),
            InstanceAccessor("size", &CollectionWrapper::GetSize, nullptr),
        });
    }

    // Constructor: Collection(config) for in-memory
    CollectionWrapper(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<CollectionWrapper>(info) {
        auto env = info.Env();
        if (info.Length() < 1) {
            Napi::TypeError::New(env, "Expected config object").ThrowAsJavaScriptException();
            return;
        }

        // Internal construction from existing Collection pointer
        if (info[0].IsExternal()) {
            auto ext = info[0].As<Napi::External<arrow::Collection>>();
            collection_ = std::unique_ptr<arrow::Collection>(ext.Data());
            return;
        }

        auto obj = info[0].As<Napi::Object>();
        std::string name = obj.Has("name") ? obj.Get("name").As<Napi::String>().Utf8Value() : "";
        auto config = js_to_collection_config(name, obj);
        collection_ = std::make_unique<arrow::Collection>(config);
    }

    // Used by Client to wrap Collection* it owns
    void SetBorrowed(arrow::Collection* ptr) {
        collection_.reset();
        borrowed_ = ptr;
    }

    arrow::Collection& col() {
        return borrowed_ ? *borrowed_ : *collection_;
    }

    const arrow::Collection& col() const {
        return borrowed_ ? *borrowed_ : *collection_;
    }

    static Napi::Value CreatePersistent(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        auto obj = info[0].As<Napi::Object>();
        std::string name = obj.Has("name") ? obj.Get("name").As<Napi::String>().Utf8Value() : "";
        auto config = js_to_collection_config(name, obj);
        std::string path = info[1].As<Napi::String>().Utf8Value();

        auto result = arrow::Collection::create(config, path);
        if (!result.ok()) {
            throw_on_error(env, result.status());
            return env.Undefined();
        }

        auto* col = new arrow::Collection(std::move(*result));
        auto ext = Napi::External<arrow::Collection>::New(env, col);
        return GetClass(env).New({ext});
    }

    static Napi::Value Load(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string path = info[0].As<Napi::String>().Utf8Value();

        auto result = arrow::Collection::load(path);
        if (!result.ok()) {
            throw_on_error(env, result.status());
            return env.Undefined();
        }

        auto* col = new arrow::Collection(std::move(*result));
        auto ext = Napi::External<arrow::Collection>::New(env, col);
        return GetClass(env).New({ext});
    }

private:
    std::unique_ptr<arrow::Collection> collection_;
    arrow::Collection* borrowed_ = nullptr;

    // Properties
    Napi::Value GetName(const Napi::CallbackInfo& info) {
        return Napi::String::New(info.Env(), col().name());
    }
    Napi::Value GetDimension(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), col().dimension());
    }
    Napi::Value GetSpace(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), static_cast<int>(col().space()));
    }
    Napi::Value GetSize(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), static_cast<double>(col().size()));
    }

    // insert(embedding, metadata?) -> string  OR  insert(id, embedding, metadata?)
    Napi::Value Insert(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        if (info.Length() >= 2 && info[0].IsString() && (info[1].IsArray() || info[1].IsTypedArray())) {
            // insert(id, embedding, metadata?)
            std::string id = info[0].As<Napi::String>().Utf8Value();
            auto vec = js_to_float_vector(info[1]);
            arrow::Metadata meta;
            if (info.Length() >= 3 && info[2].IsObject()) {
                meta = js_to_metadata(info[2].As<Napi::Object>());
            }
            throw_on_error(env, col().insert(id, vec, std::move(meta)));
            return env.Undefined();
        }
        // insert(embedding, metadata?) -> auto-ID
        auto vec = js_to_float_vector(info[0]);
        arrow::Metadata meta;
        if (info.Length() >= 2 && info[1].IsObject()) {
            meta = js_to_metadata(info[1].As<Napi::Object>());
        }
        auto result = unwrap(env, col().insert(vec, std::move(meta)));
        if (env.IsExceptionPending()) return env.Undefined();
        return Napi::String::New(env, result);
    }

    Napi::Value InsertBatch(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        auto arr = info[0].As<Napi::Array>();
        std::vector<arrow::Document> docs;
        docs.reserve(arr.Length());
        for (uint32_t i = 0; i < arr.Length(); ++i) {
            auto docObj = arr.Get(i).As<Napi::Object>();
            arrow::Document doc;
            if (docObj.Has("id")) doc.id = docObj.Get("id").As<Napi::String>().Utf8Value();
            doc.embedding = js_to_float_vector(docObj.Get("embedding"));
            if (docObj.Has("metadata") && docObj.Get("metadata").IsObject()) {
                doc.metadata = js_to_metadata(docObj.Get("metadata").As<Napi::Object>());
            }
            docs.push_back(std::move(doc));
        }
        auto result = unwrap(env, col().insertBatch(std::move(docs)));
        if (env.IsExceptionPending()) return env.Undefined();

        auto obj = Napi::Object::New(env);
        obj.Set("successCount", Napi::Number::New(env, static_cast<double>(result.successCount)));
        obj.Set("failureCount", Napi::Number::New(env, static_cast<double>(result.failureCount)));
        auto results = Napi::Array::New(env, result.results.size());
        for (size_t i = 0; i < result.results.size(); ++i) {
            auto r = Napi::Object::New(env);
            r.Set("id", Napi::String::New(env, result.results[i].id));
            r.Set("ok", Napi::Boolean::New(env, result.results[i].status.ok()));
            r.Set("message", Napi::String::New(env, result.results[i].status.message()));
            results.Set(i, r);
        }
        obj.Set("results", results);
        return obj;
    }

    Napi::Value Get(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string id = info[0].As<Napi::String>().Utf8Value();
        auto vec = unwrap(env, col().get(id));
        if (env.IsExceptionPending()) return env.Undefined();
        return float_vector_to_js(env, vec);
    }

    Napi::Value Update(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string id = info[0].As<Napi::String>().Utf8Value();
        auto vec = js_to_float_vector(info[1]);
        arrow::Metadata meta;
        if (info.Length() >= 3 && info[2].IsObject()) {
            meta = js_to_metadata(info[2].As<Napi::Object>());
        }
        throw_on_error(env, col().update(id, vec, std::move(meta)));
        return env.Undefined();
    }

    Napi::Value Upsert(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string id = info[0].As<Napi::String>().Utf8Value();
        auto vec = js_to_float_vector(info[1]);
        arrow::Metadata meta;
        if (info.Length() >= 3 && info[2].IsObject()) {
            meta = js_to_metadata(info[2].As<Napi::Object>());
        }
        throw_on_error(env, col().upsert(id, vec, std::move(meta)));
        return env.Undefined();
    }

    Napi::Value Remove(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string id = info[0].As<Napi::String>().Utf8Value();
        throw_on_error(env, col().remove(id));
        return env.Undefined();
    }

    // search(query, k, opts?) where opts = { ef?, filter? }
    Napi::Value Search(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        auto query = js_to_float_vector(info[0]);
        uint32_t k = info[1].As<Napi::Number>().Uint32Value();
        uint32_t ef = 200;
        std::optional<arrow::MetadataFilter> filter;

        if (info.Length() >= 3 && info[2].IsObject()) {
            auto opts = info[2].As<Napi::Object>();
            if (opts.Has("ef")) ef = opts.Get("ef").As<Napi::Number>().Uint32Value();
            if (opts.Has("filter")) filter = js_to_filter(opts.Get("filter"));
        }

        if (filter) {
            return search_results_to_js(env, col().search(query, k, *filter, ef));
        }
        return search_results_to_js(env, col().search(query, k, ef));
    }

    // query(query, k, opts?) -> { hits: [...] }
    Napi::Value Query(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        auto query = js_to_float_vector(info[0]);
        uint32_t k = info[1].As<Napi::Number>().Uint32Value();
        uint32_t ef = 200;
        std::optional<arrow::MetadataFilter> filter;

        if (info.Length() >= 3 && info[2].IsObject()) {
            auto opts = info[2].As<Napi::Object>();
            if (opts.Has("ef")) ef = opts.Get("ef").As<Napi::Number>().Uint32Value();
            if (opts.Has("filter")) filter = js_to_filter(opts.Get("filter"));
        }

        arrow::SearchResult result = filter
            ? col().query(query, k, *filter, ef)
            : col().query(query, k, ef);
        return search_result_to_js(env, result);
    }

    Napi::Value SearchBatch(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        auto queriesArr = info[0].As<Napi::Array>();
        uint32_t k = info[1].As<Napi::Number>().Uint32Value();
        uint32_t ef = info.Length() >= 3 ? info[2].As<Napi::Number>().Uint32Value() : 200;

        std::vector<std::vector<float>> queries;
        queries.reserve(queriesArr.Length());
        for (uint32_t i = 0; i < queriesArr.Length(); ++i) {
            queries.push_back(js_to_float_vector(queriesArr.Get(i)));
        }

        auto results = unwrap(env, col().searchBatch(queries, k, ef));
        if (env.IsExceptionPending()) return env.Undefined();

        auto arr = Napi::Array::New(env, results.size());
        for (size_t i = 0; i < results.size(); ++i) {
            arr.Set(i, search_results_to_js(env, results[i]));
        }
        return arr;
    }

    Napi::Value SetMetadata(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string id = info[0].As<Napi::String>().Utf8Value();
        auto meta = js_to_metadata(info[1].As<Napi::Object>());
        throw_on_error(env, col().setMetadata(id, meta));
        return env.Undefined();
    }

    Napi::Value GetMetadata(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string id = info[0].As<Napi::String>().Utf8Value();
        auto meta = unwrap(env, col().getMetadata(id));
        if (env.IsExceptionPending()) return env.Undefined();
        return metadata_to_js(env, meta);
    }

    Napi::Value Optimize(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        throw_on_error(env, col().optimize());
        return env.Undefined();
    }

    Napi::Value Save(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string path = info[0].As<Napi::String>().Utf8Value();
        throw_on_error(env, col().save(path));
        return env.Undefined();
    }

    Napi::Value Close(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        throw_on_error(env, col().close());
        return env.Undefined();
    }

    Napi::Value GetStats(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        auto s = col().stats();
        auto obj = Napi::Object::New(env);
        obj.Set("vectorCount", Napi::Number::New(env, static_cast<double>(s.vectorCount)));
        obj.Set("metadataCount", Napi::Number::New(env, static_cast<double>(s.metadataCount)));
        obj.Set("maxCapacity", Napi::Number::New(env, static_cast<double>(s.maxCapacity)));
        obj.Set("dimensions", Napi::Number::New(env, static_cast<double>(s.dimensions)));
        return obj;
    }
};
