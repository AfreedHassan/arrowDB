#include "helpers.h"
#include <arrow/client.h>
#include <arrow/collection.h>

// Forward declaration
arrow::CollectionConfig js_to_collection_config(std::string name, Napi::Object obj);

class ClientWrapper : public Napi::ObjectWrap<ClientWrapper> {
public:
    static Napi::Function GetClass(Napi::Env env) {
        return DefineClass(env, "Client", {
            InstanceMethod("createCollection", &ClientWrapper::CreateCollection),
            InstanceMethod("getCollection", &ClientWrapper::GetCollection),
            InstanceMethod("getOrCreateCollection", &ClientWrapper::GetOrCreateCollection),
            InstanceMethod("dropCollection", &ClientWrapper::DropCollection),
            InstanceMethod("listCollections", &ClientWrapper::ListCollections),
            InstanceMethod("hasCollection", &ClientWrapper::HasCollection),
            InstanceMethod("close", &ClientWrapper::Close),
            InstanceAccessor("dataDir", &ClientWrapper::GetDataDir, nullptr),
        });
    }

    ClientWrapper(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<ClientWrapper>(info) {
        auto env = info.Env();
        if (info.Length() < 1) {
            Napi::TypeError::New(env, "Expected data directory path").ThrowAsJavaScriptException();
            return;
        }
        std::string dataDir = info[0].As<Napi::String>().Utf8Value();
        client_ = std::make_unique<arrow::Client>(std::filesystem::path(dataDir));
    }

private:
    std::unique_ptr<arrow::Client> client_;

    Napi::Value WrapCollection(Napi::Env env, arrow::Collection* ptr) {
        // Create a CollectionWrapper in "borrowed" mode
        auto ctor = CollectionWrapper::GetClass(env);
        // Create with dummy config, then override with borrowed pointer
        auto configObj = Napi::Object::New(env);
        configObj.Set("name", Napi::String::New(env, ""));
        configObj.Set("dimensions", Napi::Number::New(env, ptr->dimension()));
        auto wrapper = ctor.New({configObj});
        auto* cw = Napi::ObjectWrap<CollectionWrapper>::Unwrap(wrapper.As<Napi::Object>());
        cw->SetBorrowed(ptr);
        return wrapper;
    }

    Napi::Value CreateCollection(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string name = info[0].As<Napi::String>().Utf8Value();
        auto configObj = info[1].As<Napi::Object>();
        auto config = js_to_collection_config(name, configObj);
        auto* ptr = unwrap(env, client_->createCollection(name, config));
        if (env.IsExceptionPending()) return env.Undefined();
        return WrapCollection(env, ptr);
    }

    Napi::Value GetCollection(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string name = info[0].As<Napi::String>().Utf8Value();
        auto* ptr = unwrap(env, client_->getCollection(name));
        if (env.IsExceptionPending()) return env.Undefined();
        return WrapCollection(env, ptr);
    }

    Napi::Value GetOrCreateCollection(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string name = info[0].As<Napi::String>().Utf8Value();
        arrow::Collection* ptr;
        if (info.Length() >= 2 && info[1].IsObject()) {
            auto config = js_to_collection_config(name, info[1].As<Napi::Object>());
            ptr = unwrap(env, client_->getOrCreateCollection(name, config));
        } else {
            ptr = unwrap(env, client_->getOrCreateCollection(name));
        }
        if (env.IsExceptionPending()) return env.Undefined();
        return WrapCollection(env, ptr);
    }

    Napi::Value DropCollection(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string name = info[0].As<Napi::String>().Utf8Value();
        throw_on_error(env, client_->dropCollection(name));
        return env.Undefined();
    }

    Napi::Value ListCollections(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        auto names = client_->listCollections();
        auto arr = Napi::Array::New(env, names.size());
        for (size_t i = 0; i < names.size(); ++i) {
            arr.Set(i, Napi::String::New(env, names[i]));
        }
        return arr;
    }

    Napi::Value HasCollection(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        std::string name = info[0].As<Napi::String>().Utf8Value();
        return Napi::Boolean::New(env, client_->hasCollection(name));
    }

    Napi::Value Close(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        throw_on_error(env, client_->close());
        return env.Undefined();
    }

    Napi::Value GetDataDir(const Napi::CallbackInfo& info) {
        return Napi::String::New(info.Env(), client_->dataDir().string());
    }
};
