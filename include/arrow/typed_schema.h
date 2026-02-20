// Copyright 2025 ArrowDB
//
// Compile-time typed schema for search results. Header-only, fully opt-in.
//
// Usage:
//   using Docs = arrow::Schema<
//       arrow::Field<"category", arrow::FieldType::String, true>,
//       arrow::Field<"score",    arrow::FieldType::Double>,
//       arrow::Field<"count",    arrow::FieldType::Int64>
//   >;
//
//   auto results = arrow::bind<Docs>(col.query(vec, 10));
//   for (auto& hit : results.hits) {
//       std::string cat = hit.get<"category">();  // compile-time typed
//       // hit.get<"typo">() → compile error
//   }
//
#ifndef ARROW_TYPED_SCHEMA_H
#define ARROW_TYPED_SCHEMA_H

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include "arrow/collection.h"
#include "arrow/types.h"
#include "arrow/utils/result.h"
#include "arrow/utils/status.h"

namespace arrow {

// ── FixedString ─────────────────────────────────────────────
// Structural type enabling string literals as NTTP (C++20/23).

template <std::size_t N>
struct FixedString {
    char data[N]{};

    constexpr FixedString(const char (&str)[N]) {
        for (std::size_t i = 0; i < N; ++i) data[i] = str[i];
    }

    constexpr bool operator==(const FixedString&) const = default;

    template <std::size_t M>
    constexpr bool operator==(const FixedString<M>& other) const {
        if constexpr (N != M) return false;
        else {
            for (std::size_t i = 0; i < N; ++i)
                if (data[i] != other.data[i]) return false;
            return true;
        }
    }

    constexpr operator std::string_view() const {
        return std::string_view(data, N - 1);
    }

    std::string str() const { return std::string(data, N - 1); }
};

// ── FieldType → C++ type mapping ────────────────────────────

template <FieldType FT>
struct FieldTypeToCpp;

template <>
struct FieldTypeToCpp<FieldType::Int64> { using type = int64_t; };
template <>
struct FieldTypeToCpp<FieldType::Double> { using type = double; };
template <>
struct FieldTypeToCpp<FieldType::String> { using type = std::string; };
template <>
struct FieldTypeToCpp<FieldType::Bool> { using type = bool; };

template <FieldType FT>
using FieldTypeToCpp_t = typename FieldTypeToCpp<FT>::type;

// ── Field ───────────────────────────────────────────────────

template <FixedString Name, FieldType Type, bool Required = false>
struct Field {
    static constexpr auto name = Name;
    static constexpr FieldType fieldType = Type;
    static constexpr bool required = Required;

    using value_type = FieldTypeToCpp_t<Type>;
    using storage_type = std::conditional_t<Required, value_type, std::optional<value_type>>;
};

// ── Schema ──────────────────────────────────────────────────

namespace detail {

// Find the index of a field by name at compile time.
template <FixedString Name, typename... Fields>
struct IndexOfImpl;

template <FixedString Name>
struct IndexOfImpl<Name> {
    static constexpr std::size_t value = static_cast<std::size_t>(-1);
};

template <FixedString Name, typename Head, typename... Tail>
struct IndexOfImpl<Name, Head, Tail...> {
    static constexpr std::size_t value =
        (Name == Head::name) ? 0 : 1 + IndexOfImpl<Name, Tail...>::value;
};

// Check if a field name exists.
template <FixedString Name, typename... Fields>
struct HasFieldImpl : std::bool_constant<(... || (Name == Fields::name))> {};

// Extract a single field value from metadata.
template <typename FieldT>
typename FieldT::storage_type extractField(const Metadata& meta) {
    auto it = meta.find(std::string(std::string_view(FieldT::name)));
    if (it == meta.end()) {
        if constexpr (FieldT::required) {
            throw std::runtime_error(
                "Required field missing: " + std::string(std::string_view(FieldT::name)));
        } else {
            return std::nullopt;
        }
    }
    if constexpr (FieldT::fieldType == FieldType::Int64) {
        if constexpr (FieldT::required) {
            return it->second.asInt64();
        } else {
            if (!holds_alternative<int64_t>(it->second)) return std::nullopt;
            return it->second.asInt64();
        }
    } else if constexpr (FieldT::fieldType == FieldType::Double) {
        if constexpr (FieldT::required) {
            return it->second.asDouble();
        } else {
            if (!holds_alternative<double>(it->second)) return std::nullopt;
            return it->second.asDouble();
        }
    } else if constexpr (FieldT::fieldType == FieldType::String) {
        if constexpr (FieldT::required) {
            return std::string(it->second.asString());
        } else {
            if (!holds_alternative<std::string>(it->second)) return std::nullopt;
            return std::string(it->second.asString());
        }
    } else if constexpr (FieldT::fieldType == FieldType::Bool) {
        if constexpr (FieldT::required) {
            return it->second.asBool();
        } else {
            if (!holds_alternative<bool>(it->second)) return std::nullopt;
            return it->second.asBool();
        }
    }
}

}  // namespace detail

template <typename... Fields>
struct Schema {
    using tuple_type = std::tuple<typename Fields::storage_type...>;

    static constexpr std::size_t fieldCount = sizeof...(Fields);

    template <FixedString Name>
    static constexpr std::size_t indexOf =
        detail::IndexOfImpl<Name, Fields...>::value;

    template <FixedString Name>
    static constexpr bool hasField =
        detail::HasFieldImpl<Name, Fields...>::value;

    template <FixedString Name>
    using StorageAt = std::tuple_element_t<indexOf<Name>, tuple_type>;

    static MetadataSchema toRuntimeSchema() {
        MetadataSchema schema;
        (schema.field(
            std::string(std::string_view(Fields::name)),
            Fields::fieldType,
            Fields::required), ...);
        return schema;
    }

    // Build a tuple from metadata.
    static tuple_type extract(const Metadata& meta) {
        return tuple_type{detail::extractField<Fields>(meta)...};
    }
};

// ── Hit<S> ──────────────────────────────────────────────────

template <typename S>
struct Hit {
    VectorID id;
    float score;
    typename S::tuple_type fields;

    template <FixedString Name>
    decltype(auto) get() const {
        static_assert(S::template hasField<Name>,
            "Field not found in schema. "
            "Available fields are visible in the "
            "Schema<arrow::Field<...>, ...> type shown in the instantiation chain above.");
        constexpr auto idx = S::template indexOf<Name>;
        return std::get<idx>(fields);
    }

    template <FixedString Name>
    decltype(auto) get() {
        static_assert(S::template hasField<Name>,
            "Field not found in schema. "
            "Available fields are visible in the "
            "Schema<arrow::Field<...>, ...> type shown in the instantiation chain above.");
        constexpr auto idx = S::template indexOf<Name>;
        static_assert(idx < S::fieldCount, "Field not found in schema");
        return std::get<idx>(fields);
    }
};

// ── Results<S> ──────────────────────────────────────────────

template <typename S>
struct Results {
    std::vector<Hit<S>> hits;
};

// ── bind / tryBind ──────────────────────────────────────────

template <typename S>
Results<S> bind(const SearchResult& result) {
    Results<S> out;
    out.hits.reserve(result.hits.size());
    for (const auto& doc : result.hits) {
        out.hits.push_back(Hit<S>{
            .id = doc.id,
            .score = doc.score,
            .fields = S::extract(doc.metadata),
        });
    }
    return out;
}

template <typename S>
Results<S> bind(SearchResult&& result) {
    Results<S> out;
    out.hits.reserve(result.hits.size());
    for (auto& doc : result.hits) {
        out.hits.push_back(Hit<S>{
            .id = std::move(doc.id),
            .score = doc.score,
            .fields = S::extract(doc.metadata),
        });
    }
    return out;
}

template <typename S>
utils::Result<Results<S>> tryBind(const SearchResult& result) {
    try {
        return bind<S>(result);
    } catch (const std::exception& e) {
        return utils::Status(utils::StatusCode::kInvalidArgument, e.what());
    }
}

// ── query<S> free functions ─────────────────────────────────

template <typename S>
Results<S> query(const Collection& col,
                 const std::vector<float>& queryVec,
                 uint32_t k,
                 uint32_t ef = 200) {
    return bind<S>(col.query(queryVec, k, ef));
}

template <typename S>
Results<S> query(const Collection& col,
                 const std::vector<float>& queryVec,
                 uint32_t k,
                 const MetadataFilter& filter,
                 uint32_t ef = 200) {
    return bind<S>(col.query(queryVec, k, filter, ef));
}

template <typename S>
Results<S> query(const Collection& col,
                 const std::vector<float>& queryVec,
                 uint32_t k,
                 const PreparedFilter& filter,
                 uint32_t ef = 200) {
    return bind<S>(col.query(queryVec, k, filter, ef));
}

template <typename S>
Results<S> query(const Collection& col,
                 const std::string& queryStr,
                 uint32_t k,
                 uint32_t ef = 200) {
    return bind<S>(col.query(queryStr, k, ef));
}

}  // namespace arrow

#endif  // ARROW_TYPED_SCHEMA_H
