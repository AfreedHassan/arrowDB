// Copyright 2025 ArrowDB
#ifndef ARROW_FILTER_H
#define ARROW_FILTER_H

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "arrow/types.h"

namespace arrow {

/// Unified metadata filter: DSL factories + implicit construction from lambdas.
///
/// Usage:
///   // DSL
///   col.search(query, 10, MetadataFilter::Eq("category", "tech"));
///
///   // Raw lambda — implicit construction
///   col.search(query, 10, [](const Metadata& m) { return m.contains("x"); });
///
///   // Composed
///   MetadataFilter::And(
///       MetadataFilter::Where<std::string>("name", [](auto& s) { return s.size() > 5; }),
///       MetadataFilter::Gt("year", int64_t(2020)));
class MetadataFilter {
public:
    /// Construct from any callable (lambda, std::function, function pointer, etc.).
    template <typename Fn,
              typename = std::enable_if_t<
                  std::is_invocable_r_v<bool, Fn, const Metadata&> &&
                  !std::is_same_v<std::decay_t<Fn>, MetadataFilter>>>
    MetadataFilter(Fn&& fn)
        : pred_(std::make_shared<std::function<bool(const Metadata&)>>(std::forward<Fn>(fn))) {}

    // ── Comparison operators ────────────────────────────────

    /// Exact equality (works on all MetadataValue types).
    static MetadataFilter Eq(std::string field, MetadataValue value) {
        return MetadataFilter([f = std::move(field), v = std::move(value)](const Metadata& meta) {
            auto it = meta.find(f);
            return it != meta.end() && it->second == v;
        });
    }

    /// Not-equal (works on all MetadataValue types).
    static MetadataFilter Neq(std::string field, MetadataValue value) {
        return MetadataFilter([f = std::move(field), v = std::move(value)](const Metadata& meta) {
            auto it = meta.find(f);
            return it != meta.end() && it->second != v;
        });
    }

    /// Greater-than (numeric types only, with cross-type int64/double comparison).
    static MetadataFilter Gt(std::string field, MetadataValue value) {
        return MetadataFilter([f = std::move(field), v = std::move(value)](const Metadata& meta) {
            auto it = meta.find(f);
            if (it == meta.end()) return false;
            return compareNumeric(it->second, v) > 0;
        });
    }

    /// Greater-than-or-equal (numeric types only).
    static MetadataFilter Gte(std::string field, MetadataValue value) {
        return MetadataFilter([f = std::move(field), v = std::move(value)](const Metadata& meta) {
            auto it = meta.find(f);
            if (it == meta.end()) return false;
            return compareNumeric(it->second, v) >= 0;
        });
    }

    /// Less-than (numeric types only).
    static MetadataFilter Lt(std::string field, MetadataValue value) {
        return MetadataFilter([f = std::move(field), v = std::move(value)](const Metadata& meta) {
            auto it = meta.find(f);
            if (it == meta.end()) return false;
            return compareNumeric(it->second, v) < 0;
        });
    }

    /// Less-than-or-equal (numeric types only).
    static MetadataFilter Lte(std::string field, MetadataValue value) {
        return MetadataFilter([f = std::move(field), v = std::move(value)](const Metadata& meta) {
            auto it = meta.find(f);
            if (it == meta.end()) return false;
            return compareNumeric(it->second, v) <= 0;
        });
    }

    /// Field value is in the given set (works on all types).
    static MetadataFilter In(std::string field, std::vector<MetadataValue> values) {
        return MetadataFilter([f = std::move(field), vals = std::move(values)](const Metadata& meta) {
            auto it = meta.find(f);
            if (it == meta.end()) return false;
            for (const auto& v : vals) {
                if (it->second == v) return true;
            }
            return false;
        });
    }

    // ── Custom predicate ─────────────────────────────────────

    /// Type-safe custom predicate on a single field.
    /// T must be one of: int64_t, double, std::string, bool.
    /// Returns false if the field is missing or holds a different type.
    ///
    /// Usage:
    ///   MetadataFilter::Where<std::string>("name", [](const std::string& s) {
    ///       return s.size() > 5;
    ///   })
    template <typename T>
    static MetadataFilter Where(std::string field, std::function<bool(const T&)> pred) {
        return MetadataFilter([f = std::move(field), p = std::move(pred)](const Metadata& meta) {
            auto it = meta.find(f);
            if (it == meta.end()) return false;
            auto* val = std::get_if<T>(&it->second);
            return val && p(*val);
        });
    }

    // ── Logical operators ───────────────────────────────────

    /// Logical AND of two filters.
    static MetadataFilter And(MetadataFilter a, MetadataFilter b) {
        return MetadataFilter([pa = std::move(a.pred_), pb = std::move(b.pred_)](const Metadata& meta) {
            return (*pa)(meta) && (*pb)(meta);
        });
    }

    /// Logical OR of two filters.
    static MetadataFilter Or(MetadataFilter a, MetadataFilter b) {
        return MetadataFilter([pa = std::move(a.pred_), pb = std::move(b.pred_)](const Metadata& meta) {
            return (*pa)(meta) || (*pb)(meta);
        });
    }

    /// Logical NOT.
    static MetadataFilter Not(MetadataFilter inner) {
        return MetadataFilter([p = std::move(inner.pred_)](const Metadata& meta) {
            return !(*p)(meta);
        });
    }

    /// Variadic AND — all filters must match.
    static MetadataFilter And(std::initializer_list<MetadataFilter> filters) {
        std::vector<Predicate> preds;
        preds.reserve(filters.size());
        for (auto& f : filters) {
            preds.push_back(f.pred_);
        }
        return MetadataFilter([ps = std::move(preds)](const Metadata& meta) {
            for (const auto& p : ps) {
                if (!(*p)(meta)) return false;
            }
            return true;
        });
    }

    /// Variadic OR — at least one filter must match.
    static MetadataFilter Or(std::initializer_list<MetadataFilter> filters) {
        std::vector<Predicate> preds;
        preds.reserve(filters.size());
        for (auto& f : filters) {
            preds.push_back(f.pred_);
        }
        return MetadataFilter([ps = std::move(preds)](const Metadata& meta) {
            for (const auto& p : ps) {
                if ((*p)(meta)) return true;
            }
            return false;
        });
    }

    /// Direct evaluation.
    bool operator()(const Metadata& meta) const { return (*pred_)(meta); }

private:
    using Predicate = std::shared_ptr<std::function<bool(const Metadata&)>>;
    Predicate pred_;

    /// Compare two MetadataValues numerically.
    /// Returns <0, 0, or >0 like strcmp. Returns 0 for non-numeric types.
    static int compareNumeric(const MetadataValue& lhs, const MetadataValue& rhs) {
        auto toDouble = [](const MetadataValue& v) -> std::pair<bool, double> {
            if (auto* i = std::get_if<int64_t>(&v)) return {true, static_cast<double>(*i)};
            if (auto* d = std::get_if<double>(&v)) return {true, *d};
            return {false, 0.0};
        };
        auto [lOk, lVal] = toDouble(lhs);
        auto [rOk, rVal] = toDouble(rhs);
        if (!lOk || !rOk) return 0;
        if (lVal < rVal) return -1;
        if (lVal > rVal) return 1;
        return 0;
    }
};

} // namespace arrow

#endif // ARROW_FILTER_H
