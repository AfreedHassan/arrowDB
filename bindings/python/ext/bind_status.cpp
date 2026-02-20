#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <arrow/utils/status.h>
#include <arrow/utils/result.h>
#include <stdexcept>

namespace nb = nanobind;
using namespace arrow::utils;

// C++ exception — safe to throw without GIL held.
// nanobind catches it at the Python/C boundary (where GIL is re-acquired).
struct ArrowDBException : std::runtime_error {
    StatusCode code;
    ArrowDBException(StatusCode c, const std::string& msg)
        : std::runtime_error(msg), code(c) {}
};

void throw_on_error(const Status& status) {
    if (status.ok()) return;
    std::string msg = status.message().empty()
        ? "ArrowDB error (code " + std::to_string(static_cast<int>(status.code())) + ")"
        : status.message();
    throw ArrowDBException(status.code(), msg);
}

void bind_status(nb::module_& m) {
    // StatusCode enum
    nb::enum_<StatusCode>(m, "StatusCode")
        .value("OK", StatusCode::kOk)
        .value("INVALID_ARGUMENT", StatusCode::kInvalidArgument)
        .value("NOT_FOUND", StatusCode::kNotFound)
        .value("ALREADY_EXISTS", StatusCode::kAlreadyExists)
        .value("UNIMPLEMENTED", StatusCode::kUnimplemented)
        .value("DIMENSION_MISMATCH", StatusCode::kDimensionMismatch)
        .value("IO_ERROR", StatusCode::kIoError)
        .value("EOF", StatusCode::kEof)
        .value("CORRUPTION", StatusCode::kCorruption)
        .value("CHECKSUM_MISMATCH", StatusCode::kChecksumMismatch)
        .value("BAD_RECORD", StatusCode::kBadRecord)
        .value("BAD_HEADER", StatusCode::kBadHeader)
        .value("VERSION_MISMATCH", StatusCode::kVersionMismatch)
        .value("INTERNAL", StatusCode::kInternal);

    // Register ArrowDBError as a Python exception (subclass of RuntimeError)
    nb::exception<ArrowDBException>(m, "ArrowDBError");
}
