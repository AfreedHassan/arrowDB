// Copyright 2025 ArrowDB
//
// Single public header for ArrowDB vector database library.
// Users should include only this header: #include <arrow/arrow.h>
//
#ifndef ARROW_ARROW_H
#define ARROW_ARROW_H

// Public types
#include "arrow/types.h"

// Error handling
#include "arrow/utils/status.h"
#include "arrow/utils/result.h"

// Configuration
#include "arrow/options.h"

// Database interface
#include "arrow/db.h"

// Collection interface (public API only)
// Note: Collection currently exposes some internals; this will be cleaned up
// when Collection is refactored to use Pimpl pattern.
#include "arrow/collection.h"

#endif // ARROW_ARROW_H
