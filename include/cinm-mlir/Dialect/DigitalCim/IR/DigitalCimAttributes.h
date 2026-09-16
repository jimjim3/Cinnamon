/// Declaration of the DigitalCim dialect attributes.
#pragma once
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimBase.h"
#include "mlir/IR/Attributes.h"

// Enums first so scoped enumerators (e.g.
// ::mlir::digitalcim::MultiplierKind::Exact) exist.
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimAttributes.h.inc"
