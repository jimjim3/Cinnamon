/// Declaration of the DigitalFlashCim dialect attributes.
#pragma once
#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimBase.h"
#include "mlir/IR/Attributes.h"

// Enums first so scoped enumerators (e.g.
// ::mlir::digitalflashcim::MultiplierKind::Exact) exist.
#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimAttributes.h.inc"
