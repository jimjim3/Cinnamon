/// Declaration of the conversion passes for the DigitalFlashCim dialect.
///
/// @file

#pragma once

#include "cinm-mlir/Conversion/DigitalFlashCimToFunc/DigitalFlashCimToFunc.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/DigitalFlashCimPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
