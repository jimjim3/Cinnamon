/// Declaration of the conversion passes for the DigitalCim dialect.
///
/// @file

#pragma once

#include "cinm-mlir/Conversion/DigitalCimToFunc/DigitalCimToFunc.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/DigitalCimPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
