/// Declaration of the DigitalCim dialect ops.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimAttributes.h"
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimBase.h"

#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimOps.h.inc"

//===----------------------------------------------------------------------===//
