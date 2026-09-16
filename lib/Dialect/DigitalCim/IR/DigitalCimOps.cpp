/// Implements the DigitalCim dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimOps.h"
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "digitalcim-ops"

using namespace mlir;
using namespace mlir::digitalcim;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimOps.cpp.inc"

//===----------------------------------------------------------------------===//
// DigitalCimDialect
//===----------------------------------------------------------------------===//

void DigitalCimDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimOps.cpp.inc"
      >();
}

void AcquireMacroOp::getAsmResultNames(
    ::mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "dcim_macro");
}

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

::mlir::LogicalResult WriteWeightsOp::verify() {
  auto weightsTy = dyn_cast<MemRefType>(getWeights().getType());
  if (!weightsTy || weightsTy.getRank() != 2)
    return emitOpError("weights must be a rank-2 memref (macroRows x macroCols)");

  if (!isa<IntegerType>(weightsTy.getElementType()))
    return emitOpError("weights element type must be an integer type "
                       "(digital CIM stores quantized weights, not floats)");

  return success();
}

::mlir::LogicalResult TriggerMacOp::verify() {
  auto inputTy = dyn_cast<MemRefType>(getInput().getType());
  if (!inputTy || inputTy.getRank() != 1)
    return emitOpError("input must be a rank-1 memref (length macroRows)");

  if (!isa<IntegerType>(inputTy.getElementType()))
    return emitOpError("input element type must be an integer type "
                       "(digital CIM streams quantized activations, not floats)");

  // Best-effort: if the future's payload is itself a memref, it should be
  // rank-1 (length macroCols) too. The macroCols value itself lives on the
  // defining AcquireMacroOp, not on MacroIdType, so we can't check the
  // exact length here without tracing def-use back to it.
  if (auto memrefPayload = dyn_cast<MemRefType>(getResult().getType().getValueType())) {
    if (memrefPayload.getRank() != 1)
      return emitOpError("result future's memref payload must be rank-1 (length macroCols)");
  }

  return success();
}
