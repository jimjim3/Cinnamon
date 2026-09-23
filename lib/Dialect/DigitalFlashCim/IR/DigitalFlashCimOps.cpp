/// Implements the DigitalFlashCim dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimOps.h"
#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "digitalflashcim-ops"

using namespace mlir;
using namespace mlir::digitalflashcim;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimOps.cpp.inc"

//===----------------------------------------------------------------------===//
// DigitalFlashCimDialect
//===----------------------------------------------------------------------===//

void DigitalFlashCimDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimOps.cpp.inc"
      >();
}

void AcquireMacroOp::getAsmResultNames(
    ::mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "dfcim_macro");
}

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

// Same checks as the sibling DigitalCim dialect's WriteWeightsOp::verify --
// erase_block/program_rows split the "write the whole matrix" step in two,
// but program_rows carries the same weights operand shape constraint
// write_weights did.
::mlir::LogicalResult ProgramRowsOp::verify() {
  auto weightsTy = dyn_cast<MemRefType>(getWeights().getType());
  if (!weightsTy || weightsTy.getRank() != 2)
    return emitOpError("weights must be a rank-2 memref (macroRows x macroCols)");

  if (!isa<IntegerType>(weightsTy.getElementType()))
    return emitOpError("weights element type must be an integer type "
                       "(digital flash CIM stores quantized weights, not floats)");

  return success();
}

::mlir::LogicalResult TriggerMacOp::verify() {
  auto inputTy = dyn_cast<MemRefType>(getInput().getType());
  if (!inputTy || inputTy.getRank() != 1)
    return emitOpError("input must be a rank-1 memref (length macroRows)");

  if (!isa<IntegerType>(inputTy.getElementType()))
    return emitOpError("input element type must be an integer type "
                       "(digital flash CIM streams quantized activations, not floats)");

  // Best-effort: if the future's payload is itself a memref, it should be
  // rank-1 (length macroCols) too -- see the sibling DigitalCim dialect's
  // TriggerMacOp::verify for why the exact length can't be checked here.
  if (auto memrefPayload = dyn_cast<MemRefType>(getResult().getType().getValueType())) {
    if (memrefPayload.getRank() != 1)
      return emitOpError("result future's memref payload must be rank-1 (length macroCols)");
  }

  return success();
}
