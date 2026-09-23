//===- Types.td - DigitalFlashCim dialect types ------*- tablegen -*-===//
//
// This is the definitions file for the DigitalFlashCim dialect types.
// Identical shape to the sibling DigitalCim dialect's types -- a macro
// handle and a future -- since the compute/result datapath is unchanged
// under Option B (see DigitalFlashCimBase.td).
//
//===----------------------------------------------------------------------===//

#ifndef DIGITALFLASHCIM_TYPES
#define DIGITALFLASHCIM_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "DigitalFlashCimBase.td"

def DigitalFlashCim_MacroIdType : DigitalFlashCim_Type<"MacroId"> {
  let mnemonic = "macroId";
  let summary = "Handle to an acquired digital flash CIM macro.";
}

def DigitalFlashCim_FutureType : DigitalFlashCim_Type<"Future"> {
  let mnemonic = "future";
  let summary = "Result of a digitalflashcim.trigger_mac, not yet materialized.";
  let description = [{
    Wraps the eventual payload type (normally a 1-D memref of length
    macro_cols). Resolved into a value by `digitalflashcim.barrier`.
  }];

  let parameters = (ins "Type":$valueType);

  let hasCustomAssemblyFormat = 1;
}

#endif
