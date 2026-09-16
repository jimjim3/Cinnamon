//===- Types.td - DigitalCim dialect types -----------*- tablegen -*-===//
//
// This is the definitions file for the DigitalCim dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef DIGITALCIM_TYPES
#define DIGITALCIM_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "DigitalCimBase.td"

def DigitalCim_MacroIdType : DigitalCim_Type<"MacroId"> {
  let mnemonic = "macroId";
  let summary = "Handle to an acquired digital CIM macro.";
}

def DigitalCim_FutureType : DigitalCim_Type<"Future"> {
  let mnemonic = "future";
  let summary = "Result of a digitalcim.trigger_mac, not yet materialized.";
  let description = [{
    Wraps the eventual payload type (normally a 1-D memref of length
    macro_cols). Resolved into a value by `digitalcim.barrier`.
  }];

  let parameters = (ins "Type":$valueType);

  let hasCustomAssemblyFormat = 1;
}

#endif
