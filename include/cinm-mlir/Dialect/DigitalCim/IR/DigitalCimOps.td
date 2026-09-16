//===- Ops.td - DigitalCim dialect ops -------------------*- tablegen -*-===//
//
// Finalized ISA (~/cim-compiler/docs/architecture.md section 4):
//   write_weights, trigger_mac, barrier, configure_compressor.
// trigger_mac has a fixed known cost of macro_rows * cycles_per_mac cycles
// (row-serial accumulation -- see the "重要修正" section of that doc for
// why this is a row-serial, not spatially-parallel, design), which is what
// lets the scheduler's S3 macro-busy constraint stay simple.
//
//===----------------------------------------------------------------------===//

#ifndef DIGITALCIM_OPS
#define DIGITALCIM_OPS

include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimBase.td"
include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimTypes.td"
include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimAttributes.td"

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Administrative ops
//===----------------------------------------------------------------------===//

def DigitalCim_AcquireMacroOp : DigitalCim_Op<"acquire_macro",
    [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Acquire a digital CIM macro of the given geometry, returns its handle.";
  let description = [{
    macroRows is the SRAM reduction depth (row-serial accumulation cycles
    per trigger_mac); macroCols is the number of parallel output lanes
    (our extension beyond OpenACM's single-output PE demo -- see
    docs/architecture.md section 4).
  }];
  let arguments = (ins
    I64Attr:$macroRows,
    I64Attr:$macroCols,
    I64Attr:$bitWidth,
    DefaultValuedAttr<DigitalCim_MultiplierKindAttr,
                      "::mlir::digitalcim::MultiplierKind::Exact">:$multiplierKind
  );
  let results = (outs DigitalCim_MacroIdType:$macroId);
  // multiplierKind is named explicitly (not left to attr-dict) because a
  // bare I32EnumAttr like DigitalCim_MultiplierKindAttr has no dialect
  // attribute-parsing hook for the generic `#digitalcim<...>` form --
  // naming it here prints/parses it as its plain enum keyword instead,
  // same as the sibling Cim dialect's Cim_ActivateOp does with $kind.
  let assemblyFormat = "$multiplierKind attr-dict `->` qualified(type($macroId))";
}

def DigitalCim_ReleaseMacroOp : DigitalCim_Op<"release_macro", []> {
  let summary = "Release a digital CIM macro.";
  let arguments = (ins DigitalCim_MacroIdType:$macroId);
  let results = (outs);
  let assemblyFormat = "$macroId attr-dict `:` qualified(type($macroId))";
}

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

def DigitalCim_ConfigureCompressorOp : DigitalCim_Op<"configure_compressor", []> {
  let summary = "Select the multiplier implementation (and optional approximate-compressor config) for a macro.";
  let description = [{
    Sets the accuracy-energy tradeoff knob for subsequent trigger_mac calls
    on this macro. multiplierKind selects among OpenACM's three families
    (exact / approx_compressor / logarithmic, see
    src/openacm_calibration.py for their calibrated energy/accuracy
    numbers); compressorConfig optionally carries the specific compressor
    assignment (mirrors OpenACM's per-column compressor-type vector, see
    OpenACMv2 section 4.1.1) as a dialect-agnostic dictionary, since we
    have not yet decided a typed representation for it.
  }];
  let arguments = (ins
    DigitalCim_MacroIdType:$macroId,
    DigitalCim_MultiplierKindAttr:$multiplierKind,
    OptionalAttr<DictionaryAttr>:$compressorConfig
  );
  let results = (outs);
  let assemblyFormat = "$macroId $multiplierKind attr-dict `:` qualified(type($macroId))";
}

//===----------------------------------------------------------------------===//
// Compute ops
//===----------------------------------------------------------------------===//

def DigitalCim_WriteWeightsOp : DigitalCim_Op<"write_weights", []> {
  let summary = "Write a macroRows x macroCols weight matrix into the SRAM macro.";
  let description = [{
    Treated as one indivisible task by the scheduler, matching the analog
    pipeline's "one weight slice = one task" granularity (see
    docs/architecture.md section 4) -- callers do not manage per-row writes.
  }];
  let arguments = (ins
    DigitalCim_MacroIdType:$macroId,
    AnyStridedMemRefOfRank<2>:$weights
  );
  let results = (outs);
  let assemblyFormat =
    "$macroId `,` $weights attr-dict `:` "
    "qualified(type($macroId)) `,` type($weights)";
  let hasVerifier = 1;
}

def DigitalCim_TriggerMacOp : DigitalCim_Op<"trigger_mac", []> {
  let summary = "Stream a macroRows-length input vector into the macro, producing macroCols dot-product outputs.";
  let description = [{
    Costs macroRows * cyclesPerMac cycles (row-serial accumulation over the
    resident weights, see docs/architecture.md section 4's "重要修正").
    This is the only compute op the scheduler needs a duration for, and it
    maps directly onto the S3 macro-busy contention constraint: the macro
    is occupied for the whole call, no other write_weights/trigger_mac may
    target it concurrently.
  }];
  let arguments = (ins
    DigitalCim_MacroIdType:$macroId,
    AnyStridedMemRefOfRank<1>:$input
  );
  let results = (outs DigitalCim_FutureType:$result);
  let assemblyFormat =
    "$macroId `,` $input attr-dict `:` "
    "qualified(type($macroId)) `,` type($input) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def DigitalCim_BarrierOp : DigitalCim_Op<"barrier", []> {
  let summary = "Wait for a trigger_mac to finish; returns its memref payload.";
  let arguments = (ins DigitalCim_FutureType:$value);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat =
    "$value attr-dict `:` qualified(type($value)) `->` type($result)";
}

#endif
