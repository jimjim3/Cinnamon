//===- Ops.td - DigitalFlashCim dialect ops --------------*- tablegen -*-===//
//
// ISA: acquire_macro, release_macro, configure_compressor, erase_block,
// program_rows, trigger_mac, barrier.
//
// This is the sibling DigitalCim (SRAM) dialect's ISA with ONE change:
// `write_weights` (arbitrary overwrite) is replaced by `erase_block` +
// `program_rows`, because NOR flash requires erase-before-program and can
// only program a cell in one direction (bits go 1 -> 0; getting a 0 back
// to 1 needs a fresh erase). Every other op -- including trigger_mac's
// cost model (macro_rows * cycles_per_mac cycles, row-serial accumulation)
// -- is unchanged, because the multiply/accumulate datapath does not
// reference the storage medium at all (see
// ~/CIM_Compiler_Digital_Flash/src/hw.py's module docstring, "Option B").
//
// SCOPE NOTE (erase granularity): a real flash erase can span multiple
// macros that share one physical block
// (~/CIM_Compiler_Digital_Flash/src/placement.py's `erase_groups`, used
// for the Python-side COST model). This dialect does not represent that
// cross-macro sharing structurally -- `erase_block` is scoped to one
// macro, matching hw.py's v1 default (`erase_block_rows == macro_rows`,
// i.e. "one block = one macro"). Emitting fewer erases when several macros
// share a real block is a code-generation-time OPTIMIZATION (deciding
// which erase_block calls to skip), not a functional-correctness concern
// this IR needs to solve, since re-erasing an already-erased macro is
// simply redundant, not wrong. No codegen.py exists yet in this project to
// make that call either way (see PLAN.md Phase 6) -- left for whenever
// that lowering is built.
//
//===----------------------------------------------------------------------===//

#ifndef DIGITALFLASHCIM_OPS
#define DIGITALFLASHCIM_OPS

include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimBase.td"
include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimTypes.td"
include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimAttributes.td"

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Administrative ops
//===----------------------------------------------------------------------===//

def DigitalFlashCim_AcquireMacroOp : DigitalFlashCim_Op<"acquire_macro",
    [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Acquire a digital flash CIM macro of the given geometry, returns its handle.";
  let description = [{
    Same geometry parameters as the sibling DigitalCim dialect's
    acquire_macro (macroRows = reduction depth / row-serial accumulation
    cycles per trigger_mac; macroCols = parallel output lanes). A freshly
    acquired macro's flash cells are in an UNSPECIFIED prior state (not
    necessarily erased) -- the runtime models this by NOT pre-erasing on
    acquire, so a caller that calls program_rows before erase_block will
    observe the erase-before-program constraint's consequences (see
    erase_block's description) rather than getting a free pass on a fresh
    macro.
  }];
  let arguments = (ins
    I64Attr:$macroRows,
    I64Attr:$macroCols,
    I64Attr:$bitWidth,
    DefaultValuedAttr<DigitalFlashCim_MultiplierKindAttr,
                      "::mlir::digitalflashcim::MultiplierKind::Exact">:$multiplierKind
  );
  let results = (outs DigitalFlashCim_MacroIdType:$macroId);
  let assemblyFormat = "$multiplierKind attr-dict `->` qualified(type($macroId))";
}

def DigitalFlashCim_ReleaseMacroOp : DigitalFlashCim_Op<"release_macro", []> {
  let summary = "Release a digital flash CIM macro.";
  let arguments = (ins DigitalFlashCim_MacroIdType:$macroId);
  let results = (outs);
  let assemblyFormat = "$macroId attr-dict `:` qualified(type($macroId))";
}

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

def DigitalFlashCim_ConfigureCompressorOp : DigitalFlashCim_Op<"configure_compressor", []> {
  let summary = "Select the multiplier implementation for a macro. Unchanged from DigitalCim -- see that dialect's op of the same name.";
  let arguments = (ins
    DigitalFlashCim_MacroIdType:$macroId,
    DigitalFlashCim_MultiplierKindAttr:$multiplierKind,
    OptionalAttr<DictionaryAttr>:$compressorConfig
  );
  let results = (outs);
  let assemblyFormat = "$macroId $multiplierKind attr-dict `:` qualified(type($macroId))";
}

//===----------------------------------------------------------------------===//
// Storage ops (erase-before-program -- the one real difference from DigitalCim)
//===----------------------------------------------------------------------===//

def DigitalFlashCim_EraseBlockOp : DigitalFlashCim_Op<"erase_block", []> {
  let summary = "Erase a macro's flash storage, resetting every cell to the erased state (all bits 1).";
  let description = [{
    Must run before program_rows can (re)program rows to arbitrary values;
    see program_rows's description for what happens if it is skipped. This
    op takes no data operand -- unlike SRAM's write_weights, an erase does
    not write any particular value, it clears the block to its physical
    erased state.

    A macro that has never been programmed since acquire is in an
    UNSPECIFIED state, not necessarily already erased (see acquire_macro),
    so this op is still required even for a macro's very first
    program_rows call.
  }];
  let arguments = (ins DigitalFlashCim_MacroIdType:$macroId);
  let results = (outs);
  let assemblyFormat = "$macroId attr-dict `:` qualified(type($macroId))";
}

def DigitalFlashCim_ProgramRowsOp : DigitalFlashCim_Op<"program_rows", []> {
  let summary = "Program a macroRows x macroCols weight matrix into a macro's flash cells.";
  let description = [{
    Physically one-directional: programming a flash cell can only clear
    bits from 1 to 0, never set a 0 back to 1. The runtime therefore
    computes each cell's new contents as (current contents) AND (target
    value), not a plain overwrite -- so calling this on a macro that was
    properly erase_block'd first reproduces the target value exactly
    (every cell starts all-1s, AND with anything gives that thing back),
    while calling it WITHOUT a prior erase silently produces a
    bitwise-corrupted result wherever the target has a 1-bit the current
    cell contents do not (exactly matching what real NOR flash would do --
    this is not merely a labeled-error convenience check, see the runtime
    executor for the actual AND semantics). Same granularity convention as
    DigitalCim's write_weights: one call programs the whole matrix as one
    indivisible task (the scheduler's per-row cost accounting is a Python-
    side concern, see src/hw.py's program_cycles_per_row).
  }];
  let arguments = (ins
    DigitalFlashCim_MacroIdType:$macroId,
    AnyStridedMemRefOfRank<2>:$weights
  );
  let results = (outs);
  let assemblyFormat =
    "$macroId `,` $weights attr-dict `:` "
    "qualified(type($macroId)) `,` type($weights)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Compute ops -- unchanged from DigitalCim, see that dialect's ops of the
// same name for the full rationale (row-serial accumulation, S3 scheduler
// mapping). The multiply/accumulate datapath does not reference the
// storage medium.
//===----------------------------------------------------------------------===//

def DigitalFlashCim_TriggerMacOp : DigitalFlashCim_Op<"trigger_mac", []> {
  let summary = "Stream a macroRows-length input vector into the macro, producing macroCols dot-product outputs.";
  let arguments = (ins
    DigitalFlashCim_MacroIdType:$macroId,
    AnyStridedMemRefOfRank<1>:$input
  );
  let results = (outs DigitalFlashCim_FutureType:$result);
  let assemblyFormat =
    "$macroId `,` $input attr-dict `:` "
    "qualified(type($macroId)) `,` type($input) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def DigitalFlashCim_BarrierOp : DigitalFlashCim_Op<"barrier", []> {
  let summary = "Wait for a trigger_mac to finish; returns its memref payload.";
  let arguments = (ins DigitalFlashCim_FutureType:$value);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat =
    "$value attr-dict `:` qualified(type($value)) `->` type($result)";
}

#endif
