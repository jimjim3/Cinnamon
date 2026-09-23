//===- Base.td - DigitalFlashCim dialect base ------------*- tablegen -*-===//
//
// This is the definitions file for the DigitalFlashCim dialect base.
//
// DigitalFlashCim models a digital NOR-flash in-memory-MAC CIM macro:
// bit-exact digital multiply/accumulate (same datapath as the sibling
// `DigitalCim` dialect, which targets SRAM), but with flash cells as the
// weight-storage medium instead of SRAM. See
// ~/CIM_Compiler_Digital_Flash/docs/memos/2026-09-23-sram-to-flash-scope.md
// (Option B: real-time multiplier, flash as pure non-volatile bitcell) and
// ~/CIM_Compiler_Digital_Flash/docs/memos/2026-09-23-lut-vs-realtime-cost-
// assessment.md for why this ISA stays close to DigitalCim's rather than
// adopting a LUT-multiplier design (kept in a sibling repo,
// ~/CIM_Compiler_Digital_Flash_LUT, for future study).
//
// THE ONE REAL PHYSICAL DIFFERENCE THIS DIALECT MODELS: NOR flash requires
// erase-before-program. A flash cell can only be PROGRAMMED in one
// direction (each bit can be cleared from 1 to 0, never set back to 1
// without a fresh erase), and erase resets a block to all-1s. DigitalCim's
// single `write_weights` op (arbitrary overwrite, matching SRAM) is
// therefore split into `erase_block` + `program_rows` here -- see
// DigitalCimOps.td's sibling for the op it replaces, and this dialect's
// Ops.td for how the runtime enforces the one-directional program
// semantics functionally (bitwise AND against the current cell contents),
// not just as a cost-model label.
//
//===----------------------------------------------------------------------===//

#ifndef DIGITALFLASHCIM_BASE
#define DIGITALFLASHCIM_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def DigitalFlashCim_Dialect : Dialect {
    let name = "digitalflashcim";
    let cppNamespace = "::mlir::digitalflashcim";

    let summary = "Digital NOR-flash in-memory-MAC compute-in-memory macros.";
    let description = [{
        Models a digital compute-in-memory macro built on NOR flash cells: a
        flash array holding a stationary weight matrix, coupled to one (or
        `macro_cols` parallel) digital multiplier(s) that stream an input
        vector in over `macro_rows` cycles, accumulating a dot product per
        column -- identical compute datapath to the sibling `DigitalCim`
        (SRAM) dialect. The difference is entirely in the storage path:
        flash requires an explicit erase before a macro's weights can be
        (re)programmed, modeled here as two ops instead of DigitalCim's one
        `write_weights`.
    }];

    let useDefaultTypePrinterParser = 1;

    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
class DigitalFlashCim_Attr<string name, list<Trait> traits = []>
        : AttrDef<DigitalFlashCim_Dialect, name, traits>;

// Template for ops.
class DigitalFlashCim_Op<string mnemonic, list<Trait> traits = []>
        : Op<DigitalFlashCim_Dialect, mnemonic, traits>;

// Template for types.
class DigitalFlashCim_Type<string name, list<Trait> traits = []>
        : TypeDef<DigitalFlashCim_Dialect, name, traits>;

#endif
