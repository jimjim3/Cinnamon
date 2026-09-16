//===- Base.td - DigitalCim dialect base -----------------*- tablegen -*-===//
//
// This is the definitions file for the DigitalCim dialect base.
//
// DigitalCim models a digital SRAM in-memory-MAC CIM macro (the class of
// architecture demonstrated by the open-source OpenACM project:
// https://github.com/ShenShan123/OpenACM), as opposed to the analog
// memristor crossbars the sibling `Memristor` dialect targets. See
// ~/cim-compiler/docs/architecture.md section 4 for the ISA design
// rationale (row-serial accumulation, macro_cols as parallel output lanes).
//
//===----------------------------------------------------------------------===//

#ifndef DIGITALCIM_BASE
#define DIGITALCIM_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def DigitalCim_Dialect : Dialect {
    let name = "digitalcim";
    let cppNamespace = "::mlir::digitalcim";

    let summary = "Digital SRAM in-memory-MAC compute-in-memory macros.";
    let description = [{
        Models a digital compute-in-memory macro: an SRAM array holding a
        stationary weight matrix, coupled to one (or `macro_cols` parallel)
        digital multiplier(s) that stream an input vector in over
        `macro_rows` cycles, accumulating a dot product per column. Unlike
        the `Memristor` dialect's analog crossbars, there is no shared
        analog summing line -- each output lane is an independent digital
        circuit.
    }];

    let useDefaultTypePrinterParser = 1;

    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
class DigitalCim_Attr<string name, list<Trait> traits = []>
        : AttrDef<DigitalCim_Dialect, name, traits>;

// Template for ops.
class DigitalCim_Op<string mnemonic, list<Trait> traits = []>
        : Op<DigitalCim_Dialect, mnemonic, traits>;

// Template for types.
class DigitalCim_Type<string name, list<Trait> traits = []>
        : TypeDef<DigitalCim_Dialect, name, traits>;

#endif
