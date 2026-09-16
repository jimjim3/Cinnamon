//===- Attributes.td - DigitalCim dialect attributes --*- tablegen -*-===//
//
// This is the definitions file for the DigitalCim dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef DIGITALCIM_ATTRIBUTES
#define DIGITALCIM_ATTRIBUTES

include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/CommonAttrConstraints.td"

// Mirrors src/hw.py's MultiplierKind Literal["exact", "approx_compressor",
// "logarithmic"], calibrated against OpenACM Table II
// (src/openacm_calibration.py). Follows the same pattern as the sibling
// `Cim` dialect's Cim_ActivationKindAttr (a bare I32EnumAttr used directly
// as an attribute constraint, not wrapped in EnumAttr<Dialect,...>).
def DigitalCim_MultiplierKindAttr : I32EnumAttr<
    "MultiplierKind", "digital CIM macro multiplier implementation", [
  I32EnumAttrCase<"Exact",            0, "exact">,
  I32EnumAttrCase<"ApproxCompressor", 1, "approx_compressor">,
  I32EnumAttrCase<"Logarithmic",      2, "logarithmic">
]> {
  let cppNamespace = "::mlir::digitalcim";
}

#endif
