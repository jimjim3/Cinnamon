//===- Attributes.td - DigitalFlashCim dialect attributes -*- tablegen -*-===//
//
// This is the definitions file for the DigitalFlashCim dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef DIGITALFLASHCIM_ATTRIBUTES
#define DIGITALFLASHCIM_ATTRIBUTES

include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/CommonAttrConstraints.td"

// Mirrors ~/CIM_Compiler_Digital_Flash/src/hw.py's MultiplierKind and the
// sibling DigitalCim_MultiplierKindAttr -- the multiplier circuit is
// unchanged under Option B (see DigitalFlashCimBase.td), so the same three
// kinds apply, calibrated against the same OpenACM numbers
// (src/nor_flash_calibration.py's OPENACM_EXACT_MULTIPLIER -- only "exact"
// is populated there so far).
def DigitalFlashCim_MultiplierKindAttr : I32EnumAttr<
    "MultiplierKind", "digital flash CIM macro multiplier implementation", [
  I32EnumAttrCase<"Exact",            0, "exact">,
  I32EnumAttrCase<"ApproxCompressor", 1, "approx_compressor">,
  I32EnumAttrCase<"Logarithmic",      2, "logarithmic">
]> {
  let cppNamespace = "::mlir::digitalflashcim";
}

#endif
