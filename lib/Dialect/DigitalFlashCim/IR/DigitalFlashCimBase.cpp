#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimBase.h"
#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimDialect.h"

using namespace mlir;
using namespace mlir::digitalflashcim;

#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimBase.cpp.inc"

// Bring in enum helpers (stringify/symbolize) once.
#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimEnums.cpp.inc"

// Bring in attribute class definitions once (empty: MultiplierKind is a
// bare enum attr, not an AttrDef -- see DigitalFlashCimAttributes.td).
#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimAttributes.cpp.inc"

void DigitalFlashCimDialect::initialize() {
  registerOps();
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimAttributes.cpp.inc"
      >();
}
