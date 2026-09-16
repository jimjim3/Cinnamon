#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimBase.h"
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimDialect.h"

using namespace mlir;
using namespace mlir::digitalcim;

#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimBase.cpp.inc"

// Bring in enum helpers (stringify/symbolize) once.
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimEnums.cpp.inc"

// Bring in attribute class definitions once (empty: MultiplierKind is a
// bare enum attr, not an AttrDef -- see DigitalCimAttributes.td).
#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimAttributes.cpp.inc"

void DigitalCimDialect::initialize() {
  registerOps();
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimAttributes.cpp.inc"
      >();
}
