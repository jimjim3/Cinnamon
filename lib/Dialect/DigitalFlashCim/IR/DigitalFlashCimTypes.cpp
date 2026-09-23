/// Implements the DigitalFlashCim dialect types.
///
/// @file

#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "digitalflashcim-types"

using namespace mlir;
using namespace mlir::digitalflashcim;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DigitalFlashCimDialect
//===----------------------------------------------------------------------===//

void DigitalFlashCimDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimTypes.cpp.inc"
      >();
}

Type mlir::digitalflashcim::FutureType::parse(mlir::AsmParser &parser) {
  Type valueType;

  if (parser.parseLess() || //
      parser.parseType(valueType) || //
      parser.parseGreater()) {
    return Type();
  }

  return digitalflashcim::FutureType::get(parser.getContext(), valueType);
}

void mlir::digitalflashcim::FutureType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getValueType() << ">";
}
