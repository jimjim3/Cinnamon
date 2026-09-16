/// Implements the DigitalCim dialect types.
///
/// @file

#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "digitalcim-types"

using namespace mlir;
using namespace mlir::digitalcim;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DigitalCimDialect
//===----------------------------------------------------------------------===//

void DigitalCimDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimTypes.cpp.inc"
      >();
}

Type mlir::digitalcim::FutureType::parse(mlir::AsmParser &parser) {
  Type valueType;

  if (parser.parseLess() || //
      parser.parseType(valueType) || //
      parser.parseGreater()) {
    return Type();
  }

  return digitalcim::FutureType::get(parser.getContext(), valueType);
}

void mlir::digitalcim::FutureType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getValueType() << ">";
}
