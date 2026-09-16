#pragma once

#include "cinm-mlir/Dialect/DigitalCim/IR/DigitalCimDialect.h"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::digitalcim {
void populateDigitalCimToFuncConversionPatterns(RewritePatternSet &patterns,
                                                TypeConverter &converter,
                                                MLIRContext *context);
std::unique_ptr<Pass> createConvertDigitalCimToFuncPass();
} // namespace mlir::digitalcim
