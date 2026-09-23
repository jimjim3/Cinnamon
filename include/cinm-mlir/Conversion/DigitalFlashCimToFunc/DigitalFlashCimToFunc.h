#pragma once

#include "cinm-mlir/Dialect/DigitalFlashCim/IR/DigitalFlashCimDialect.h"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::digitalflashcim {
void populateDigitalFlashCimToFuncConversionPatterns(RewritePatternSet &patterns,
                                                      TypeConverter &converter,
                                                      MLIRContext *context);
std::unique_ptr<Pass> createConvertDigitalFlashCimToFuncPass();
} // namespace mlir::digitalflashcim
