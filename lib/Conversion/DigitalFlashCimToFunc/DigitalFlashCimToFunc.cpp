/// Lowers digitalflashcim ops to calls into the DigitalFlashCim runtime
/// library (runtime/DigitalFlashCim/). See DigitalFlashCimPasses.td for the
/// overall design rationale; this file mirrors the sibling
/// DigitalCimToFunc.cpp almost exactly (same macroId/future erasure, same
/// out-param memref convention, same dynamic-shape casting helper and its
/// documented bug-fix history) -- the only op-level difference is
/// WriteWeightsOpLowering being replaced by EraseBlockOpLowering +
/// ProgramRowsOpLowering.

#include "cinm-mlir/Conversion/DigitalFlashCimToFunc/DigitalFlashCimToFunc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <string>

#define DEBUG_TYPE "digitalflashcim-to-func"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::digitalflashcim;

namespace mlir::digitalflashcim {
#define GEN_PASS_DEF_CONVERTDIGITALFLASHCIMTOFUNC
#include "cinm-mlir/Conversion/DigitalFlashCimPasses.h.inc"
} // namespace mlir::digitalflashcim

namespace {

// Only integer element types are meaningful here -- digital flash CIM
// stores quantized weights/activations, enforced already by
// ProgramRowsOp's and TriggerMacOp's verifiers.
StringRef mangleElementType(Type elementType) {
  auto intTy = cast<IntegerType>(elementType);
  switch (intTy.getWidth()) {
  case 8:
    return "i8";
  case 16:
    return "i16";
  case 32:
    return "i32";
  case 64:
    return "i64";
  default:
    llvm_unreachable("unsupported DigitalFlashCim element bit width (expected 8/16/32/64)");
  }
}

CallOp emitRuntimeCall(PatternRewriter &rewriter, Location loc,
                       ModuleOp module, StringRef fnName,
                       TypeRange resultTypes, ValueRange operands) {
  if (!module.lookupSymbol(fnName)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    auto fnType = FunctionType::get(rewriter.getContext(),
                                    operands.getTypes(), resultTypes);
    auto funcOp =
        FuncOp::create(rewriter, loc, fnName, fnType, ArrayRef<NamedAttribute>{});
    funcOp.setVisibility(FuncOp::Visibility::Nested);
  }
  return CallOp::create(rewriter, loc, fnName, resultTypes, operands);
}

// Same fix as DigitalCimToFunc.cpp's castToDynamicShape (2026-09-16 there):
// cast to a fully-dynamic STRIDED layout, not the default identity layout,
// so a memref.subview with a nonzero offset remains a legal cast target.
Value castToDynamicShape(PatternRewriter &rewriter, Location loc, Value memref) {
  auto memrefTy = cast<MemRefType>(memref.getType());
  SmallVector<int64_t> dynShape(memrefTy.getRank(), ShapedType::kDynamic);
  SmallVector<int64_t> dynStrides(memrefTy.getRank(), ShapedType::kDynamic);
  auto layout = StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic, dynStrides);
  auto dynTy = MemRefType::get(dynShape, memrefTy.getElementType(), layout);
  return memref::CastOp::create(rewriter, loc, dynTy, memref);
}

Value materializeI32(PatternRewriter &rewriter, Location loc, int32_t v) {
  return arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                   rewriter.getI32IntegerAttr(v));
}
Value materializeI64(PatternRewriter &rewriter, Location loc, int64_t v) {
  return arith::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                   rewriter.getI64IntegerAttr(v));
}

struct AcquireMacroOpLowering : OpConversionPattern<AcquireMacroOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AcquireMacroOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    SmallVector<Value, 4> args = {
        materializeI64(rewriter, loc, op.getMacroRows()),
        materializeI64(rewriter, loc, op.getMacroCols()),
        materializeI64(rewriter, loc, op.getBitWidth()),
        materializeI32(rewriter, loc,
                       static_cast<int32_t>(op.getMultiplierKind())),
    };
    auto call = emitRuntimeCall(rewriter, loc, module,
                                "digitalflashcim_acquire_macro",
                                {rewriter.getI32Type()}, args);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct ReleaseMacroOpLowering : OpConversionPattern<ReleaseMacroOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReleaseMacroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    emitRuntimeCall(rewriter, op.getLoc(), module, "digitalflashcim_release_macro",
                    {}, {adaptor.getMacroId()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConfigureCompressorOpLowering
    : OpConversionPattern<ConfigureCompressorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConfigureCompressorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    SmallVector<Value, 2> args = {
        adaptor.getMacroId(),
        materializeI32(rewriter, loc,
                       static_cast<int32_t>(op.getMultiplierKind())),
    };
    emitRuntimeCall(rewriter, loc, module, "digitalflashcim_configure_compressor",
                    {}, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseBlockOpLowering : OpConversionPattern<EraseBlockOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(EraseBlockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    emitRuntimeCall(rewriter, op.getLoc(), module, "digitalflashcim_erase_block",
                    {}, {adaptor.getMacroId()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ProgramRowsOpLowering : OpConversionPattern<ProgramRowsOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ProgramRowsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto elemTy =
        cast<MemRefType>(op.getWeights().getType()).getElementType();
    std::string fnName =
        ("digitalflashcim_program_rows_" + mangleElementType(elemTy)).str();
    Value dynWeights = castToDynamicShape(rewriter, loc, op.getWeights());
    emitRuntimeCall(rewriter, loc, module, fnName, {},
                    {adaptor.getMacroId(), dynWeights});
    rewriter.eraseOp(op);
    return success();
  }
};

struct TriggerMacOpLowering : OpConversionPattern<TriggerMacOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TriggerMacOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto elemTy = cast<MemRefType>(op.getInput().getType()).getElementType();
    std::string fnName =
        ("digitalflashcim_trigger_mac_" + mangleElementType(elemTy)).str();
    Value dynInput = castToDynamicShape(rewriter, loc, op.getInput());
    auto call = emitRuntimeCall(rewriter, loc, module, fnName,
                                {rewriter.getI32Type()},
                                {adaptor.getMacroId(), dynInput});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct BarrierOpLowering : OpConversionPattern<BarrierOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto resultTy = cast<MemRefType>(op.getResult().getType());
    std::string fnName =
        ("digitalflashcim_barrier_" + mangleElementType(resultTy.getElementType()))
            .str();

    Value alloc = memref::AllocOp::create(rewriter, loc, resultTy);
    Value dynOut = castToDynamicShape(rewriter, loc, alloc);
    emitRuntimeCall(rewriter, loc, module, fnName, {},
                    {adaptor.getValue(), dynOut});
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

} // namespace

void mlir::digitalflashcim::populateDigitalFlashCimToFuncConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &converter,
    MLIRContext *ctx) {
  patterns.insert<AcquireMacroOpLowering, ReleaseMacroOpLowering,
                  ConfigureCompressorOpLowering, EraseBlockOpLowering,
                  ProgramRowsOpLowering, TriggerMacOpLowering,
                  BarrierOpLowering>(converter, ctx);
}

namespace mlir::digitalflashcim {

class ConvertDigitalFlashCimToFunc
    : public mlir::digitalflashcim::impl::ConvertDigitalFlashCimToFuncBase<
          ConvertDigitalFlashCimToFunc> {
public:
  void runOnOperation() override {
    TypeConverter converter;
    converter.addConversion([](Type t) { return t; });
    converter.addConversion([](MacroIdType t) -> Type {
      return IntegerType::get(t.getContext(), 32);
    });
    converter.addConversion([](FutureType t) -> Type {
      return IntegerType::get(t.getContext(), 32);
    });

    RewritePatternSet patterns{&getContext()};
    populateDigitalFlashCimToFuncConversionPatterns(patterns, converter,
                                                     &getContext());

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](auto *) { return true; });
    target.addIllegalDialect<DigitalFlashCimDialect>();
    target.addLegalDialect<FuncDialect>();
    target.addLegalOp<FuncOp>();

    Operation *operation = getOperation();
    FrozenRewritePatternSet frozenPatterns{std::move(patterns)};
    if (failed(applyPartialConversion(operation, target, frozenPatterns)))
      signalPassFailure();
  }
};

} // namespace mlir::digitalflashcim

std::unique_ptr<Pass> digitalflashcim::createConvertDigitalFlashCimToFuncPass() {
  return std::make_unique<ConvertDigitalFlashCimToFunc>();
}
