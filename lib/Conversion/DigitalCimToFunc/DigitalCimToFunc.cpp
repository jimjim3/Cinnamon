/// Lowers digitalcim ops to calls into the DigitalCim runtime library
/// (runtime/DigitalCim/). See DigitalCimPasses.td for the overall design
/// rationale (macroId/future both erase to i32, out-param convention for
/// memref results mirroring the sibling MemristorToFunc pass).

#include "cinm-mlir/Conversion/DigitalCimToFunc/DigitalCimToFunc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <string>

#define DEBUG_TYPE "digitalcim-to-func"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::digitalcim;

namespace mlir::digitalcim {
#define GEN_PASS_DEF_CONVERTDIGITALCIMTOFUNC
#include "cinm-mlir/Conversion/DigitalCimPasses.h.inc"
} // namespace mlir::digitalcim

namespace {

// Only integer element types are meaningful here -- digital CIM stores
// quantized weights/activations, enforced already by WriteWeightsOp's and
// TriggerMacOp's verifiers.
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
    llvm_unreachable("unsupported DigitalCim element bit width (expected 8/16/32/64)");
  }
}

// Looks up (or inserts, at module scope) the external declaration for a
// runtime call, and emits the call itself at the rewriter's current
// insertion point.
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

// Casts a statically-shaped memref to an equal-rank, fully-dynamic-shape
// memref, so every call site of a given (rank, element type) shares one
// external symbol regardless of the macro's concrete geometry -- mirrors
// MemristorToFunc.cpp's identical trick.
Value castToDynamicShape(PatternRewriter &rewriter, Location loc, Value memref) {
  auto memrefTy = cast<MemRefType>(memref.getType());
  SmallVector<int64_t> dynShape(memrefTy.getRank(), ShapedType::kDynamic);
  auto dynTy = MemRefType::get(dynShape, memrefTy.getElementType());
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
                                "digitalcim_acquire_macro",
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
    emitRuntimeCall(rewriter, op.getLoc(), module, "digitalcim_release_macro",
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
    // compressorConfig (the optional per-column compressor-assignment dict)
    // has no typed C ABI representation yet -- dropped for this v1 runtime.
    // It only ever affected the offline energy/accuracy model
    // (src/openacm_calibration.py), not functional correctness, since this
    // runtime only implements exact-multiplier semantics regardless of
    // multiplierKind (see runtime/DigitalCim/executor.cpp's header comment).
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    SmallVector<Value, 2> args = {
        adaptor.getMacroId(),
        materializeI32(rewriter, loc,
                       static_cast<int32_t>(op.getMultiplierKind())),
    };
    emitRuntimeCall(rewriter, loc, module, "digitalcim_configure_compressor",
                    {}, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct WriteWeightsOpLowering : OpConversionPattern<WriteWeightsOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(WriteWeightsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto elemTy =
        cast<MemRefType>(op.getWeights().getType()).getElementType();
    std::string fnName =
        ("digitalcim_write_weights_" + mangleElementType(elemTy)).str();
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
        ("digitalcim_trigger_mac_" + mangleElementType(elemTy)).str();
    Value dynInput = castToDynamicShape(rewriter, loc, op.getInput());
    // The "future" is just the macro id: only one trigger_mac can be in
    // flight per macro at a time (the scheduler's S3 constraint already
    // guarantees this), so the runtime resolves synchronously and hands
    // the same id back as the future's i32-erased representation.
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
        ("digitalcim_barrier_" + mangleElementType(resultTy.getElementType()))
            .str();

    // Out-param convention (mirrors MemristorOps' result-as-operand
    // pattern): the runtime can't return a memref by C-ABI value, so the
    // caller allocates the statically-shaped output buffer and the runtime
    // call fills it in-place.
    Value alloc = memref::AllocOp::create(rewriter, loc, resultTy);
    Value dynOut = castToDynamicShape(rewriter, loc, alloc);
    emitRuntimeCall(rewriter, loc, module, fnName, {},
                    {adaptor.getValue(), dynOut});
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

} // namespace

void mlir::digitalcim::populateDigitalCimToFuncConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &converter,
    MLIRContext *ctx) {
  patterns.insert<AcquireMacroOpLowering, ReleaseMacroOpLowering,
                  ConfigureCompressorOpLowering, WriteWeightsOpLowering,
                  TriggerMacOpLowering, BarrierOpLowering>(converter, ctx);
}

namespace mlir::digitalcim {

class ConvertDigitalCimToFunc
    : public mlir::digitalcim::impl::ConvertDigitalCimToFuncBase<
          ConvertDigitalCimToFunc> {
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
    populateDigitalCimToFuncConversionPatterns(patterns, converter,
                                               &getContext());

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](auto *) { return true; });
    target.addIllegalDialect<DigitalCimDialect>();
    target.addLegalDialect<FuncDialect>();
    target.addLegalOp<FuncOp>();

    Operation *operation = getOperation();
    FrozenRewritePatternSet frozenPatterns{std::move(patterns)};
    if (failed(applyPartialConversion(operation, target, frozenPatterns)))
      signalPassFailure();
  }
};

} // namespace mlir::digitalcim

std::unique_ptr<Pass> digitalcim::createConvertDigitalCimToFuncPass() {
  return std::make_unique<ConvertDigitalCimToFunc>();
}
