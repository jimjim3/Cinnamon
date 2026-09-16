//===- DigitalCimPasses.td - DigitalCim conversion passes -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIGITALCIM_CONVERSION_PASSES
#define DIGITALCIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertDigitalCimToFunc : Pass<"convert-digitalcim-to-func", operation="func::FuncOp"> {
    let summary = "Convert digitalcim dialect to func dialect calls into the DigitalCim runtime library";
    let description = [{
      Lowers digitalcim ops to calls into libDigitalCimDialectRuntime (see
      runtime/DigitalCim/), which computes MAC results in software for
      correctness testing -- this is a pure-software simulator, not a
      hardware backend (see ~/cim-compiler/docs/architecture.md's
      "純軟體路線" scope decision). !digitalcim.macroId and
      !digitalcim.future<T> both erase to i32: a macro can only have one
      trigger_mac in flight at a time (the scheduler's S3 macro-busy
      constraint already guarantees this), so a "future" handle is just the
      macro id itself, and the runtime resolves it synchronously.
    }];
    let constructor = "mlir::digitalcim::createConvertDigitalCimToFuncPass()";
    let dependentDialects = ["digitalcim::DigitalCimDialect", "func::FuncDialect",
        "memref::MemRefDialect", "arith::ArithDialect"];
}

#endif // DIGITALCIM_CONVERSION_PASSES
