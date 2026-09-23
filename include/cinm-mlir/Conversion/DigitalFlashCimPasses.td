//===- DigitalFlashCimPasses.td - DigitalFlashCim conversion passes -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIGITALFLASHCIM_CONVERSION_PASSES
#define DIGITALFLASHCIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertDigitalFlashCimToFunc : Pass<"convert-digitalflashcim-to-func", operation="func::FuncOp"> {
    let summary = "Convert digitalflashcim dialect to func dialect calls into the DigitalFlashCim runtime library";
    let description = [{
      Lowers digitalflashcim ops to calls into libDigitalFlashCimDialectRuntime
      (see runtime/DigitalFlashCim/), which computes MAC results in software
      for correctness testing -- a pure-software simulator, not a hardware
      backend, same scope as the sibling DigitalCim dialect's conversion
      pass. !digitalflashcim.macroId and !digitalflashcim.future<T> both
      erase to i32, for the same reason as DigitalCim's (only one
      trigger_mac in flight per macro at a time, so a "future" is just the
      macro id).
    }];
    let constructor = "mlir::digitalflashcim::createConvertDigitalFlashCimToFuncPass()";
    let dependentDialects = ["digitalflashcim::DigitalFlashCimDialect", "func::FuncDialect",
        "memref::MemRefDialect", "arith::ArithDialect"];
}

#endif // DIGITALFLASHCIM_CONVERSION_PASSES
