// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s

// CHECK-LABEL: simple
func.func @simple(%weights: memref<64x32xi8>, %input: memref<64xi8>, %out: memref<32xi8>) {
  %macro = digitalcim.acquire_macro logarithmic { macroRows = 64 : i64, macroCols = 32 : i64, bitWidth = 8 : i64 } -> !digitalcim.macroId
  digitalcim.configure_compressor %macro logarithmic : !digitalcim.macroId
  digitalcim.write_weights %macro, %weights : !digitalcim.macroId, memref<64x32xi8>
  %fut = digitalcim.trigger_mac %macro, %input : !digitalcim.macroId, memref<64xi8> -> !digitalcim.future<memref<32xi8>>
  %res = digitalcim.barrier %fut : !digitalcim.future<memref<32xi8>> -> memref<32xi8>
  memref.copy %res, %out : memref<32xi8> to memref<32xi8>
  digitalcim.release_macro %macro : !digitalcim.macroId
  return
}

// CHECK-LABEL: exact_and_approx
func.func @exact_and_approx(%macro: !digitalcim.macroId) {
  digitalcim.configure_compressor %macro exact : !digitalcim.macroId
  digitalcim.configure_compressor %macro approx_compressor : !digitalcim.macroId
  return
}
