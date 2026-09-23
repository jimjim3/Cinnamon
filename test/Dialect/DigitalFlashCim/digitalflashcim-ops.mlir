// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s

// CHECK-LABEL: simple
func.func @simple(%weights: memref<64x32xi8>, %input: memref<64xi8>, %out: memref<32xi8>) {
  %macro = digitalflashcim.acquire_macro logarithmic { macroRows = 64 : i64, macroCols = 32 : i64, bitWidth = 8 : i64 } -> !digitalflashcim.macroId
  digitalflashcim.configure_compressor %macro logarithmic : !digitalflashcim.macroId
  // erase-before-program: unlike DigitalCim's single write_weights, this
  // dialect requires an explicit erase_block before program_rows.
  digitalflashcim.erase_block %macro : !digitalflashcim.macroId
  digitalflashcim.program_rows %macro, %weights : !digitalflashcim.macroId, memref<64x32xi8>
  %fut = digitalflashcim.trigger_mac %macro, %input : !digitalflashcim.macroId, memref<64xi8> -> !digitalflashcim.future<memref<32xi8>>
  %res = digitalflashcim.barrier %fut : !digitalflashcim.future<memref<32xi8>> -> memref<32xi8>
  memref.copy %res, %out : memref<32xi8> to memref<32xi8>
  digitalflashcim.release_macro %macro : !digitalflashcim.macroId
  return
}

// CHECK-LABEL: exact_and_approx
func.func @exact_and_approx(%macro: !digitalflashcim.macroId) {
  digitalflashcim.configure_compressor %macro exact : !digitalflashcim.macroId
  digitalflashcim.configure_compressor %macro approx_compressor : !digitalflashcim.macroId
  return
}

// CHECK-LABEL: reprogram_needs_a_fresh_erase
// A macro reused for a second layer's weights: erase_block runs again
// before the second program_rows, since the first program_rows may have
// left cells that are not all-1s (only a fresh erase guarantees that).
func.func @reprogram_needs_a_fresh_erase(%w1: memref<64x32xi8>, %w2: memref<64x32xi8>) {
  %macro = digitalflashcim.acquire_macro exact { macroRows = 64 : i64, macroCols = 32 : i64, bitWidth = 8 : i64 } -> !digitalflashcim.macroId
  digitalflashcim.erase_block %macro : !digitalflashcim.macroId
  digitalflashcim.program_rows %macro, %w1 : !digitalflashcim.macroId, memref<64x32xi8>
  digitalflashcim.erase_block %macro : !digitalflashcim.macroId
  digitalflashcim.program_rows %macro, %w2 : !digitalflashcim.macroId, memref<64x32xi8>
  digitalflashcim.release_macro %macro : !digitalflashcim.macroId
  return
}
