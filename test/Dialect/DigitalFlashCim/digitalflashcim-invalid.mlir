// RUN: cinm-opt %s --split-input-file -verify-diagnostics

// -----

func.func @program_rows_float(%macro: !digitalflashcim.macroId, %weights: memref<64x32xf32>) {
  // expected-error @below {{weights element type must be an integer type (digital flash CIM stores quantized weights, not floats)}}
  digitalflashcim.program_rows %macro, %weights : !digitalflashcim.macroId, memref<64x32xf32>
  return
}

// -----

func.func @program_rows_bad_rank(%macro: !digitalflashcim.macroId, %weights: memref<64xi8>) {
  // expected-error @below {{'digitalflashcim.program_rows' op operand #1 must be strided memref of any type values of rank 2}}
  digitalflashcim.program_rows %macro, %weights : !digitalflashcim.macroId, memref<64xi8>
  return
}

// -----

func.func @trigger_mac_float(%macro: !digitalflashcim.macroId, %input: memref<64xf32>) {
  // expected-error @below {{input element type must be an integer type (digital flash CIM streams quantized activations, not floats)}}
  %fut = digitalflashcim.trigger_mac %macro, %input : !digitalflashcim.macroId, memref<64xf32> -> !digitalflashcim.future<memref<32xf32>>
  return
}

// -----

func.func @trigger_mac_bad_rank(%macro: !digitalflashcim.macroId, %input: memref<64x1xi8>) {
  // expected-error @below {{'digitalflashcim.trigger_mac' op operand #1 must be strided memref of any type values of rank 1}}
  %fut = digitalflashcim.trigger_mac %macro, %input : !digitalflashcim.macroId, memref<64x1xi8> -> !digitalflashcim.future<memref<32xi8>>
  return
}
