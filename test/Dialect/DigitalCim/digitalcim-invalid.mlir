// RUN: cinm-opt %s --split-input-file -verify-diagnostics

// -----

func.func @write_weights_float(%macro: !digitalcim.macroId, %weights: memref<64x32xf32>) {
  // expected-error @below {{weights element type must be an integer type (digital CIM stores quantized weights, not floats)}}
  digitalcim.write_weights %macro, %weights : !digitalcim.macroId, memref<64x32xf32>
  return
}

// -----

func.func @write_weights_bad_rank(%macro: !digitalcim.macroId, %weights: memref<64xi8>) {
  // expected-error @below {{'digitalcim.write_weights' op operand #1 must be strided memref of any type values of rank 2}}
  digitalcim.write_weights %macro, %weights : !digitalcim.macroId, memref<64xi8>
  return
}

// -----

func.func @trigger_mac_float(%macro: !digitalcim.macroId, %input: memref<64xf32>) {
  // expected-error @below {{input element type must be an integer type (digital CIM streams quantized activations, not floats)}}
  %fut = digitalcim.trigger_mac %macro, %input : !digitalcim.macroId, memref<64xf32> -> !digitalcim.future<memref<32xf32>>
  return
}

// -----

func.func @trigger_mac_bad_rank(%macro: !digitalcim.macroId, %input: memref<64x1xi8>) {
  // expected-error @below {{'digitalcim.trigger_mac' op operand #1 must be strided memref of any type values of rank 1}}
  %fut = digitalcim.trigger_mac %macro, %input : !digitalcim.macroId, memref<64x1xi8> -> !digitalcim.future<memref<32xi8>>
  return
}
