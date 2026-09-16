#include "executor_interface.hpp"

// C ABI matching what MLIR's --convert-func-to-llvm generates for external
// calls with memref arguments: a memref<N x T> flattens to (base_ptr,
// aligned_ptr, offset, size0..sizeN-1, stride0..strideN-1). Mirrors the
// sibling runtime/Memristor/runtime_c_interface.cpp's exact convention.

extern "C" {

#define DEF_DIGITALCIM_WRITE_WEIGHTS(name, type)                             \
  void digitalcim_write_weights_##name(                                     \
      int32_t macroId, type *base, type *data, int64_t offset,              \
      int64_t size0, int64_t size1, int64_t stride0, int64_t stride1) {     \
    digitalcim_runtime::memref_descriptor<2> w = {                          \
        base, data, offset, {size0, size1}, {stride0, stride1}};            \
    digitalcim_runtime::write_weights<type>(macroId, w);                    \
  }

#define DEF_DIGITALCIM_TRIGGER_MAC(name, type)                               \
  int32_t digitalcim_trigger_mac_##name(int32_t macroId, type *base,        \
                                        type *data, int64_t offset,          \
                                        int64_t size0, int64_t stride0) {    \
    digitalcim_runtime::memref_descriptor<1> in = {                         \
        base, data, offset, {size0}, {stride0}};                            \
    return digitalcim_runtime::trigger_mac<type>(macroId, in);              \
  }

#define DEF_DIGITALCIM_BARRIER(name, type)                                   \
  void digitalcim_barrier_##name(int32_t futureId, type *base, type *data,  \
                                 int64_t offset, int64_t size0,              \
                                 int64_t stride0) {                          \
    digitalcim_runtime::memref_descriptor<1> out = {                        \
        base, data, offset, {size0}, {stride0}};                            \
    digitalcim_runtime::barrier<type>(futureId, out);                       \
  }

int32_t digitalcim_acquire_macro(int64_t rows, int64_t cols, int64_t bitWidth,
                                 int32_t multiplierKind) {
  return digitalcim_runtime::acquire_macro(rows, cols, bitWidth,
                                           multiplierKind);
}

void digitalcim_release_macro(int32_t macroId) {
  digitalcim_runtime::release_macro(macroId);
}

void digitalcim_configure_compressor(int32_t macroId, int32_t multiplierKind) {
  digitalcim_runtime::configure_compressor(macroId, multiplierKind);
}

DEF_DIGITALCIM_WRITE_WEIGHTS(i8, int8_t)
DEF_DIGITALCIM_WRITE_WEIGHTS(i16, int16_t)
DEF_DIGITALCIM_WRITE_WEIGHTS(i32, int32_t)
DEF_DIGITALCIM_WRITE_WEIGHTS(i64, int64_t)

DEF_DIGITALCIM_TRIGGER_MAC(i8, int8_t)
DEF_DIGITALCIM_TRIGGER_MAC(i16, int16_t)
DEF_DIGITALCIM_TRIGGER_MAC(i32, int32_t)
DEF_DIGITALCIM_TRIGGER_MAC(i64, int64_t)

DEF_DIGITALCIM_BARRIER(i8, int8_t)
DEF_DIGITALCIM_BARRIER(i16, int16_t)
DEF_DIGITALCIM_BARRIER(i32, int32_t)
DEF_DIGITALCIM_BARRIER(i64, int64_t)

} // extern "C"
