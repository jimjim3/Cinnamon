#include "executor_interface.hpp"

// C ABI matching what MLIR's --convert-func-to-llvm generates for external
// calls with memref arguments -- same convention as the sibling
// runtime/DigitalCim/runtime_c_interface.cpp.

extern "C" {

#define DEF_DIGITALFLASHCIM_PROGRAM_ROWS(name, type)                         \
  void digitalflashcim_program_rows_##name(                                 \
      int32_t macroId, type *base, type *data, int64_t offset,              \
      int64_t size0, int64_t size1, int64_t stride0, int64_t stride1) {     \
    digitalflashcim_runtime::memref_descriptor<2> w = {                     \
        base, data, offset, {size0, size1}, {stride0, stride1}};            \
    digitalflashcim_runtime::program_rows<type>(macroId, w);                \
  }

#define DEF_DIGITALFLASHCIM_TRIGGER_MAC(name, type)                          \
  int32_t digitalflashcim_trigger_mac_##name(int32_t macroId, type *base,   \
                                             type *data, int64_t offset,     \
                                             int64_t size0, int64_t stride0) { \
    digitalflashcim_runtime::memref_descriptor<1> in = {                    \
        base, data, offset, {size0}, {stride0}};                            \
    return digitalflashcim_runtime::trigger_mac<type>(macroId, in);         \
  }

#define DEF_DIGITALFLASHCIM_BARRIER(name, type)                              \
  void digitalflashcim_barrier_##name(int32_t futureId, type *base,         \
                                      type *data, int64_t offset,            \
                                      int64_t size0, int64_t stride0) {      \
    digitalflashcim_runtime::memref_descriptor<1> out = {                   \
        base, data, offset, {size0}, {stride0}};                            \
    digitalflashcim_runtime::barrier<type>(futureId, out);                  \
  }

int32_t digitalflashcim_acquire_macro(int64_t rows, int64_t cols, int64_t bitWidth,
                                      int32_t multiplierKind) {
  return digitalflashcim_runtime::acquire_macro(rows, cols, bitWidth,
                                                multiplierKind);
}

void digitalflashcim_release_macro(int32_t macroId) {
  digitalflashcim_runtime::release_macro(macroId);
}

void digitalflashcim_configure_compressor(int32_t macroId, int32_t multiplierKind) {
  digitalflashcim_runtime::configure_compressor(macroId, multiplierKind);
}

void digitalflashcim_erase_block(int32_t macroId) {
  digitalflashcim_runtime::erase_block(macroId);
}

DEF_DIGITALFLASHCIM_PROGRAM_ROWS(i8, int8_t)
DEF_DIGITALFLASHCIM_PROGRAM_ROWS(i16, int16_t)
DEF_DIGITALFLASHCIM_PROGRAM_ROWS(i32, int32_t)
DEF_DIGITALFLASHCIM_PROGRAM_ROWS(i64, int64_t)

DEF_DIGITALFLASHCIM_TRIGGER_MAC(i8, int8_t)
DEF_DIGITALFLASHCIM_TRIGGER_MAC(i16, int16_t)
DEF_DIGITALFLASHCIM_TRIGGER_MAC(i32, int32_t)
DEF_DIGITALFLASHCIM_TRIGGER_MAC(i64, int64_t)

DEF_DIGITALFLASHCIM_BARRIER(i8, int8_t)
DEF_DIGITALFLASHCIM_BARRIER(i16, int16_t)
DEF_DIGITALFLASHCIM_BARRIER(i32, int32_t)
DEF_DIGITALFLASHCIM_BARRIER(i64, int64_t)

} // extern "C"
