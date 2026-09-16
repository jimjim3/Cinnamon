#include "executor_interface.hpp"

#include <iostream>

namespace digitalcim_runtime {

std::array<MacroSlot, kMaxMacros> macros{};

int32_t acquire_macro(int64_t rows, int64_t cols, int64_t bitWidth,
                      int32_t multiplierKind) {
  for (size_t i = 0; i < kMaxMacros; ++i) {
    if (!macros[i].inUse) {
      macros[i].inUse = true;
      macros[i].rows = rows;
      macros[i].cols = cols;
      macros[i].bitWidth = bitWidth;
      macros[i].multiplierKind = multiplierKind;
      macros[i].weights.assign(static_cast<size_t>(rows * cols), 0);
      macros[i].lastResult.assign(static_cast<size_t>(cols), 0);
      return static_cast<int32_t>(i);
    }
  }
  std::cerr << "digitalcim_runtime: no free macro slots (max "
           << kMaxMacros << ")" << std::endl;
  std::exit(1);
}

void release_macro(int32_t macroId) { macros[macroId].inUse = false; }

void configure_compressor(int32_t macroId, int32_t multiplierKind) {
  // See executor_interface.hpp's header comment: stored, but does not
  // currently change trigger_mac's computed value (known gap).
  macros[macroId].multiplierKind = multiplierKind;
}

template <typename T>
void write_weights(int32_t macroId, memref_descriptor<2> weights) {
  auto &slot = macros[macroId];
  auto *data = static_cast<T *>(weights.data);
  for (int64_t r = 0; r < slot.rows; ++r) {
    for (int64_t c = 0; c < slot.cols; ++c) {
      int64_t idx = weights.offset + r * weights.strides[0] +
                    c * weights.strides[1];
      slot.weights[static_cast<size_t>(r * slot.cols + c)] =
          static_cast<int64_t>(data[idx]);
    }
  }
}

template <typename T>
int32_t trigger_mac(int32_t macroId, memref_descriptor<1> input) {
  auto &slot = macros[macroId];
  auto *data = static_cast<T *>(input.data);
  // Row-serial accumulation over macro_rows SRAM words (see
  // docs/architecture.md section 4) -- computed here in one shot for
  // functional correctness; the actual macro_rows-cycle timing cost lives
  // in the Python-side scheduler model, not this runtime.
  for (int64_t c = 0; c < slot.cols; ++c) {
    int64_t acc = 0;
    for (int64_t r = 0; r < slot.rows; ++r) {
      int64_t inIdx = input.offset + r * input.strides[0];
      acc += slot.weights[static_cast<size_t>(r * slot.cols + c)] *
             static_cast<int64_t>(data[inIdx]);
    }
    slot.lastResult[static_cast<size_t>(c)] = acc;
  }
  return macroId; // the "future" is just the macro id -- see
                  // DigitalCimToFunc.cpp's header comment.
}

template <typename T>
void barrier(int32_t futureId, memref_descriptor<1> out) {
  auto &slot = macros[futureId];
  auto *data = static_cast<T *>(out.data);
  for (int64_t c = 0; c < slot.cols; ++c) {
    int64_t idx = out.offset + c * out.strides[0];
    data[idx] = static_cast<T>(slot.lastResult[static_cast<size_t>(c)]);
  }
}

#define INSTANTIATE_FOR_TYPE(type)                                          \
  template void write_weights<type>(int32_t, memref_descriptor<2>);        \
  template int32_t trigger_mac<type>(int32_t, memref_descriptor<1>);       \
  template void barrier<type>(int32_t, memref_descriptor<1>);

INSTANTIATE_FOR_TYPE(int8_t)
INSTANTIATE_FOR_TYPE(int16_t)
INSTANTIATE_FOR_TYPE(int32_t)
INSTANTIATE_FOR_TYPE(int64_t)

} // namespace digitalcim_runtime
