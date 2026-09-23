#include "executor_interface.hpp"

#include <algorithm>
#include <iostream>

namespace digitalflashcim_runtime {

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
      // Deliberately NOT erased -- see executor_interface.hpp's header
      // comment for why an unspecified/all-0 initial state (not all-1s) is
      // the right modeling choice.
      macros[i].weights.assign(static_cast<size_t>(rows * cols), 0);
      macros[i].lastResult.assign(static_cast<size_t>(cols), 0);
      return static_cast<int32_t>(i);
    }
  }
  std::cerr << "digitalflashcim_runtime: no free macro slots (max "
           << kMaxMacros << ")" << std::endl;
  std::exit(1);
}

void release_macro(int32_t macroId) { macros[macroId].inUse = false; }

void configure_compressor(int32_t macroId, int32_t multiplierKind) {
  macros[macroId].multiplierKind = multiplierKind;
}

void erase_block(int32_t macroId) {
  auto &slot = macros[macroId];
  // Erased == every cell reads all-1s. int64_t(-1) is exactly that
  // pattern at every bit width (two's complement all-ones), so this needs
  // no per-bitWidth masking -- see executor_interface.hpp.
  std::fill(slot.weights.begin(), slot.weights.end(), static_cast<int64_t>(-1));
}

template <typename T>
void program_rows(int32_t macroId, memref_descriptor<2> weights) {
  auto &slot = macros[macroId];
  auto *data = static_cast<T *>(weights.data);
  bool anyMismatch = false;
  for (int64_t r = 0; r < slot.rows; ++r) {
    for (int64_t c = 0; c < slot.cols; ++c) {
      int64_t idx = weights.offset + r * weights.strides[0] +
                    c * weights.strides[1];
      int64_t target = static_cast<int64_t>(data[idx]);
      size_t cell = static_cast<size_t>(r * slot.cols + c);
      // Physical program semantics: a flash cell can only be PULLED from 1
      // to 0, never pushed back up -- so the achievable result is
      // (current contents) AND (target), not a plain overwrite. This is
      // the identity when the cell was properly erase_block'd first
      // (all-1s AND anything == that thing); it silently loses any bit the
      // target needed set to 1 that the cell did not already have,
      // otherwise -- exactly what real hardware would do.
      int64_t achieved = slot.weights[cell] & target;
      if (achieved != target)
        anyMismatch = true;
      slot.weights[cell] = achieved;
    }
  }
  if (anyMismatch) {
    std::cerr << "digitalflashcim_runtime: macro " << macroId
             << " programmed without (or after an insufficient) erase_block"
                " -- one or more cells could not reach their target value"
                " (real NOR flash cannot set a bit from 0 to 1 without"
                " erasing first). This is not a crash: the achieved,"
                " bitwise-AND-corrupted value was stored, same as real"
                " hardware would silently do." << std::endl;
  }
}

template <typename T>
int32_t trigger_mac(int32_t macroId, memref_descriptor<1> input) {
  auto &slot = macros[macroId];
  auto *data = static_cast<T *>(input.data);
  // Row-serial accumulation, identical to the sibling DigitalCim runtime --
  // the multiply/accumulate datapath does not reference the storage
  // medium at all.
  for (int64_t c = 0; c < slot.cols; ++c) {
    int64_t acc = 0;
    for (int64_t r = 0; r < slot.rows; ++r) {
      int64_t inIdx = input.offset + r * input.strides[0];
      acc += slot.weights[static_cast<size_t>(r * slot.cols + c)] *
             static_cast<int64_t>(data[inIdx]);
    }
    slot.lastResult[static_cast<size_t>(c)] = acc;
  }
  return macroId; // the "future" is just the macro id, same as DigitalCim.
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
  template void program_rows<type>(int32_t, memref_descriptor<2>);         \
  template int32_t trigger_mac<type>(int32_t, memref_descriptor<1>);       \
  template void barrier<type>(int32_t, memref_descriptor<1>);

INSTANTIATE_FOR_TYPE(int8_t)
INSTANTIATE_FOR_TYPE(int16_t)
INSTANTIATE_FOR_TYPE(int32_t)
INSTANTIATE_FOR_TYPE(int64_t)

} // namespace digitalflashcim_runtime
