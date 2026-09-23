#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// Pure-software functional simulator for the DigitalFlashCim dialect, same
// scope as the sibling DigitalCim runtime (correctness, not timing/energy --
// see ~/CIM_Compiler_Digital_Flash/src/nor_flash_calibration.py for where
// the actual cycle/energy numbers live, on the Python side).
//
// THE ONE THING THIS RUNTIME MODELS THAT DigitalCim'S DOES NOT:
// erase-before-program, and it is modeled FUNCTIONALLY, not as a labeled
// error check. Real NOR flash: erase sets every cell in a block to logical
// 1; program can only pull a bit from 1 down to 0, never push 0 back up to
// 1. So programming a cell whose current contents are C with a target
// value T physically achieves C & T, not T outright -- which equals T
// exactly when C is all-1s (freshly erased), and silently drops any bit
// of T that needed to be 1 where C already held a 0 otherwise. This
// runtime computes exactly that AND, at the full int64_t width the value
// is sign-extended into (see executor.cpp's program_rows for why that is
// safe regardless of the macro's bitWidth: an all-1s erased cell
// sign-extends to int64_t(-1) at ANY bit width, and ANDing with that is
// the identity).
//
// A freshly acquired macro's cells start at 0 (all-CLEAR), the OPPOSITE of
// erased -- not because that is claimed to be physically realistic (a real
// fab's post-manufacture state is unspecified), but because starting at
// "already erased" would let a caller who never calls erase_block get away
// with it by simulator accident, defeating the whole point of modeling
// this constraint. Starting at all-0 means any program_rows before the
// macro's first erase_block visibly fails to reproduce the target value
// (AND with 0 is always 0), exactly as it should.
//
// KNOWN GAP (same as DigitalCim): multiplierKind is stored but does not
// change the computed dot product -- see that runtime's header comment for
// why (no real RTL or faithful compressor truth tables to model against).
namespace digitalflashcim_runtime {

struct MacroSlot {
  bool inUse = false;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t bitWidth = 0;
  int32_t multiplierKind = 0;
  std::vector<int64_t> weights;    // row-major, rows*cols, sign-extended; erased == -1 per cell
  std::vector<int64_t> lastResult; // cols, valid after trigger_mac
};

constexpr size_t kMaxMacros = 64;
extern std::array<MacroSlot, kMaxMacros> macros;

template <size_t Rank> struct memref_descriptor {
  void *base;
  void *data;
  int64_t offset;
  std::array<int64_t, Rank> sizes;
  std::array<int64_t, Rank> strides;
};

int32_t acquire_macro(int64_t rows, int64_t cols, int64_t bitWidth,
                      int32_t multiplierKind);
void release_macro(int32_t macroId);
void configure_compressor(int32_t macroId, int32_t multiplierKind);

void erase_block(int32_t macroId);

template <typename T>
void program_rows(int32_t macroId, memref_descriptor<2> weights);

template <typename T>
int32_t trigger_mac(int32_t macroId, memref_descriptor<1> input);

template <typename T>
void barrier(int32_t futureId, memref_descriptor<1> out);

} // namespace digitalflashcim_runtime
