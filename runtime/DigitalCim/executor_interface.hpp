#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// Pure-software functional simulator for the DigitalCim dialect (see
// ~/cim-compiler/docs/architecture.md's "純軟體路線" scope decision -- this
// is not a hardware backend). It computes CORRECT dot products; it does not
// model timing or energy (that lives in the Python-side
// src/hw.py/src/openacm_calibration.py model, mirroring how the sibling
// Memristor runtime also only does functional simulation).
//
// KNOWN GAP: multiplierKind ("exact" / "approx_compressor" / "logarithmic")
// is stored per macro but does NOT currently change the computed value --
// every kind computes the bit-exact integer dot product. Modeling
// OpenACM's actual approximate-compressor/logarithmic error characteristics
// would need either the real generated RTL or a faithful reimplementation
// of its specific compressor truth tables, neither of which we have; this
// is left as a documented future gap rather than a guessed approximation.
namespace digitalcim_runtime {

struct MacroSlot {
  bool inUse = false;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t bitWidth = 0;
  int32_t multiplierKind = 0;
  std::vector<int64_t> weights;    // row-major, rows*cols, sign-extended
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

template <typename T>
void write_weights(int32_t macroId, memref_descriptor<2> weights);

template <typename T>
int32_t trigger_mac(int32_t macroId, memref_descriptor<1> input);

template <typename T>
void barrier(int32_t futureId, memref_descriptor<1> out);

} // namespace digitalcim_runtime
