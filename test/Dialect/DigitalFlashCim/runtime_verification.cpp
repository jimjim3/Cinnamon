// Direct numeric verification of libDigitalFlashCimDialectRuntime.so,
// bypassing MLIR entirely -- mirrors the ad-hoc C++ cross-language check
// cim-compiler's own memo describes running (uncommitted there) for the
// sibling DigitalCim runtime. Committed here instead of left in a scratch
// directory, so the claim this dialect exists to make -- that
// erase-before-program is enforced FUNCTIONALLY (bitwise AND against the
// current cell contents), not merely documented -- stays reproducible.
//
// NOT wired into CTest/lit yet (see PLAN.md Phase 6): build and run it
// manually after `pixi run configure`:
//
//   CXX=.pixi/envs/default/bin/x86_64-conda-linux-gnu-clang++
//   $CXX -std=c++17 -o /tmp/dfcim_runtime_test \
//     test/Dialect/DigitalFlashCim/runtime_verification.cpp \
//     -L build/lib -lDigitalFlashCimDialectRuntime -Wl,-rpath,build/lib
//   /tmp/dfcim_runtime_test   # exits 0 iff every check passes
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" {
int32_t digitalflashcim_acquire_macro(int64_t rows, int64_t cols, int64_t bitWidth, int32_t multiplierKind);
void digitalflashcim_release_macro(int32_t macroId);
void digitalflashcim_erase_block(int32_t macroId);
void digitalflashcim_program_rows_i8(int32_t macroId, int8_t *base, int8_t *data, int64_t offset,
                                     int64_t size0, int64_t size1, int64_t stride0, int64_t stride1);
int32_t digitalflashcim_trigger_mac_i8(int32_t macroId, int8_t *base, int8_t *data, int64_t offset,
                                       int64_t size0, int64_t stride0);
void digitalflashcim_barrier_i32(int32_t futureId, int32_t *base, int32_t *data, int64_t offset,
                                 int64_t size0, int64_t stride0);
}

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
                              else std::fprintf(stderr, "ok:   %s\n", msg); } while (0)

int main() {
  const int64_t rows = 2, cols = 2;

  // ---- Test 1: program WITHOUT erase corrupts the value ----
  {
    int32_t m = digitalflashcim_acquire_macro(rows, cols, 8, 0);
    int8_t target[4] = {5, -3, 127, -128}; // row-major 2x2
    digitalflashcim_program_rows_i8(m, target, target, 0, rows, cols, cols, 1);
    // Freshly-acquired macro starts at 0 (see runtime/DigitalFlashCim/
    // executor_interface.hpp); AND with 0 is always 0, so EVERY cell should
    // read back as 0, not the target -- proving programming without erase
    // does NOT silently succeed like SRAM's plain overwrite would.
    int8_t input[2] = {1, 1};
    int32_t fut = digitalflashcim_trigger_mac_i8(m, input, input, 0, rows, 1);
    int32_t out[2] = {-99, -99};
    digitalflashcim_barrier_i32(fut, out, out, 0, cols, 1);
    CHECK(out[0] == 0 && out[1] == 0,
          "program_rows without erase_block corrupts to 0 (AND with never-erased state)");
    digitalflashcim_release_macro(m);
  }

  // ---- Test 2: erase_block then program_rows reproduces the exact value ----
  {
    int32_t m = digitalflashcim_acquire_macro(rows, cols, 8, 0);
    digitalflashcim_erase_block(m);
    int8_t target[4] = {5, -3, 127, -128};
    digitalflashcim_program_rows_i8(m, target, target, 0, rows, cols, cols, 1);
    int8_t input[2] = {1, 0}; // isolate row 0: dot product = target[0][*] * 1
    int32_t fut = digitalflashcim_trigger_mac_i8(m, input, input, 0, rows, 1);
    int32_t out[2] = {-99, -99};
    digitalflashcim_barrier_i32(fut, out, out, 0, cols, 1);
    CHECK(out[0] == 5 && out[1] == -3,
          "erase_block then program_rows reproduces the exact target (row 0 isolated)");

    int8_t input2[2] = {0, 1}; // isolate row 1
    fut = digitalflashcim_trigger_mac_i8(m, input2, input2, 0, rows, 1);
    digitalflashcim_barrier_i32(fut, out, out, 0, cols, 1);
    CHECK(out[0] == 127 && out[1] == -128,
          "erase_block then program_rows reproduces the exact target (row 1 isolated, incl. extremes)");
    digitalflashcim_release_macro(m);
  }

  // ---- Test 3: reprogramming without a fresh erase corrupts relative to the NEW target ----
  {
    int32_t m = digitalflashcim_acquire_macro(rows, cols, 8, 0);
    digitalflashcim_erase_block(m);
    int8_t first[4] = {0b00000101, 0, 0, 0}; // cell(0,0) = 5 = 0b0101
    digitalflashcim_program_rows_i8(m, first, first, 0, rows, cols, cols, 1);
    // Reprogram cell(0,0) to a value needing a bit that's already 0 (bit1):
    // target = 0b0111 (7). Physically achievable = old(0b0101) & new(0b0111) = 0b0101 = 5, NOT 7.
    int8_t second[4] = {0b00000111, 0, 0, 0};
    digitalflashcim_program_rows_i8(m, second, second, 0, rows, cols, cols, 1);
    int8_t input[2] = {1, 0};
    int32_t fut = digitalflashcim_trigger_mac_i8(m, input, input, 0, rows, 1);
    int32_t out[2] = {-99, -99};
    digitalflashcim_barrier_i32(fut, out, out, 0, cols, 1);
    CHECK(out[0] == 5,
          "reprogramming without a fresh erase: achieved = old_bits AND new_bits (5 & 7 = 5, not 7)");
    digitalflashcim_release_macro(m);
  }

  // ---- Test 4: a SECOND erase_block clears the corruption from test 3's pattern ----
  {
    int32_t m = digitalflashcim_acquire_macro(rows, cols, 8, 0);
    digitalflashcim_erase_block(m);
    int8_t first[4] = {0b00000101, 0, 0, 0};
    digitalflashcim_program_rows_i8(m, first, first, 0, rows, cols, cols, 1);
    digitalflashcim_erase_block(m); // fresh erase before reprogramming
    int8_t second[4] = {0b00000111, 0, 0, 0};
    digitalflashcim_program_rows_i8(m, second, second, 0, rows, cols, cols, 1);
    int8_t input[2] = {1, 0};
    int32_t fut = digitalflashcim_trigger_mac_i8(m, input, input, 0, rows, 1);
    int32_t out[2] = {-99, -99};
    digitalflashcim_barrier_i32(fut, out, out, 0, cols, 1);
    CHECK(out[0] == 7,
          "a fresh erase_block before reprogramming DOES reach the new target (7)");
    digitalflashcim_release_macro(m);
  }

  std::fprintf(stderr, "\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
  return failures == 0 ? 0 : 1;
}
