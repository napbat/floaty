/*
 * Runs the double-double arithmetic of QD, `dd_real`, and its remainders,
 * as an oracle.
 *
 * Each function takes the binary64 bit patterns of the operand halves and a
 * rounding direction. It sets the direction, clears the exception flags,
 * computes with `dd_real`, and reads the flags. It writes the bit patterns
 * of the result halves to `result`, and returns the raised flags as the
 * bits below. It then restores the floating-point environment of the
 * caller: the rounding direction and the flags.
 *
 * The operators of `dd_real` are inline functions of the QD headers, so
 * this file compiles them. It must use the compiler options of the QD
 * library build.
 *
 * QD's `sqrt` of a negative value returns QD's NaN and writes an error
 * message to standard error. The shim discards the message.
 */

#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iostream>

#include <qd/dd_real.h>

namespace {

/* The rounding directions of the interface. */
enum : int {
  ROUNDING_NEAREST = 0,
  ROUNDING_ZERO = 1,
  ROUNDING_UP = 2,
  ROUNDING_DOWN = 3,
};

/* The flag bits of the interface. */
enum : unsigned {
  FLAG_INVALID = 0x01,
  FLAG_DIVIDE_BY_ZERO = 0x02,
  FLAG_OVERFLOW = 0x04,
  FLAG_UNDERFLOW = 0x08,
  FLAG_INEXACT = 0x10,
  /* The value that reports a rounding direction that the interface does not
     have. It is not a flag. */
  FLAG_BAD_ROUNDING = 0x80,
};

enum class Operation { add, sub, mul, div, sqrt };

double from_bits(std::uint64_t bits) {
  double value;
  std::memcpy(&value, &bits, sizeof value);
  return value;
}

std::uint64_t to_bits(double value) {
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  return bits;
}

/* Returns the `FE_*` mode of an interface direction, or -1. */
int fenv_mode(int rounding) {
  switch (rounding) {
  case ROUNDING_NEAREST:
    return FE_TONEAREST;
  case ROUNDING_ZERO:
    return FE_TOWARDZERO;
  case ROUNDING_UP:
    return FE_UPWARD;
  case ROUNDING_DOWN:
    return FE_DOWNWARD;
  default:
    return -1;
  }
}

/*
 * Returns the interface flag bits of the raised `FE_*` exceptions. The
 * function inlines into both callers, so `run` keeps its machine code and
 * its offsets, which `floaty/src/double_double/qd.rs` cites.
 */
__attribute__((always_inline)) inline unsigned interface_flags(int raised) {
  unsigned flags = 0;
  if (raised & FE_INVALID) {
    flags |= FLAG_INVALID;
  }
  if (raised & FE_DIVBYZERO) {
    flags |= FLAG_DIVIDE_BY_ZERO;
  }
  if (raised & FE_OVERFLOW) {
    flags |= FLAG_OVERFLOW;
  }
  if (raised & FE_UNDERFLOW) {
    flags |= FLAG_UNDERFLOW;
  }
  if (raised & FE_INEXACT) {
    flags |= FLAG_INEXACT;
  }
  return flags;
}

/*
 * Runs one operation under `rounding`.
 *
 * The operands pass through volatile objects that the function reads after
 * it sets the direction and clears the flags. The result passes through a
 * volatile object that the function writes before it reads the flags. The
 * compiler keeps volatile accesses and calls in program order, so every
 * floating-point step of the operation runs between the two.
 */
unsigned run(Operation operation, const std::uint64_t *a, const std::uint64_t *b, int rounding,
             std::uint64_t *result) {
  int mode = fenv_mode(rounding);
  if (mode < 0) {
    return FLAG_BAD_ROUNDING;
  }
  std::fenv_t saved;
  std::fegetenv(&saved);
  std::fesetround(mode);
  std::feclearexcept(FE_ALL_EXCEPT);

  volatile double input[4] = {from_bits(a[0]), from_bits(a[1]), from_bits(b[0]),
                              from_bits(b[1])};
  dd_real x(input[0], input[1]);
  dd_real y(input[2], input[3]);
  dd_real z;
  switch (operation) {
  case Operation::add:
    z = x + y;
    break;
  case Operation::sub:
    z = x - y;
    break;
  case Operation::mul:
    z = x * y;
    break;
  case Operation::div:
    z = x / y;
    break;
  case Operation::sqrt: {
    /* QD writes an error for a negative value. The test runs that case
       often, so the shim discards the text. */
    std::streambuf *error = std::cerr.rdbuf(nullptr);
    z = sqrt(x);
    std::cerr.rdbuf(error);
    std::cerr.clear();
    break;
  }
  }
  volatile double output[2] = {z.x[0], z.x[1]};
  int raised = std::fetestexcept(FE_ALL_EXCEPT);
  std::fesetenv(&saved);

  result[0] = to_bits(output[0]);
  result[1] = to_bits(output[1]);
  return interface_flags(raised);
}

/*
 * Runs QD's remainder `drem`, or its truncated remainder `fmod`, under
 * `rounding`, as `run` runs an operator. `drem` is an inline function of
 * the QD headers, and `fmod` is a function of `libqd.a`. A function apart
 * from `run` leaves the machine code of `run` as it is.
 */
unsigned run_remainder(bool truncated, const std::uint64_t *a, const std::uint64_t *b,
                       int rounding, std::uint64_t *result) {
  int mode = fenv_mode(rounding);
  if (mode < 0) {
    return FLAG_BAD_ROUNDING;
  }
  std::fenv_t saved;
  std::fegetenv(&saved);
  std::fesetround(mode);
  std::feclearexcept(FE_ALL_EXCEPT);

  volatile double input[4] = {from_bits(a[0]), from_bits(a[1]), from_bits(b[0]),
                              from_bits(b[1])};
  dd_real x(input[0], input[1]);
  dd_real y(input[2], input[3]);
  dd_real z = truncated ? fmod(x, y) : drem(x, y);
  volatile double output[2] = {z.x[0], z.x[1]};
  int raised = std::fetestexcept(FE_ALL_EXCEPT);
  std::fesetenv(&saved);
  result[0] = to_bits(output[0]);
  result[1] = to_bits(output[1]);
  return interface_flags(raised);
}

} // namespace

extern "C" {

unsigned floaty_qd_add(const std::uint64_t *a, const std::uint64_t *b, int rounding,
                       std::uint64_t *result) {
  return run(Operation::add, a, b, rounding, result);
}

unsigned floaty_qd_sub(const std::uint64_t *a, const std::uint64_t *b, int rounding,
                       std::uint64_t *result) {
  return run(Operation::sub, a, b, rounding, result);
}

unsigned floaty_qd_mul(const std::uint64_t *a, const std::uint64_t *b, int rounding,
                       std::uint64_t *result) {
  return run(Operation::mul, a, b, rounding, result);
}

unsigned floaty_qd_div(const std::uint64_t *a, const std::uint64_t *b, int rounding,
                       std::uint64_t *result) {
  return run(Operation::div, a, b, rounding, result);
}

unsigned floaty_qd_drem(const std::uint64_t *a, const std::uint64_t *b, int rounding,
                        std::uint64_t *result) {
  return run_remainder(false, a, b, rounding, result);
}

unsigned floaty_qd_fmod(const std::uint64_t *a, const std::uint64_t *b, int rounding,
                        std::uint64_t *result) {
  return run_remainder(true, a, b, rounding, result);
}

/* `sqrt` has one operand. It passes `a` again as the unused second one. */
unsigned floaty_qd_sqrt(const std::uint64_t *a, int rounding, std::uint64_t *result) {
  return run(Operation::sqrt, a, a, rounding, result);
}

/* Metadata lives outside the machine-code sections of the arithmetic oracle. */
extern const unsigned floaty_qd_radix = std::numeric_limits<dd_real>::radix;
extern const unsigned floaty_qd_precision = std::numeric_limits<dd_real>::digits;

} // extern "C"
