/*
 * Runs QD 2.3.24's sqr and inv with the pinned arithmetic configuration.
 * A separate object keeps the existing arithmetic oracle's machine code.
 * Each call restores the caller's floating-point environment.
 */

#include <cfenv>
#include <cstdint>
#include <cstring>

#include <qd/dd_real.h>

namespace {

enum class Operation { square, inverse };

int fenv_mode(int rounding) {
  switch (rounding) {
  case 0:
    return FE_TONEAREST;
  case 1:
    return FE_TOWARDZERO;
  case 2:
    return FE_UPWARD;
  case 3:
    return FE_DOWNWARD;
  default:
    return -1;
  }
}

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

unsigned interface_flags(int raised) {
  unsigned flags = 0;
  if (raised & FE_INVALID) {
    flags |= 0x01;
  }
  if (raised & FE_DIVBYZERO) {
    flags |= 0x02;
  }
  if (raised & FE_OVERFLOW) {
    flags |= 0x04;
  }
  if (raised & FE_UNDERFLOW) {
    flags |= 0x08;
  }
  if (raised & FE_INEXACT) {
    flags |= 0x10;
  }
  return flags;
}

template <Operation operation>
unsigned run(const std::uint64_t *a, int rounding, std::uint64_t *result) {
  int mode = fenv_mode(rounding);
  if (mode < 0) {
    return 0x80;
  }
  std::fenv_t saved;
  std::fegetenv(&saved);
  std::fesetround(mode);
  std::feclearexcept(FE_ALL_EXCEPT);
  /* Volatile inputs and outputs keep the arithmetic inside the flag check. */
  volatile double input[2] = {from_bits(a[0]), from_bits(a[1])};
  dd_real x(input[0], input[1]);
  dd_real z = operation == Operation::square ? sqr(x) : inv(x);
  volatile double output[2] = {z.x[0], z.x[1]};
  int raised = std::fetestexcept(FE_ALL_EXCEPT);
  std::fesetenv(&saved);
  result[0] = to_bits(output[0]);
  result[1] = to_bits(output[1]);
  return interface_flags(raised);
}

} // namespace

extern "C" {

unsigned floaty_qd_sqr(const std::uint64_t *a, int rounding, std::uint64_t *result) {
  return run<Operation::square>(a, rounding, result);
}

unsigned floaty_qd_inv(const std::uint64_t *a, int rounding, std::uint64_t *result) {
  return run<Operation::inverse>(a, rounding, result);
}

} // extern "C"
