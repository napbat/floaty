/*
 * Runs QD 2.3.24's sqr, inv, and npwr with the pinned arithmetic
 * configuration. A separate object keeps the existing arithmetic oracle's
 * machine code. Each call restores the caller's floating-point environment.
 */

#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iostream>

#include <qd/dd_real.h>

namespace {

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

/* Runs `function` on `a` under `rounding`. */
template <typename Function>
unsigned run(const std::uint64_t *a, int rounding, std::uint64_t *result, Function function) {
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
  dd_real z = function(x);
  volatile double output[2] = {z.x[0], z.x[1]};
  int raised = std::fetestexcept(FE_ALL_EXCEPT);
  std::fesetenv(&saved);
  result[0] = to_bits(output[0]);
  result[1] = to_bits(output[1]);
  return interface_flags(raised);
}

/*
 * QD's npwr source loop with an unsigned magnitude. The library's npwr
 * applies std::abs to INT_MIN, which is undefined, and its loop never
 * returns. This copy computes 1 / a^(2^31) for INT_MIN. For 0 it returns
 * `a`, unlike npwr.
 */
dd_real power_of_magnitude(const dd_real &a, int n) {
  dd_real r = a;
  dd_real s = 1.0;
  unsigned count = n < 0 ? 0u - static_cast<unsigned>(n) : static_cast<unsigned>(n);
  if (count > 1) {
    while (count > 0) {
      if (count % 2 == 1) {
        s *= r;
      }
      count /= 2;
      if (count > 0) {
        r = sqr(r);
      }
    }
  } else {
    s = r;
  }
  if (n < 0) {
    return 1.0 / s;
  }
  return s;
}

/* Runs QD's npwr, and discards the error text of 0^0. */
dd_real power(const dd_real &a, int n) {
  std::streambuf *error = std::cerr.rdbuf(nullptr);
  dd_real z = npwr(a, n);
  std::cerr.rdbuf(error);
  std::cerr.clear();
  return z;
}

} // namespace

extern "C" {

unsigned floaty_qd_sqr(const std::uint64_t *a, int rounding, std::uint64_t *result) {
  return run(a, rounding, result, [](const dd_real &x) { return sqr(x); });
}

unsigned floaty_qd_inv(const std::uint64_t *a, int rounding, std::uint64_t *result) {
  return run(a, rounding, result, [](const dd_real &x) { return inv(x); });
}

/* The library's npwr. `n` must not be INT_MIN, for which npwr never returns. */
unsigned floaty_qd_npwr(const std::uint64_t *a, int n, int rounding, std::uint64_t *result) {
  return run(a, rounding, result, [n](const dd_real &x) { return power(x, n); });
}

/* QD's npwr source with an unsigned magnitude, for INT_MIN. */
unsigned floaty_qd_npwr_magnitude(const std::uint64_t *a, int n, int rounding,
                                  std::uint64_t *result) {
  return run(a, rounding, result, [n](const dd_real &x) { return power_of_magnitude(x, n); });
}

} // extern "C"
