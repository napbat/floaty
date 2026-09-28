# rustc_apfloat Does Not Signal Overflow for a Finite Overflow Result

Status: resolved. The `rustc_apfloat` tests remove the overflow flag of a
finite result before the comparison.

## Affected rule

The overflow flag. IEEE 754-2019 section 7.4 signals overflow when the
rounded result with an unbounded exponent range exceeds the largest finite
value, in every rounding direction. A direction that rounds toward zero gives
the largest finite value and still signals overflow and inexact.

## Conflicting reference evidence

- `rustc_apfloat` 0.2.3 (`src/ieee.rs` line 2003, `overflow_result`) returns
  only `Status::INEXACT` with the largest finite value, as LLVM
  `APFloat::handleOverflow` does. It signals overflow only with an infinity.
- TestFloat, MPFR through `floaty-verify/src/mpfr.rs`, and the x86
  processor signal overflow for both results. floaty matches them.

## Resolution

floaty follows IEEE 754. `floaty-verify/tests/apfloat-operations.rs` removes
the overflow flag of floaty's finite results before it compares the flags
with `rustc_apfloat`.

## Regression evidence

- `floaty-verify/tests/testfloat-arithmetic.rs` and
  `floaty-verify/tests/sse-hardware.rs` check the overflow flag of every
  directed rounding against TestFloat and the processor.
- `floaty-verify/tests/apfloat-operations.rs` compares every other flag of
  the `from_int` results, and removes overflow only from the largest finite
  value, which the format parameters give.
