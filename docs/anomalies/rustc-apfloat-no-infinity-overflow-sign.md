# rustc_apfloat Rounds a Negative Overflow the Wrong Way Without Infinity

Status: resolved. The `rustc_apfloat` tests skip the affected results, and
the MPFR tests check them.

## Affected rule

Overflow in a directed rounding for a format without an infinity, such as
OCP FP8 E4M3: a negative value that rounds past the largest finite value
gives the NaN toward negative and the largest finite value toward positive,
as IEEE 754-2019 section 7.4 gives an infinity or the largest finite value.

## Conflicting reference evidence

- `rustc_apfloat` 0.2.3 (`src/ieee.rs` line 2000) documents that
  `overflow_result` handles a positive overflow, and that a caller must
  negate the rounding direction for a negative overflow. `normalize` does so
  at line 2064.
- For `NonfiniteBehavior::NanOnly`, `normalize` calls
  `overflow_result(round)` at lines 2101 and 2149 without the negation. A
  negative value whose magnitude rounds to the NaN pattern, below
  `2^(emax + 1)`, then gives the NaN toward positive and the largest finite
  value toward negative: the reverse of IEEE 754 and of the positive case.
- MPFR, through the rounding oracle in `floaty-verify/src/mpfr.rs`, gives
  the IEEE 754 result, and floaty matches it.

## Resolution

floaty follows IEEE 754. `floaty-verify/tests/apfloat-operations.rs`
(`negative_overflow_without_infinity`) skips the `scale_b` results that the
error changes. The crate stays unmodified.

## Regression evidence

- `floaty-verify/tests/operations-mpfr.rs` and
  `floaty-verify/tests/arithmetic-mpfr.rs` check every FP8 overflow in every
  direction against MPFR.
- `floaty-verify/tests/apfloat-operations.rs` checks every other `scale_b`
  result against `rustc_apfloat`.
