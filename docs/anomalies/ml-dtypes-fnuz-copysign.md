# ml_dtypes copysign Turns the FNUZ Zero Into the NaN

Status: resolved. The FP8 tests check this case by `DESIGN.md`.

## Affected rule

The sign operations of the FNUZ formats. `DESIGN.md` keeps the one zero
encoding and the one NaN encoding of `Fnuz`, because the formats have no
negative zero.

## Conflicting reference evidence

- `ml_dtypes` 0.6.0 gives `np.copysign(0x00, -2.0)` as `0x80` in
  `float8_e4m3fnuz` and `float8_e5m2fnuz`. `0x80` is the NaN of these
  formats.
- The same version gives `np.negative(0x00)` as `0x00`, so `ml_dtypes`
  disagrees with itself: one sign operation keeps the zero, the other makes
  a NaN.
- IEEE 754-2019 section 5.5.1 defines `copySign` as a copy of the operand
  with another sign bit. The FNUZ formats have no negative zero, so the zero
  has no other sign, and the result `0x80` is a value of another class.

## Resolution

floaty keeps the zero. `floaty-verify/tests/fp8-operations.rs` checks
`copy_sign` of the FNUZ zero by `DESIGN.md` and compares every other pair
with `ml_dtypes`.

## Regression evidence

- `floaty-verify/tests/fp8-operations.rs` compares every other `copysign`
  pair, and every `negative` and `abs` result, with `ml_dtypes`.
- `floaty/src/float/tests.rs` (`sign_operations_change_only_the_sign_bit`)
  checks the FNUZ zero and NaN.
