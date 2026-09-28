# The Intel Decimal Library Rounds Some Narrow Fused Multiply-Adds Wrongly

Status: resolved. The random tests compare these cases with the library's
own fused multiply-add in the next wider format.

## Affected rule

The fused multiply-add of IEEE 754-2019 section 5.4.1, rounded once. An
exact zero result signals nothing, and its sign is `+0` except when rounding
toward negative (section 6.3). A tiny inexact result keeps its sign.

## Conflicting reference evidence

The Intel Decimal Floating-Point Math Library 2.0 Update 2
(`LIBRARY/src/bid32_fma.c` and `LIBRARY/src/bid64_fma.c`), in 11 of the
seeded random fused multiply-adds: 10 in decimal32 and 1 in decimal64.

- `bid32_fma(0x972625a0, 0x972625a0, 0x8000186a)`, rounding toward zero:
  `-2500000E-55 * -2500000E-55 - 6250E-101` is exactly zero. The library
  signals underflow and inexact. floaty gives `+0` with no IEEE flag. It
  reports `DENORMAL_INPUT`, because the addend is subnormal.
- `bid64_fma(0x608386f26fc0ffff, 0x2d40000000002710, 0x8000000000000001)`,
  rounding toward zero: `9999999999999999E-382 * 1E-32 - 1E-398` is exactly
  `-1E-414`, below the smallest subnormal value. The library gives `+0`.
  floaty gives `-0`, with underflow and inexact.
- The operands convert exactly to the next wider format, and the library's
  fused multiply-add there is exact for both cases. Its result converts back
  to the correct one, which agrees with floaty.
- `readtest.in` has no line that shows these cases.

## Resolution

floaty follows IEEE 754. When the library's fused multiply-add in the next
wider format is exact and differs from the narrow function,
`floaty-verify/tests/decimal-intel-random.rs` takes the wider result,
converted back to the format, as the expected result.

## Regression evidence

- `floaty-verify/tests/decimal-intel.rs` checks every `fma` line of
  `readtest.in` against the library.
- `floaty-verify/tests/decimal-intel-random.rs` checks random triples of
  each width.
