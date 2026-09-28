# The Intel Decimal Library Computes `quantum` Wrongly

Status: resolved. The Intel tests take the expected `quantum` from the
IEEE 754 definition.

## Affected rule

`quantum(x)`, IEEE 754-2019 section 5.3.2: the value `1 * 10^q` for a finite
`x` with exponent `q`, positive infinity for an infinity, and a quiet NaN for
a NaN. A signaling NaN signals invalid, as every general-computational
operation does (section 6.2).

## Conflicting reference evidence

The Intel Decimal Floating-Point Math Library 2.0 Update 2:

- `bid32_quantum` (`LIBRARY/src/bid32_quantumd.c`, lines 44 to 73) tests the
  steering bits of the 32-bit operand with `MASK_STEERING_BITS`, the 64-bit
  mask `0x6000000000000000` of `bid_internal.h` line 2718. The test never
  matches, so an encoding with the steering bits `11` reads its exponent
  from the wrong field. `bid32_quantum(0x6dbffac1)` gives `0x6d800001`. The
  correct result is `0x36800001`, and floaty gives it.
- `bid32_quantum` and `bid64_quantum` test for an infinity with
  `MASK_INF32` or `MASK_INF` before they test for a NaN. A NaN matches the
  infinity mask, so the functions return the operand with a clear sign bit.
  A signaling NaN stays signaling, invalid is not signaled, and the trailing
  bits stay.
- `bid128_quantum` (`LIBRARY/src/bid128_quantumd.c`, line 59) sets only the
  high 64 bits of a NaN result. The low 64 bits are not initialized.
- `readtest.in` has no line that shows these cases. 33,723 of the seeded
  random cases do.

## Resolution

floaty follows IEEE 754. `floaty-verify/tests/decimal-intel-random.rs`
computes the expected `quantum` from the definition for an operand with the
steering bits `11` in decimal32, and for every NaN and infinity. The other
`quantum` cases compare with the library.

## Regression evidence

- `floaty-verify/tests/decimal-intel.rs` checks every `quantum` line of
  `readtest.in` against the library.
- `floaty-verify/tests/decimal-intel-random.rs` checks random operands of
  each width, and the cases above against the definition.
- The unit test `quantum_reads_a_subnormal_operand_through_daz` in
  `floaty/src/decimal/scale.rs` checks the result of a subnormal operand.
