# SoftFloat's ARM Specialization Does Not Quiet x87 NaNs

Status: resolved. The x87 TestFloat tests quiet the expected NaN.

## Affected rule

NaN propagation in x87 extended-precision arithmetic under the `SignalingFirst`
rule, which the default mode uses: an operation with a signaling NaN operand
returns that NaN made quiet.

## Conflicting reference evidence

- Berkeley SoftFloat Release 3e, `source/ARM-VFPv2/s_propagateNaNExtF80UI.c`
  line 79, is `uiZ.v0 | UINT64_C( 0xC000000000000000 );`. The statement
  discards its result, so the function returns a signaling NaN unchanged.
  The upstream `master` branch (commit a0c6494, March 2025) has the same
  line. Every other specialization, and the other formats of the ARM
  specialization, set the quiet bit.
- IEEE 754-2019 section 6.2 requires an operation with a signaling NaN
  operand to signal invalid and to deliver a quiet NaN. ARM's
  `FPProcessNaN` also sets the quiet bit.
- TestFloat, built against the ARM specialization, therefore expects a
  signaling NaN result for `extF80_add`, `extF80_sub`, `extF80_mul`,
  `extF80_div`, and `extF80_sqrt` with a signaling NaN operand. No real ARM
  processor has the x87 format, so no hardware shows the combination.

## Resolution

floaty returns the quiet NaN. `floaty-verify/src/testfloat.rs`
(`quiet_extended_nan`) sets the quiet bit of an expected x87 signaling NaN
before the comparison, for the x87 tests only. The flags stay as TestFloat
gives them. The submodule stays unmodified.

## Regression evidence

- `floaty-verify/tests/testfloat-arithmetic.rs` (`extended::*`) passes with
  the correction and fails without it.
- `floaty-verify/tests/x87-hardware.rs` checks x87 NaN results against the
  processor under the `LargerSignificand` rule.
