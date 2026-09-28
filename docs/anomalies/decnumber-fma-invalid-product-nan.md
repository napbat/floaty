# decNumber's Two Fused Multiply-Adds Disagree on `0 * inf + NaN`

Status: resolved. The decimal32 tests take a special fused multiply-add from
`decDoubleFMA`, as the decimal64 and decimal128 tests do.

## Affected rule

A fused multiply-add whose product is the invalid `0 * inf` and whose
addend is a NaN. IEEE 754-2019 section 7.2 (c) leaves it to the
implementation whether a quiet NaN addend signals invalid. Section 6.2.3
asks that the result keeps the payload of an input NaN.

## Conflicting reference evidence

decNumber 3.68 has two fused multiply-adds:

- `decFloatFMA` (`decBasic.c`, lines 2022 to 2025), under `decDoubleFMA`
  and `decQuadFMA`, handles a NaN operand before the invalid product.
  `fma(0, Inf, NaN7)` gives `NaN7` with no condition, and
  `fma(0, Inf, sNaN7)` gives `NaN7` with `Invalid_operation`. floaty gives
  these results with `InvalidProduct::YieldsToNan`. The decTest 2.62
  vectors have no such case.
- `decNumberFMA` (`decNumber.c`, lines 1143 to 1147) handles the invalid
  product first and ignores the addend. `fma(0, Inf, NaN7)` gives `NaN` with
  `Invalid_operation`. `fma(0, Inf, sNaN7)` gives `NaN` with
  `Invalid_operation`, and the payload of the signaling NaN is lost.

## Resolution

floaty keeps its `InvalidProduct` field, and the tests run decNumber's
cases with `YieldsToNan`. decSingle has no fused multiply-add, so the
decimal32 random tests compute it with decNumber's arbitrary-precision
numbers. A decimal32 fused multiply-add with a special operand takes
`decDoubleFMA` of the operands widened exactly to decimal64 instead.

## Regression evidence

- `floaty-verify/tests/decimal-decnumber/vectors.rs` checks every `ddFMA`
  and `dqFMA` vector.
- `floaty-verify/tests/decimal-decnumber/random.rs` checks random decimal32,
  decimal64, and decimal128 triples with special operands.
