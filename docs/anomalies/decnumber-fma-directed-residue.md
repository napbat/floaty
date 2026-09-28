# decNumber Loses a Small Addend of a Fused Multiply-Add

Status: resolved. The random fused multiply-add tests compute the decNumber
result at `2p + 3` digits and round it to the format.

## Affected rule

The correctly rounded fused multiply-add. IEEE 754-2019 section 5.4.1 gives
`fusedMultiplyAdd(x, y, z)` as `(x * y) + z` computed with an unbounded range
and precision, rounded once to the destination format.

## Conflicting reference evidence

- decNumber 3.68 `decNumberFMA` (`decNumber.c`, lines 1092 to 1167) and
  `decFloatFMA` (`decBasic.c`, from line 1996) keep the low digits of a long
  product as a rounding residue. These are the digits of a product of more
  than `p` digits below the rounding position.
  An addend far below the product joins the same residue, and the product
  digits replace it. An addend of the other sign is then lost in a directed
  rounding.
- decimal64 `fma(-9999999999999999E+7, -9999999999999999E+7,
  -9999999999999999E+7)`, rounded toward negative: decNumber gives
  `9999999999999998E+30`. The exact value is
  `9.99999999999999799999990000000010000001E+45`, so the correct result is
  `9999999999999997E+30`. Python's `decimal` module and floaty give
  `9999999999999997E+30`.
- The decTest 2.62 `ddFMA` and `dqFMA` vectors do not contain such a case.
  Each random run of 30,000 triples finds 16 to 20 of them.

## Resolution

floaty follows IEEE 754. `floaty_verify::decnumber::Arithmetic::fma_wide`
runs `decNumberFMA` at `2p + 3` digits, where no product digit is below the
rounding position. It rounds with 05up, which keeps a later rounding to
fewer digits correct, and converts the result to the format. The decTest
vectors still run through decNumber's `decDoubleFMA` and `decQuadFMA`.

## Regression evidence

- The unit test `works_around_the_fma_residue` in
  `floaty-verify/src/decnumber.rs` checks the example above.
- `floaty-verify/tests/decimal-decnumber/random.rs` compares 180,000
  fused multiply-add results for each of decimal64 and decimal128 with the
  wide oracle: 30,000 triples in six rounding directions.
