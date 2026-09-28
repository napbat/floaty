# decNumber Orders NaN Payloads Wrongly in the Fixed-Size Formats

Status: resolved. The random total-order tests use decNumber's
arbitrary-precision `decNumberCompareTotal`.

## Affected rule

`totalOrder` of two NaNs of one kind and sign. floaty orders them by
payload, as the arbitrary-precision decNumber functions and Python's
`decimal` module do: the lower payload orders first for a positive NaN.
IEEE 754-2019 section 5.10 leaves the order of payloads to the
implementation.

## Conflicting reference evidence

- decNumber 3.68 `decFloatCompareTotal` (`decBasic.c`, lines 1727 to 1783)
  compares the payloads in groups of four digits. The inner loop stops at
  the first different digit, but the outer loop goes on to the next group,
  and a later difference replaces the order.
- `NaN6039` against `NaN74744532`: `decDoubleCompareTotal` orders
  `NaN6039` above. `decNumberCompareTotal` in `decNumber.c`, Python's
  `decimal` module, and floaty order it below, as the payload values do.
- `decFloatCompareTotalMag` (`decBasic.c`, line 1793) calls
  `decFloatCompareTotal` on the magnitudes, so it has the same fault.
- The decTest 2.62 vectors pass, because none of their NaN pairs differ in
  more than one group.

## Resolution

floaty keeps the order of the payload values.
`floaty_verify::decnumber::Arithmetic::total_order`
converts both operands to `decNumber` and calls `decNumberCompareTotal` or
`decNumberCompareTotalMag`. The decTest vectors still run through the
fixed-size functions.

## Regression evidence

- The unit test `orders_nan_payloads` in `floaty-verify/src/decnumber.rs`
  checks the example above against both decNumber functions.
- `floaty-verify/tests/decimal-decnumber/random.rs` compares the total order
  of random NaN pairs of decimal64 and decimal128 with the wide oracle.
