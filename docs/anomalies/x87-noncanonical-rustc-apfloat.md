# rustc_apfloat and the Processor Disagree on Non-Canonical x87 Encodings

Status: resolved. The processor is the reference for these encodings.

## Affected rule

The `X87` encoding of `Binary<15, X87>` at width 80: the class and the value
of an unnormal, a pseudo-NaN, a pseudo-infinity, and a pseudo-denormal.

## Conflicting reference evidence

- The Intel SDM Volume 1, revision 253665-093US, section 8.2.2 and Table 8-3
  on page 8-14, says that the 387 and later processors do not support
  pseudo-NaNs, pseudo-infinities, and unnormals. They raise an
  invalid-operation exception for these operands. The same section says that
  a pseudo-denormal operand is handled with the biased exponent 1, as a
  denormal. The conflict was checked in the text of the PDF of that revision.
- The `FXAM` instruction of the host processor reports these classes:
  Unsupported for an unnormal, a pseudo-NaN, and a pseudo-infinity, and
  Denormal for a pseudo-denormal.
- `rustc_apfloat` 0.2.3 (`src/ieee.rs`, the comment on
  `X87DoubleExtendedS::from_bits`) treats the first three encodings as NaNs
  and a pseudo-denormal as a normal value. It follows LLVM, not the processor.

## Resolution

floaty follows the processor. An unnormal, a pseudo-NaN, and a
pseudo-infinity have the class `Unsupported`. A pseudo-denormal has the class
`Subnormal`, it is not canonical, and it decodes to the value of the normal
encoding with exponent field 1. The `rustc_apfloat` comparison covers the
canonical x87 encodings only.

## Regression evidence

- `floaty-verify/tests/x87-hardware.rs` compares the class and the sign of
  every boundary encoding and 200,000 random encodings with `FXAM`. It also
  multiplies 10,000 random pseudo-denormals by 1.0 on the processor and checks
  that each one decodes to the value of the normal result.
- `floaty-verify/tests/classification.rs` (`x87_encodings`) checks
  `is_canonical` against the Table 8-3 rule for the same inputs, and compares
  the canonical encodings with `rustc_apfloat`.
- `floaty/src/binary/tests.rs` checks the decoded value of a pseudo-denormal
  and the class of each unsupported encoding.
