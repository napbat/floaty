# decNumber Refuses a Remainder with a Long Quotient

Status: resolved. The decNumber tests exclude the cases that decNumber marks
`Division_impossible`.

## Affected rule

`remainder(x, y)`, IEEE 754-2019 section 5.3.1: `x - y * n`, where `n` is the
integer nearest `x / y`, with ties to even. The result is always exact, for
every size of `n`.

## Conflicting reference evidence

- decNumber 3.68 gives a quiet NaN and signals `Division_impossible`, an
  invalid operation, when `n` needs more than `p` digits. The division
  routine of the fixed-size formats does so in `decBasic.c`, lines 591 to
  595, and `decNumberRemainderNear` does the same.
- decTest 2.62 expects that result: `ddRemainderNear.decTest` lines 341,
  467, 468, and 603 to 606, and the matching `dqRemainderNear` lines. An
  example is `1E+384 remaindernear 1`.
- Python's `decimal` module signals `InvalidOperation` for the same cases.
- The Intel Decimal Floating-Point Math Library 2.0 Update 2 gives the exact
  remainder for every quotient. For example, line 13533 of `readtest.in`
  divides a decimal128 value near `5.7E-1222` by one near `2.2E-6159`, a
  quotient of about 4,900 digits. floaty
  agrees with the library on every `rem` line of `readtest.in` and on random
  operands of each width.

## Resolution

floaty follows IEEE 754 and gives the exact remainder: `1E+384 remainder 1`
is `0`. `floaty-verify/tests/decimal-decnumber` skips the vectors and random
cases that decNumber marks `Division_impossible`. It identifies them by that
condition, not by floaty's result, and it asserts their count.

## Regression evidence

- `floaty-verify/tests/decimal-intel.rs` and
  `floaty-verify/tests/decimal-intel-random.rs` check the exact remainder
  against the Intel library.
- `floaty-verify/tests/decimal-decnumber` checks every other
  `remaindernear` vector and random case against decNumber.
