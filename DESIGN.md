# floaty Design

Status: design approved on 2026-09-27. Build steps 1, encoding, 2, rounding,
3, arithmetic, 4, the other binary operations, 5, the x86 SSE and x87
presets, 6, performance, 7, decimal, and 8, double-double, are complete. Two
speedup passes of the binary engine followed step 8. The second made the
behavior a type.

This file is the source of truth for every design decision in floaty. Update
it in the same change that alters a decision.

## Purpose

floaty is a base layer for software floating point. It emulates
floating-point formats and the behavior of the hardware that uses them. Every
operation gives the same bits on every host.

floaty serves these consumers:

- Binary lifters and decompilers that evaluate or fold floating-point
  instructions. `napbat/dasm` is the first reference use case.
- Compilers that fold constants.
- Software FPU emulators.

floaty is not a fixed list of standards. It describes a format with
parameters. A new format is a new set of parameters, not a new engine.

## Goals

- Give bit-identical results on every host. The engine does not use host
  floating-point arithmetic, except in a fast path that the oracle tests prove
  identical.
- Round add, subtract, multiply, divide, square root, fused multiply-add, and
  every conversion correctly.
- Model platform behavior as data: rounding direction, flush-to-zero,
  denormals-are-zero, tininess detection, NaN rules, and x87 precision
  control.
- Support binary formats up to 512 bits, decimal formats in both encodings,
  and double-double.
- Emulate fast. The memory size of a value is not a constraint.

## Non-Goals

These items are out of scope now. Some can return later.

- Transcendental functions such as `sin`, `exp`, and `log`. Deferred.
- Decimal text parsing and printing. Deferred. Correct results for binary512
  extremes need large integers and an `alloc` feature.
- `const fn` evaluation. Stable Rust cannot call a trait method in a
  `const fn`, and the engine uses traits. Revisit later.
- Traps. floaty reports flags and never traps.
- Instruction-level quirks of one instruction set. Consumers build them from
  floaty primitives. See [What presets do not cover](#what-presets-do-not-cover).
- Formats, modes, or engines from outside the crate. Every trait is sealed.

## Constraints

- Rust edition 2024. The minimum supported Rust version is 1.85.
- The `floaty` crate is `no_std` and has no dependencies.
- The workspace denies warnings, Clippy `pedantic`, rustdoc `all`, and
  `missing_docs`.
- Every C reference library lives in `floaty-verify`.
- The repository is private.

## The Value Type

```rust
pub struct Float<S: Standard<W>, const W: usize, M: Mode = mode::Ieee> { /* private */ }
```

| Parameter | Meaning |
| --- | --- |
| `S` | The standard: a format family and its parameters. |
| `W` | The total width of the encoding, in bits. |
| `M` | The default mode: the behavior that operators use. |

A value is a format and its bits, like a value in a register. The mode is only
a default for operations. It does not change the value.

- An invalid combination does not compile. `Float<Binary<8>, 600>` fails,
  because no storage exists for width 600. An exponent too wide for its width
  fails a compile-time assertion when the compiler generates code for the
  format. `cargo check` does not report that assertion.
- Each standard and width pair selects its storage and its engine through its
  trait implementation. This gives static dispatch without specialization.
- The storage layout is private. Step 6 kept packed bits for every format,
  by benchmark; see [Engine and Performance](#engine-and-performance).
- `from_bits` ignores the storage bits above `W`. For example, TF32 in a
  `u32` ignores bits 19 to 31. `to_bits` returns those bits as zero.
- `Float::PRECISION`, `Float::EMAX`, and `Float::EMIN` give the precision and
  the IEEE 754 exponent range of the format.
- `classify` returns a `Class`: `Zero`, `Subnormal`, `Normal`, `Infinite`,
  `QuietNan`, `SignalingNan`, or `Unsupported`. The class does not include
  the sign. `is_canonical` implements IEEE 754 `isCanonical`: an encoding is
  canonical when encoding its decoded value gives the same bits.
- `decode::<N>()` returns the exact value as `Decoded<N>`: a zero, a finite
  value `significand * RADIX^exponent`, an infinity, a NaN with its sign and
  payload, or an unsupported encoding. `N` is the limb count of the
  significand. A consumer reads its operands with `decode` and rounds its
  result with `round`.
- A trait item marked `#[doc(hidden)]` is internal to the crate. It is not
  part of the API.
- Generic conversion is a method, `convert` or `cast`. A blanket
  `From<Float<A, _>> for Float<B, _>` overlaps `impl<T> From<T> for T`. A
  lossless pair can add a concrete `From` implementation.

## Formats

### Binary Family

`Binary<E, Enc = Ieee>` has a sign bit, `E` exponent bits, and `W - 1 - E`
fraction bits. `Enc` is the encoding. It sets the special values, the bias,
and the explicit integer bit. Encodings are sealed marker types, because
stable Rust has no enum const parameters.

| Encoding | Infinity | NaN | Negative zero | Bias | Formats |
| --- | --- | --- | --- | --- | --- |
| `Ieee` | Yes | Exponent all ones, fraction not zero | Yes | 2^(E-1) - 1 | IEEE 754 binary formats, bfloat16, TF32, OCP E5M2 |
| `NoInf` | No | Exponent and fraction all ones | Yes | 2^(E-1) - 1 | OCP E4M3 |
| `Fnuz` | No | One pattern: sign set, all other bits zero | No | 2^(E-1) | FNUZ E4M3 and E5M2 |
| `X87` | Yes | As `Ieee` | Yes | 16383 | x87 extended precision |
| `Unsigned` (later) | To be decided | To be decided | No sign bit | To be decided | R11G11B10 channels |
| `Finite` (later) | No | No | Yes | To be decided | fp4 and fp6 microscaling elements |

`X87` stores the integer bit explicitly, so its 64 fraction bits give 64 bits
of precision. The engine follows the 387 and later processors: an unnormal, a
pseudo-NaN, or a pseudo-infinity input is an invalid operand, and a
pseudo-denormal input is accepted.

Step 1 fixed these encoding details:

- A binary format has 2 to 28 exponent bits and at least one fraction bit.
  The limit of 28 keeps every exponent sum inside `i32`. `X87` exists only as
  `Binary<15, X87>` at width 80. Another layout fails at compile time.
- The NaN of `NoInf` and `Fnuz` is quiet, because those formats have no
  signaling NaN.
- An x87 unnormal, pseudo-NaN, or pseudo-infinity has the class
  `Unsupported`. An x87 pseudo-denormal has the class `Subnormal`, as the
  `FXAM` instruction reports it. Its value equals the normal value with
  exponent field 1, and it is not canonical.
- `Fnuz` has no negative zero, so the encoder writes a negative zero as
  positive zero.

The crate names common formats with type aliases. The alias names are
provisional.

| Alias | Type | Precision |
| --- | --- | --- |
| `F16`, `F32`, `F64`, `F128` | `Float<Binary<5>, 16>` to `Float<Binary<15>, 128>` | 11, 24, 53, 113 |
| `F160` to `F512` | Every IEEE width in steps of 32, below | See below |
| `BF16` | `Float<Binary<8>, 16>` | 8 |
| `TF32` | `Float<Binary<8>, 19>` | 11 |
| `F8E4M3` | `Float<Binary<4, NoInf>, 8>` | 4 |
| `F8E5M2` | `Float<Binary<5>, 8>` | 3 |
| `F8E4M3Fnuz` | `Float<Binary<4, Fnuz>, 8>` | 4 |
| `F8E5M2Fnuz` | `Float<Binary<5, Fnuz>, 8>` | 3 |
| `F80` | `Float<Binary<15, X87>, 80>` | 64 |

IEEE 754 defines binary formats for 16, 32, 64, and every multiple of 32 from
128. For those widths, the exponent width is `round(4 * log2(k)) - 13`.

| Format | Exponent bits | Precision | Maximum exponent |
| --- | --- | --- | --- |
| binary16 | 5 | 11 | 15 |
| binary32 | 8 | 24 | 127 |
| binary64 | 11 | 53 | 1023 |
| binary128 | 15 | 113 | 16383 |
| binary160 | 16 | 144 | 32767 |
| binary192 | 17 | 175 | 65535 |
| binary224 | 18 | 206 | 131071 |
| binary256 | 19 | 237 | 262143 |
| binary288 | 20 | 268 | 524287 |
| binary320 | 20 | 300 | 524287 |
| binary352 | 21 | 331 | 1048575 |
| binary384 | 21 | 363 | 1048575 |
| binary416 | 22 | 394 | 2097151 |
| binary448 | 22 | 426 | 2097151 |
| binary480 | 23 | 457 | 4194303 |
| binary512 | 23 | 489 | 4194303 |

### Width-to-Storage Table

Stable Rust cannot compute a type from a const parameter. A macro implements
`Storage` for `Width<W>` for every `W` from 1 to 512. The storage is the
smallest of `u8`, `u16`, `u32`, `u64`, and `u128`, then `[u64; N]`. The
integer types `Int<BITS>` and `UInt<BITS>` use the same table.

### Decimal Family

`Decimal<Enc>` covers the IEEE 754 formats decimal32, decimal64, and
decimal128. `Enc` is `Bid` or `Dpd`. The two encodings hold the same values
with different bits.

- BID, the binary integer decimal encoding, is the encoding of Intel's
  software library.
- DPD, the densely packed decimal encoding, is the encoding of IBM POWER
  hardware.

A decimal value can have more than one representation, for example 1.0 and
1.00. IEEE 754 fixes the exponent of every result through its preferred
exponent rules. floaty follows those rules exactly.

The decimal engine has its own exact value, a coefficient times a power of
10 plus a sticky bit, and its own rounding routine. Conversion between binary
and decimal formats needs its own correctly rounded path.

Step 7 fixed these rules:

- The widths are 32, 64, and 128 bits, with the parameters of IEEE 754-2019
  Table 3.6. Another width fails at compile time. The aliases are `D32Bid`,
  `D64Bid`, `D128Bid`, `D32Dpd`, `D64Dpd`, and `D128Dpd`. No encoding is the
  default.
- The API is radix-generic. `Float::RADIX` is 2 or 10. `PRECISION` counts
  digits of the radix, and `EMAX` and `EMIN` are exponents of the radix.
  `Decoded` and `Exact` hold `significand * RADIX^exponent`. A decimal
  `decode` keeps the exponent of the encoding, its quantum, so it tells 1.0
  from 1.00. `round` takes an exact value in the radix of the format.
- The decimal rounding routine takes a preferred exponent. An exact result
  is the member of its cohort whose exponent is nearest the preferred one.
  An inexact result has the least possible exponent, so it keeps every
  digit. An exponent can be above the largest exponent of a full
  coefficient. It then clamps down when the coefficient has room for
  trailing zeros, as IEEE 754 requires.
- The decimal rounding routine and the binary one share the choice of the
  direction. For a decimal result, `ToOdd` is IBM's round to prepare for
  shorter precision: it rounds toward zero and, when the result is inexact
  and its last digit is 0 or 5, adds one. That keeps a later rounding to
  fewer digits correct, as binary round to odd does for bits.
- The precision limit counts decimal digits for a decimal format. It moves
  only the rounding position. The result then takes its exponent by the
  rules of the format at its full precision, so an inexact result has
  trailing zeros. FTZ and DAZ treat a value below `10^EMIN` as subnormal.
- A BID coefficient above `10^p - 1` is non-canonical and reads as zero. A
  non-canonical DPD declet reads as its digit value. An infinity with a bit
  set below its five leading combination bits is non-canonical. A NaN with
  a bit set between its signaling bit and its trailing field, or with a
  payload above `10^(p - 1) - 1`, is non-canonical. `is_canonical` reports
  these encodings. An operation returns canonical encodings.
- `Decoded::Zero` has an `exponent` field: the quantum of a decimal zero,
  and 0 for a binary zero. The field changes the public enum.
- `quantize_with` signals no underflow or overflow, as IEEE 754-2019
  section 5.3.2 states. It reports no `TINY`, and FTZ does not apply. It
  reports `ROUNDED_UP` when its rounding increments the coefficient.
- `log_b_with`, the comparisons, and the sign operations report neither
  `TINY` nor `ROUNDED_UP`.
- The decimal operations are those of the binary family, and the decimal
  operations of IEEE 754: `quantize`, `same_quantum`, `quantum`, `log_b`,
  and `scale_b`, which scales by a power of 10. Comparison, `total_cmp`, and
  the minimum and maximum families order the members of one cohort as
  IEEE 754 and decTest do.
- Conversions go between the decimal formats, between BID and DPD, to and
  from integers, and to and from every binary format. A binary-to-decimal or
  decimal-to-binary conversion rounds correctly. It divides by a power of 5
  and shifts to reach a few digits or bits more than the precision, with a
  sticky bit. The numbers stay below 16,384 bits, although 10^6144 is about
  2^20410. A value far outside the decimal range rounds as a stand-in on the
  same side, which rounds alike in every direction.
- A decimal result is tiny when its exact value is below `10^EMIN`, before
  rounding, as IEEE 754 section 7.5 requires for decimal formats and
  decNumber does. The `tininess` field of `Env` applies only to binary
  formats.
- A conversion from a decimal format keeps the exponent of the source where
  the destination holds it, as IEEE 754 prefers. A conversion from a binary
  format prefers exponent 0, as the Intel library gives it: binary64 0.5
  converts to `5E-1`.
- A finite value divided by an infinity gives a zero with the least
  exponent, as decTest requires.
- `remainder_with` is exact at every quotient size, as IEEE 754 requires.
  decNumber reports `Division_impossible` when the integer quotient has
  more than `p` digits. floaty follows IEEE 754, as the Intel library does,
  and the decNumber tests of that case are excluded with this reason. See
  `docs/anomalies/decnumber-remainder-near-division-impossible.md`.
- `log_b` gives the exponent of the leading digit as a decimal value. Only
  the decimal formats have it: the binary `logB` returns an integer, which
  a consumer reads from `decode`.
- decTest's `half_down` and `up` rounding modes are not `Rounding`
  directions. They are part of the open question on the IBM POWER decimal
  rounding modes.

### Double-Double Family

`DoubleDouble<Alg, M>` stores a value as the sum of two binary64 values,
`hi + lo`. It is a type of its own, not a `Float` format. No standard
defines its correct results, and different published algorithms give
different low halves for the same inputs. Each algorithm therefore matches
one reference implementation bit for bit. The type has the operations that
its reference defines, and the operations on its exact value.

| Variant | Reference | Arithmetic |
| --- | --- | --- |
| `Gcc` | libgcc `config/rs6000/ibm-ldouble.c` of GCC 15.2.0, as its powerpc64le build compiles it: `__gcc_qadd`, `__gcc_qsub`, `__gcc_qmul`, and `__gcc_qdiv`. This is the IBM `long double` of PowerPC. | add, sub, mul, div |
| `Qd` | QD 2.3.24 `dd_real`, configured with `--enable-ieee-add`, `--disable-sloppy-div`, and `--enable-fma=c99`, and built by g++ 15.2.0 with `-O2 -ffp-contract=off` for x86-64 | add, sub, mul, div, sqrt |

Step 8 fixed these rules:

- Each algorithm is a fixed sequence of binary64 operations and
  comparisons. floaty runs each step on the floaty binary64 engine under
  the behavior of the call. The result has the union of the flags of the
  steps, as the hardware sets them when it runs the reference.
- With the behavior of the reference platform, the result and the five
  IEEE flags match the reference in every rounding direction. `TINY`,
  `ROUNDED_UP`, and `DENORMAL_INPUT` say that some step was tiny, rounded
  up, or read a subnormal operand. They do not describe the double-double
  result.
- The platform of `Gcc` is PowerPC, as QEMU's PowerPC target gives it. Its
  behavior has the `FirstOperand` NaN rule and a positive default NaN. It
  also has the `AddendSecond` fused NaN order, the `SignalsAndYieldsToNan`
  invalid product rule, and tininess before rounding. The platform of `Qd`
  is x86-64 with SSE, `Env::X86_SSE`.
- Each algorithm follows the machine code that the pinned compiler makes of
  its reference, not the source. The compiler orders the operands of each
  instruction, and the operand order decides which NaN a step returns.
  `build.rs` pins the machine code, so another compiler build stops the
  build and does not change the reference silently.
- `Gcc` follows `ibm-ldouble.o` of the pinned libgcc. GCC fuses
  `a*d + b*c` of `__gcc_qmul` into one fused multiply-add.
- A pair with a finite high half and a NaN low half is malformed. The
  libgcc routines combine such NaNs in fused multiply-adds. PowerPC takes
  their NaN in the order first factor, addend, second factor, which
  `AddendSecond` gives. `fmsub` selects a NaN operand before it negates the
  addend, so a NaN addend keeps its sign; the `Gcc` steps do the same. So
  `Gcc` matches libgcc for every pair, malformed pairs included.
- `Qd` follows the machine code that g++ 15.2.0 makes of QD for x86-64:
  addition, subtraction, and multiplication as the harness shim inlines
  them, `dd_real::accurate_div`, and `sqrt` of `dd_real.o`. The compiler
  swaps the operands of some additions, and x86 returns the NaN of the first
  operand. It also drops the error of a `quick_two_sum` whose low half is
  unused: in the second remainder of the division, and in the square root.
  So that step signals nothing.
- QD's own results stay: the square root of a zero is `+0`, and the square
  root of a negative value is QD's NaN,
  `(0x7FF8000000000000, 0x7FF8000000000000)`. One signaling comparison of
  the high half with zero decides both cases.
- `from_parts(hi, lo)` keeps any pair, and `from_f64(x)` is `(x, +0)`.
  `hi` and `lo` return the halves. Negation negates both halves, as GCC
  negates an `__ibm128` and QD negates a `dd_real`. `abs` negates both
  halves when the exact value has a negative sign. A pair whose high half
  has the other sign, such as `(-0, 1)`, keeps its halves. Neither signals.
- The operations on the exact value `hi + lo`: `decode`, conversion to every
  `Float` format rounded once, the quiet and signaling comparisons, and
  `PartialEq` and `PartialOrd`. A NaN high half makes the value that NaN,
  and an infinite high half makes the value that infinity. With a finite
  high half, an infinite or NaN low half makes the value that low half.
  Otherwise the value is the exact sum, and a zero sum takes the sign of
  the high half.
- One value can have more than one pair, for example with the sign of a
  zero low half. Arithmetic returns the pair that the reference returns.

## Behavior

### Env

`Env` is a small, immutable value. It holds every behavior that changes the
bits of a result.

```rust
#[non_exhaustive]
pub struct Env {
    pub rounding: Rounding,       // one of six directions
    pub flush_to_zero: bool,      // FTZ: a tiny result becomes zero
    pub denormals_are_zero: bool, // DAZ: a subnormal input reads as zero
    pub tininess: Tininess,       // BeforeRounding or AfterRounding
    pub nan: NanRule,             // which NaN an operation returns
    pub precision: Option<NonZeroU32>, // x87 precision control
    pub saturate: bool,           // FP8 saturating overflow
    pub total_order: TotalOrder,  // Datum or Encoding: two encodings of one datum
}

#[non_exhaustive]
pub struct NanRule {
    pub propagation: NanPropagation, // SignalingFirst, FirstOperand, LargerSignificand, or DefaultNan
    pub default_negative: bool,      // the sign of the default NaN
    pub invalid_product: InvalidProduct, // Signals, YieldsToNan, or SignalsAndYieldsToNan: fma 0 * inf + NaN
    pub fused_order: FusedNanOrder,  // ProductFirst, AddendFirst, or AddendSecond: the NaN operands of fma
}
```

`Env`, `NanRule`, `InvalidProduct`, `FusedNanOrder`, `TotalOrder`,
`Tininess`, `Behavior`, `Override`, and `Mode` live in the module
`floaty::env`. The
crate root re-exports `Env`, `Flags`, `Rounding`, `TotalOrder`, and the
module `mode`.

- `Env` is `#[non_exhaustive]`. Build a new value from `Env::IEEE` or a
  preset with builder methods such as `with_rounding`. A new field then does
  not break callers.
- `Flags` is a bit set. It has `contains`, `is_empty`, `union`, and
  `difference`, and it combines with `|`.
- `Rounding` has six directions: `NearestEven`, `NearestAway`,
  `TowardPositive`, `TowardNegative`, `TowardZero`, and `ToOdd`. Decimal
  needs `NearestAway`. IBM POWER binary128 instructions and internal
  algorithms use `ToOdd`. The names are provisional.
- `NanRule` selects the NaN that an operation returns when an input is a NaN,
  made quiet. It also sets the sign of the default NaN of an invalid
  operation. The default NaN is quiet and has a zero payload. `NanRule` is
  `#[non_exhaustive]`: build it with `NanRule::new(propagation)` and the
  builder methods `with_default_negative`, `with_invalid_product`, and
  `with_fused_order`.
- `invalid_product` decides a fused multiply-add whose product is the invalid
  `0 * inf` and whose addend is a NaN. IEEE 754-2019, section 7.2 (c), lets
  the implementation decide whether a quiet NaN addend signals invalid there.
  `Signals`, the default, signals invalid and makes the product a default NaN
  that meets the addend under the propagation rule, as SoftFloat does. With
  `SignalingFirst` this gives the result of the Arm `FPMulAdd` pseudocode.
  `YieldsToNan` gives the NaN addend precedence and signals invalid only for
  a signaling addend, as x86 does. `SignalsAndYieldsToNan` signals invalid
  and gives the NaN addend precedence, as QEMU's PowerPC target does for
  `fmadd` and `fmsub`. The rule is a field of its own, not a part of a
  propagation rule, because implementations combine the two independently.
- `fused_order` decides the order in which a fused multiply-add offers its
  NaN operands to the propagation rule. `ProductFirst`, the default, selects
  from the two factors, makes that NaN quiet, and then selects from it and
  the addend, as SoftFloat does. With `FirstOperand` it gives the first NaN
  of the three operands, as x86 does. `AddendFirst` selects from the addend,
  the first factor, and the second factor, with no quieting between them.
  With `SignalingFirst` it gives the result of the Arm `FPMulAdd`
  pseudocode, which passes its operands to `FPProcessNaNs3` in that order.
  QEMU's Arm target uses the same order. `AddendSecond` selects from the
  first factor, the addend, and the second factor. With `FirstOperand` it
  gives the result of QEMU's PowerPC target for `fmadd` and `fmsub`. IEEE
  754-2019 section 6.2.3 does not say which input NaN gives the payload.
  The orders differ for a quiet NaN factor with a quiet NaN addend.
- `total_order` decides how `total_cmp` orders two encodings of one datum:
  an x87 pseudo-denormal and the normal encoding of its value, or a
  non-canonical decimal encoding and its canonical twin. `Datum`, the
  default, makes them equal, as the note in IEEE 754-2019 section 5.10
  states. `Encoding` orders them by their bits, so two different encodings
  are never equal. No processor has a total-order instruction, so every
  preset keeps `Datum`. `TotalOrder` is an `Override`:
  `a.total_cmp_with(b, TotalOrder::Encoding)`.
- `precision` rounds the significand to fewer bits and keeps the exponent
  range of the format, as x87 precision control does. The format then acts
  as a `p`-bit format with the same exponent range: a tiny result rounds at
  the quantum `2^(emin - p + 1)`. SoftFloat rounds x87 results with
  precision control the same way. `NonZeroU32` makes a zero precision
  impossible to write.
- FTZ flushes a result that is tiny by the tininess rule to a zero with its
  sign, and reports `UNDERFLOW`, `INEXACT`, and `TINY`. The SSE unit reports
  the same flags, and the SSE hardware test checks them.
- An overflow of `NoInf` to its all-ones significand, the NaN pattern, gives
  the largest finite value when the direction does not round to infinity. So
  `ToOdd` gives an even result there, for example 448 for 452 in OCP E4M3,
  with `OVERFLOW` and `INEXACT`. The format has no odd value between 448 and
  the NaN.
- `saturate` makes an overflow in an encoding without infinity give the
  largest finite value instead of a NaN.

### Modes

```rust
pub trait Behavior: Sealed + Copy { fn env(self) -> Env; } // an Env, or a mode

pub trait Mode: Sealed + Copy + Default + 'static {
    const ENV: Env;
}
```

A mode names one constant `Env`, so a type can carry a default behavior. A
mode is a type of size zero, so its value `M::default()` is also a behavior
that a call can pass.
The modes live in the module `floaty::mode`, because the encoding markers
`Ieee` and `X87` already use those names at the crate root. `mode::Ieee` is
the default mode. It rounds to nearest even, does not flush, and detects
tininess after rounding. Its NaN rule, chosen in step 2, is
`NanRule::new(NanPropagation::SignalingFirst)`: a signaling NaN before a
quiet NaN, then the earlier operand, and a positive default NaN. A conversion keeps the high-order payload bits.
SoftFloat's ARM-VFPv2 specialization follows the same rule, so TestFloat
checks the default mode directly.

A combinator makes a mode from another mode and one changed field, at
compile time. The fields that processors change at run time each have one:

| Combinator | Field | Processor control |
| --- | --- | --- |
| `Rounded<M, R>` | `rounding`, from a type in `mode::direction` | MXCSR.RC, x87 RC, Arm FPCR.RMode |
| `FlushToZero<M, S>` | `flush_to_zero`, from `mode::switch::On` or `Off` | MXCSR.FTZ, Arm FPCR.FZ |
| `DenormalsAreZero<M, S>` | `denormals_are_zero` | MXCSR.DAZ |
| `Precision<M, DIGITS>` | `precision` | x87 PC |
| `FullPrecision<M>` | `precision` set to `None` | |

`Precision<M, 0>` fails to compile where a program uses the behavior of the
mode. A later combinator replaces the field that an earlier one set.

### Operations and Overrides

```rust
type F32 = Float<Binary<8>, 32>; // default mode: mode::Ieee

a + b                                // the type's mode, fixed at compile time; flags dropped
a.add_with(b, Rounding::TowardZero)  // override the rounding only: an Env; returns (value, Flags)
a.add_with(b, env)                   // replace the whole behavior with an Env
a.add_with(b, mode::X86Sse)          // replace it with a mode, fixed at compile time
a.add_with(b, F32::ENV)              // the type's behavior as an Env; returns the flags
a.with_mode::<Other>()               // same bits, new default mode, no cost
```

- An operation rounds with the mode of its result type, unless the call
  overrides it. A conversion uses the mode of the destination type.
- The operands of `a + b` must have the same type. Mixed modes do not compile.
- Only the `_with` methods return flags. Operators drop them.
- A bare `Rounding` override keeps every other field of the type's `Env`.
  AVX-512 embedded rounding, the RISC-V rounding field of an instruction, and
  the x87 constant loads change only the rounding direction.
- The engine is generic over the format and over the behavior. An `Env` is a
  behavior that the program chooses at run time, and the engine reads its
  fields. A mode is a behavior fixed at compile time. Each mode gets its own
  compiled copy of the common path of an operation, with every field of the
  mode as a constant. The rare paths, the special values and the tiny
  results, take the `Env` by reference for every behavior.
- A program changes the behavior in one of three ways:
  1. It passes an `Env` at run time. This way is always available.
  2. It maps a state that it reads at run time to one of a known set of
     modes, with a `match`. Each arm runs a copy of the code for its mode.
     An emulator does this for a control register: MXCSR has 16 states of
     rounding, FTZ, and DAZ, and the x87 control word has 12 states of
     rounding and precision control.
  3. It calls `with_mode` at a point fixed in the source.
- Stable Rust cannot use a struct as a const generic parameter. So a
  behavior is a type with an associated constant, and a combinator derives
  one mode from another. The unstable `adt_const_params` gives the same code
  and needs the incomplete `generic_const_exprs` to derive a mode.

### Flags

| Flag | Meaning |
| --- | --- |
| `INVALID` | IEEE invalid operation. |
| `DIVIDE_BY_ZERO` | IEEE division by zero. |
| `OVERFLOW` | IEEE overflow. |
| `UNDERFLOW` | IEEE underflow: the result is tiny and inexact. |
| `INEXACT` | IEEE inexact. |
| `TINY` | The result is tiny, even when exact. A consumer needs this flag to emulate x86 with the underflow exception unmasked. |
| `ROUNDED_UP` | The rounding incremented the significand. x87 reports this in status bit C1. |
| `DENORMAL_INPUT` | An input was subnormal. x86 reports this as DE, and ARM as IDC. |

### Presets

A preset is a named constant `Env` and a matching mode type. It records how
the arithmetic of one platform behaves. Start with x86 SSE and x87. Add ARM,
RISC-V, and Direct3D later.

A preset records two kinds of facts: rules that the hardware fixes, and the
reset values of control registers. A preset is a starting `Env`. A consumer
overrides fields for each instruction, for example the rounding field of
MXCSR.

| Property | x86 SSE | x87 |
| --- | --- | --- |
| Rounding at reset | `NearestEven` (MXCSR 0x1F80) | `NearestEven` (control word 0x037F after `FNINIT`) |
| FTZ and DAZ at reset | Off | Not available |
| Precision control at reset | Not available | 64 bits |
| Tininess | After rounding | After rounding |
| Two NaN inputs | The first source operand, quieted | A QNaN before an SNaN. With two of one kind, the larger significand. Quieted. |
| One NaN input | That NaN, quieted | That NaN, quieted |
| Default NaN of an invalid operation | Negative QNaN, the QNaN floating-point indefinite | Negative QNaN, the QNaN floating-point indefinite |
| `0 * inf + NaN` in a fused multiply-add | The NaN addend, made quiet; invalid only for a signaling addend (`YieldsToNan`) | No fused multiply-add; `YieldsToNan` by the same exception priority |
| Denormal-input flag | MXCSR DE | Status word DE |
| Round-up flag | Not available | Status word C1 |

Every preset field is a claim about hardware. Cite the vendor manual beside
the field. The source is the Intel SDM Volume 1, revision 253665-093US. The
NaN rules are in Table 4-8 on page 4-17, and tininess is in section 4.9.1.5
on page 4-23.
Step 5 confirms every row that the hardware can show. No x87 instruction
shows the fused multiply-add row.

```rust
impl Env {
    pub const X86_SSE: Self; // MXCSR 1F80H after reset
    pub const X87: Self;     // control word 037FH after FNINIT
}

pub mod mode {
    pub enum X86Sse {} // ENV = Env::X86_SSE
    pub enum X87 {}    // ENV = Env::X87
}
```

- `Env::X86_SSE` and `Env::X87` are the presets, and `mode::X86Sse` and
  `mode::X87` are their modes. Each field of a preset has its citation in a
  comment beside it in `env.rs`.
- A consumer overrides the fields of the control register for each
  instruction. For SSE they are the MXCSR rounding control, FTZ, and DAZ.
  For x87 they are the rounding and precision controls.
- `Env::X87` limits the precision to 64 bits, the reset value of the
  precision control. That is the full precision of the x87 format, so it
  changes no x87 result. Binary128 arithmetic under `Env::X87` rounds to 64
  bits, so the preset is for x87 values.
- The DE flag is `DENORMAL_INPUT` with two exceptions. A NaN operand, an
  invalid operation, or a division by zero comes first, by the exception
  priority of section 4.9.2. DAZ also clears DE, by section 10.2.3.4. Some
  instructions never report DE, such as `CVTSS2SI`, `ROUNDSS`, and `FISTP`.
  These mappings are the consumer's. `floaty-verify/src/x86.rs` has the
  mappings that the hardware tests use: `mxcsr_flags` for SSE and
  `x87_arithmetic_status` for x87.

#### What Presets Do Not Cover

Instruction-level behavior stays in the consumer. Examples:

- `MINSS` returns its second operand when an input is a NaN. A consumer
  builds it from a compare.
- `VCVTNEPS2BF16` ignores MXCSR, rounds to nearest even, and flushes
  denormals. A consumer passes a matching `Env`.
- `CVTSS2SI` returns 0x80000000 for an out-of-range input. A consumer maps
  `ToInt::OutOfRange` to that value.
- Estimate instructions such as `RCPSS` and the x87 transcendental results.

## Rounding Core

Every operation has the same shape:

```text
unpack -> special values -> exact core -> round(Exact, env) -> pack
```

- Special values never reach the rounding routine. Each operation handles
  NaN selection through `NanRule`, infinity arithmetic, DAZ, invalid
  operations, and the sign of an exact zero. For example, `x - x` is `+0`, or
  `-0` when rounding toward negative.
- The rounding routine handles everything that depends on the destination:
  precision, including x87 precision control, the subnormal range, tininess,
  FTZ, overflow, all six directions, and the flags. Overflow gives an
  infinity, the largest finite value, or a NaN for an encoding without
  infinity, unless `saturate` is set.

### Exact Values

The exact value is public, so a consumer can build a new instruction and
round its result.

```rust
pub struct Exact<const N: usize> {
    pub negative: bool,
    pub exponent: i32,         // value = significand * RADIX^exponent: the weight of the lowest digit
    pub significand: [u64; N], // a binary integer of any width; the rounding routine finds the top digit
    pub sticky: bool,          // the true magnitude is above the value by less than one lowest digit
}

impl<S: Standard<W>, const W: usize, M: Mode> Float<S, W, M> {
    pub fn round<const N: usize>(exact: Exact<N>, behavior: impl Override) -> (Self, Flags);
    pub fn convert<T: FloatType>(self) -> T;
    pub fn convert_with<T: FloatType>(self, behavior: impl Override) -> (T, Flags);
}
```

- The field names are `exponent` and `significand`, not `exp` and `sig`, to
  match `Decoded` and the naming rules in `AGENTS.md`.
- An integer `n` is `significand = n` and `exponent = 0`. A product is the
  product of the significands, with the sum of the exponents. The caller
  never normalizes.
- `N` is generic. A 1,024-bit binary512 product and a 66-bit x87 constant use
  the same routine.
- Contract: when `sticky` is set, a binary `significand` must have at least
  `p + 2` significant bits, where `p` is the target precision. With fewer
  bits the true value is not known well enough to round correctly. The rule
  comes from the round-to-odd property. A decimal `significand` must have
  more than `p` digits, because decimal tininess does not depend on the
  rounded value. A debug assertion checks the contract.
- `exponent` is `i32`. The binary512 maximum exponent is 4,194,303. A product
  doubles it, and the result still fits.

### Conversions

`convert` rounds with the mode of the destination type. `convert_with` takes
an override of that behavior and returns the flags. `FloatType` is the sealed
trait of every `Float` type.

- A subnormal input sets `DENORMAL_INPUT`. The flag means that an input was
  subnormal before DAZ: x86 reports DE when DAZ is off, and ARM reports IDC
  when it flushes an input. With DAZ set, the input reads as a zero with its
  sign.
- A signaling NaN input signals invalid. A NaN converts by the NaN rule.
  `DefaultNan` gives the default NaN. The other rules keep the sign and the
  high-order payload bits and set the quiet bit. A format with one NaN
  encoding gives that NaN.
- An unsupported x87 input signals invalid and gives the default NaN.
- An infinity converts to an infinity when the destination has one. Otherwise
  it converts to the NaN, or to the largest finite value with the same sign
  at the precision of the behavior when `saturate` is set, and signals
  invalid, as a conversion of an infinity to an integer does. No external
  reference gives flags for FP8 conversions, so these flags are a decision of
  this design.

### Arithmetic

`add_with`, `sub_with`, `mul_with`, `div_with`, `sqrt_with`, and
`mul_add_with` take an override and return the flags. The operators `+`,
`-`, `*`, and `/`, and `sqrt` and `mul_add`, use the default mode and drop the
flags. Each operation follows IEEE 754 for special values and rounds once.

- A NaN operand gives the NaN that the rule selects, made quiet. An
  unsupported x87 operand signals invalid and gives the default NaN, before
  any NaN operand, as the Intel SDM Volume 1, section 4.9.2, orders them.
- An exact zero sum of values with different signs is `+0`, or `-0` when
  rounding toward negative.
- A division by zero in a format without an infinity gives the NaN, or the
  largest finite value when `saturate` is set, and signals divide-by-zero.
- `mul_add` selects a NaN in the order of the `fused_order` field. The
  default, `ProductFirst`, is SoftFloat's order: the NaN of the two factors
  first, then the NaN of that result and the addend. With `FirstOperand` the
  order gives the first NaN of the three operands, as x86 does. The
  `invalid_product` field of the NaN rule decides `0 * inf + NaN`. TestFloat
  checks `Signals` with every propagation rule. The processor checks
  `YieldsToNan` with `FirstOperand`. Arm takes the addend first among NaN
  operands, which `AddendFirst` gives; an Arm preset needs that order.
  PowerPC takes the addend second, which `AddendSecond` gives, and QEMU
  checks it with `SignalsAndYieldsToNan`.
- x86 does not report DE for every subnormal operand. It reports DE only when
  DAZ is off and no NaN operand, invalid operation, or divide-by-zero occurs,
  by the precedence of the Intel SDM Volume 1, section 4.9.2. An x87 store
  never reports DE. `DENORMAL_INPUT` stays a fact about the operands; a
  consumer maps it to the DE flag of each instruction.
- An instruction maps its operands to the operation. For example,
  `VFMADD213SS x, y, z` computes `y * x + z` and takes NaNs in that order, so
  it is `y.mul_add_with(x, z, env)`.

### Intermediate Sizes

Each operation reduces its intermediate result before it rounds. Two
binary512 operands can have exponents millions apart, so a full aligned sum
can need millions of bits.

| Operation | Intermediate |
| --- | --- |
| Add and subtract | Both operands exact in the limb width of the storage, or the lower operand shifted right with its lost bits in the lowest bit |
| Multiply | The exact `2p`-bit product |
| Divide and square root | `p + 2` bits, with a nonzero remainder as the sticky bit |
| Fused multiply-add | The exact product, and the addend aligned with a sticky bit |
| Conversion | The exact source significand |
| Decimal operations | Coefficients in 512 bits, 154 digits. A product is exact, with at most `2p` digits. A sum keeps both terms exact, or cuts a term that is far smaller below the round digit with a digit that stands for the lost part: at most `2p + 2` digits. A square root scales its operand to at most `2p + 3` digits, 71 for decimal128 |
| Conversion between binary and decimal | Exact values in 1,024 bits when they fit, and in 16,384 bits otherwise: the value divided by a power of 5 and of 2 to a few digits more than the precision, with a sticky bit. The 16,384-bit arrays take tens of KiB of stack, which a `no_std` consumer with a small stack must allow for |

The engine keeps the product, quotient, and root in a limb array of twice
the storage width. Stable Rust cannot name `[u64; 2 * N]` for a generic `N`,
so a sealed table, `Widen`, names the double width of each limb count from 1
to 8. The design first proposed separate high and low halves; the table
keeps one value type for the rounding routine.

An aligned sum keeps both operands exact when its width holds them.
Otherwise the lower operand is at least four times smaller. It shifts right
with its lost bits in the lowest bit, which stays far below the rounding
position. That needs a width of each term plus 3 bits, and of the rounding
precision plus 5 bits. Addition uses the limb width of the storage when it
has that room, and twice that width otherwise. The fused multiply-add sums at
twice the width.

## Operations

| Group | Operations | Decisions at the operation level |
| --- | --- | --- |
| Rounded | `add`, `sub`, `mul`, `div`, `sqrt`, `mul_add`, `scale_b`, float-to-float conversion, `round_to_integral` | The NaN to return. The sign of an exact zero. Whether round to integer signals inexact: IEEE 754 has both variants, and floaty reports the flag for the consumer to keep or drop. |
| Exact | IEEE `remainder`, negation, `abs`, `copy_sign`, classification, `next_up`, `next_down` | Never rounds. The sign operations never signal. |
| Compare | `compare_quiet_with`, `compare_signaling_with`, `total_cmp` | Which comparisons signal invalid for a NaN. |
| Minimum and maximum | IEEE 754-2019 `minimum`, `maximum`, `minimum_number`, `maximum_number`, and IEEE 754-2008 `min_num`, `max_num` | floaty ships all three families. ARM `FMINNM` uses the 2008 rule. |
| Float to integer | Rounds with the `Env` | Out-of-range and NaN results. See below. |
| Integer to float | Rounds with the `Env` | None. |

### Integer Conversions

Integer conversions accept the Rust primitive integers and the sealed
`Int<BITS>` and `UInt<BITS>` types for every width from 1 to 512. The range
check uses the exact width, so `Int<24>` overflows at 2^23.

A float-to-integer conversion returns an enum and the flags. Each consumer
maps the enum to its instruction set.

```rust
pub enum ToInt<I> {
    Value(I),
    OutOfRange { negative: bool },
    Nan,
}
```

| Instruction set | Out of range | NaN |
| --- | --- | --- |
| x86 | 0x8000_0000, the integer indefinite | 0x8000_0000 |
| ARM | Saturate | 0 |
| RISC-V | Saturate | The largest value |

- `Int<BITS>` and `UInt<BITS>` live in `integer.rs`. Each stores its bits in
  the storage type of the width table, with the bits above `BITS` clear.
  `Int` uses two's complement. `from_bits` masks, and `to_bits` returns the
  bits. The sealed `Integer` trait covers these types and the primitives
  `i8` to `i128` and `u8` to `u128`. `isize` and `usize` are not integer
  types here, because their width depends on the host.
- `from_int` and `from_int_with` round like every other result, so the
  precision limit applies. A zero converts to `+0`.
- `to_int` and `to_int_with` round in the direction of the behavior.
  `mode::Ieee` rounds to nearest even, not toward zero as Rust `as` does. An
  infinity or a rounded value out of range gives `OutOfRange` with the sign
  of the float. A NaN or an unsupported encoding gives `Nan`. Both signal
  `INVALID` and not `INEXACT`.
- `to_int_with` rounds to an integer, not to the format, so the precision
  limit does not apply. SoftFloat's `extF80_to_*` conversions also ignore
  precision control.
- `ToInt` is not `#[non_exhaustive]`: its three cases are complete.

### Decisions for the Other Operations

Step 4 fixed these rules. Each keeps the neutral form: a consumer builds the
instruction-level behavior from the result and the flags.

- Every operation that takes a behavior applies DAZ to its inputs and reports
  `DENORMAL_INPUT` for a subnormal input. The sign operations and
  `total_cmp` take no behavior.
- `round_to_integral_with` and `to_int_with` report `INEXACT` whenever the
  result differs from the operand, as `roundToIntegralExact` and
  `convertToIntegerExact` do. The IEEE 754 operations that do not signal
  inexact are the same operations with `INEXACT` ignored, so floaty has no
  second method for them. They also report `ROUNDED_UP`.
- `round_to_integral_with` and `remainder_with` give results that the format
  holds exactly at its full precision, so the precision limit and
  flush-to-zero do not apply. One exception: in a format whose largest finite
  value is below `2^(p - 1)`, such as `Binary<3>` at width 8, a rounded
  integer can exceed that value, and it overflows by the usual rules.
  SoftFloat and the x87 `FRNDINT` and `FPREM1` instructions also ignore
  precision control. `scale_b_with` rounds, so every rounding field applies.
- `remainder_with` is the IEEE 754 `remainder`: the quotient rounds to
  nearest even, whatever the rounding direction. It is not the Rust `%`
  operator, which truncates, so `Float` does not implement `Rem`. A
  subnormal remainder reports `TINY`, as a subnormal result of a rounded
  operation does. The operations that return an operand or a neighbor of an
  operand, such as `minimum` and `next_up`, do not report `TINY`.
- `scale_b_with` takes an `i32` scale. A scale beyond `2^30` in magnitude
  acts as `2^30`, which overflows or underflows every format.
- `next_up_with` and `next_down_with` signal invalid only for a signaling
  NaN or an unsupported encoding. The step does not round, so the precision
  limit and flush-to-zero do not apply. In a format without an infinity, the
  value past the largest finite value is the NaN, or the largest finite
  value of the full precision when `saturate` is set.
- With DAZ, a minimum or maximum operation that selects a subnormal operand
  returns the zero that DAZ reads, because DAZ replaces the input before the
  operation.
- `compare_quiet_with` and `compare_signaling_with` return
  `Option<Ordering>`: `None` means unordered. The quiet form signals invalid
  for a signaling NaN, and the signaling form for every NaN. Every IEEE 754
  comparison predicate follows from the order and one of the two forms.
  `PartialEq` and `PartialOrd` on `Float` are the quiet predicates with the
  default mode; compare `to_bits` for equal encodings.
- `total_cmp` is IEEE 754 `totalOrder`. It orders by the sign bit and then
  the magnitude bits. The NaN of `Fnuz` orders first, as a negative NaN
  does. `total_cmp_with` takes a `TotalOrder` override.
- Under `TotalOrder::Datum`, the default, an x87 pseudo-denormal orders as
  the normal encoding of its value, so the two are equal. An unsupported x87
  encoding is no datum and orders by its bits under both rules. Step 4
  ordered every non-canonical x87 encoding by its bits; step 7 made that
  order `TotalOrder::Encoding` and the default the IEEE 754 rule, for both
  radices.
- A decimal `total_cmp` decodes both encodings. Members of one cohort order
  by exponent. Under `TotalOrder::Datum`, two encodings of one datum are
  equal, as the note in IEEE 754-2019 section 5.10 states: for example, a
  non-canonical infinity and the canonical one. decNumber and the Intel
  library agree. Under `TotalOrder::Encoding` they order by the bits below
  the sign, reversed for a negative sign.
- The minimum and maximum families return an operand in its canonical
  encoding, or a NaN that the NaN rule selects, and never round. Every
  family orders `-0` below `+0`, including `minNum` and `maxNum`, where IEEE
  754-2008 lets an implementation choose. `minNum` returns a NaN for a
  signaling NaN operand, and `minimumNumber` returns the number and signals
  invalid.
- `abs`, `copy_sign`, and negation change only the sign bit and signal
  nothing. The zero and the NaN of `Fnuz` each have one encoding, so they do
  not change.

### NaN Payloads in Conversions

A conversion keeps the high-order payload bits, as x86 and ARM do. A
narrowing conversion drops the low-order bits. A conversion quiets the NaN,
so the result stays a NaN. An encoding with one NaN pattern loses the
payload. A canonical `NanRule` ignores payloads, as RISC-V does.

The decimal formats follow the Intel decimal library:

- Between two decimal formats, the payload keeps its high-order digits. A
  widening conversion multiplies it by a power of 10.
- Between a binary and a decimal format, the payload bits align with the
  trailing significand field of the decimal format, and keep their
  high-order bits. The field holds the payload value as a binary integer of
  the field width, in DPD as in BID; the declet bits do not align. The
  Intel library confirms the rule for BID.
- A payload above `10^(p - 1) - 1` is non-canonical, and becomes zero.

## Engine and Performance

- The engine computes on `[u64; N]` limbs. A widening multiply returns a
  `[u64; 2 * N]` value that the sealed `Widen` table names for each `N` from 1
  to 8. A generic `[u64; 2 * N]` type needs the unstable `generic_const_exprs`
  feature, and the table avoids it.
- The engine has one generic path. Step 6 made its integer steps faster, and
  the speedup pass extended them:
  - An array of one limb computes on a native `u64`, and an array of two
    limbs on a native `u128`.
  - Division of values of at most 64 bits uses the native `u64` division, and
    of at most 128 bits the native `u128` division. Wider values use Knuth's
    Algorithm D, one quotient limb per step.
  - The integer square root of at most 64 bits uses `u64::isqrt`, and of at
    most 128 bits `u128::isqrt`. A wider root uses Newton's iteration from
    the root of the top 128 bits.
  - Addition works at the limb width of the storage when that width holds
    the precision plus 5 bits, which every named format does. Otherwise it
    works at twice that width.
- Step 6 kept packed bits as the storage of every binary format. On the
  benchmark host a binary64 `decode` takes 4 ns. A conversion, which decodes,
  rounds, and encodes, takes 19 ns, and `add_with` takes 38 ns. Pre-split
  fields would save the two decodes and the encode of an addition, at most a
  quarter of it. They also must keep the raw bits. An unsupported or
  non-canonical x87 encoding round-trips through `from_bits` and `to_bits`.
- The speedup pass cut the fixed cost of a rounded operation. It kept one
  arithmetic path and one rounding routine:
  - The binary `add`, `sub`, `mul`, `div`, `sqrt`, and `mul_add` read a
    normal operand directly from its encoding. A normal operand needs no
    special case, no denormal flag, and no DAZ. When every operand is normal,
    and positive for `sqrt`, the operation goes to the finite code that the
    general path also runs. Every other operand takes the full decode, and
    so does a `NoInf` number with the largest exponent field. The direct
    read saves most of the decode cost that pre-split fields would save,
    with no new storage form.
  - The rounding routine has a branch for a value whose top bit is at or
    above `emin`. The result is normal or overflows, and it is never tiny.
    So the branch skips the tininess rules, and it keeps the top `precision`
    bits of the value. The rest of the routine stays out of line, so that
    the common branch inlines into its callers.
  - A binary conversion to an integer of at most 64 bits rounds in two
    limbs, not nine. A value whose top bit reaches the width of the integer
    is out of range in every rounding direction, so the conversion stops
    before it rounds.
  - These branches are not fast paths: they run the code of the general
    path on fewer steps.
- A second pass specialized the common path for each format and each
  behavior:
  - Each binary format is a target type of the rounding routine,
    `exact::Format`. So each format has its own copy of the routine, with
    the precision, `emin`, and `emax` of the format as constants.
  - The rare path of each operation, the special values and the subnormal
    operands, stays out of line with `#[inline(never)]`. The common path is
    then short enough to inline into the operation.
  - The common path takes the behavior as a type, down to the rounding
    routine. A mode makes every field a constant there, and the precision
    limit matters most: without a limit, the rounding shifts are constant.
  - LLVM does not inline three shared functions for `#[inline]`: the
    rounding routine, its normal branch, and the sum of two terms. They carry
    `#[inline(always)]` with a narrow allowance of `clippy::inline_always`,
    and each one states the measured reason.
  - The sum aligns both terms to their common exponent, so its branch
    predicts well. An order of the terms by exponent branches on random
    data. The encoder sets the sign as a field of one bit, without a branch.
  - The decimal engine follows the same pattern. Each decimal format is a
    target type of the decimal rounding routine, `DecimalRoundingTarget`,
    and the operations carry the behavior as a type down to it. A mode gains
    less than 1% there: the decimal engine computes every format on 512-bit
    integers, and multiplication and digit counting of those integers take
    about half of the time of an operation.
- A fast path must pass the same oracle tests as the generic path, and this
  file must list it. There is one fast path.
  - The operators `+`, `-`, `*`, and `/` of binary32 and binary64 compute on
    the host `f32` and `f64` on x86-64 with SSE2. The mode must round to
    nearest even without FTZ, DAZ, or a precision limit below the format
    precision. MXCSR must also round to nearest even without FTZ or DAZ.
    Each call reads MXCSR with `STMXCSR`, because an emulator or a library
    built with `-ffast-math` can change it. That read is the one `unsafe`
    block of `floaty`. The block stores MXCSR in a stack slot of its own and
    loads the value into a register. A store into a slot of the frame of the
    caller made a later load of the frame wait on store forwarding, which
    doubled the time of an operator.
  - The operators return no flags. A NaN result goes back to the engine,
    which selects the NaN by the rule of the mode. The `_with` methods never
    take the path, because portable Rust cannot read the host flags. `sqrt`
    and `mul_add` are not in `core`, so they stay in the engine.
  - The TestFloat arithmetic tests check the operators in every run that
    rounds to nearest even with the NaN rule of a mode. The SSE hardware
    tests check them under the MXCSR value at reset. They also check that
    the operators give the engine result under FTZ, DAZ, and each directed
    rounding of MXCSR. A test checks that the operators of every format give
    the `_with` result of the default mode.
  - Other hosts use the engine until the oracle tests run there.
- Step 6 rejected two fast paths:
  - Lookup tables for FP8. A table takes 64 KiB for each operation, format,
    and behavior, and the engine computes an FP8 operation in 25 to 47 ns.
  - An exact computation in a wider host format and one software rounding.
    The software rounding is most of the cost, and the path would be a
    second arithmetic path for a small gain.
- `cargo bench -p floaty-verify --bench operations` measures the main
  operations of each format against the host types and `rustc_apfloat`.
  Results on an Intel i9-9900K, in nanoseconds per operation, before and
  after step 6:

| Operation | Before | After |
| --- | --- | --- |
| binary64 `+` operator | 45 | 1.8 |
| binary64 `add_with` | 45 | 38 |
| binary64 `div_with` | 302 | 47 |
| binary64 `sqrt` | 247 | 65 |
| binary128 `div_with` | 851 | 117 |
| binary128 `sqrt` | 767 | 316 |
| binary512 `add_with` | 159 | 92 |
| binary512 `div_with` | 9,092 | 406 |
| binary512 `sqrt` | 16,938 | 1,902 |

The speedup pass, before and after, on the same host:

| Operation | Before | After |
| --- | --- | --- |
| binary64 `add_with` | 39 | 28 |
| binary64 `mul_add` | 49 | 34 |
| binary64 `div_with` | 53 | 45 |
| binary64 `sqrt` | 71 | 63 |
| binary64 to `i64` | 23 | 14 |
| binary16 `*` operator | 32 | 21 |
| binary16 to binary64 | 22 | 17 |
| OCP FP8 E4M3 `+` operator | 37 | 28 |
| binary128 `add_with` | 51 | 35 |
| binary128 `sqrt` | 261 | 255 |
| binary512 `add_with` | 99 | 71 |

The specialization per format and behavior, before and after, in
nanoseconds per operation. Other work loaded the host, so each figure is the
lower of two interleaved runs; the `rustc_apfloat` rows did not change. The
`Env` column passes an `Env`, and the mode column passes the mode of the
type. Before the change, both took the path of an `Env`.

| Operation | Before | After, `Env` | After, mode |
| --- | --- | --- | --- |
| binary64 add | 29 | 22 | 16 |
| binary64 fused multiply-add | 36 | 26 | 25 |
| binary64 divide | 46 | 37 | 35 |
| binary64 square root | 65 | | 61 |
| binary64 round to integral | 18 | | 9 |
| binary64 to binary16 | 18 | | 9 |
| binary16 add | 31 | 20 | 16 |
| binary16 multiply | 22 | 15 | 13 |
| OCP FP8 E4M3 add | 29 | 19 | 16 |
| binary128 add | 36 | 25 | 20 |
| binary512 add | 76 | 65 | 56 |
| binary64 `+` operator, host path | 2 | | 2 |

The three forced functions, measured against `#[inline]` on the same code:
binary64 add with a mode takes 15.9 ns against 20.5 ns, round to integral
takes 8.9 ns against 16.6 ns, and binary128 add with a mode takes 19.7 ns
against 25.8 ns.

## Verification

A bit-exact claim needs an independent reference. floaty tests against
established implementations. Where no implementation exists, the test
evaluates the published definition, from IEEE 754 or a vendor manual, with an
established library such as MPFR. A reference written only for floaty is not
an oracle.

| Scope | Oracle | Coverage |
| --- | --- | --- |
| Decoding of binary16, bfloat16, TF32, binary32, binary64, binary128, OCP FP8, canonical x87 encodings, and custom layouts whose exponent field crosses a limb boundary | `rustc_apfloat`, the Rust port of LLVM APFloat | Class, sign, and exact value of every 8-, 16-, and 19-bit encoding, and of boundary and random wider encodings. Precision, `emax`, and `emin`. Every decoding test also checks the documented form of `Decoded` and the NaN payload that the format definition gives. |
| FP8, including the FNUZ variants | A table that `ml_dtypes` generates, in `floaty-verify/data` | Class, sign, and value of every encoding. Precision, `emax`, and `emin`. |
| x87 classification | The host processor, on x86-64: `FXAM`, and a multiply by 1.0 | Class and sign, including unsupported encodings and pseudo-denormals. The value of a pseudo-denormal. |
| x87 canonical encodings | The rule of the Intel SDM Volume 1 Table 8-3: the integer bit is set exactly when the exponent field is not zero | `is_canonical` for boundary and random encodings |
| binary160 to binary512, and a 200-bit layout whose exponent field crosses a limb boundary | The definition in IEEE 754-2019 section 3.4, evaluated exactly with MPFR. No established library decodes these widths. | Class, sign, and exact value of boundary and random encodings |
| binary32 and binary64 classification | The host `f32` and `f64` types | Random encodings, and every binary32 encoding in an ignored sweep |
| IEEE interchange parameters | The formulas of IEEE 754-2019 table 3.5 | Precision, `emax`, and `emin` of every IEEE width |
| binary16, binary32, binary64, x87 extended, binary128 | Berkeley TestFloat and SoftFloat Release 3e, as git submodules. The ARM-VFPv2 NaN specialization matches the default mode. | Every conversion between these formats at TestFloat level 2, in every rounding direction including round to odd, with both tininess rules: result bits and the five IEEE flags. |
| Rounding of every binary format: binary16 to binary512, bfloat16, TF32, the FP8 formats, and x87 with precision control at 24, 53, and 64 bits | MPFR, through the `rug` crate | Exact inputs of up to 500 bits near every boundary of the precision in use, in every direction, with both tininess rules, flush-to-zero, saturation, and precision limits: the value, the canonical form, and every flag. An ignored sweep rounds every significand below 2^8 at every exponent to each FP8 format. |
| Conversions from bfloat16, TF32, the FP8 formats, binary256, and binary512 | MPFR for the finite values, and the conversion rules of this design for the special values | Every encoding of the 8- and 16-bit sources, and boundary and random wider ones, in six behaviors |
| The `DefaultNan` rule | TestFloat with SoftFloat's ARM-VFPv2-defaultNaN specialization | Every TestFloat conversion and arithmetic case with that rule |
| Arithmetic of binary16, binary32, binary64, x87 extended, and binary128 | TestFloat | `add`, `sub`, `mul`, and `div` at level 1 and `sqrt` at level 2, in every direction with both tininess rules, and for x87 at precision control 32, 64, and 80. The first million level 1 `mulAdd` cases of each format in the normal run, and all 318 million in an ignored sweep. |
| Arithmetic of the FP8 formats, bfloat16, TF32, binary256, binary512, x87 extended with precision control, and a layout whose exponent field crosses a limb boundary | MPFR computes each result at 64 extra bits with a sticky bit; the special values follow IEEE 754 and this design | Every FP8 operand pair and operand for `add`, `sub`, `mul`, `div`, and `sqrt`, random `mul_add` triples, triples whose addend cancels the product, and every pair and triple of special values, in twelve behaviors, one of them with the x86 NaN rule and `YieldsToNan`. A NaN result must be quiet and must be a NaN operand, with its sign and payload, or the default NaN of an invalid operation. x87 arithmetic and `mul_add` at precision 24, 53, and 64 in every direction. |
| FP8 arithmetic, rounding to nearest even | A table that `ml_dtypes` generates: it computes in binary32 and rounds again, which gives the correctly rounded result at FP8 precision | Every operand pair for `add`, `sub`, `mul`, and `div`, and every operand for `sqrt`, with the `FirstOperand` rule: result bits, including the NaN of an overflow or a division by zero. `ml_dtypes` does not keep the sign of a NaN operand, so a result with a NaN operand only has to be a NaN. |
| The `LargerSignificand` and `FirstOperand` NaN rules | TestFloat with SoftFloat's 8086 and 8086-SSE specializations | Every TestFloat arithmetic case at the default behavior. The 8086-SSE run skips x87 extended precision, whose code there is the 8086 code. SoftFloat follows `InvalidProduct::Signals` for every rule. |
| SSE arithmetic | The host processor: `ADDSS`, `SUBSS`, `MULSS`, `DIVSS`, `SQRTSS`, their `SD` forms, and `VFMADD213SS` and `VFMADD213SD` | Every MXCSR setting: result bits and the IE, DE, ZE, OE, UE, and PE flags, with the `FirstOperand` rule, a negative default NaN, and `YieldsToNan` |
| x87 arithmetic | The host processor: `FADD`, `FSUB`, `FMUL`, `FDIV`, and `FSQRT` | Every rounding control at precision control 24, 53, and 64, including unsupported operands and pseudo-denormals: result bits, IE, DE, ZE, OE, UE, PE, and C1 |
| SSE conversions | The host processor: `CVTSD2SS` and `CVTSS2SD` under MXCSR | Every MXCSR rounding direction with FTZ and DAZ on and off, including exact, halfway, and near-halfway results at every binary32 boundary: result bits and the IE, DE, OE, UE, and PE flags |
| x87 stores | The host processor: `FLD` and `FSTP` to 64 and 32 bits | Every rounding control, including unsupported encodings, pseudo-denormals, and exact, halfway, and near-halfway results at every binary32 and binary64 boundary: result bits, IE, OE, UE, PE, and the C1 round-up bit. `FSTP` never reports DE. |
| Comparison, remainder, rounding to an integral value, and integer conversions of binary16, binary32, binary64, x87 extended, and binary128 | TestFloat | The six comparison predicates once, `rem` at level 1 in every direction, and `roundToInt` and the conversions to and from 32- and 64-bit integers at level 2 in every direction: result and the five IEEE flags. `rem` and `roundToInt` also run with the default-NaN, 8086, and 8086-SSE NaN rules, and for x87 at precision control 32 and 64, which both ignore. `-notexact` runs show that the non-exact IEEE operations are the same results with `INEXACT` ignored. The conversions to integers map `ToInt` to the ARM-VFPv2 values and to the x86 integer indefinite; the x86 runs cannot tell `Nan` from `OutOfRange`, and the ARM runs can. |
| The step 4 operations of the FP8 formats, bfloat16, TF32, x87 extended, binary256, binary512, and a layout whose exponent field crosses a limb boundary | MPFR, and the definitions of IEEE 754-2019 and this design for the special values and NaN selection | Every FP8 operand pair for the comparisons, `total_cmp` under both `TotalOrder` rules, the six minimum and maximum operations, and `remainder`, and every FP8 operand for `round_to_integral`, `to_int` into five integer types, `scale_b`, `next_up`, and `next_down`, in eight behaviors that use every direction, DAZ, FTZ, saturation, a precision limit, and every NaN rule. Boundary, special, and random operands of the wider formats, `to_int` into integers of 24 to 512 bits, and `from_int` from integers of 24 to 512 bits. |
| `next_up`, `next_down`, `remainder`, rounding to an integral value, integer conversions of 7 to 128 bits, and scaling | `rustc_apfloat` | Every FP8 operand and pair of E4M3 and E5M2, every binary16 and bfloat16 operand, and boundary and random operands of TF32, binary32, binary64, x87 extended, and binary128, in five directions. The differences of `rustc_apfloat` from IEEE 754 are in `docs/anomalies/` and in the module documentation of the test. |
| `next_up`, `next_down`, negation, `abs`, and `copy_sign` of the FP8 formats | A table that `ml_dtypes` generates, in `floaty-verify/data` | Every operand, and every pair for `copy_sign`. The FNUZ `copysign` of `ml_dtypes` makes a NaN from the zero; see `docs/anomalies/ml-dtypes-fnuz-copysign.md`. |
| SSE comparisons, integer conversions, and rounding | The host processor: `UCOMISS`, `COMISS`, their `SD` forms, `CVTSS2SI`, `CVTTSS2SI`, `CVTSD2SI`, `CVTTSD2SI`, `CVTSI2SS`, `CVTSI2SD`, `ROUNDSS`, and `ROUNDSD` | Every MXCSR setting: order, result bits, the integer indefinite for `ToInt::OutOfRange` and `ToInt::Nan`, and the IE, DE, and PE flags. The test maps DE and the suppressed precision exception of `ROUNDSS` as the Intel SDM states. |
| x87 comparisons, integer conversions, rounding, remainder, scaling, and sign operations | The host processor: `FUCOMIP`, `FCOMIP`, `FUCOMPP`, `FCOMPP`, `FRNDINT`, `FISTP`, `FISTTP`, `FILD`, `FPREM1`, `FSCALE`, `FCHS`, and `FABS` | Every rounding control, and precision control for `FRNDINT` and `FSCALE`, including unsupported operands and pseudo-denormals: result bits, IE, DE, OE, UE, PE, and C1. `FSCALE` and `FILD` ignore precision control, so the test runs them at the full 64-bit precision. |
| Conversion of every binary16 encoding to the FP8 formats, rounding to nearest even | A table that `ml_dtypes` generates, in `floaty-verify/data` | Result bits, and NaN and sign for a NaN result |
| The `AddendFirst` fused NaN order | The `FPMulAdd` and `FPProcessNaNs3` pseudocode of the Arm Architecture Reference Manual, evaluated in the test; QEMU's `pickNaNMulAdd` for Arm agrees | Every triple of ten special values of binary32, binary64, and decimal64, with the default-NaN mode off and on: the NaN result and the flags. A case without a NaN result gives the result of the default order. |
| `TotalOrder` of decimal encodings of one datum | decNumber's `canonical` and its arbitrary-precision total order | Random decimal64 and decimal128 encodings against their canonical twins and random operands, for the values and the magnitudes, under both rules |
| Modes and mode combinators | TestFloat, QD, decNumber, and the host processor | Every run of the ARM generator with its NaN rule, tininess after rounding, and no precision limit, through `Rounded<Ieee, R>` in its rounding direction: add, sub, mul, div, and square root of binary16, binary32, binary64, x87 extended, and binary128, and the 20 conversions between those formats. The fused multiply-add at nearest even in the normal run, and in every direction in the ignored sweep. `Qd` in four directions through `Rounded<X86Sse, R>`. The random decimal32, decimal64, and decimal128 add, subtract, multiply, and divide of the decNumber test in the six shared directions through `Rounded<Ieee, R>`. The SSE arithmetic in the 16 MXCSR states of rounding, FTZ, and DAZ, and the x87 arithmetic in the 12 control word states of rounding and precision control, each through the mode that a `match` selects for the state, as an emulator does. Each test also checks that the selected mode has the `Env` of its run. |
| x86 SSE and x87 presets | The host processor, through inline assembly on x86-64 | The MXCSR of the process and the control word after `FNINIT`, decoded to the preset fields. Products that only the tininess rule tells apart, in both units. NaN selection with one and two operands, the default NaN, the fused multiply-add NaN addend, DE, precision control, and C1. Every other SSE and x87 hardware test compares floaty under behaviors built from the presets. |
| Decimal, DPD | The decTest 2.62 vectors, and the decNumber 3.68 library | Every decimal64 and decimal128 vector of an operation that floaty has. That includes the `canonical` vectors, for `is_canonical` and a conversion to the same format, and the `apply` vectors of an explicit encoding, for `decode`, which the decimal32 vectors also give. Random operands at the edges of decimal32, decimal64, and decimal128 for every such operation, square root, the conversions between widths, FTZ, and DAZ, in the six shared rounding directions: result bits, the five IEEE flags, `TINY`, and `ROUNDED_UP`. Add, subtract, multiply, divide, fused multiply-add, square root, and `scale_b` of decimal64 and decimal128 under precision limits of 1, 2, 3, 7, and `p - 1` digits. |
| Decimal, BID, and conversions between decimal and binary32, binary64, x87 extended, and binary128 | The Intel Decimal Floating-Point Math Library 2.0 Update 2 and its `readtest.in` vectors | Every `readtest.in` vector of a function that floaty has: 65,300 operation lines and 41,505 conversion lines. About 22 million seeded random cases in the five directions of the library. They include exact square roots, values on both sides of every integer bound, binary ties at every sign of decimal exponent, and minimum and maximum operands that compare equal. The tests compare result bits with NaN payloads, the five IEEE flags, and the denormal flag of a conversion from binary. Each test asserts every rule and skip count. |
| Conversions between decimal and binary16, bfloat16, TF32, the FP8 formats, binary256, and binary512 | MPFR, which gives the correctly rounded leading digits of a binary value, and rounds a decimal value that GMP reduces to an integer and a sticky bit | Every encoding of the 8- and 16-bit sources. Boundary and random wide values, and 3,000 random binary256 and binary512 values in the range of each decimal format. Decimal values at the edges of each format, and NaNs with edge payloads. Seven behaviors each: the value, the decimal exponent, and every flag. A NaN result is quiet, with the sign of the source and the payload rule of this design. |
| The `AddendSecond` fused NaN order and the `SignalsAndYieldsToNan` invalid product rule | The PowerPC `fmadd` and `fmsub` instructions, run under QEMU 10.2.1 `qemu-ppc64le` in the libgcc batch program | Every triple of twelve binary64 values, the special values of the Arm test, the smallest subnormal, and the largest finite value, in four rounding directions under the PowerPC behavior: result bits and the five IEEE flags |
| Double-double `Gcc` | libgcc's `__gcc_qadd`, `__gcc_qsub`, `__gcc_qmul`, and `__gcc_qdiv` of the pinned powerpc64le GCC 15.2.0, run under QEMU 10.2.1 `qemu-ppc64le` in one batch process | Random pairs at the edges of binary64 in four rounding directions under the PowerPC behavior, malformed pairs included: both halves bit for bit and the five IEEE flags |
| Double-double `Qd` | QD 2.3.24, built by `build.rs` in the pinned configuration and called through a C++ shim | Random pairs at the edges of binary64 in four rounding directions under `Env::X86_SSE`: both halves bit for bit, with the payload and sign of a NaN half, and the five IEEE flags |
| The exact value of a double-double pair | MPFR, which adds the halves exactly | `decode`, conversion to binary16, bfloat16, binary32, x87 extended, binary128, decimal64, and decimal128 in five behaviors, and the quiet and signaling comparisons |

- `rustc_apfloat` follows LLVM, not the processor, for non-canonical x87
  encodings. The processor is the reference for those encodings. See
  `docs/anomalies/x87-noncanonical-rustc-apfloat.md`.
- MPFR has no round to odd, and `rug` does not expose MPFR's ties-away
  wrapper, which is a C macro. The harness takes both from the
  round-toward-zero and round-away-from-zero results: round to odd picks the
  odd one, and ties away picks by the midpoint. MPFR's subnormal
  emulation gives the subnormal quantum. FTZ, tininess, overflow, and the FP8
  overflow encodings are small wrappers around the MPFR results.
- TestFloat and SoftFloat after Release 3e add bfloat16 support that does not
  build with the ARM specialization. The submodules stay at Release 3e.
- SoftFloat's ARM specialization does not quiet a signaling x87 NaN. The x87
  tests correct the expected NaN; see
  `docs/anomalies/softfloat-arm-extf80-nan-quieting.md`.
- The workspace manifest optimizes `floaty-verify` in test builds, because
  the normal run makes tens of millions of oracle comparisons. The whole run
  takes about one and a half minutes.
- `ml_dtypes` gives every NaN a canonical payload, so its conversion table
  checks only NaN and sign for a NaN result.
- Test every FP8 input pair for every operation and rounding direction in the
  normal test run.
- Test every 16-bit input pair as an ignored test that runs on a schedule.
  One binary operation in one direction has about 4.3 billion pairs.
- `floaty-verify` holds the whole harness. It is not a default workspace
  member, so a plain build never needs a C toolchain.
- `build.rs` downloads the decTest, decNumber, and Intel archives once into
  `floaty-verify/reference/downloads/`. It checks the SHA-256 that it pins
  for each archive before it uses the archive. A change of archive is a
  design change.
- decNumber's decDouble and decQuad functions never report `Clamped`,
  `Rounded`, or `Subnormal`, as the decNumber manual states. The tests
  compare the five IEEE conditions.
- The Intel library builds with its rounding direction and status flags as
  function arguments, and without optimization: at `-O2`, `bid128_pow`
  does not return on line 11425 of `readtest.in`. It compiles with
  `-std=gnu99`. The build removes `CFLAGS`, `CPPFLAGS`, `LDFLAGS`, and `CC`
  from the environment of `make`, because the Intel makefile takes an
  exported `CFLAGS` in place of its own settings.
- decNumber and the Intel library both detect decimal tininess before
  rounding, as floaty does.
- `build.rs` builds the libgcc batch program with the cross compiler. It
  stops when the compiler is not GCC 15.2.0. It also stops when
  `ibm-ldouble.o` in the libgcc of the compiler does not have the pinned
  SHA-256. floaty's `Gcc` follows that object.
- `build.rs` and each QEMU test stop when `qemu-ppc64le` is not QEMU 10.2.1.
  The tests remove the `QEMU_*` variables that select another processor.
- QEMU's PowerPC target stands in for POWER hardware. No POWER processor
  confirms the `Gcc` results or the PowerPC fused multiply-add rules.
- `build.rs` downloads QD 2.3.24 with a pinned SHA-256, configures it with
  `--enable-ieee-add --disable-sloppy-div --enable-fma=c99` and the
  build-only `--disable-fortran --disable-shared --with-pic`, and checks
  `qd_config.h` for those settings. It stops when `g++` is not GCC 15.2.0,
  or when the code and constant sections of the shim object or of
  `dd_real.o` in `libqd.a` do not have the pinned SHA-256. floaty's `Qd`
  follows that machine code, so a change to the shim needs a new
  transcription and a new digest.
- QD's `fma` is the C library `fma`. glibc selects its implementation at
  run time. On a processor with FMA3 it is `vfmadd213sd`, which computes
  `fma(x, y, z)` as `y * x + z` and takes the first NaN in that order. So
  the fused steps of `Qd` pass `y` as the first factor. The `Qd` test first
  checks the `fma` of the host against floaty on every triple of six special
  values, and stops on another `fma`.
- The `Gcc` tests compare every pair, and count the malformed operands. The
  fused NaN order and the `fmsub` NaN rule decide those cases.
- The `Gcc` tests aim some operands so that a cross product of
  `__gcc_qmul` lands just below 2^-1022 while the main product stays normal.
  There, tininess before rounding decides the underflow flag.
- The C library of the references does not report the x86 denormal flag, so
  the double-double tests compare the five IEEE flags. QD's `sqrt` of a
  negative value signals nothing and writes an error to standard error.
- decNumber's fixed-size fused multiply-add can lose a small addend, and
  its fixed-size total order can order NaN payloads wrongly. The random
  tests use decNumber's arbitrary-precision functions for those operations.
  See `docs/anomalies/decnumber-fma-directed-residue.md` and
  `docs/anomalies/decnumber-comparetotal-nan-payload.md`.
- decNumber gives a quiet NaN addend for `0 * inf + NaN`, which is
  `InvalidProduct::YieldsToNan`, so its fused multiply-add tests use that
  rule. IEEE 754-2019 section 7.2 leaves the choice to the implementation.
- The Intel library selects NaNs with `FirstOperand` and `YieldsToNan`, and
  its fused multiply-add takes the NaN of `y`, then `z`, then `x`. The tests
  run the library's `fma(x, y, z)` as `y.mul_add_with(x, z, env)`. A NaN `x`
  and `z` with a number `y` stays different, because floaty takes the NaN
  of the factors first. The tests count those cases and skip them.
- The Intel library reports its denormal flag only for a conversion from a
  binary format, when the binary operand is subnormal or an x87
  pseudo-denormal. The tests compare it with `DENORMAL_INPUT` there. An
  invalid conversion to an integer gives the integer with only its top bit
  set.
- The Intel `minnum` and `maxnum` return the second operand for two zeros
  and for two members of one cohort. Where the operands compare equal and
  the library gives an equal value, the tests take the value and the flags
  from the library. They take the operand from the rule of this design,
  ordered by the library's `totalOrder`.
- floaty has no IEEE 754 `encodeDecimal` or `decodeDecimal`. A conversion
  between BID and DPD is `convertFormat`: it quiets a signaling NaN, signals
  invalid, and gives a canonical encoding. The Intel `bid_to_dpd` and
  `bid_dpd_to_bid` functions re-encode the bits and keep a signaling NaN.
  For a NaN, the tests take the sign and the payload from the library, and
  expect a quiet NaN and invalid for a signaling NaN.
- The Intel library's `quantum` and some of its decimal32 and decimal64
  fused multiply-adds are wrong. See `docs/anomalies/intel-decimal-quantum.md`
  and `docs/anomalies/intel-decimal-narrow-fma.md`.
- decNumber's width conversions keep a signaling NaN signaling, and a
  narrowing conversion keeps the low-order payload digits. The DPD width
  tests take the sign and payload from decNumber and apply floaty's payload
  rule. The Intel library checks the payload rule for BID.

## Workspace Layout

```text
floaty/                  the workspace
├── floaty/              the crate: no_std, no dependencies
│   └── src/
│       ├── float.rs     Float, Class, and the format aliases
│       ├── format.rs    Standard, Binary<E, Enc>, the width-to-storage table
│       ├── env.rs       Env, Rounding, NanRule, Flags, Behavior, presets
│       ├── env/mode.rs  the modes and the mode combinators
│       ├── limbs.rs     [u64; N] arithmetic
│       ├── integer.rs   Int<BITS>, UInt<BITS>, Integer, ToInt
│       ├── unpacked.rs  the decoded value that every engine computes on
│       ├── nan.rs       NaN selection and payloads, shared by both radices
│       ├── exact.rs     Exact<N>, the binary rounding routine, and the
│       │                direction choice that both radices share
│       ├── binary.rs    unpack, pack, and every binary operation
│       ├── decimal.rs   the BID and DPD codecs and every decimal operation
│       ├── radix.rs     exact arithmetic for binary and decimal conversions
│       └── double_double.rs  the double-double type, and the Gcc and Qd
│                        algorithms
└── floaty-verify/       TestFloat, MPFR, decimal, and hardware harness
    ├── build.rs         builds testfloat_gen from the submodules with make,
    │                    decNumber, the Intel decimal library, and QD from
    │                    pinned archives, and the libgcc batch program with
    │                    the pinned PowerPC cross compiler
    ├── reference/       Berkeley SoftFloat and TestFloat, git submodules,
    │                    and the ignored download cache `downloads/`
    ├── shim/            C wrappers of the Intel binary80 conversions, which
    │                    take a `long double` that Rust has no type for, the
    │                    libgcc batch program for QEMU, and the QD shim
    ├── data/            generated reference tables
    └── scripts/         the generators of the reference tables
```

## Build Order

Each step passes its oracle tests before the next step starts.

1. Encoding: limbs, the width table, and `Float<Binary<E, Enc>, W>` with
   encode, decode, and classify for every encoding. Test exhaustive round
   trips on small formats.
2. Rounding: the rounding routine, `Env`, and flags. Test through
   float-to-float conversion, which is only unpack and round, against
   TestFloat and MPFR.
3. Arithmetic: `add`, `sub`, `mul`, `div`, `sqrt`, and `fma`. Test against
   TestFloat and MPFR, and test FP8 exhaustively.
4. The other binary operations: compare, minimum and maximum, integer
   conversions, round to integer, and `rem`.
5. Presets: x86 SSE and x87. Test against the processor.
6. Performance: benchmarks, the storage layout decision, and fast paths.
7. Decimal: BID and DPD. Test against decTest and the Intel decimal library
   tests.
8. Double-double: `Gcc` and `Qd`.

## Open Questions

- decNumber rounds square root only to nearest. The oracle takes its root
  at `2p + 4` digits and rounds that to the format.
- No decimal library has FTZ or DAZ. The tests apply the definitions of
  this design to decNumber's results. DAZ replaces each subnormal operand
  by a zero with its sign and exponent. A result is tiny when decNumber's
  result rounded toward zero is subnormal, or is an inexact zero.
- `TINY` and `ROUNDED_UP` come from decNumber's result rounded toward zero.
  A result is rounded up when it is inexact and differs from that result.
- No library has a decimal precision limit. The test rounds decNumber's
  exact result once to the limit, with the exponent range of the format
  and no clamp. It then applies the representation rule of the precision
  limit. decNumber first computes at `2p + 4` digits with 05up, because
  `decNumberAdd` and `decNumberScaleB` do not round correctly when an
  operand has more digits than the context.
- decSingle has no arithmetic. The decimal32 random tests use decNumber's
  arbitrary-precision numbers in the decimal32 context. They leave out the
  copy operations, which give canonical encodings there. A decimal32 fused
  multiply-add with a special operand takes `decDoubleFMA` of the widened
  operands. See `docs/anomalies/decnumber-fma-invalid-product-nan.md`.
- The tests decide every skip from decNumber's class and string, not from
  floaty. They skip these vectors: the rounding modes `half_down` and `up`;
  the decNumber operations `abs`, `minus`, `plus`, `reduce`, `divideint`,
  `remainder`, `maxmag`, `minmag`, and `nexttoward`, and the logical ones;
  text conversions; null operands; `remaindernear` with
  `Division_impossible`; `scaleb` operands that are not integers with
  exponent 0 within `2 * (emax + p)`; and a fused multiply-add of two
  different signaling NaNs.
- The final names of the aliases, the `Rounding` directions, and the
  `NanPropagation` variants.
- The extra IBM POWER decimal rounding modes, and decimal widths above 128.
- The details of the `Unsigned` and `Finite` encodings.
- Presets for ARM, RISC-V, and Direct3D.
- An optional layer that carries flags on values through a computation.
- Decide whether a reciprocal square root estimate, as SoftFloat uses,
  replaces the integer square root and the wide division. The speedup pass
  left them slow: binary128 `sqrt` takes 255 ns.
- Decide whether each decimal format computes on its own working width, in
  place of 512 bits for every format: about 57 bits for decimal32, 117 bits
  for decimal64, and 236 bits for decimal128. A decimal64 addition runs about
  3,150 instructions, ten times a binary64 addition.
- Measure the specialization per format and behavior again on an idle host.
  Other work loaded the host during the measurement in this file.
- Add mode combinators for the NaN rule and the tininess rule with the Arm
  preset: Arm FPCR.DN switches the NaN rule at run time.
- Confirm the `Gcc` double-double results, and the PowerPC `fmadd` and
  `fmsub` NaN rules, on POWER hardware. QEMU stands in for it now.
- `saturate` applies only to encodings without an infinity. OCP FP8
  conversions and the x86 and ARM FP8 instructions also saturate E5M2, which
  has an infinity. Decide when a consumer needs those semantics.

## Rust Notes

Sketches on stable Rust 1.98 confirmed these points during the design. Step 1
confirms them on Rust 1.85 through the minimum-version gate.

- A struct field can use the projected type `<S as Standard<W>>::Bits`.
- One generic `impl` of `Standard<W>` for `Binary<E, Enc>`, bounded by the
  width table, covers every width. An unsupported width or an exponent too
  wide for its width fails to compile.
- `a + b` with operands of different modes fails to compile.
- A widening multiply over `[u64; N]` needs no unstable features when a
  table of trait implementations names the double-width type for each `N`.
- A `const fn` can read an associated constant of a generic type and do
  const-generic limb arithmetic, but cannot call a trait method. This is why
  the engine is not `const`.
