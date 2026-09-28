# floaty Design

Status: design approved on 2026-09-27. Build steps 1, encoding, 2, rounding,
and 3, arithmetic, are complete.

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
- The storage layout is private. Step 6 of the build order chooses packed bits
  or pre-split fields for each format, by benchmark.
- `from_bits` ignores the storage bits above `W`. For example, TF32 in a
  `u32` ignores bits 19 to 31. `to_bits` returns those bits as zero.
- `Float::PRECISION`, `Float::EMAX`, and `Float::EMIN` give the precision and
  the IEEE 754 exponent range of the format.
- `classify` returns a `Class`: `Zero`, `Subnormal`, `Normal`, `Infinite`,
  `QuietNan`, `SignalingNan`, or `Unsupported`. The class does not include
  the sign. `is_canonical` implements IEEE 754 `isCanonical`: an encoding is
  canonical when encoding its decoded value gives the same bits.
- `decode::<N>()` returns the exact value as `Decoded<N>`: a zero, a finite
  value `significand * 2^exponent`, an infinity, a NaN with its sign and
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

### Double-Double Family

`DoubleDouble<Alg>` at `W = 128` stores a value as the sum of two binary64
values, `hi + lo`. No standard defines its correct results. Different
published algorithms give different low halves for the same inputs. Each
variant therefore matches one reference implementation bit for bit.

| Variant | Reference |
| --- | --- |
| `Gcc` | libgcc `config/rs6000/ibm-ldouble.c`: `__gcc_qadd`, `__gcc_qsub`, `__gcc_qmul`, and `__gcc_qdiv`. This is the IBM `long double` of PowerPC. |
| `Qd` | The QD library `dd_real`, with its accurate IEEE-style addition and accurate division, and its multiply and square root. |

Pin the GCC version, the QD version, and the QD configuration when step 8
starts. The engine builds on the floaty binary64 format, so it inherits
bit-exact behavior. One value can have more than one bit pattern, for example
with the sign of a zero low half.

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
}

#[non_exhaustive]
pub struct NanRule {
    pub propagation: NanPropagation, // SignalingFirst, FirstOperand, LargerSignificand, or DefaultNan
    pub default_negative: bool,      // the sign of the default NaN
    pub invalid_product: InvalidProduct, // Signals or YieldsToNan: fma 0 * inf + NaN
}
```

`Env`, `NanRule`, `InvalidProduct`, `Tininess`, `Override`, and `Mode` live in the module
`floaty::env`. The crate root re-exports `Env`, `Flags`, `Rounding`, and the
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
  builder methods `with_default_negative` and `with_invalid_product`.
- `invalid_product` decides a fused multiply-add whose product is the invalid
  `0 * inf` and whose addend is a NaN. IEEE 754-2019, section 7.2 (c), lets
  the implementation decide whether a quiet NaN addend signals invalid there.
  `Signals`, the default, signals invalid and makes the product a default NaN
  that meets the addend under the propagation rule, as SoftFloat does. With
  `SignalingFirst` this gives the result of the Arm `FPMulAdd` pseudocode.
  `YieldsToNan` gives the NaN addend precedence and signals invalid only for
  a signaling addend, as x86 does. The rule is a field of its own, not a part
  of a propagation rule, because implementations combine the two
  independently.
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
pub trait Mode: Sealed + 'static {
    const ENV: Env;
}
```

A mode names one constant `Env`, so a type can carry a default behavior.
The modes live in the module `floaty::mode`, because the encoding markers
`Ieee` and `X87` already use those names at the crate root. `mode::Ieee` is
the default mode. It rounds to nearest even, does not flush, and detects
tininess after rounding. Its NaN rule, chosen in step 2, is
`NanRule::new(NanPropagation::SignalingFirst)`: a signaling NaN before a
quiet NaN, then the earlier operand, and a positive default NaN. A conversion keeps the high-order payload bits.
SoftFloat's ARM-VFPv2 specialization follows the same rule, so TestFloat
checks the default mode directly.

### Operations and Overrides

```rust
type F32 = Float<Binary<8>, 32>; // default mode: mode::Ieee

a + b                                // the type's mode; flags dropped
a.add_with(b, Rounding::TowardZero)  // override the rounding only; returns (value, Flags)
a.add_with(b, env)                   // replace the whole Env
a.add_with(b, F32::ENV)              // no override; returns the flags
a.with_mode::<Other>()               // same bits, new default mode, no cost
```

- An operation rounds with the mode of its result type, unless the call
  overrides it. A conversion uses the mode of the destination type.
- The operands of `a + b` must have the same type. Mixed modes do not compile.
- Only the `_with` methods return flags. Operators drop them.
- A bare `Rounding` override keeps every other field of the type's `Env`.
  AVX-512 embedded rounding, the RISC-V rounding field of an instruction, and
  the x87 constant loads change only the rounding direction.
- The engine is generic over the format only. A mode reaches the engine as a
  constant `Env` argument. When the `Env` is constant, the compiler removes
  the branches that it does not use. Every mode shares one compiled engine for
  each format.

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
| Default NaN of an invalid operation | Negative QNaN, the real indefinite | Negative QNaN, the floating-point indefinite |
| Denormal-input flag | MXCSR DE | Status word DE |

Every preset field is a claim about hardware. Cite the vendor manual beside
the field. In the Intel SDM Volume 1, revision 253665-093US, the NaN rules are
in Table 4-8 on page 4-17, and tininess is in section 4.9.1.5 on page 4-23.
Step 5 confirms every row on hardware.

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
    pub exponent: i32,         // value = significand * 2^exponent: the weight of the lowest bit
    pub significand: [u64; N], // any width; the rounding routine finds the top bit
    pub sticky: bool,          // the true magnitude is above the value by less than one lowest bit
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
- Contract: when `sticky` is set, `significand` must have at least `p + 2`
  significant bits, where `p` is the target precision. With fewer bits the
  true value is not known well enough to round correctly. The rule comes from
  the round-to-odd property. A debug assertion checks it.
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
- `mul_add` selects a NaN in SoftFloat's order: the NaN of the two factors
  first, then the NaN of that result and the addend. With `FirstOperand` the
  order gives the first NaN of the three operands, as x86 does. The
  `invalid_product` field of the NaN rule decides `0 * inf + NaN`. TestFloat
  checks `Signals` with every propagation rule. The processor checks
  `YieldsToNan` with `FirstOperand`. Real ARM hardware takes the addend
  first among NaN operands; an ARM preset needs that order.
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
| Add and subtract | Both operands exact in twice the storage width, or the lower operand shifted right with its lost bits in the lowest bit |
| Multiply | The exact `2p`-bit product |
| Divide and square root | `p + 2` bits, with a nonzero remainder as the sticky bit |
| Fused multiply-add | The exact product, and the addend aligned with a sticky bit |
| Conversion | The exact source significand |

The engine keeps these values in a limb array of twice the storage width.
Stable Rust cannot name `[u64; 2 * N]` for a generic `N`, so a sealed table,
`Widen`, names the double width of each limb count from 1 to 8. The design
first proposed separate high and low halves; the table keeps one value type
for the rounding routine. An aligned sum keeps both operands exact when the
double width holds them. Otherwise the lower operand is at least four times
smaller, and it shifts right with its lost bits in the lowest bit, which stays
far below the rounding position. Division and square root compute bit by bit;
step 6 decides faster algorithms.

## Operations

| Group | Operations | Decisions at the operation level |
| --- | --- | --- |
| Rounded | `add`, `sub`, `mul`, `div`, `sqrt`, `fma`, `scalb`, float-to-float conversion, round to integer | The NaN to return. The sign of an exact zero. Whether round to integer signals inexact: IEEE 754 has both variants. |
| Exact | IEEE `rem`, `neg`, `abs`, `copysign`, classification, `next_up`, `next_down` | Never rounds. The sign operations never signal. |
| Compare | Quiet and signaling predicates, `total_order` | Which comparisons signal invalid for a NaN. |
| Minimum and maximum | IEEE 754-2019 `minimum`, `maximum`, `minimumNumber`, `maximumNumber`, and IEEE 754-2008 `minNum`, `maxNum` | floaty ships all three families. ARM `FMINNM` uses the 2008 rule. |
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

### NaN Payloads in Conversions

A conversion keeps the high-order payload bits, as x86 and ARM do. A
narrowing conversion drops the low-order bits. A conversion quiets the NaN,
so the result stays a NaN. An encoding with one NaN pattern loses the
payload. A canonical `NanRule` ignores payloads, as RISC-V does.

## Engine and Performance

- The engine computes on `[u64; N]` limbs. A widening multiply returns a
  `[u64; 2 * N]` value that the sealed `Widen` table names for each `N` from 1
  to 8. A generic `[u64; 2 * N]` type needs the unstable `generic_const_exprs`
  feature, and the table avoids it.
- Start with one generic path. For `N` of 1 or 2 the compiler unrolls the
  loops. Specialize a format only when a benchmark shows a gap.
- Step 6 evaluates these fast paths:
  - Host `f32` and `f64` arithmetic for `NearestEven` when the caller does not
    want flags, with NaN results fixed in software.
  - An exact computation in a wider host format and one software rounding.
    For example, a binary16 product is exact in binary32.
  - Lookup tables for FP8.
- A fast path must pass the same oracle tests as the generic path, and this
  file must list it.
- Step 6 also decides the storage layout of each format.

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
| binary16, binary32, binary64, x87 extended, binary128 | Berkeley TestFloat and SoftFloat Release 3e, as git submodules. The ARM-VFPv2 NaN specialization matches the default mode. | Every conversion between these formats at TestFloat level 2, in every rounding direction including round to odd, with both tininess rules: result bits and the five IEEE flags. Later steps add arithmetic, integer conversions, and x87 precision control at 32, 64, and 80 bits. |
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
| Conversion of every binary16 encoding to the FP8 formats, rounding to nearest even | A table that `ml_dtypes` generates, in `floaty-verify/data` | Result bits, and NaN and sign for a NaN result |
| x86 SSE and x87 presets | The host processor, through inline assembly on x86-64 | NaN selection, the denormal-input flag, FTZ, DAZ, x87 C1, and precision control |
| Decimal | The decTest vectors (DPD) and the Intel decimal library tests (BID) | Arithmetic, rounding, flags, and result exponents |
| Double-double | libgcc on PowerPC under QEMU, and QD | Bit-exact match to each reference |

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
  takes about 40 seconds.
- `ml_dtypes` gives every NaN a canonical payload, so its conversion table
  checks only NaN and sign for a NaN result.
- Test every FP8 input pair for every operation and rounding direction in the
  normal test run.
- Test every 16-bit input pair as an ignored test that runs on a schedule.
  One binary operation in one direction has about 4.3 billion pairs.
- `floaty-verify` holds the whole harness. It is not a default workspace
  member, so a plain build never needs a C toolchain.

## Workspace Layout

```text
floaty/                  the workspace
├── floaty/              the crate: no_std, no dependencies
│   └── src/
│       ├── float.rs     Float, Class, and the format aliases
│       ├── format.rs    Standard, Binary<E, Enc>, the width-to-storage table
│       ├── env.rs       Env, Rounding, NanRule, Flags, modes, presets
│       ├── limbs.rs     [u64; N] arithmetic, Int<BITS>, UInt<BITS>
│       ├── exact.rs     Exact<N> and the rounding routine
│       ├── binary.rs    unpack, pack, and every binary operation
│       ├── decimal.rs   step 7
│       └── double_double.rs  step 8
└── floaty-verify/       TestFloat, MPFR, and hardware harness
    ├── build.rs         builds testfloat_gen from the submodules with make
    ├── reference/       Berkeley SoftFloat and TestFloat, git submodules
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

- The final names of the aliases, the `Rounding` directions, and the
  `NanPropagation` variants.
- The extra IBM POWER decimal rounding modes, and decimal widths above 128.
- The details of the `Unsigned` and `Finite` encodings.
- Presets for ARM, RISC-V, and Direct3D.
- An optional layer that carries flags on values through a computation.
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
