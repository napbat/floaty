# floaty Design

Status: design approved on 2026-09-27. The crate has no arithmetic yet.

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
pub struct Float<S: Standard<W>, const W: usize, M: Mode = Ieee> { /* private */ }
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
  fails a compile-time assertion.
- Each standard and width pair selects its storage and its engine through its
  trait implementation. This gives static dispatch without specialization.
- The storage layout is private. Step 6 of the build order chooses packed bits
  or pre-split fields for each format, by benchmark.
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
    pub precision: Option<u32>,   // x87 precision control
    pub saturate: bool,           // FP8 saturating overflow
}
```

- `Env` is `#[non_exhaustive]`. Build a new value from a preset with builder
  methods such as `with_rounding`. A new field then does not break callers.
- `Rounding` has six directions: `NearestEven`, `NearestAway`,
  `TowardPositive`, `TowardNegative`, `TowardZero`, and `ToOdd`. Decimal
  needs `NearestAway`. IBM POWER binary128 instructions and internal
  algorithms use `ToOdd`. The names are provisional.
- `NanRule` selects the NaN that an operation returns when an input is a NaN.
  It also sets the default NaN of an invalid operation: its sign and payload.
- `precision` rounds the significand to fewer bits and keeps the exponent
  range of the format, as x87 precision control does.
- `saturate` makes an overflow in an encoding without infinity give the
  largest finite value instead of a NaN.

### Modes

```rust
pub trait Mode: Sealed {
    const ENV: Env;
}
```

A mode names one constant `Env`, so a type can carry a default behavior.
`Ieee` is the default mode. It rounds to nearest even, does not flush, and
detects tininess after rounding. Its NaN rule is an open question.

### Operations and Overrides

```rust
type F32 = Float<Binary<8>, 32>; // default mode: Ieee

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
the field: the NaN rules are in the Intel SDM Volume 1, Table 4-7, and
tininess is in the Intel SDM Volume 1, section 4.9.1.5. Step 5 confirms every
row on hardware.

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
    pub exp: i32,      // value = sig * 2^exp; exp is the weight of the lowest bit
    pub sig: [u64; N], // any width; the rounding routine finds the top bit
    pub sticky: bool,  // the true magnitude is above sig * 2^exp by less than one lowest bit
}

impl<S: Standard<W>, const W: usize, M: Mode> Float<S, W, M> {
    pub fn round<const N: usize>(x: Exact<N>, o: impl Override) -> (Self, Flags);
}
```

- An integer `n` is `sig = n` and `exp = 0`. A product is the product of the
  significands, with the sum of the exponents. The caller never normalizes.
- `N` is generic. A 1,024-bit binary512 product and a 66-bit x87 constant use
  the same routine.
- Contract: when `sticky` is set, `sig` must have at least `p + 2`
  significant bits, where `p` is the target precision. With fewer bits the
  true value is not known well enough to round correctly. The rule comes from
  the round-to-odd property. A debug assertion checks it.
- `exp` is `i32`. The binary512 maximum exponent is 4,194,303. A product
  doubles it, and the result still fits.

### Intermediate Sizes

Each operation reduces its intermediate result before it rounds. Two
binary512 operands can have exponents millions apart, so a full aligned sum
can need millions of bits.

| Operation | Intermediate |
| --- | --- |
| Add and subtract | `p + 3` bits and a sticky bit |
| Multiply | The exact `2p`-bit product |
| Divide and square root | `p + 2` bits, with a nonzero remainder as the sticky bit |
| Fused multiply-add | The exact product, and the addend aligned with a sticky bit |
| Conversion | The exact source significand |

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

- The engine computes on `[u64; N]` limbs. A widening multiply returns the
  high and low halves as two `[u64; N]` values. This avoids a `[u64; 2 * N]`
  type, which needs the unstable `generic_const_exprs` feature.
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
established implementations, not a reference written for floaty.

| Scope | Oracle | Coverage |
| --- | --- | --- |
| binary16, binary32, binary64, x87 extended, binary128 | Berkeley TestFloat | Every rounding direction including round to odd, tininess before and after rounding, x87 precision control at 32, 64, and 80 bits, integer conversions, and flags |
| bfloat16, TF32, FP8, binary160 to binary512, and a second check of the formats above | MPFR, through the `rug` crate | Correct rounding at any precision and exponent range, with subnormals |
| x86 SSE and x87 presets | The host processor, through inline assembly on x86-64 | NaN selection, the denormal-input flag, FTZ, DAZ, x87 C1, and precision control |
| Decimal | The decTest vectors (DPD) and the Intel decimal library tests (BID) | Arithmetic, rounding, flags, and result exponents |
| Double-double | libgcc on PowerPC under QEMU, and QD | Bit-exact match to each reference |

- MPFR has no round to odd. The harness rounds toward zero and sets the lowest
  bit when the result is inexact. For ties away from zero the harness uses
  `mpfr_round_nearest_away`. FTZ and the FP8 overflow encodings are small
  wrappers around the correctly rounded MPFR result.
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
│       ├── format.rs    Standard, Binary<E, Enc>, the width-to-storage table
│       ├── env.rs       Env, Rounding, NanRule, Flags, modes, presets
│       ├── limbs.rs     [u64; N] arithmetic, Int<BITS>, UInt<BITS>
│       ├── exact.rs     Exact<N> and the rounding routine
│       ├── binary.rs    unpack, pack, and every binary operation
│       ├── decimal.rs   step 7
│       └── double_double.rs  step 8
└── floaty-verify/       TestFloat, MPFR, and hardware harness
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

- The NaN rule of the `Ieee` mode. IEEE 754 does not fix which NaN payload an
  operation returns. Decide in step 2.
- The final names of the aliases, the `Rounding` directions, and the
  `NanRule` variants.
- The extra IBM POWER decimal rounding modes, and decimal widths above 128.
- The details of the `Unsigned` and `Finite` encodings.
- Presets for ARM, RISC-V, and Direct3D.
- An optional layer that carries flags on values through a computation.

## Rust Notes

Sketches on stable Rust 1.98 confirmed these points during the design. Step 1
confirms them on Rust 1.85 through the minimum-version gate.

- A struct field can use the projected type `<S as Standard<W>>::Bits`.
- One generic `impl` of `Standard<W>` for `Binary<E, Enc>`, bounded by the
  width table, covers every width. An unsupported width or an exponent too
  wide for its width fails to compile.
- `a + b` with operands of different modes fails to compile.
- A widening multiply over `[u64; N]` that returns two halves needs no
  unstable features.
- A `const fn` can read an associated constant of a generic type and do
  const-generic limb arithmetic, but cannot call a trait method. This is why
  the engine is not `const`.
