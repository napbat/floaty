# floaty

Bit-exact, platform-independent software floating point for Rust.

floaty emulates binary, decimal, and double-double floating-point formats,
and the behavior of the hardware that computes them. Every operation gives
the same bits and the same flags on every host. floaty is a base layer for
binary lifters, decompilers, constant folders, and FPU emulators.

- **Formats.** Every IEEE 754 binary interchange format from 16 to 512
  bits, bfloat16, TF32, OCP FP8 E4M3 and E5M2, the FNUZ FP8 variants, the
  OCP MX formats FP4 and FP6, x87 80-bit extended precision, and custom
  binary layouts up to 512 bits.
  decimal32, decimal64, and decimal128 in the BID and DPD encodings. Double-double values that match
  the IBM `long double` of libgcc or the `dd_real` of QD, bit for bit.
- **Correct rounding.** Addition, subtraction, multiplication, division,
  square root, fused multiply-add, and every conversion round correctly in
  eight directions: the five of IEEE 754, round to odd, and the two extra
  directions of IBM POWER decimal floating point.
- **The IEEE 754 operations.** Comparisons, total order, the IEEE 754-2019
  and IEEE 754-2008 minimum and maximum operations, the remainder, rounding
  to an integral value, integer conversions up to 512 bits, `scale_b`,
  `next_up` and `next_down`, the sign operations, and the decimal quantum
  operations.
- **Hardware behavior as data.** Flush-to-zero, denormals-are-zero, tininess
  detection, NaN propagation rules, x87 precision control, and saturation
  of overflows. Presets give the x86 SSE and x87 behavior.
- **Flags.** The five IEEE 754 flags, and `TINY`, `ROUNDED_UP`, and
  `DENORMAL_INPUT`, which emulators need.
- **Vector lanes.** `Lanes<T, N>` applies each operation to the lanes of a
  vector register and returns the union of the flags of the lanes.
- **Fast where possible.** Where the build targets a floating-point unit
  that gives the same bits, the operations without flags use it.
- **Small.** `no_std`, no `alloc`, no dependencies, and no `unsafe` code
  outside the host paths.

## Installation

floaty is not on crates.io. Add it as a Git dependency, and pin a revision:

```toml
[dependencies]
floaty = { git = "https://github.com/napbat/floaty", rev = "<commit>" }
```

floaty needs Rust 1.85 or later.

## Quick start

Operators round with the default mode of the type and drop the flags. Each
operation also has a `_with` form that takes a behavior and returns the
flags.

```rust
use floaty::{F32, Flags, Rounding};

let one = F32::from_bits(0x3F80_0000); // 1.0
let tiny = F32::from_bits(0x3380_0000); // 2^-24, half an ulp of 1.0

// The default mode rounds to nearest even, so the tie goes to 1.0.
assert_eq!((one + tiny).to_bits(), 0x3F80_0000);

// Override the rounding direction for one operation, and read the flags.
let (sum, flags) = one.add_with(tiny, Rounding::TowardPositive);
assert_eq!(sum.to_bits(), 0x3F80_0001);
assert_eq!(flags, Flags::INEXACT | Flags::ROUNDED_UP);
```

## Formats

A value has the type `Float<S, W, M>`: a standard `S`, a width of `W` bits,
and a default mode `M`. A value is only its bits, like a value in a
register. Type aliases name the common formats.

| Alias | Format | Precision |
| --- | --- | --- |
| `F16`, `F32`, `F64`, `F128` | IEEE 754 binary16, binary32, binary64, binary128 | 11, 24, 53, 113 bits |
| `F160` to `F512` | IEEE 754 binary formats in steps of 32 bits | 144 to 489 bits |
| `BF16` | bfloat16 | 8 bits |
| `TF32` | NVIDIA TensorFloat-32, 19 bits | 11 bits |
| `F8E4M3Fn`, `F8E5M2` | OCP FP8. E4M3 has no infinity, so LLVM and `ml_dtypes` call it E4M3FN. | 4, 3 bits |
| `F8E4M3Fnuz`, `F8E5M2Fnuz` | FP8 with one NaN and no negative zero | 4, 3 bits |
| `F4E2M1Fn`, `F6E2M3Fn`, `F6E3M2Fn` | OCP MX FP4 and FP6: no infinity and no NaN | 2, 4, 3 bits |
| `F80` | x87 extended precision, with an explicit integer bit | 64 bits |
| `D32Bid`, `D64Bid`, `D128Bid` | IEEE 754 decimal formats, BID encoding | 7, 16, 34 digits |
| `D32Dpd`, `D64Dpd`, `D128Dpd` | IEEE 754 decimal formats, DPD encoding | 7, 16, 34 digits |

`Binary<E, Enc>` describes other binary layouts: 2 to 28 exponent bits `E`,
at least one fraction bit, a width up to 512 bits, and an encoding of the
special values, `Ieee`, `NoInf`, `Fnuz`, `Finite`, or `X87`. An invalid
layout fails to compile.

```rust
use floaty::{Binary, Decoded, Float};

// An 8-bit format with 4 exponent bits and IEEE special values.
type Minifloat = Float<Binary<4>, 8>;

assert_eq!((Minifloat::PRECISION, Minifloat::EMAX, Minifloat::EMIN), (4, 7, -6));
let largest = Minifloat::from_bits(0x77).decode::<1>();
assert_eq!(largest, Decoded::Finite { negative: false, exponent: 4, significand: [15] });
```

An overflow in a format without an infinity gives the NaN when the format
has one. A saturating behavior gives the largest finite value instead, in
every binary format. The MX formats have neither an infinity nor a NaN, so
they always saturate, and an invalid operation gives `+0` with `INVALID`:

```rust
use floaty::{F4E2M1Fn, F8E4M3Fn, F8E5M2, Flags};

let largest = F8E4M3Fn::from_bits(0x7E); // 448
let (nan, flags) = largest.add_with(largest, F8E4M3Fn::ENV);
assert_eq!((nan.to_bits(), flags), (0x7F, Flags::OVERFLOW | Flags::INEXACT));

let (saturated, _) = largest.add_with(largest, F8E4M3Fn::ENV.with_saturate(true));
assert_eq!(saturated.to_bits(), 0x7E);

// E5M2 has an infinity, which saturation replaces with 57344.
let largest = F8E5M2::from_bits(0x7B);
let (saturated, flags) = largest.add_with(largest, F8E5M2::ENV.with_saturate(true));
assert_eq!((saturated.to_bits(), flags), (0x7B, Flags::OVERFLOW | Flags::INEXACT));

// FP4 E2M1: 6 + 6 saturates to 6, and 0 / 0 gives +0.
let (zero, six) = (F4E2M1Fn::from_bits(0x0), F4E2M1Fn::from_bits(0x7));
assert_eq!(six.add_with(six, F4E2M1Fn::ENV).0.to_bits(), 0x7);
assert_eq!(zero.div_with(zero, F4E2M1Fn::ENV), (zero, Flags::INVALID));
```

### Unsigned floats of R11G11B10

The packed format R11G11B10 of Direct3D, Vulkan, and OpenGL holds unsigned
floats of 11 and 10 bits, with 5 exponent bits and no sign bit. Their values
are the non-negative values of `Float<Binary<5>, 12>` and
`Float<Binary<5>, 11>`, so floaty builds them from those formats. The
example follows GL_EXT_packed_float as Mesa implements it: round to nearest
even, saturate, map a negative value to zero, and give one NaN.
`floaty-verify` checks it against Mesa for every rounding case.

```rust
use floaty::{Binary, Env, F32, Float};

type Eleven = Float<Binary<5>, 12>;

/// Returns the 11-bit channel of a binary32 value.
fn to_channel(value: F32) -> u16 {
    let (signed, _) = value.convert_with::<Eleven>(Env::IEEE.with_saturate(true));
    if signed.is_nan() {
        0x7C1
    } else if signed.is_sign_negative() {
        0
    } else {
        signed.to_bits()
    }
}

assert_eq!(to_channel(F32::from_bits(0x3F80_0000)), 0x3C0); // 1.0
assert_eq!(to_channel(F32::from_bits(0xBF80_0000)), 0); // -1.0 gives zero
assert_eq!(to_channel(F32::from_bits(0x4E6E_6B28)), 0x7BF); // 1e9 gives 65024
let one: F32 = Eleven::from_bits(0x3C0).convert();
assert_eq!(one.to_bits(), 0x3F80_0000);
```

## Behavior

`Env` holds every setting that changes the bits of a result:

| Field | Meaning |
| --- | --- |
| `rounding` | `TiesToEven`, `TiesToAway`, `TowardPositive`, `TowardNegative`, and `TowardZero` from IEEE 754; `ToOdd`; and `TiesTowardZero` and `AwayFromZero` from IBM POWER decimal |
| `flush_to_zero` | A tiny result becomes a zero (FTZ). |
| `denormals_are_zero` | A subnormal operand reads as a zero (DAZ). |
| `tininess` | Tininess detection before or after rounding. |
| `nan` | The NaN that an operation returns: the propagation rule, the sign of the default NaN, and the fused multiply-add rules. |
| `precision` | A precision limit below the format precision, as x87 precision control gives it. |
| `saturate` | A finite result that overflows gives the largest finite value instead of an infinity or a NaN, as OCP FP8 saturation does. Double-double arithmetic ignores it, because libgcc and QD do not saturate. |
| `total_order` | How `total_cmp` orders two encodings of one value. |

`Env::IEEE` is the IEEE 754 default. `Env::X86_SSE` and `Env::X87` give the
behavior of those units after reset: for example, the negative default NaN.
The builder methods change one field.

```rust
use floaty::{Env, F64, Flags};

let zero = F64::from_bits(0);
let (nan, flags) = zero.div_with(zero, F64::ENV);
assert_eq!((nan.to_bits(), flags), (0x7FF8_0000_0000_0000, Flags::INVALID));

// The SSE unit gives the negative QNaN floating-point indefinite.
let (nan, _) = zero.div_with(zero, Env::X86_SSE);
assert_eq!(nan.to_bits(), 0xFFF8_0000_0000_0000);

// With FTZ, a tiny result becomes a zero and signals underflow.
let smallest_normal = F64::from_bits(0x0010_0000_0000_0000);
let half = F64::from_bits(0x3FE0_0000_0000_0000);
let (flushed, flags) = smallest_normal.mul_with(half, Env::X86_SSE.with_flush_to_zero(true));
assert_eq!(flushed.to_bits(), 0);
assert_eq!(flags, Flags::UNDERFLOW | Flags::INEXACT | Flags::TINY);
```

An emulator builds an `Env` from the control register of each
instruction:

```rust
use floaty::{Env, F32, Flags, Rounding};

/// Runs `ADDSS` under an MXCSR value.
fn addss(left: F32, right: F32, mxcsr: u32) -> (F32, Flags) {
    let rounding = match (mxcsr >> 13) & 3 {
        0 => Rounding::TiesToEven,
        1 => Rounding::TowardNegative,
        2 => Rounding::TowardPositive,
        _ => Rounding::TowardZero,
    };
    let env = Env::X86_SSE
        .with_rounding(rounding)
        .with_flush_to_zero(mxcsr & (1 << 15) != 0)
        .with_denormals_are_zero(mxcsr & (1 << 6) != 0);
    left.add_with(right, env)
}

let (one, tiny) = (F32::from_bits(0x3F80_0000), F32::from_bits(0x3380_0000));
let (sum, _) = addss(one, tiny, 0x1F80 | (2 << 13));
assert_eq!(sum.to_bits(), 0x3F80_0001);
```

floaty models the rules of a unit, not the quirks of one instruction. For
example, `MINSS` returns its second operand for a NaN, and `CVTSS2SI`
returns `0x8000_0000` for an out-of-range value. An emulator builds such an
instruction from a comparison, or from the `ToInt` result.

### Modes

A mode is a behavior fixed at compile time. Each type carries a default
mode, `mode::Ieee` unless the type names another. The engine compiles the
common path of each operation once for each mode, with every field as a
constant. A combinator changes one field of a mode: `Rounded`,
`FlushToZero`, `DenormalsAreZero`, `Precision`, `FullPrecision`,
`Propagation` for the NaN propagation rule, and `DetectTininess`. An
emulator maps each state of a control register, such as MXCSR or the Arm
FPCR, to one mode.

```rust
use floaty::mode::direction::TowardZero;
use floaty::mode::switch::On;
use floaty::mode::{FlushToZero, Rounded, X86Sse};
use floaty::{Binary, Float, Rounding};

// MXCSR with round toward zero and FTZ set.
type Truncating = FlushToZero<Rounded<X86Sse, TowardZero>, On>;
type Double = Float<Binary<11>, 64, Truncating>;

assert_eq!(Double::ENV.rounding, Rounding::TowardZero);
assert!(Double::ENV.flush_to_zero);
```

A `_with` method takes a `Rounding`, which changes only the direction, an
`Env` chosen at run time, or a mode such as `mode::X86Sse`. `with_mode`
gives the same bits with another default mode, at no cost.

### Flags

| Flag | Meaning |
| --- | --- |
| `INVALID`, `DIVIDE_BY_ZERO`, `OVERFLOW`, `UNDERFLOW`, `INEXACT` | The IEEE 754 exceptions. |
| `TINY` | A rounded result or a remainder is tiny, even when exact. x86 needs this flag when the underflow exception is unmasked. |
| `ROUNDED_UP` | The magnitude of the result is larger than the magnitude of the exact value. x87 reports this in C1. |
| `DENORMAL_INPUT` | An operand was subnormal, before denormals-are-zero. x86 reports this as DE, and ARM as IDC. |

## Conversions and integers

`convert` rounds with the mode of the destination type. `to_int` and
`from_int` convert to and from the primitive integers, and to and from
`Int<BITS>` and `UInt<BITS>` of up to 512 bits.

```rust
use floaty::{BF16, D64Bid, Decoded, F32, F64, Flags, Rounding, ToInt};

let pi = F32::from_bits(0x4049_0FDB);
let rounded: BF16 = pi.convert();
assert_eq!(rounded.to_bits(), 0x4049);

assert_eq!(pi.to_int::<i32>(), ToInt::Value(3));
let three_hundred = F32::from_bits(0x4396_0000);
assert_eq!(three_hundred.to_int::<i8>(), ToInt::OutOfRange { negative: false });

// binary64 0.1 is not 1/10. Its conversion to decimal64 rounds.
let tenth = F64::from_bits(0x3FB9_9999_9999_999A);
let (decimal, flags) = tenth.convert_with::<D64Bid>(Rounding::TiesToEven);
let expected = Decoded::Finite {
    negative: false,
    exponent: -16,
    significand: [1_000_000_000_000_000],
};
assert_eq!((decimal.decode::<1>(), flags), (expected, Flags::INEXACT));
```

## Decimal formats

A decimal value keeps its exponent: 3.30 and 3.3 are different encodings of
one value. Every result takes the exponent that the IEEE 754 preferred
exponent rules give.

```rust
use floaty::{D64Bid, D64Dpd, Decoded, Exact, Rounding};

// Exact values round to a format. Here each price is exact.
let price = |cents: u64| {
    let exact = Exact { negative: false, exponent: -2, significand: [cents], sticky: false };
    D64Bid::round(exact, Rounding::TiesToEven).0
};
let total = price(110) + price(220);
let expected = Decoded::Finite { negative: false, exponent: -2, significand: [330] };
assert_eq!(total.decode::<1>(), expected);

// The same value in the DPD encoding.
let dpd: D64Dpd = total.convert();
assert_eq!(dpd.decode::<1>(), expected);
```

## Double-double

`DoubleDouble<Gcc>` matches the IBM `long double` of libgcc on PowerPC.
`DoubleDouble<Qd>` matches `dd_real` of QD 2.3.24 on x86-64. Each follows
the machine code of its reference, so the low halves, the NaN payloads, and
the flags match too.

```rust
use floaty::{DoubleDouble, F64, Gcc};

let one = DoubleDouble::<Gcc>::from_f64(F64::from_bits(0x3FF0_0000_0000_0000));
let ulp = DoubleDouble::<Gcc>::from_f64(F64::from_bits(0x3CA0_0000_0000_0000)); // 2^-53

// binary64 cannot hold 1 + 2^-53. The pair keeps it in the low half.
let sum = one + ulp;
assert_eq!(sum.hi().to_bits(), 0x3FF0_0000_0000_0000);
assert_eq!(sum.lo().to_bits(), 0x3CA0_0000_0000_0000);
let rounded: F64 = sum.convert();
assert_eq!(rounded.to_bits(), 0x3FF0_0000_0000_0000);
```

## Lanes

`Lanes<T, N>` holds the lanes of a vector register. Each lane gives the bits
of the scalar operation. A `_with` method returns the union of the flags of
the lanes, as a vector unit accumulates them.

```rust
use floaty::{Env, F32, Flags, Lanes};

let x = Lanes::<F32, 4>::from_bits([0x3F80_0000, 0x7F7F_FFFF, 0x4000_0000, 0x4040_0000]);
let y = Lanes::<F32, 4>::from_bits([0x3380_0000, 0x7F7F_FFFF, 0x4000_0000, 0x4040_0000]);
let (sum, flags) = x.add_with(y, Env::X86_SSE);
assert_eq!(sum.to_bits(), [0x3F80_0000, 0x7F80_0000, 0x4080_0000, 0x40C0_0000]);
assert_eq!(flags, Flags::OVERFLOW | Flags::INEXACT | Flags::ROUNDED_UP);
```

## Hardware acceleration

The engine computes every result in integer arithmetic. A host path
computes an operation on the floating-point unit of the host instead, only
where the unit gives the bits of the engine. The oracle tests check each
path.

- The build selects the host paths at compile time, from `target_arch` and
  `target_feature`. floaty does not detect the processor at run time.
  Enable more paths with `-C target-cpu` or `-C target-feature`, for
  example `RUSTFLAGS="-C target-cpu=x86-64-v3"`.
- Only the entry points without flags use a host path: the operators, and
  the methods that drop the flags of the default mode. The `_with` methods
  always run the engine.
- The mode must round to nearest even without FTZ, DAZ, saturation, or a
  precision limit below the format precision. `round_to_integral` also
  takes the other directions that its instructions encode.
- Each call reads MXCSR, the x87 control word, or FPCR. Another setting, or
  an unmasked exception, sends the operation to the engine. So does a NaN
  result: the engine selects the NaN by the rule of the mode.
- Build with `RUSTFLAGS="--cfg floaty_engine_only"` to remove every host
  path and all `unsafe` code.

| Host path | Target feature | Formats and operations | Also goes to the engine |
| --- | --- | --- | --- |
| SSE | x86-64 with SSE2 | binary32 and binary64: `+`, `-`, `*`, `/`, `sqrt`, `convert` between them, `to_int`, `from_int`, comparisons, and the minimum and maximum operations | The integer indefinite; an unordered comparison; a minimum or maximum of a NaN or of two zeros |
| SSE4.1 rounding | x86-64 with SSE4.1 | `round_to_integral` of binary32 and binary64, and through binary32 of bfloat16 and of binary16 with F16C, in `TiesToEven`, `TowardPositive`, `TowardNegative`, and `TowardZero` | |
| FMA | x86-64 with FMA | `mul_add` of binary32 and binary64 | |
| F16C | x86-64 with F16C | binary16 through binary32: `+`, `-`, `*`, `/`, `sqrt`, `to_int`, `from_int`, comparisons, and the minimum and maximum operations. `convert` from binary16 to binary32 and binary64, and from binary32 to binary16. | `mul_add`, and `convert` from binary64, which two roundings can get wrong |
| bfloat16 | x86-64 with SSE2, or AArch64 | bfloat16 widened by a shift: `convert` to binary32 and binary64, `to_int`, `round_to_integral` with SSE4.1 on x86-64, comparisons, and the minimum and maximum operations | `from_int`, and the arithmetic without `FEAT_BF16` |
| x87 | x86-64 | x87 extended: `+`, `-`, `*`, `/`, `sqrt`, `round_to_integral` to nearest even, `to_int`, `from_int`, and `convert` to and from binary32 and binary64 | A control word other than round to nearest at 64-bit precision; the integer indefinite |
| x87 remainder | x86-64 | `remainder` of binary32, binary64, and x87 extended, by `FPREM1` | A dividend exponent more than 630 above the divisor exponent; a subnormal dividend, or a divisor small enough to give a subnormal result |
| Packed SSE and AVX | x86-64 with SSE2, and AVX for 256 bits | `Lanes` of binary32 and binary64: the operators, `sqrt`, `mul_add` with FMA, `round_to_integral` with SSE4.1, `convert` between them, `compare_quiet`, and the minimum and maximum operations. binary16 lanes with F16C, and bfloat16 `round_to_integral` and `convert`. | A NaN in any lane, or for the minimum and maximum a NaN or two zeros in any pair of lanes, sends each lane to its scalar path |
| AArch64 | AArch64 | binary32, binary64, and binary16: `+`, `-`, `*`, `/`, `sqrt`, `convert`, `to_int`, `from_int`, `round_to_integral` in the five IEEE 754 directions, comparisons, and the minimum and maximum operations. `mul_add` of binary32 and binary64. | FPCR with a nonzero `RMode`, FZ, FZ16, FIZ, AH, AHP, or a trap enable; a saturated integer |
| `FEAT_FP16` | AArch64 with `+fp16` | `mul_add` of binary16 | |
| `FEAT_BF16` | AArch64 with `+bf16` | bfloat16 through binary32: `+`, `-`, `*`, `/`, `sqrt`, and `convert` from binary32 | `mul_add` |
| Packed AArch64 | AArch64 | As the packed SSE paths, at 128 bits | As the packed SSE paths |
| Double-double | The binary64 paths of the build | The operators of `Gcc` and `Qd`, and `sqrt` of `Qd`, with one check of the environment for all steps | |

[docs/x86-64-acceleration.md](docs/x86-64-acceleration.md) lists the x86-64
instructions that could give more paths, and what blocks each one.

## Performance

Nanoseconds per operation on an Intel Core i9-9900K, from
`cargo bench -p floaty-verify --bench operations`: the median of 11 samples
of 1,024 operations on random normal operands. Figures change from run to
run and from host to host.

The engine, through the `_with` methods with an `Env` chosen at run time.
Each cell gives floaty, then `rustc_apfloat` 0.2.3 where it has the format.

| Format | add | mul | div | mul_add |
| --- | --- | --- | --- | --- |
| binary16 | 19.8 / 38.1 | 14.6 / 32.3 | 19.1 / 54.8 | 27.5 / 48.2 |
| bfloat16 | 19.9 / 37.3 | 15.1 / 31.7 | 18.5 / 73.7 | 27.2 / 49.3 |
| binary32 | 20.2 / 38.5 | 15.7 / 32.0 | 20.6 / 45.2 | 26.9 / 48.5 |
| binary64 | 20.5 / 38.3 | 17.4 / 32.2 | 35.8 / 202.2 | 25.8 / 48.0 |
| x87 extended | 27.6 / 36.5 | 19.5 / 31.1 | 56.1 / 238.8 | 38.5 / 50.0 |
| binary128 | 23.0 / 33.8 | 22.3 / 31.9 | 69.7 / 404.0 | 38.6 / 48.4 |
| binary256 | 42.8 | 42.9 | 127.8 | 66.6 |
| binary512 | 60.6 | 89.7 | 281.0 | 148.6 |
| decimal64, BID | 80.3 | 57.8 | 84.8 | 87.5 |
| decimal128, BID | 82.6 | 74.1 | 126.8 | 131.7 |
| decimal64, DPD | 105.5 | 77.8 | 111.5 | 111.6 |
| decimal128, DPD | 119.3 | 114.5 | 163.3 | 175.6 |

The entry points without flags, which take a host path where the build has
one. A `Lanes` figure is per lane.

| Operation | Engine | Default build | x86-64-v3 build | Host `f32` or `f64`, x86-64-v3 |
| --- | --- | --- | --- | --- |
| binary32 `+` | 20.2 | 1.9 | 1.9 | 0.8 |
| binary64 `/` | 35.8 | 1.9 | 2.0 | 0.9 |
| binary32 `mul_add` | 26.9 | 24.3 | 1.7 | 1.0 |
| binary16 `+` | 19.8 | 16.1 | 2.3 | |
| x87 extended `/` | 56.1 | 4.5 | 4.0 | |
| `Lanes<F32, 8>` `+` | 17.1 | 0.4 | 0.3 | |
| `Lanes<F16, 8>` `+` | 15.8 | 11.5 | 0.3 | |

## Verification

Every behavior has a test against an established reference. Where no
implementation exists, a test evaluates the published definition from
IEEE 754 or a vendor manual with MPFR or decNumber.

| Reference | What it checks |
| --- | --- |
| Berkeley TestFloat and SoftFloat 3e | Arithmetic, conversions, comparisons, the remainder, rounding to an integral value, and integer conversions of binary16, binary32, binary64, binary128, and x87 extended, in the six directions of TestFloat, under four NaN rules |
| MPFR, through `rug` | Rounding to every binary format up to 512 bits. The arithmetic and the other operations of the formats that TestFloat lacks. Conversions between those formats and the decimal formats. |
| `rustc_apfloat` 0.2.3 | Decoding and classification, `next_up` and `next_down`, the remainder, rounding to an integral value, integer conversions, and `scale_b` |
| `ml_dtypes` 0.6.0 | Every FP8, FP6, and FP4 encoding and operand pair, as generated tables |
| The host processor | The SSE and x87 presets under every MXCSR and control word state, the packed instructions and their flags, and every host path. The AArch64 tests run under QEMU 10.2.1. |
| decTest 2.62 and decNumber 3.68 | DPD vectors, and random decimal32, decimal64, and decimal128 operations, with FTZ, DAZ, and precision limits |
| Intel Decimal Floating-Point Math Library 2.0 Update 2 | The BID vectors of `readtest.in`, about 22 million random cases, and conversions to and from binary formats |
| libgcc of GCC 15.2.0, under QEMU `qemu-ppc64le` | `DoubleDouble<Gcc>`, and the PowerPC fused multiply-add NaN rules |
| QD 2.3.24 | `DoubleDouble<Qd>` |
| Mesa 25.2.0, `format_r11g11b10f.h` | The R11G11B10 recipe: every rounding case of both channels, and every channel code |

The normal test run tests every FP8, FP6, and FP4 operand pair of every
operation. Ignored sweeps test every binary16 and bfloat16 operand pair of
the host paths, every binary32 encoding, and all 318 million TestFloat
`mulAdd` cases.
A reference that disagrees with IEEE 754 or with the processor has a comment
beside its test with the evidence and the resolution.

## Platform support

| Target | Status |
| --- | --- |
| `x86_64-unknown-linux-gnu` | Tested, in the baseline build and with `-C target-cpu=x86-64-v3` |
| `aarch64-unknown-linux-gnu` | Tested under QEMU, without and with `+fp16,+bf16` |
| `i686-unknown-linux-gnu` | Tested under QEMU, as a 32-bit target |
| `s390x-unknown-linux-gnu` | Tested under QEMU, as a big-endian target |
| Other targets | The engine has no target-specific code. The gates do not test these targets. |

The minimum supported Rust version is 1.85. The crate uses edition 2024.

## Limitations

- No transcendental functions, such as `sin`, `exp`, and `log`.
- No decimal text parsing or printing.
- No `const fn` evaluation. The engine uses traits, which a `const fn` on
  stable Rust cannot call.
- No traps. floaty reports flags and never traps.
- Presets for x86 SSE and x87 only.
- No decimal formats wider than 128 bits. No hardware or major library
  uses them.
- QEMU stands in for POWER hardware in the `Gcc` tests.

## Development

The workspace has two packages:

| Package | Purpose |
| --- | --- |
| `floaty` | The library. `no_std`, no dependencies. |
| `floaty-verify` | The oracle tests and the benchmark. Not a default member. |

`floaty-verify` builds its C references on Linux x86-64 only. It needs the
submodules, a C and C++ toolchain, the PowerPC cross compiler, and QEMU.
The first build downloads the decimal and QD archives and two Mesa headers.
The AArch64 gates also need the AArch64 cross compiler.

```text
git submodule update --init
cargo test --workspace
cargo bench -p floaty-verify --bench operations
```

[AGENTS.md](AGENTS.md) lists the tools with their pinned releases, the
quality gates of each build, and the contributor rules.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
