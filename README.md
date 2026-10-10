# floaty

Bit-exact, platform-independent software floating point for Rust.

floaty emulates binary, decimal, and double-double floating-point formats,
and the behavior of the hardware that computes them. Every operation gives
the same bits and the same flags on every host. floaty is a base layer for
binary lifters, decompilers, constant folders, and FPU emulators.

- **Formats.** Every IEEE 754 binary interchange format from 16 to 512
  bits, bfloat16, TF32, OCP FP8 E4M3 and E5M2, the FP8 variants of LLVM
  and `ml_dtypes`, the OCP MX formats FP4 and FP6, x87 80-bit extended
  precision, and custom
  binary layouts up to 512 bits.
  decimal32, decimal64, and decimal128 in the BID and DPD encodings. Double-double values that match
  the IBM `long double` of libgcc or the `dd_real` of QD, bit for bit.
- **Correct rounding.** Addition, subtraction, multiplication, division,
  square root, fused multiply-add, and every conversion round correctly in
  eight directions: the five of IEEE 754, round to odd, and the two extra
  directions of IBM POWER decimal floating point.
- **The IEEE 754 operations.** Comparisons, total order, the IEEE 754-2019
  minimum, maximum, and magnitude operations and the IEEE 754-2008 `minNum`
  and `maxNum`, the remainder, rounding to an integral value, integer
  conversions up to 512 bits, `scale_b` and `log_b`, the exponentials and
  the logarithms of every format in base e, 2, and 10, with their forms
  `b^x - 1` and `log_b(1 + x)`, the hyperbolic functions and their
  inverses, and the functions scaled by pi, `sinPi`, `cosPi`, and `tanPi`,
  correctly rounded, `hypot`, the reciprocal square root,
  `compound`, `pown`, and `rootn` of the binary formats, correctly rounded,
  the last two for `|n|` up to 64, the IEEE 754-2019 augmented addition,
  subtraction, and multiplication of the binary formats, `next_up` and
  `next_down`, the sign operations, the NaN payload operations, the
  reduction operations of the binary formats, and the decimal quantum
  operations. The sums of the reductions round once. The operators include
  `%`, the truncated remainder of C `fmod`, and the compound assignments
  such as `+=`.
- **Hardware behavior as data.** Flush-to-zero, denormals-are-zero, tininess
  detection, NaN propagation rules, x87 precision control, and saturation
  of overflows. Presets give the x86 SSE and x87 behavior.
- **Flags.** The five IEEE 754 flags, and `TINY`, `ROUNDED_UP`, and
  `DENORMAL_INPUT`, which emulators need.
- **Vector lanes.** `Lanes<T, N>` applies each operation to the lanes of a
  vector register and returns the union of the flags of the lanes.
- **Slice kernels.** Sums, dot products, squares of distances, and norms
  of whole vectors of binary32, bfloat16, binary16, and integer codes,
  accumulated in `N` binary32 lanes in one documented order, with each
  step rounded or fused. The order gives the same bits on every host, and
  the host path runs at the speed of a plain host loop.
- **Elementwise slice operations.** Views such as `Sum`, `Product`,
  `MinimumNumber`, and `RoundToIntegral` compute value `i` of a vector from
  value `i` of their operands. A store, a conversion to integers, a minimum
  or maximum reduction, or a kernel reads a view in one pass.
- **Fast where possible.** Where the build targets a floating-point unit
  that gives the same bits, the operations without flags use it.
- **Small.** `no_std`, no `alloc`, no dependencies, and no `unsafe` code
  outside the host paths. The optional feature `std` adds one check of the
  processor for the slice paths.

## Installation

floaty is not on crates.io. Add it as a Git dependency, and pin a revision:

```toml
[dependencies]
floaty = { git = "https://github.com/napbat/floaty", rev = "<commit>" }
```

floaty needs Rust 1.89 or later. Enable the feature `std` to let the slice
kernels and the elementwise slice operations use the largest instruction set
of the processor, as [Hardware acceleration](#hardware-acceleration) states:

```toml
floaty = { git = "https://github.com/napbat/floaty", rev = "<commit>", features = ["std"] }
```

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
register. Type aliases name the common formats. `F32` and `F64` convert to
and from the host `f32` and `f64` with `From`. The conversion is a bit cast,
so it keeps a signaling NaN.

| Alias | Format | Precision |
| --- | --- | --- |
| `F16`, `F32`, `F64`, `F128` | IEEE 754 binary16, binary32, binary64, binary128 | 11, 24, 53, 113 bits |
| `F160` to `F512` | IEEE 754 binary formats in steps of 32 bits | 144 to 489 bits |
| `BF16` | bfloat16 | 8 bits |
| `TF32` | NVIDIA TensorFloat-32, 19 bits | 11 bits |
| `F8E4M3Fn`, `F8E5M2` | OCP FP8. E4M3 has no infinity, so LLVM and `ml_dtypes` call it E4M3FN. | 4, 3 bits |
| `F8E4M3`, `F8E3M4` | FP8 with IEEE 754 special values, as LLVM and `ml_dtypes` define them | 4, 5 bits |
| `F8E4M3Fnuz`, `F8E5M2Fnuz` | FP8 with one NaN and no negative zero | 4, 3 bits |
| `F8E4M3B11Fnuz` | FNUZ FP8 E4M3 with the exponent bias 11 | 4 bits |
| `F4E2M1Fn`, `F6E2M3Fn`, `F6E3M2Fn` | OCP MX FP4 and FP6: no infinity and no NaN | 2, 4, 3 bits |
| `F80` | x87 extended precision, with an explicit integer bit | 64 bits |
| `D32Bid`, `D64Bid`, `D128Bid` | IEEE 754 decimal formats, BID encoding | 7, 16, 34 digits |
| `D32Dpd`, `D64Dpd`, `D128Dpd` | IEEE 754 decimal formats, DPD encoding | 7, 16, 34 digits |

`Binary<E, Enc>` describes other binary layouts: 2 to 28 exponent bits `E`,
at least one fraction bit, a width up to 512 bits, and an encoding of the
special values and the bias, `Ieee`, `NoInf`, `Fnuz`, `B11Fnuz`, `Finite`,
or `X87`. An invalid layout fails to compile.

```rust
use floaty::{Binary, Decoded, Float};

// An 8-bit format with 4 exponent bits and IEEE special values.
type Minifloat = Float<Binary<4>, 8>;

assert_eq!((Minifloat::PRECISION, Minifloat::EMAX, Minifloat::EMIN), (4, 7, -6));
let largest = Minifloat::from_bits(0x77).decode::<1>();
assert_eq!(largest, Decoded::Finite { negative: false, exponent: 4, significand: [15] });
```

An overflow in a format without an infinity gives the NaN when the format
has one, with the sign of the result. A saturating behavior gives the
largest finite value instead, in every binary format, and so does every
other infinite result, such as an infinite operand. The MX formats have
neither an infinity nor a NaN, so they always saturate, and an invalid
operation gives `+0` with `INVALID`:

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

MX 1.0 leaves the conversion of a NaN to FP4 or FP6 to the implementation.
floaty gives `+0` with `INVALID`, as `ml_dtypes` does. PTX `cvt` with
`.satfinite` gives the largest positive value instead (PTX ISA 9.4). A
conversion that follows PTX tests for a NaN first:

```rust
use floaty::{F32, F4E2M1Fn};

/// Converts binary32 to FP4 E2M1 as PTX `cvt` with `.satfinite` does.
fn to_fp4(value: F32) -> F4E2M1Fn {
    if value.is_nan() { F4E2M1Fn::from_bits(0x7) } else { value.convert() }
}

assert_eq!(to_fp4(F32::from_bits(0xFFC0_0000)).to_bits(), 0x7); // -NaN gives +6
assert_eq!(to_fp4(F32::from_bits(0x7F80_0000)).to_bits(), 0x7); // +inf gives +6
assert_eq!(to_fp4(F32::from_bits(0x3FC0_0000)).to_bits(), 0x3); // 1.5
```

### Unsigned floats of R11G11B10

The packed format R11G11B10 of Direct3D, Vulkan, and OpenGL holds unsigned
floats of 11 and 10 bits, with 5 exponent bits and no sign bit. Their values
are the non-negative values of `Float<Binary<5>, 12>` and
`Float<Binary<5>, 11>`, so floaty builds them from those formats. The
example follows GL_EXT_packed_float as Mesa implements it: round to nearest
even, saturate a finite value, keep positive infinity, map a negative value
to zero, and give one NaN.
`floaty-verify` checks it against Mesa for every rounding case.

```rust
use floaty::{Binary, Env, F32, Float};

type Eleven = Float<Binary<5>, 12>;

/// Returns the 11-bit channel of a binary32 value.
fn to_channel(value: F32) -> u16 {
    // Saturation also clamps an infinity, and GL keeps positive infinity.
    if value.is_infinite() && !value.is_sign_negative() {
        return 0x7C0;
    }
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
assert_eq!(to_channel(F32::from_bits(0x7F80_0000)), 0x7C0); // +inf stays infinite
let one: F32 = Eleven::from_bits(0x3C0).convert();
assert_eq!(one.to_bits(), 0x3F80_0000);
```

### The MX scale type E8M0

E8M0FNU, the scale of the OCP microscaling formats, is an exponent field
alone: code `c` holds 2^(c - 127), and code 255 is the NaN. It has no sign,
no zero, and no fraction, so floaty builds it from binary64 and a precision
limit of one bit. The example rounds as `ml_dtypes` does: to the nearest
power of two with a tie up, to 2^-127 below that value, and to the NaN for
an overflow, a zero, a negative value, or a NaN. `floaty-verify` checks it
against `ml_dtypes` for every code and every rounding case of binary32.
`ml_dtypes` 0.6.0 differs in one range: it rounds a binary32 subnormal above
2^-127 and below 1.5 * 2^-127 to 2^-126, although 2^-127 is nearer. The
example gives 2^-127 there.

```rust
use core::num::NonZeroU32;
use floaty::{Env, Exact, F32, F64, Rounding};

/// Returns the E8M0 code of a binary32 value.
fn to_scale(value: F32) -> u8 {
    let one_bit = Env::IEEE
        .with_rounding(Rounding::TiesToAway)
        .with_precision(NonZeroU32::new(1));
    let (power, _) = value.convert_with::<F64>(one_bit);
    if power.is_nan() || power.is_sign_negative() || power.is_zero() {
        return 0xFF;
    }
    // The exponent field of a positive binary64 value is its bits above 52.
    let exponent = i32::try_from(power.to_bits() >> 52).expect("11 bits") - 1023;
    if exponent > 127 {
        return 0xFF;
    }
    u8::try_from(exponent.max(-127) + 127).expect("the code is below 255")
}

/// Returns the binary32 value of an E8M0 code.
fn from_scale(code: u8) -> F32 {
    if code == 0xFF {
        return F32::from_bits(0x7FC0_0000);
    }
    let power = Exact {
        negative: false,
        exponent: i32::from(code) - 127,
        significand: [1],
        sticky: false,
    };
    F32::round(power, Env::IEEE).0
}

assert_eq!(to_scale(F32::from_bits(0x3FC0_0000)), 128); // 1.5 gives 2.0
assert_eq!(to_scale(F32::from_bits(0x3FBF_FFFF)), 127); // below 1.5 gives 1.0
assert_eq!(to_scale(F32::from_bits(0x7F40_0000)), 0xFF); // 1.5 * 2^127 overflows
assert_eq!(to_scale(F32::from_bits(0x0000_0001)), 0); // 2^-149 gives 2^-127
assert_eq!(from_scale(0).to_bits(), 0x0040_0000); // 2^-127, subnormal in binary32
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
| `saturate` | Every infinite result gives the largest finite value of its sign instead of an infinity or a NaN: an overflow, an infinite operand, and an exact infinity such as `1 / 0`, as OCP FP8 saturation and PTX `.satfinite` do. The flags do not change. Double-double arithmetic ignores it, because libgcc and QD do not saturate. |
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
mode, unless the type names another: `mode::Ieee` for a `Float`, and the
reference mode of the algorithm for a `DoubleDouble`. The engine compiles the
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

`RADIX` is 2. `PRECISION` is the nominal precision of the reference:
106 binary digits for `Gcc`, and 104 for `Qd`. A double-double has no
fixed-width significand. The low half can hold bits beyond the nominal
precision.

Each algorithm takes the behavior of its reference as its default mode:
`mode::Libgcc` for `Gcc`, and `mode::X86Sse` for `Qd`. So the operators
match the reference without a behavior argument. Under another behavior,
each step is a binary64 operation under that behavior, and a `_with` method
returns the flags of every step.

| Operations | `Gcc` | `Qd` |
| --- | --- | --- |
| `+`, `-`, `*`, `/` | libgcc `__gcc_qadd` and the others | `dd_real` operators |
| `sqrt`, `remainder`, `%` | glibc 2.43 `sqrtl`, `remainderl`, `fmodl` | QD `sqrt`, `drem`, `fmod` |
| `mul_add`, `next_up`, `next_down` | glibc 2.43 `fmal`, `nextupl`, `nextdownl` | QD has no such function: `mul_add` rounds the exact value once, as IEEE 754 `fusedMultiplyAdd` does, and `next_up` and `next_down` give the next canonical pair, as IEEE 754 `nextUp` and `nextDown` do |
| `sqr`, `inv`, `npwr`, `nroot` | none: these are QD operations | QD `sqr`, `inv`, `npwr`, `nroot`, with a correctly rounded seed for `nroot` |
| Conversions, integer conversions, `scale_b`, `log_b`, `round_to_integral`, `copy_sign`, classification, total order, minimum and maximum | the exact value `hi + lo`, by floaty's rule. For canonical pairs, glibc 2.43 matches except for the cases listed below. | the same |
| `payload`, `from_payload`, `from_payload_signaling` | IEEE 754-2019 section 9.7, on the exact value, with 51 payload bits | the same |

The payload operations ignore the mode and give a +0 low half.
The constructors accept exact integral payloads, including noncanonical pairs.
`from_payload` accepts +0. `from_payload_signaling` rejects zero.
Both constructors reject -0, negative values, fractions, and values at or above `2^51`.

QD's `npwr` takes `std::abs(n)`, which is undefined for `INT_MIN`, and its
compiled loop never returns for that exponent. For `i32::MIN`, floaty's
`npwr` takes the magnitude 2^31, as the source of QD intends.

QD seeds `nroot` with `exp(-log(|hi|) / n)` of the C library, and glibc's
`exp` does not always round correctly. floaty's `nroot` takes its own
correctly rounded binary64 `exp` and `log` there, in the behavior of the
call, so its result does not depend on the C library. In 1.4 million test
pairs and exponents, glibc 2.43 gives another seed in 852 cases, and
another root in 790.

`floaty-verify` checks the rule of the operations on the exact value with
exact rationals and MPFR. It also compares them with the IBM `long double`
functions of glibc 2.43 on canonical pairs: `logbl`, `copysignl`, `ilogbl`,
`floorl`, `ceill`, `truncl`, `roundl`, `roundevenl`, `rintl`, `nearbyintl`,
`scalbnl`, `llrintl`, `lroundl`, the ten minimum and maximum functions,
`totalorderl`, and `totalordermagl`. glibc reads the halves, so it differs
from floaty in five cases, which the tests record: the sign of a zero low
half, the low half of a NaN or an infinity, the minimum and maximum of two
pairs of one value, a scaled value outside the normal range, and an extra
inexact flag of `rintl`, and of `llrintl` and `lroundl` out of range.

`total_cmp` follows the `TotalOrder` of the default mode, `Datum`, so two
pairs of one value, such as `(1, 0)` and `(2, -1)`, are equal, as
`totalorderl` treats two canonical pairs of one value. `total_cmp_with`
and `TotalOrder::Encoding` order such pairs by their halves, as the minimum
and maximum operations do.

An exact value rounds to a pair in two steps: the high half is the value
rounded to nearest even, and the low half is the rest rounded in the
direction of the behavior. The sum of the halves then splits into its
canonical pair, so a value that a canonical pair holds gives that pair in
every direction.

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
the lanes, as a vector unit accumulates them. Besides the arithmetic,
`Lanes` has the exponentials, the logarithms, the hyperbolic functions, the
functions scaled by pi, the operations of the binary formats alone, such as
`hypot`, `pown`, `compound`, and the augmented operations, and those of the
decimal formats alone, such as `quantize`. `pown`, `rootn`, and `compound`
take one exponent for each lane. The augmented operations return the heads
and the tails as two `Lanes`.

```rust
use floaty::{Env, F32, Flags, Lanes};

let x = Lanes::<F32, 4>::from_bits([0x3F80_0000, 0x7F7F_FFFF, 0x4000_0000, 0x4040_0000]);
let y = Lanes::<F32, 4>::from_bits([0x3380_0000, 0x7F7F_FFFF, 0x4000_0000, 0x4040_0000]);
let (sum, flags) = x.add_with(y, Env::X86_SSE);
assert_eq!(sum.to_bits(), [0x3F80_0000, 0x7F80_0000, 0x4080_0000, 0x40C0_0000]);
assert_eq!(flags, Flags::OVERFLOW | Flags::INEXACT | Flags::ROUNDED_UP);
```

### Slice kernels

The slice kernels of `Lanes<F32, N>` read whole vectors: `&[F32]`,
`&[f32]`, `&[BF16]`, `&[F16]`, `&[u8]` codes, `LittleEndian` encodings in
bytes at any alignment, and `ScaledCodes`. Each value widens exactly to
binary32, or a code converts and multiplies by its scale. Value `i` adds its
term into lane `i % N`, every lane starts at +0, and the lanes then add by
halves: lane `j` plus lane `j + N/2`, down to one lane. `N` is a power of
two.

| Kernel | Term of value `i` |
| --- | --- |
| `sum` | `x` |
| `dot`, `dot_fused` | `x * y` |
| `distance_square`, `distance_square_fused` | `d * d` with `d = x - y` rounded |
| `norm`, `norm_fused` | the square root of `dot` of `x` with itself |
| `dot_rows`, `distance_square_rows` | `dot` or `distance_square` of each row of a matrix with a query |

A separate kernel rounds the term and then the sum. A fused kernel adds
the product in one fused multiply-add. The fused kernels give the same bits
on a host without FMA, from the engine, as on a host with FMA.
`Lanes::convert_slice` converts a slice of the lane type, or of host `f32`
values, for example binary32 to bfloat16. Each `_with` method runs the
engine and returns the flags.

```rust
use floaty::{BF16, F32, Lanes, LittleEndian};

let query = [1.0_f32, 2.0, 3.0];
let row = [0x80, 0x3F, 0x00, 0x40, 0x40, 0x40]; // bfloat16 1, 2, 3
let dot = Lanes::<F32, 32>::dot(LittleEndian::<BF16>::new(&row), &query[..]);
assert_eq!(dot.to_bits(), 0x4160_0000); // 14
```

### Elementwise slice operations

The views of `floaty::elementwise` are vectors whose value `i` is one
operation of value `i` of their operands: `Sum`, `Difference`, `Product`,
`Quotient`, `Minimum`, `Maximum`, `MinimumNumber`, `MaximumNumber`, `Abs`,
and `RoundToIntegral` in a fixed direction. `Splat` repeats one value.
Integral rounding keeps each value's sign, including the sign of zero.
Views nest, and each step rounds. `Lanes<F32, N>` reads a view `N` values
at a time, and `store` and `to_int_slice` read the values past the last
chunk of `N` eight at a time:

| Operation | Result |
| --- | --- |
| `store` | value `i` into element `i` of `&mut [T]`, or of `&[Cell<T>]` for a store into a vector that the view reads |
| `to_int_slice` | value `i` converted to an integer type, as `Float::to_int` converts it |
| `minimum_of`, `maximum_of`, `minimum_number_of`, `maximum_number_of` | one value, in the order of the kernels, from +∞ or -∞ in every lane |

A cell destination must be disjoint from every cell operand, or use the
same cells at the same indices. The rule applies to every operand of a
nested view. `store` and `store_with` panic on shifted overlap before
the first write.

A kernel also reads a view, so the norm of a reconstruction needs no buffer.

```rust
use floaty::elementwise::{Difference, MinimumNumber, Product, RoundToIntegral, Splat};
use floaty::{F32, Lanes, Rounding, ToInt};

let x = [0.25_f32, 0.5, 2.0];
let lo = [0.0_f32; 3];
let levels = Splat::new(F32::from_bits(0x437F_0000), 3); // 255
let one = Splat::new(F32::from_bits(0x3F80_0000), 3);
let scaled = RoundToIntegral {
    values: Product(MinimumNumber(Difference(&x[..], &lo[..]), one), levels),
    rounding: Rounding::TiesToAway,
};
let mut codes = [ToInt::Value(0_u8); 3];
Lanes::<F32, 32>::to_int_slice(scaled, &mut codes);
assert_eq!(codes, [64, 128, 255].map(ToInt::Value)); // 63.75, 127.5, 255
```

### Blocks

`floaty::block` runs a chain of binary16, bfloat16, binary32, or binary64
steps for each lane of slices. A type that implements `Chain<IN, P>`
states the steps of one lane once, generic over the sealed trait `Steps`,
so one chain runs in each format. The steps are every operation of the
format from values of the format to one value of the format: `+`, `-`,
`*`, `/`, negation, `mul_add`, `sqrt`, `abs`, `copy_sign`, `next_up`,
`next_down`, `round_to_integral`, `round_to_integral_by` a direction,
`remainder`, `truncated_remainder`, `scale_b`, `log_b`, the exponentials,
the logarithms, the hyperbolic functions, the functions scaled by pi,
`compound`, `hypot`, `pown`, `rootn`, `reciprocal_sqrt`, `minimum`,
`maximum`, `minimum_number`, `maximum_number`, their four magnitude forms,
`min_num`, and `max_num`. Each step gives the result of its entry point in
the mode of the type, so a block gives the bits of the same steps one at a
time on every host.

| Function | Result |
| --- | --- |
| `F32::map(&chain, x, p, out)`, and `map` of `F16`, `BF16`, and `F64` | the chain of value `i` of each slice of `x` and the parameters `p`, into element `i` of `out`; slices of the type, or of `f32` for `F32` and of `f64` for `F64`. Returns a `Report`: `lane_count()` and `engine_lane_count()`, the lanes that the engine computed |
| `F32::evaluate(&chain, x, p)`, and `evaluate` of `F16`, `BF16`, and `F64` | the chain of one lane |

A block checks the environment and selects the instruction set once for
the call. Its steps then run as Rust operations on `f32` or `f64`, so LLVM
sees the whole chain and computes many lanes in vector instructions.
binary16 and bfloat16 compute each step in `f32` and round it to their
format, as their host paths do. A lane whose result is a NaN runs the
chain again in the engine, which selects the NaN by the rule of the mode.
So does a lane with a step that the instruction set cannot compute
exactly, and a lane where a step reads a bit of a NaN that can change a
result that is not a NaN. The host path table lists these steps. The
`Report` of `map` counts those lanes: every lane when the call takes no
host path, as `host_path` tells why, and otherwise each lane that ran again.
LLVM vectorizes the loop only where it inlines `apply`, so mark `apply`
`#[inline(always)]` in a long chain, and in a binary16 chain, whose steps
round in more instructions.

```rust
use floaty::F32;
use floaty::block::{Chain, Steps};

/// `(x - mean) * scale + offset`, with the product and the sum rounded once.
struct Normalize;

impl Chain<1, 3> for Normalize {
    fn apply<S: Steps>(&self, [x]: [S; 1], [mean, scale, offset]: [S; 3]) -> S {
        (x - mean).mul_add(scale, offset)
    }
}

let x = [1.5_f32, 2.5, -0.5];
let mut out = [0.0_f32; 3];
F32::map(&Normalize, [&x[..]], [0.5_f32, 2.0, 1.0].map(F32::from), &mut out[..]);
assert_eq!(out, [3.0, 5.0, -1.0]);
```

## Hardware acceleration

The engine computes every result in integer arithmetic. A host path
computes an operation on the floating-point unit of the host instead, only
where the unit gives the bits of the engine. The oracle tests check each
path.

- The build selects the host paths at compile time, from `target_arch` and
  `target_feature`. Enable more paths with `-C target-cpu` or
  `-C target-feature`, for example `RUSTFLAGS="-C target-cpu=x86-64-v3"`.
- With the feature `std`, the slice kernels, the elementwise slice
  operations, and `convert_slice` of `Lanes`, and blocks, also select an
  instruction set at run time. The first call checks the processor, and an
  atomic keeps the answer, so a later call pays one load and one branch. On
  x86-64 and
  32-bit x86, a processor with the features of x86-64-v3 (AVX, AVX2, BMI1,
  BMI2, F16C, FMA, LZCNT, and MOVBE) runs a copy of these paths compiled for
  them. A processor that also has those of x86-64-v4 (AVX-512F, BW, CD, DQ,
  and VL) runs a copy with 512-bit registers. A build that enables every
  feature of a set runs it with no check. Every set gives the bits of the
  engine. AArch64 has its NEON paths in every build, and no path uses SVE,
  whose vector length the processor selects. s390x runs the scalar
  instruction of each lane in every build. Without `std`, floaty does not
  check the processor.
- The test feature `override-host-level` reads the environment variable
  `FLOATY_HOST_LEVEL`: `build`, `v3`, or `v4` caps the instruction set of
  the check, so one processor runs the tests in each set. A set that the
  processor does not have panics.
- Only the entry points without flags use a host path: the operators, and
  the methods that drop the flags of the default mode. The `_with` methods
  always run the engine.
- The mode must round to nearest even without FTZ, DAZ, saturation, or a
  precision limit below the format precision. `round_to_integral` also
  takes the other directions that its instructions encode, and on AArch64
  the conversion of binary64 to binary32 also takes `ToOdd`, which `FCVTXN`
  encodes. The slice kernels and the elementwise slice operations also take
  `TowardPositive`, `TowardNegative`, and `TowardZero` where the build or
  the processor has AVX-512F: the embedded rounding control of its 512-bit
  forms encodes the direction.
- Each call reads MXCSR, the x87 control word, FPCR, or the FPC register of
  s390x, except on the paths of the next item. Another setting, or an
  unmasked exception, sends the operation to the engine. So does a NaN
  result: the engine selects the NaN by the rule of the mode.
- Three paths do not read the environment, because no field of it can
  change their result. The comparisons of binary16, bfloat16, binary32,
  and binary64 order the encodings in integer instructions. The conversions
  between bfloat16 and binary32 shift and round in integer instructions,
  except where `FEAT_BF16` rounds in `BFCVT`. `from_int` of an integer that
  binary32 or binary64 holds exactly does not round. The mode must still
  allow each path.
- `host_path` of a type tells whether its host paths run on this thread,
  and otherwise why its operations take the engine: `Unavailable` where the
  build has no host path for the format, `Mode` with the field of `Env`
  that the host unit does not give, or `Environment` where the setting of
  the unit is not the default. The call costs one read of the setting, and
  changes no result.

  ```rust
  use floaty::{F32, F8E4M3, HostPath};

  assert_eq!(F8E4M3::host_path(), HostPath::Unavailable);
  if F32::host_path() != HostPath::Ready {
      eprintln!("binary32 takes the engine: {:?}", F32::host_path());
  }
  ```
- Build with `RUSTFLAGS="--cfg floaty_engine_only"` to remove every host
  path and all `unsafe` code.

| Host path | Target feature | Formats and operations | Also goes to the engine |
| --- | --- | --- | --- |
| SSE | x86 or x86-64 with SSE2 | binary32 and binary64: `+`, `-`, `*`, `/`, `sqrt`, `convert` between them, `to_int`, `from_int`, comparisons, and the minimum and maximum operations | The integer indefinite; an unordered comparison; a minimum or maximum of a NaN or of two zeros; on 32-bit x86, `to_int` and `from_int` of an integer outside the range of `i32` |
| SSE4.1 rounding | x86 or x86-64 with SSE4.1 | `round_to_integral` of binary32 and binary64, and through binary32 of bfloat16 and of binary16 with F16C, in `TiesToEven`, `TowardPositive`, `TowardNegative`, and `TowardZero` | |
| FMA | x86 or x86-64 with FMA | `mul_add` of binary32 and binary64 | |
| binary16 | x86 or x86-64 with SSE2, or AArch64 | binary16 through binary32: `+`, `-`, `*`, `/`, `sqrt`, `to_int`, `from_int`, `remainder` by `FPREM1` on x86-64, `convert` from binary16 to binary32 and binary64, and from binary32 and binary64 to binary16. `mul_add` through binary64, which holds the exact product: the `Host::Half` documentation proves that the second rounding cannot change the result. Without F16C, integer instructions widen binary16 and round binary32 results to binary16. On x86-64, integer instructions round binary64 results to binary16. | |
| F16C | x86 or x86-64 with F16C | The binary16 paths in `VCVTPH2PS` and `VCVTPS2PH`, and also comparisons, the minimum and maximum operations, and `round_to_integral` with SSE4.1 | |
| bfloat16 | x86 or x86-64 with SSE2, or AArch64 | bfloat16 through binary32: `+`, `-`, `*`, `/`, `sqrt`, `to_int`, `remainder` by `FPREM1` on x86-64, `round_to_integral` with SSE4.1 on x86-64 and in the five IEEE 754 directions on AArch64, comparisons, the minimum and maximum operations, and `convert` to binary32 and binary64 and from binary32. On AArch64, also `convert` from binary64 and `from_int` of an integer below 2^53 in magnitude: `FCVTXN` rounds binary64 to binary32 with round to odd, which keeps the rounding to bfloat16. A shift widens bfloat16, and integer instructions round binary32 results to bfloat16. | `mul_add`, and on x86-64 `from_int` and `convert` from binary64, which two roundings can get wrong |
| x87 | x86 with SSE2, or x86-64 | x87 extended: `+`, `-`, `*`, `/`, and `sqrt` except on Windows, whose threads start at the 53-bit precision; `round_to_integral` to nearest even, `to_int`, `from_int`, and `convert` to and from binary32 and binary64 | A control word other than round to nearest with every exception masked, and for `+`, `-`, `*`, `/`, and `sqrt` a precision below 64 bits; the integer indefinite |
| x87 remainder | x86 with SSE2, or x86-64 | `remainder` of binary32, binary64, and x87 extended, by `FPREM1`, and of binary16 and bfloat16 widened to binary32, at any precision of the control word | A control word other than round to nearest with every exception masked; a dividend exponent more than 315 above the divisor exponent in binary64, or more than 630 in the other formats; a dividend that is subnormal in the format that `FPREM1` loads, or a divisor small enough to give a subnormal result |
| Packed SSE and AVX | x86 or x86-64 with SSE2, and AVX for 256 bits | `Lanes` of binary32 and binary64: `+`, `-`, `*`, `/`, `sqrt`, `mul_add` with FMA, `round_to_integral` with SSE4.1, `convert` between them and to binary16 and bfloat16, `compare_quiet`, the minimum and maximum operations, and `to_int`. Without F16C, integer instructions and one binary32 sum of each magnitude and 0.5 round binary32 lanes to binary16. bfloat16 lanes, and binary16 lanes with F16C: `+`, `-`, `*`, `/`, `sqrt`, `round_to_integral` with SSE4.1, `compare_quiet`, and the minimum and maximum operations. bfloat16 and binary16 lanes: `convert` to binary32 and binary64, for binary16 without F16C in integer and binary32 instructions. x87 extended lanes: `+`, `-`, `*`, `/`, and `sqrt`, with one check of the control word. | A NaN in any lane, or for the minimum and maximum a NaN or two zeros in any pair of lanes, sends each lane to its scalar path. For `to_int`, the integer indefinite sends its lane to the scalar conversion. |
| AArch64 | AArch64 | binary32, binary64, and binary16: `+`, `-`, `*`, `/`, `sqrt`, `convert`, `to_int`, `from_int`, `round_to_integral` in the five IEEE 754 directions, comparisons, and the minimum and maximum operations. `mul_add` of binary32, binary64, and binary16. `convert` from binary64 to binary32 in `ToOdd`, by `FCVTXN`. | FPCR with a nonzero `RMode`, FZ, FZ16, FIZ, AH, AHP, or a trap enable; a saturated integer |
| `FEAT_FP16` | AArch64 with `+fp16` | `mul_add` of binary16 in one instruction. `Lanes` of binary16, eight lanes at a time in binary16: `+`, `-`, `*`, `/`, `sqrt`, `mul_add` by `FMLA`, `round_to_integral` in the five IEEE 754 directions, `compare_quiet`, and the minimum and maximum operations | As the packed SSE paths |
| `FEAT_BF16` | AArch64 with `+bf16` | The scalar bfloat16 paths round in `BFCVT` | |
| Packed AArch64 | AArch64 | As the packed SSE paths, at 128 bits, but without `to_int` and x87 extended lanes | As the packed SSE paths |
| s390x | s390x | binary32 and binary64: `+`, `-`, `*`, `/`, `sqrt`, `mul_add`, `convert` between them, `to_int`, `from_int`, `round_to_integral` in the five IEEE 754 directions, comparisons, and the minimum and maximum operations. binary16 and bfloat16 through binary32, as on x86-64 without F16C, and binary16 `mul_add` through binary64. binary128 in pairs of floating-point registers: `+`, `-`, `*`, `/`, `sqrt`, `round_to_integral` in the five IEEE 754 directions, `to_int`, `from_int`, comparisons, the minimum and maximum operations, and `convert` to and from binary32 and binary64. `Lanes` of binary32, binary64, and bfloat16: the scalar instruction of each lane after one check of the FPC register | An FPC register with a nonzero binary rounding mode or an IEEE mask; a 64-bit integer at a bound of the conversion; `mul_add` and `remainder` of binary128 |
| Slice kernels | x86 or x86-64 with SSE2, AVX for 256 bits, and AVX-512F for 512 bits, in the build or, with `std`, on the processor; AArch64; or s390x | The kernels of `Lanes<F32, N>`: `sum`, `dot`, `distance_square`, `norm`, `dot_rows`, and `distance_square_rows` in the packed binary32 instructions, and the fused kernels with FMA. Rows of at most eight values sum four rows at a time, in the order of the kernels. bfloat16 widens by a shift, binary16 by F16C or `FCVTL` or in integer and binary32 instructions, and codes by `CVTDQ2PS`, `SCVTF`, or `CEGBR`. `convert_slice` takes the packed conversions. On s390x, each lane runs its scalar instruction, as the packed paths of s390x do. One check of the environment serves the call. With AVX-512F, the modes that round toward +∞, -∞, and zero compute each step in the 512-bit forms with the embedded rounding control of the direction, sixteen lanes at a time, and a row of at most eight values as any other row. | A NaN sum sends the call, or its row, to the engine. A chunk of `convert_slice` that holds a NaN converts one value at a time. Without AVX-512F, the modes that round toward +∞, -∞, and zero send the call to the engine. |
| Elementwise slice operations | As the slice kernels | The views of `floaty::elementwise` in the packed binary32 instructions: `MINPS` and `MAXPS` or `FMIN` and `FMAX`, with a NaN or two zeros settled in integer instructions; `RoundToIntegral` by `ROUNDPS` with SSE4.1, `VRNDSCALEPS` with AVX-512F, `FRINT`, or `FIEBR`, and otherwise by the sum and difference with 2^23 and an exact correction of one in the direction. `store`, `to_int_slice` by `CVTPS2DQ` on x86 and x86-64, and the reductions, with one check of the environment for the call. With AVX-512F, the modes that round toward +∞, -∞, and zero compute the sum, difference, product, quotient, and code scale of each view, and `to_int_slice`, in the 512-bit forms with the embedded rounding control of the direction. | A chunk of a store that holds a NaN, and a value that `CVTPS2DQ` does not convert, go to the engine. A reduction that gives a NaN goes to the engine. `to_int_slice` on AArch64 and s390x runs the engine. Without AVX-512F, the modes that round toward +∞, -∞, and zero send the call to the engine. |
| Double-double | The binary64 paths of the build | `+`, `-`, `*`, `/`, and `sqrt` of `Gcc` and `Qd`, with one check of the environment for all steps | |
| Blocks | As the slice kernels | `map` and `evaluate` of `F16`, `BF16`, `F32`, and `F64` in `floaty::block`: the steps of a chain as Rust operations on `f32` or `f64`, which LLVM vectorizes, after one check of the environment for the call. bfloat16 computes each step in `f32` and rounds it in the integer instructions of the bfloat16 paths. binary16 computes each step in `f32` and rounds it by the sum and difference with the power of two 2^13 above its exponent, and at least 0.5, and gives the infinity from 65520 up. The binary16 `mul_add` computes in binary64 and rounds there in the same way, with 2^42 and 2^28. The fused multiply-add and the square root take intrinsics that LLVM sees on x86, x86-64, and AArch64, and `MAEBR`, `MADBR`, `SQEBR`, and `SQDBR` on s390x. LLVM does not vectorize the square root on AArch64, so a chain with `sqrt` runs one lane at a time there. `abs`, `copy_sign`, `next_up`, `next_down`, and the minimum and maximum steps select and compute on the encodings, and `log_b` reads the exponent field, of a subnormal after an exact product by 2^23 or 2^52. The integral steps round by the sum and difference with 2^23 or 2^52 and an exact correction of one in the direction, and `scale_b` multiplies by a normal power of two of the host type. Every input passes through an empty assembly block after the check, so LLVM computes no step before the check | A NaN result, `mul_add` without FMA and in bfloat16, an integral value to odd, `scale_b` outside -126 to 127 in binary16, bfloat16, and binary32 and outside -1022 to 1023 in binary64, the remainders, the exponentials, the logarithms, the hyperbolic functions, the functions scaled by pi, `compound`, `hypot`, `pown`, `rootn`, `reciprocal_sqrt`, `copy_sign` from a NaN, and `min_num` and `max_num` of a signaling NaN: the lane runs the chain again in the engine |

## Performance

Nanoseconds per operation on an Intel Core i9-9900K, from
`cargo bench -p floaty-verify --bench operations`: the median of 11 samples
of 1,024 operations on random normal operands. Figures change from run to
run and from host to host.

The first table measures the engine, through the `_with` methods with an
`Env` chosen at run time. Each cell gives floaty, then `rustc_apfloat` 0.2.3
where it has the format.

| Format | add | mul | div | mul_add |
| --- | --- | --- | --- | --- |
| binary16 | 19.8 / 38.1 | 14.6 / 32.3 | 19.1 / 54.8 | 27.5 / 48.2 |
| bfloat16 | 19.9 / 37.3 | 15.1 / 31.7 | 18.5 / 73.7 | 27.2 / 49.3 |
| binary32 | 20.2 / 38.5 | 15.7 / 32.0 | 20.6 / 45.2 | 26.9 / 48.5 |
| binary64 | 20.5 / 38.3 | 17.4 / 32.2 | 24.6 / 202.2 | 25.8 / 48.0 |
| x87 extended | 27.6 / 36.5 | 19.5 / 31.1 | 56.1 / 238.8 | 38.5 / 50.0 |
| binary128 | 23.0 / 33.8 | 22.3 / 31.9 | 69.7 / 404.0 | 38.6 / 48.4 |
| binary256 | 42.8 | 42.9 | 127.8 | 66.6 |
| binary512 | 60.6 | 89.7 | 281.0 | 148.6 |
| decimal64, BID | 51.8 | 44.9 | 68.1 | 66.4 |
| decimal128, BID | 42.6 | 49.0 | 99.9 | 96.6 |
| decimal64, DPD | 62.5 | 53.4 | 75.7 | 76.9 |
| decimal128, DPD | 68.9 | 78.3 | 127.9 | 128.5 |

The second table measures the entry points without flags, which take a
host path where the build has one. A `Lanes` figure is per lane.

| Operation | Engine | Default build | x86-64-v3 build | Host `f32` or `f64`, x86-64-v3 |
| --- | --- | --- | --- | --- |
| binary32 `+` | 20.2 | 1.9 | 1.9 | 0.8 |
| binary64 `/` | 24.6 | 1.9 | 2.0 | 0.9 |
| binary32 `mul_add` | 26.9 | 24.3 | 1.7 | 1.0 |
| binary16 `+` | 19.8 | 6.7 | 2.3 | |
| bfloat16 `+` | 19.9 | 2.3 | 2.4 | |
| x87 extended `/` | 56.1 | 4.5 | 4.0 | |
| `Lanes<F32, 8>` `+` | 17.1 | 0.4 | 0.3 | |
| `Lanes<F16, 8>` `+` | 15.8 | 6.4 | 0.3 | |
| `Lanes<BF16, 8>` `+` | 16.5 | 1.9 | 1.6 | |

The slice kernels in nanoseconds per vector, on an AMD Ryzen 9 9950X3D.
Each cell gives the default build, then the x86-64-v3 build. The host
loop keeps 8 `f32` accumulators in the order of the kernels, and calls
`f32::mul_add` for the fused row. `_with` runs the engine.

| Kernel | Host loop, 8 accumulators | `Lanes<F32, 8>` | `Lanes<F32, 32>` | `_with`, 32 lanes |
| --- | --- | --- | --- | --- |
| `dot`, 1,024 values | 63 / 44 | 57 / 50 | 58 / 44 | 17,622 / 15,667 |
| `dot_fused`, 1,024 | 1,622 / 57 | 15,390 / 69 | 15,802 / 45 | 15,858 / 14,929 |
| `distance_square`, 1,024 | 64 / 46 | 76 / 47 | 59 / 45 | 32,585 / 27,402 |
| `norm`, 1,024 | 54 / 39 | 58 / 48 | 56 / 32 | 18,643 / 16,449 |
| `dot`, 2,560 | 142 / 119 | 136 / 122 | 134 / 102 | 49,085 / 44,738 |
| `dot_fused`, 2,560 | 3,919 / 195 | 44,107 / 212 | 44,290 / 120 | 44,418 / 43,961 |
| `distance_square`, 2,560 | 138 / 118 | 190 / 119 | 135 / 107 | 81,855 / 73,886 |
| `norm`, 2,560 | 135 / 103 | 133 / 126 | 128 / 94 | 47,622 / 42,661 |
| `dot` of `&[BF16]` and `&[f32]`, 1,280 | 69 / 62 | 75 / 63 | 70 / 51 | 32,620 / 28,636 |
| `dot` of `LittleEndian<BF16>`, 1,280 | 73 / 59 | 73 / 62 | 70 / 50 | 32,738 / 28,639 |
| `dot` of `&[F16]`, 1,280 | | 388 / 66 | 390 / 52 | 30,682 / 28,487 |
| `dot` of `&[u8]` codes, 1,280 | 86 / 62 | 98 / 63 | 92 / 48 | 31,925 / 28,953 |
| `dot` of `ScaledCodes`, 1,280 | 255 / 64 | 128 / 66 | 140 / 52 | 40,584 / 36,752 |
| `dot`, 8 values | 1.8 / 7.0 | 6.4 / 6.4 | 8.4 / 8.1 | 329 / 310 |
| `dot_rows`, 8 values, per row | | 1.9 / 2.1 | 4.7 / 4.6 | 341 / 311 |

A build without FMA computes the fused kernels in the engine.

The elementwise slice operations in nanoseconds per vector of 1,280
values, default build, on the same host. The scalar column runs the
operations of `F32` one value at a time.

| Operation | Host loop | `Lanes<F32, 32>` | `_with` | Scalar `F32` |
| --- | --- | --- | --- | --- |
| `store` of `x * keep + y * eta` into cells | 100 | 242 | 33,329 | 17,166 |
| `to_int_slice` of a quantization to codes | 1,625 | 2,436 | 56,421 | 37,177 |
| `store` of `MinimumNumber` into cells | 108 | 398 | 6,021 | 5,717 |
| `maximum_of` `Abs` | 179 | 986 | 2,430 | 5,783 |
| `convert_slice` of `f32` to bfloat16 | | 1,200 | 8,218 | 5,527 |

The decimal formats against the Intel Decimal Floating-Point Math Library
for BID, and against the `decDouble` and `decQuad` functions of decNumber
for DPD, on the same operands. Each cell gives floaty, then the reference.
decNumber has no square root of its fixed-size formats, and each decNumber
call also sets up a context.

| Format | add | mul | div | sqrt | mul_add |
| --- | --- | --- | --- | --- | --- |
| decimal64, BID | 49.7 / 46.1 | 39.9 / 67.4 | 61.8 / 65.8 | 55.6 / 40.3 | 63.0 / 110.3 |
| decimal128, BID | 38.2 / 38.1 | 45.0 / 152.6 | 95.5 / 169.2 | 164.7 / 306.0 | 89.5 / 229.2 |
| decimal64, DPD | 58.5 / 70.3 | 49.3 / 58.5 | 73.1 / 153.5 | 63.8 | 74.8 / 170.4 |
| decimal128, DPD | 66.6 / 95.4 | 72.2 / 84.5 | 123.5 / 331.7 | 187.2 | 124.0 / 219.7 |

`exp` and `log` in nanoseconds per operation, with the mode of the type, on
an AMD Ryzen AI Max+ 395. The time grows with the precision, because the
evaluation runs at twice the bits of the storage.

| Format | exp | log |
| --- | --- | --- |
| binary32 | 1,341 | 1,773 |
| binary64 | 1,350 | 1,733 |
| binary128 | 2,751 | 4,926 |
| binary256 | 7,488 | 17,246 |
| binary512 | 45,650 | 135,521 |
| decimal64, BID | 1,024 | 1,996 |
| decimal128, BID | 2,408 | 5,113 |
| decimal64, DPD | 1,096 | 1,984 |
| decimal128, DPD | 2,313 | 5,150 |

## Verification

Every behavior has a test against an established reference. Where no
implementation exists, a test evaluates the published definition from
IEEE 754 or a vendor manual, or the documented rule of floaty, with MPFR or
decNumber.

| Reference | What it checks |
| --- | --- |
| Berkeley TestFloat and SoftFloat 3e | Arithmetic, conversions, comparisons, the remainder, rounding to an integral value, and integer conversions of binary16, binary32, binary64, binary128, and x87 extended, in the six directions of TestFloat, under four NaN rules |
| MPFR, through `rug` | Rounding to every binary format up to 512 bits. The arithmetic and the other operations of every binary format but x87 extended precision, in every direction and in a set of behaviors that uses each flag setting and each NaN rule. The NaN that each NaN rule selects. The arithmetic of x87 extended precision with precision control in every direction, with FTZ, DAZ, and saturation, and with unsupported operands, and its other operations in the same set of behaviors. Conversions between the formats that TestFloat lacks, with their NaN payloads, from binary32, binary64, x87 extended, and binary128 to them, and between every binary format and the decimal formats, in every behavior of the conversion tests. The truncated remainder. `hypot`, the reciprocal square root, `pown`, and `rootn`, against `mpfr_hypot`, `mpfr_rec_sqrt`, `mpfr_pow_si`, and `mpfr_rootn_si`, and the rounded steps of `pown` past `|n| = 64`. The augmented operations, whose tail `mpfr_sum` rounds. The sums of the reduction operations, with `mpfr_sum` and `mpfr_dot`, and each step of the scaled products. The NaN payload operations of every binary format, by floaty's rule. The operations of `DoubleDouble` on its exact value, by their rules, with exact rationals. Each binary64 step of the other `DoubleDouble` operations, in the behaviors that libgcc and QD do not define. |
| MPFR, through `rug`, with published worst cases | The exponentials, the logarithms, the hyperbolic functions, and the functions scaled by pi of every binary and decimal format, `exp`, `expm1`, `exp2`, `exp2m1`, `exp10`, `exp10m1`, `log`, `log2`, `log10`, `logp1`, `log2p1`, `log10p1`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `sinPi`, `cosPi`, and `tanPi`, against the MPFR function of each at a precision that grows until its bounds decide the rounding, in every behavior of the operation tests for the binary formats and of the conversion tests for the decimal formats. GMP gives the rational results, the powers of 2 and 10 at an integer and the integer logarithms, exactly. The rules of IEEE 754-2019 section 9.2.1 give the exact values of the functions scaled by pi at the quarters. Every FP8, FP6, FP4, binary16, and bfloat16 encoding, and the integers, quarters, powers, and arguments next to each bound of the other formats. The binary64 worst cases of Lefèvre and Muller, and the decimal64 `exp` worst cases of Lefèvre, Stehlé, and Zimmermann. |
| GMP and MPFR, through `rug` | `compound` of every binary format, in every behavior of the operation tests: every FP8, FP6, and FP4 encoding with 24 exponents from `i64::MIN` to `i64::MAX`, every binary16 and bfloat16 encoding with three, and samples of the other formats with arguments next to -1, next to the thresholds of the range, and with exact powers. GMP gives an exact power, or the exact power `x^n` for a large `x`. Every other power takes bounds from `mpfr_log1p`, `mpfr_mul_si`, and `mpfr_exp`, at a precision that grows until the bounds decide the rounding. `mpfr_compound_si` of MPFR 4.2.2 gives a wrong bound in a hard case, which a comment in the oracle records. |
| `rustc_apfloat` 0.2.3 | Decoding and classification, `next_up` and `next_down`, the remainder, rounding to an integral value, integer conversions, `scale_b`, and `log_b` against `ilogb` |
| `ml_dtypes` 0.6.0 | Every FP8, FP6, and FP4 encoding and operand pair, and the E8M0 recipe, as generated tables |
| The host processor | The SSE and x87 presets under every MXCSR and control word state, the packed instructions and their flags, and every host path. The AArch64 and s390x tests run under QEMU 10.2.1. |
| decTest 2.62 and decNumber 3.68 | DPD vectors, and random decimal32, decimal64, and decimal128 operations in every direction, with FTZ, DAZ, precision limits, and every NaN rule, in DPD and, through the conversions of the Intel library, in BID. The conversions between the widths in the same behaviors, the conversions to and from every integer type, and the total order of non-canonical encodings. The minimum and maximum operations and the NaN payload operations of IEEE 754-2019, by floaty's rule. The static modes, against their behaviors. |
| Intel Decimal Floating-Point Math Library 2.0 Update 2 | The BID vectors of `readtest.in`, about 22 million random cases, and conversions to and from binary formats |
| libgcc of GCC 15.2.0, under QEMU `qemu-ppc64le` | `DoubleDouble<Gcc>`, and the PowerPC fused multiply-add NaN rules |
| glibc 2.43 libm of the powerpc64le cross C library, under QEMU | `sqrt`, `remainder`, `%`, `mul_add`, `next_up`, `next_down`, and `is_canonical` of `DoubleDouble<Gcc>`, and its operations on the exact value of canonical pairs: `log_b`, `copy_sign`, `round_to_integral` in each direction, `scale_b`, `to_int`, the minimum and maximum operations, and the total order |
| glibc 2.43 libm of the host | The NaN payload operations of binary32, binary64, x87 extended, and binary128 |
| QD 2.3.24 | `DoubleDouble<Qd>`, also under the FTZ and DAZ bits of MXCSR. `nroot` runs against the source of QD with the seed from MPFR, and that copy against the library with the seed of the C library. |
| Mesa 25.2.0, `format_r11g11b10f.h` | The R11G11B10 recipe: every rounding case of both channels, and every channel code |

The normal test run tests every FP8, FP6, and FP4 operand pair of every
operation. Ignored sweeps test every binary16 and bfloat16 operand pair of
the host paths, every binary32 encoding, and all 318 million TestFloat
`mulAdd` cases.
MPFR computes the documented order of the slice kernels of `Lanes` one
step at a time, in every behavior of the operation tests, for binary32,
bfloat16, and binary16 vectors and codes, with special values among them.
The host path of each kernel and of `convert_slice` must give the bits of
the engine in every lane count from 1 to 64.
The arithmetic, minimum and maximum, rounding, and integer oracles check
each value of the elementwise views, their nested forms, and the documented
order of the reductions, in every behavior. The host path of each
elementwise operation must give the bits of the engine in each lane count.
A reference that disagrees with IEEE 754 or with the processor has a comment
beside its test with the evidence and the resolution.

## Platform support

| Target | Status |
| --- | --- |
| `x86_64-unknown-linux-gnu` | Tested, in the baseline build and with `-C target-cpu=x86-64-v3` |
| `aarch64-unknown-linux-gnu` | Tested under QEMU, without and with `+fp16,+bf16` |
| `i686-unknown-linux-gnu` | Tested under QEMU, as a 32-bit target with the SSE2 and x87 host paths |
| `s390x-unknown-linux-gnu` | Tested under QEMU, as a big-endian target |
| Other targets | The engine has no target-specific code. The gates do not test these targets. |

The minimum supported Rust version is 1.89. The crate uses edition 2024.

## Limitations

- No `sin`, `cos`, `tan`, inverse trigonometric functions, `atan2`, `pow`,
  or `powr`.
- No text parsing or printing, in decimal or in hexadecimal.
- No `const fn` evaluation. The engine uses traits, which a `const fn` on
  stable Rust cannot call.
- No traps. floaty reports flags and never traps.
- Presets for x86 SSE and x87 only.
- No decimal formats wider than 128 bits. No hardware or major library
  uses them.
- No reduction operations or algebraic functions for the decimal formats.
- `pown` rounds once only for `|n|` up to 64, and `rootn` takes only those
  `n`: a correctly rounded result for every `n` needs memory that grows
  with `n`. Past 64, `pown` rounds each step of its squaring.
- QEMU stands in for POWER hardware in the `Gcc` tests and for
  `mode::Libgcc`.

## Development

The workspace has two packages:

| Package | Purpose |
| --- | --- |
| `floaty` | The library. `no_std`, no dependencies. |
| `floaty-verify` | The oracle tests and the benchmark. Not a default member. |

`floaty-verify` builds its C references on Linux x86-64 only. It needs the
submodules, a C and C++ toolchain, the PowerPC cross compiler, and QEMU.
The first build downloads the decimal and QD archives and two Mesa headers.
The AArch64 gates also need the AArch64 cross compiler. The tests run
under cargo-nextest, and the doctests under `cargo test --doc`.

```text
git submodule update --init
cargo nextest run --workspace
cargo test --workspace --doc
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
