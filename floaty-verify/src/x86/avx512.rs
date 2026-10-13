//! The recipes of the AVX-512 scalar instructions that an emulator builds
//! from floaty operations: `VRNDSCALE`, `VREDUCE`, `VGETEXP`, `VGETMANT`,
//! `VSCALEF`, `VRANGE`, `VFIXUPIMM`, and `VFPCLASS`, for binary32 in
//! [`single`] and binary64 in [`double`]. Each recipe gives the result and
//! the floaty flags of the instruction; `mxcsr_flags` maps the flags to
//! MXCSR. A recipe follows the operation that the Intel SDM Volume 2,
//! revision 325383-084US, gives for the instruction, or states what the
//! hardware shows. `tests/avx512-composed.rs` compares each recipe with the
//! processor.

use floaty::Rounding;

use super::SseSetting;

/// Returns the rounding direction of bits 1 and 0 of an immediate (Volume
/// 2B, Figure 5-29), or of MXCSR when bit 2 is set.
#[must_use]
pub fn direction(immediate: u8, setting: &SseSetting) -> Rounding {
    if immediate & 4 != 0 {
        return setting.rounding;
    }
    [
        Rounding::TiesToEven,
        Rounding::TowardNegative,
        Rounding::TowardPositive,
        Rounding::TowardZero,
    ][usize::from(immediate & 3)]
}

/// Defines the recipes of one format: `$float` with the encoding `$bits`.
macro_rules! recipes {
    (
        $(#[$doc:meta])* $module:ident, $float:ty, $bits:ty,
        precision: $precision:literal, half_pi: $half_pi:literal, ninety: $ninety:literal
    ) => {
        $(#[$doc])*
        pub mod $module {
            use floaty::{Env, Flags, Rounding, ToInt};

            use super::direction;
            use crate::x86::SseSetting;

            /// The bits of the encoding.
            const WIDTH: u32 = <$bits>::BITS;
            /// The sign bit.
            pub const SIGN: $bits = 1 << (WIDTH - 1);
            /// The quiet bit of a NaN.
            const QUIET: $bits = 1 << ($precision - 2);
            /// The exponent field, all ones.
            const EXPONENT: $bits = !SIGN & !((1 << ($precision - 1)) - 1);
            /// The positive infinity.
            const INFINITY: $bits = EXPONENT;
            /// The default NaN, the negative quiet NaN that Intel names the
            /// floating-point indefinite.
            const INDEFINITE: $bits = SIGN | EXPONENT | QUIET;
            /// The largest finite value.
            const LARGEST: $bits = EXPONENT - 1;
            /// The precision in bits.
            const PRECISION: i32 = $precision;

            /// Returns the value of an encoding.
            pub fn value(bits: u64) -> $float {
                <$float>::from_bits(<$bits>::try_from(bits).expect("the encoding has the width"))
            }

            /// Returns the encoding of a value.
            fn bits(value: $float) -> $bits {
                value.to_bits()
            }

            /// Returns `x` as the instruction reads it under DAZ: a
            /// subnormal reads as a zero of its sign.
            fn read(x: $float, daz: bool) -> $float {
                if daz && x.is_subnormal() {
                    <$float>::from_bits(x.to_bits() & SIGN)
                } else {
                    x
                }
            }

            /// Returns the quiet NaN of a NaN.
            fn quiet(x: $float) -> $bits {
                x.to_bits() | QUIET
            }

            /// Returns the unbiased exponent `floor(log2 |x|)` of a finite
            /// nonzero value.
            fn exponent(x: $float) -> i32 {
                match x.log_b().to_int::<i32>() {
                    ToInt::Value(exponent) => exponent,
                    other => unreachable!("a finite nonzero value has an exponent, not {other:?}"),
                }
            }

            /// `VRNDSCALE` (Volume 2B, page 5-723): `2^-M round(2^M x)` in
            /// the direction of the immediate, `M` its bits 7 to 4. The
            /// precision exception signals when the result differs from the
            /// operand, unless bit 3 suppresses it. A NaN becomes quiet, and
            /// signals invalid when signaling; an infinity and a zero stay.
            pub fn rndscale(x: $float, immediate: u8, setting: &SseSetting) -> ($bits, Flags) {
                let scale = i32::from(immediate >> 4);
                if x.is_nan() {
                    let flags = if x.is_signaling_nan() {
                        Flags::INVALID
                    } else {
                        Flags::NONE
                    };
                    return (quiet(x), flags);
                }
                let read = read(x, setting.daz);
                if read.is_infinite() || read.is_zero() {
                    return (bits(read), Flags::NONE);
                }
                let result = if exponent(read) >= PRECISION - 1 - scale {
                    // `x` is a multiple of `2^-M` already: `2^M x` needs
                    // no rounding, whatever its exponent.
                    read
                } else {
                    let rounding = direction(immediate, setting);
                    let scaled = read.scale_b(scale);
                    let (rounded, _) =
                        scaled.round_to_integral_with(Env::IEEE.with_rounding(rounding));
                    rounded.scale_b(-scale)
                };
                let precision = immediate & 8 == 0 && bits(result) != bits(read);
                let flags = if precision {
                    Flags::INEXACT
                } else {
                    Flags::NONE
                };
                (bits(result), flags)
            }

            /// `VREDUCE` (Volume 2B, page 5-692): `x - 2^-M round(2^M x)`,
            /// the subtraction in the same direction. An infinity gives +0
            /// (Table 5-29). The hardware shows that only the subtraction
            /// signals, and only the precision exception: for an inexact
            /// difference, also one that FTZ flushes, unless bit 3 suppresses
            /// it. It signals no DE and no UE.
            pub fn reduce(x: $float, immediate: u8, setting: &SseSetting) -> ($bits, Flags) {
                if x.is_nan() {
                    let flags = if x.is_signaling_nan() {
                        Flags::INVALID
                    } else {
                        Flags::NONE
                    };
                    return (quiet(x), flags);
                }
                let read = read(x, setting.daz);
                if read.is_infinite() {
                    return (0, Flags::NONE);
                }
                let (rounded, _) = rndscale(read, immediate | 8, setting);
                let rounded = <$float>::from_bits(rounded);
                let env = setting.env().with_rounding(direction(immediate, setting));
                let (difference, flags) = read.sub_with(rounded, env);
                let mut flags = flags.difference(Flags::DENORMAL_INPUT | Flags::UNDERFLOW);
                if immediate & 8 != 0 {
                    flags = flags.difference(Flags::INEXACT);
                }
                (bits(difference), flags)
            }

            /// `VGETEXP` (Volume 2B, page 5-439): `floor(log2 |x|)`, -inf for
            /// a zero and +inf for an infinity, with DE for a subnormal read
            /// without DAZ (Table 5-17).
            pub fn getexp(x: $float, daz: bool) -> ($bits, Flags) {
                if x.is_nan() {
                    let flags = if x.is_signaling_nan() {
                        Flags::INVALID
                    } else {
                        Flags::NONE
                    };
                    return (quiet(x), flags);
                }
                let read = read(x, daz);
                let flags = if x.is_subnormal() && !daz {
                    Flags::DENORMAL_INPUT
                } else {
                    Flags::NONE
                };
                if read.is_zero() {
                    return (SIGN | INFINITY, flags);
                }
                if read.is_infinite() {
                    return (INFINITY, flags);
                }
                (bits(read.log_b()), flags)
            }

            /// `VGETMANT` (Volume 2B, page 5-447, `getmant_fp32`): the
            /// significand of `|x|` in the interval of bits 1 and 0 of the
            /// immediate, with the sign that bits 3 and 2 control.
            pub fn getmant(x: $float, immediate: u8, daz: bool) -> ($bits, Flags) {
                let (interval, sign_control) = (immediate & 3, (immediate >> 2) & 3);
                if x.is_nan() {
                    let flags = if x.is_signaling_nan() {
                        Flags::INVALID
                    } else {
                        Flags::NONE
                    };
                    return (quiet(x), flags);
                }
                let one = bits(<$float>::from_int(1));
                let signed_one = if sign_control & 1 != 0 {
                    one
                } else {
                    one | SIGN
                };
                let read = read(x, daz);
                let negative = x.is_sign_negative();
                if !negative && (read.is_zero() || read.is_infinite()) {
                    return (one, Flags::NONE);
                }
                if negative {
                    if read.is_zero() {
                        return (signed_one, Flags::NONE);
                    }
                    if sign_control & 2 != 0 {
                        return (INDEFINITE, Flags::INVALID);
                    }
                    if read.is_infinite() {
                        return (signed_one, Flags::NONE);
                    }
                }
                let flags = if read.is_subnormal() {
                    Flags::DENORMAL_INPUT
                } else {
                    Flags::NONE
                };
                let exponent = exponent(read);
                let significand = read.abs().scale_b(-exponent);
                let half = match interval {
                    0 => false,
                    1 => exponent % 2 != 0,
                    2 => true,
                    _ => significand >= <$float>::from_int(3).scale_b(-1),
                };
                let magnitude = if half {
                    significand.scale_b(-1)
                } else {
                    significand
                };
                let sign = if sign_control & 1 == 0 && negative {
                    SIGN
                } else {
                    0
                };
                (bits(magnitude) | sign, flags)
            }

            /// `VSCALEF` (Volume 2B, page 5-773): `x 2^floor(y)`, rounded and
            /// flagged as a product, with the special cases of Table 5-39.
            /// DE signals for a subnormal `x` alone.
            pub fn scalef(x: $float, y: $float, setting: &SseSetting) -> ($bits, Flags) {
                let signaling = x.is_signaling_nan() || y.is_signaling_nan();
                let invalid = if signaling {
                    Flags::INVALID
                } else {
                    Flags::NONE
                };
                if x.is_signaling_nan() {
                    return (quiet(x), invalid);
                }
                if x.is_nan() {
                    if y.is_nan() {
                        return (quiet(x), invalid);
                    }
                    if y.is_infinite() {
                        let result = if y.is_sign_negative() { 0 } else { INFINITY };
                        return (result, invalid);
                    }
                    return (quiet(x), invalid);
                }
                if y.is_nan() {
                    return (quiet(y), invalid);
                }
                let (x_read, y_read) = (read(x, setting.daz), read(y, setting.daz));
                let denormal = if x.is_subnormal() && !setting.daz {
                    Flags::DENORMAL_INPUT
                } else {
                    Flags::NONE
                };
                let sign = x.to_bits() & SIGN;
                if y.is_infinite() {
                    let down = y.is_sign_negative();
                    return match (x_read.is_infinite(), x_read.is_zero(), down) {
                        (true, _, true) | (_, true, false) => (INDEFINITE, Flags::INVALID),
                        (true, _, false) | (_, true, true) => (bits(x_read), Flags::NONE),
                        (false, false, false) => (sign | INFINITY, denormal),
                        (false, false, true) => (sign, denormal),
                    };
                }
                if x_read.is_infinite() || x_read.is_zero() {
                    return (bits(x_read), denormal);
                }
                let (floor, _) = y_read.round_to_integral_with(Rounding::TowardNegative);
                let scale = match floor.to_int::<i32>() {
                    ToInt::Value(scale) => scale,
                    ToInt::OutOfRange { negative: true } => i32::MIN,
                    _ => i32::MAX,
                };
                let (result, flags) = x_read.scale_b_with(scale, setting.env());
                (bits(result), flags | denormal)
            }

            /// `VRANGE` (Volume 2B, page 5-655, `RangeSP`): the minimum,
            /// maximum, minimum magnitude, or maximum magnitude of bits 1 and
            /// 0 of the immediate, a quiet NaN taken as missing, with the sign
            /// of bits 3 and 2 (Tables 5-23 to 5-25).
            pub fn range(x: $float, y: $float, immediate: u8, daz: bool) -> ($bits, Flags) {
                if x.is_signaling_nan() {
                    return (quiet(x), Flags::INVALID);
                }
                if y.is_signaling_nan() {
                    return (quiet(y), Flags::INVALID);
                }
                let mut flags = Flags::NONE;
                if x.is_subnormal() && !daz && !y.is_nan() {
                    flags |= Flags::DENORMAL_INPUT;
                }
                if y.is_subnormal() && !daz && !x.is_nan() {
                    flags |= Flags::DENORMAL_INPUT;
                }
                let (a, b) = (read(x, daz), read(y, daz));
                let operation = immediate & 3;
                let pick = if b.is_nan() {
                    a
                } else if a.is_nan() {
                    b
                } else if a.is_zero() && b.is_zero() && a.is_sign_negative() != b.is_sign_negative()
                {
                    let negative = operation & 1 == 0;
                    if negative == a.is_sign_negative() {
                        a
                    } else {
                        b
                    }
                } else if operation >= 2
                    && a.abs().to_bits() == b.abs().to_bits()
                    && a.is_sign_negative() != b.is_sign_negative()
                {
                    let negative = operation == 2;
                    if negative == a.is_sign_negative() {
                        a
                    } else {
                        b
                    }
                } else {
                    let (first, second) = if operation >= 2 {
                        (a.abs(), b.abs())
                    } else {
                        (a, b)
                    };
                    let first_smaller = first <= second;
                    if (operation & 1 == 0) == first_smaller {
                        a
                    } else {
                        b
                    }
                };
                let tail = bits(pick) & !SIGN;
                let result = match immediate >> 2 {
                    0 => tail | (x.to_bits() & SIGN),
                    1 => bits(pick),
                    2 => tail,
                    _ => tail | SIGN,
                };
                (result, flags)
            }

            /// `VFIXUPIMM` (Volume 2B, page 5-403, `FIXUPIMM_SP`): the token
            /// of `x` selects four bits of the table, which select the
            /// result, and the immediate selects the flags of the token. The
            /// hardware shows that the quiet NaN of a number keeps its
            /// fraction and sets the exponent field and the quiet bit, and
            /// that DAZ reads a subnormal as a zero of its sign.
            pub fn fixupimm(
                destination: $bits,
                x: $float,
                table: u32,
                immediate: u8,
                daz: bool,
            ) -> ($bits, Flags) {
                let source = read(x, daz);
                let one = <$float>::from_int(1);
                let token: u32 = if source.is_nan() {
                    if source.is_signaling_nan() { 1 } else { 0 }
                } else if source.is_zero() {
                    2
                } else if bits(source) == bits(one) {
                    3
                } else if source.is_infinite() {
                    if source.is_sign_negative() { 4 } else { 5 }
                } else if source.is_sign_negative() {
                    6
                } else {
                    7
                };
                let negative = source.to_bits() & SIGN;
                let result = match (table >> (4 * token)) & 15 {
                    0 => destination,
                    1 => bits(source),
                    2 => bits(source) | EXPONENT | QUIET,
                    3 => INDEFINITE,
                    4 => SIGN | INFINITY,
                    5 => INFINITY,
                    6 => negative | INFINITY,
                    7 => SIGN,
                    8 => 0,
                    9 => bits(one) | SIGN,
                    10 => bits(one),
                    11 => bits(one.scale_b(-1)),
                    12 => $ninety,
                    13 => $half_pi,
                    14 => LARGEST,
                    _ => LARGEST | SIGN,
                };
                let mut flags = Flags::NONE;
                let set = |bit: u8| immediate & (1 << bit) != 0;
                match token {
                    2 => {
                        if set(0) {
                            flags |= Flags::DIVIDE_BY_ZERO;
                        }
                        if set(1) {
                            flags |= Flags::INVALID;
                        }
                    }
                    3 => {
                        if set(2) {
                            flags |= Flags::DIVIDE_BY_ZERO;
                        }
                        if set(3) {
                            flags |= Flags::INVALID;
                        }
                    }
                    1 if set(4) => flags |= Flags::INVALID,
                    4 if set(5) => flags |= Flags::INVALID,
                    6 if set(6) => flags |= Flags::INVALID,
                    5 if set(7) => flags |= Flags::INVALID,
                    _ => {}
                }
                (result, flags)
            }

            /// `VFPCLASS` (Volume 2B, page 5-416, `CheckFPClassSP`): whether
            /// one category that a bit of the immediate selects holds. Under
            /// DAZ, a subnormal is a zero.
            pub fn fpclass(x: $float, immediate: u8, daz: bool) -> bool {
                let zero = x.is_zero() || (daz && x.is_subnormal());
                let negative = x.is_sign_negative();
                let categories = [
                    x.is_nan() && !x.is_signaling_nan(),
                    zero && !negative,
                    zero && negative,
                    x.is_infinite() && !negative,
                    x.is_infinite() && negative,
                    x.is_subnormal() && !daz,
                    negative && x.is_finite() && !zero,
                    x.is_signaling_nan(),
                ];
                categories
                    .iter()
                    .enumerate()
                    .any(|(bit, &holds)| holds && immediate & (1 << bit) != 0)
            }
        }
    };
}

recipes!(
    /// The recipes of the binary32 instructions, `SS`.
    single, floaty::F32, u32, precision: 24, half_pi: 0x3FC9_0FDB, ninety: 0x42B4_0000
);
recipes!(
    /// The recipes of the binary64 instructions, `SD`.
    double, floaty::F64, u64, precision: 53, half_pi: 0x3FF9_21FB_5444_2D18,
    ninety: 0x4056_8000_0000_0000
);
