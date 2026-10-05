//! The integer rounding of the host paths: binary32 results round to
//! bfloat16, and to binary16 in an x86-64 build without F16C, and binary64
//! results round to binary16 on x86-64, in integer instructions where the
//! host has no instruction for the rounding. The packed paths of a build
//! without a rounding instruction round binary32 lanes to binary16 in
//! integer instructions and one binary32 sum. Blocks of binary16 and
//! bfloat16 round each step in the routines of their lanes on every host,
//! with a few integer instructions that LLVM vectorizes.
//!
//! The binary16 and bfloat16 paths compute in binary32 and round twice, as
//! the module `paths` states. These routines are the second rounding. Each
//! rounds to nearest even, which the paths require of the mode, and gives a
//! NaN for a NaN, which the caller sends to the engine. The widening of
//! binary16 is exact.

/// Widens the bits of a binary16 value exactly to binary32 with integer
/// instructions, for a build without a widening instruction. A subnormal
/// binary16 value is a normal binary32 value.
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(target_feature = "f16c")
    ),
    target_arch = "s390x"
))]
#[inline]
pub fn widen_half_bits(bits: u16) -> u32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let field = u32::from((bits >> 10) & 0x1F);
    let fraction = u32::from(bits & 0x3FF);
    let magnitude = match field {
        0 if fraction == 0 => 0,
        0 => {
            // The shift moves the leading one of the fraction to bit 10,
            // which the binary32 encoding leaves implicit.
            let shift = fraction.leading_zeros() - 21;
            ((113 - shift) << 23) | (((fraction << shift) & 0x3FF) << 13)
        }
        0x1F => 0x7F80_0000 | (fraction << 13),
        _ => ((field + 112) << 23) | (fraction << 13),
    };
    sign | magnitude
}

/// Rounds the bits of a binary32 value to binary16, to nearest even, with
/// integer instructions, for a build without a rounding instruction.
///
/// A value at or above 65520, the midpoint between the largest finite value
/// and 2^16, gives the infinity. A normal result rebiases the exponent and
/// rounds away the low 13 bits, as `round_to_bfloat` rounds away 16. A NaN
/// gives a quiet NaN, which the caller sends to the engine.
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(target_feature = "f16c")
    ),
    target_arch = "s390x"
))]
#[inline]
pub fn round_to_half(bits: u32) -> u16 {
    let sign = (bits >> 16) & 0x8000;
    let magnitude = bits & 0x7FFF_FFFF;
    let rounded = if magnitude > 0x7F80_0000 {
        0x7E00 | ((magnitude >> 13) & 0x3FF)
    } else if magnitude >= 0x477F_F000 {
        0x7C00
    } else if magnitude >= 0x3880_0000 {
        (magnitude - 0x3800_0000 + 0xFFF + ((magnitude >> 13) & 1)) >> 13
    } else {
        round_to_subnormal_half(magnitude)
    };
    u16::try_from(sign | rounded).expect("a binary16 encoding has 16 bits")
}

/// Rounds a binary32 magnitude below 2^-14 to a count of the binary16
/// quantum 2^-24, to nearest even. A count of 2^10 is the encoding of the
/// smallest normal value.
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(target_feature = "f16c")
    ),
    target_arch = "s390x"
))]
#[inline]
fn round_to_subnormal_half(magnitude: u32) -> u32 {
    // The value is `significand * 2^(field - 150)`, so the count is
    // `significand * 2^(field - 126)`. Below 2^-25 the count rounds to zero.
    let field = magnitude >> 23;
    let shift = 126 - field;
    if shift > 25 {
        return 0;
    }
    let significand = (magnitude & 0x7F_FFFF) | 0x80_0000;
    let count = significand >> shift;
    let rest = significand & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    let up = rest > half || (rest == half && count & 1 == 1);
    count + u32::from(up)
}

/// Rounds the bits of a binary64 value to binary16, to nearest even, with
/// integer instructions. x86-64 has no instruction for this rounding below
/// AVX512-FP16, and F16C rounds only binary32, which would round twice.
///
/// A value at or above 65520, the midpoint between the largest finite value
/// and 2^16, gives the infinity. A normal result rebiases the exponent and
/// rounds away the low 42 bits. A NaN gives a quiet NaN, which the caller
/// sends to the engine.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "s390x"))]
#[inline]
pub fn round_double_to_half(bits: u64) -> u16 {
    let sign = (bits >> 48) & 0x8000;
    let magnitude = bits & 0x7FFF_FFFF_FFFF_FFFF;
    let rounded = if magnitude > 0x7FF0_0000_0000_0000 {
        0x7E00 | ((magnitude >> 42) & 0x3FF)
    } else if magnitude >= 0x40EF_FE00_0000_0000 {
        0x7C00
    } else if magnitude >= 0x3F10_0000_0000_0000 {
        // The exponent field moves from the bias 1023 to the bias 15.
        (magnitude - (1008 << 52) + 0x1FF_FFFF_FFFF + ((magnitude >> 42) & 1)) >> 42
    } else {
        round_double_to_subnormal_half(magnitude)
    };
    u16::try_from(sign | rounded).expect("a binary16 encoding has 16 bits")
}

/// Rounds a binary64 magnitude below 2^-14 to a count of the binary16
/// quantum 2^-24, to nearest even. A count of 2^10 is the encoding of the
/// smallest normal value.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "s390x"))]
#[inline]
fn round_double_to_subnormal_half(magnitude: u64) -> u64 {
    // The value is `significand * 2^(field - 1075)`, so the count is
    // `significand * 2^(field - 1051)`. Below 2^-25 the count rounds to zero,
    // and so does a zero or a subnormal binary64 value.
    let field = magnitude >> 52;
    let shift = 1051 - field;
    if shift > 53 {
        return 0;
    }
    let significand = (magnitude & 0xF_FFFF_FFFF_FFFF) | (1 << 52);
    let count = significand >> shift;
    let rest = significand & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    let up = rest > half || (rest == half && count & 1 == 1);
    count + u64::from(up)
}

/// The encoding of 0.5, whose unit in the last place is 2^-24, the quantum
/// of the binary16 subnormal values.
pub const SUBNORMAL_BIAS: u32 = 0x3F00_0000;

/// Rounds the bits of a binary32 value to binary16, to nearest even, in
/// masks without a branch, which LLVM vectorizes, for the packed paths of a
/// build without a rounding instruction. It gives the bits of
/// `round_to_half` for every value but a NaN, which gives a quiet NaN.
///
/// `subnormal` is the encoding of the binary32 sum of the magnitude of the
/// value and 0.5 ([`SUBNORMAL_BIAS`]), rounded to nearest even by the host.
/// Below 2^-14 the sum is less than 0.5 + 2^-14, and the unit of its last
/// place is 2^-24, so the sum rounds the magnitude to a count of 2^-24, to
/// nearest even, and the count is the difference of the encodings. A count
/// of 2^10 is the encoding of the smallest normal value. The other cases are
/// those of `round_to_half`.
#[inline]
pub fn round_to_half_lanes(bits: u32, subnormal: u32) -> u16 {
    let sign = (bits >> 16) & 0x8000;
    let magnitude = bits & 0x7FFF_FFFF;
    let mask = |condition: bool| 0u32.wrapping_sub(u32::from(condition));
    // Each mask selects one case; the cases do not overlap.
    let tiny = mask(magnitude < 0x3880_0000);
    let normal = mask((0x3880_0000..0x477F_F000).contains(&magnitude));
    let infinite = mask((0x477F_F000..=0x7F80_0000).contains(&magnitude));
    let nan = mask(magnitude > 0x7F80_0000);
    // The lanes of the other cases compute garbage, which their masks drop,
    // so the arithmetic wraps.
    let count = subnormal.wrapping_sub(SUBNORMAL_BIAS);
    let rounded = magnitude
        .wrapping_sub(0x3800_0000)
        .wrapping_add(0xFFF + ((magnitude >> 13) & 1))
        >> 13;
    let quiet = 0x7E00 | ((magnitude >> 13) & 0x3FF);
    let result = sign | (tiny & count) | (normal & rounded) | (infinite & 0x7C00) | (nan & quiet);
    // The result has 16 bits: the low two bytes hold it on every host.
    let [low, high, _, _] = result.to_le_bytes();
    u16::from_le_bytes([low, high])
}

/// The encoding of 2^-24, the quantum of the binary16 subnormal values: the
/// fraction of a subnormal binary16 value times 2^-24 is its magnitude.
pub const HALF_QUANTUM: u32 = 0x3380_0000;

/// Returns the binary32 encoding of a binary16 encoding, in masks without a
/// branch, which LLVM vectorizes. `subnormal` is the binary32 encoding of
/// the magnitude of the value when its exponent field is zero: the fraction
/// times 2^-24 ([`HALF_QUANTUM`]), a normal binary32 value, whose conversion
/// and product are exact. Otherwise the exponent field and the fraction move
/// to their binary32 positions, and the exponent field takes the bias of
/// binary32: 127 - 15 = 112 more for a normal value, and 255 - 31 = 224 more
/// for an infinity or a NaN.
#[inline]
pub fn widen_half_lanes(bits: u16, subnormal: u32) -> u32 {
    let bits = u32::from(bits);
    let sign = (bits & 0x8000) << 16;
    let magnitude = (bits & 0x7FFF) << 13;
    let field = bits & 0x7C00;
    let zero_field = 0u32.wrapping_sub(u32::from(field == 0));
    let full_field = 0u32.wrapping_sub(u32::from(field == 0x7C00));
    let other = magnitude + (112 << 23) + (full_field & (112 << 23));
    sign | (zero_field & subnormal) | (!zero_field & other)
}

/// Returns the encoding of the power of two whose sum with the magnitude of
/// a binary32 value, rounded to nearest even, rounds the magnitude to the
/// precision of binary16: 2^13 times the power of two of the exponent of
/// the magnitude, and at least 0.5 ([`SUBNORMAL_BIAS`]). The unit of the
/// last place of the sum is then the unit of binary16 at that exponent, and
/// below 2^-14 the binary16 quantum 2^-24. From 2^115 up the encoding is not
/// that power, and `round_to_half_precision` gives the infinity.
#[inline]
pub fn half_precision_bias(bits: u32) -> u32 {
    ((bits & 0x7F80_0000) + (13 << 23)).max(SUBNORMAL_BIAS)
}

/// Rounds the bits of a binary32 value to the nearest binary16 value, ties
/// to even, and returns that value as binary32 bits, for the steps of a
/// binary16 block. It gives `round_to_half` widened to binary32 for every
/// value but a NaN, which gives a quiet NaN.
///
/// `rounded` is the encoding of `(magnitude + bias) - bias`, each rounded to
/// nearest even by the host, with the bias of `half_precision_bias`. The sum
/// rounds the magnitude to binary16 precision, and the difference is exact.
/// From 65520 up, the result is the infinity. A NaN gives the NaN of the
/// host sum, which is quiet. The few instructions keep a chain small enough
/// that LLVM inlines it into the loop of a block and vectorizes the loop.
#[inline]
pub fn round_to_half_precision(bits: u32, rounded: u32) -> u32 {
    let magnitude = bits & 0x7FFF_FFFF;
    let result = if (0x477F_F000..=0x7F80_0000).contains(&magnitude) {
        0x7F80_0000
    } else {
        rounded
    };
    (bits & 0x8000_0000) | result
}

/// Returns the encoding of the power of two whose sum with the magnitude of
/// a binary64 value, rounded to nearest even, rounds the magnitude to the
/// precision of binary16: 2^42 times the power of two of the exponent of
/// the magnitude, and at least 2^28, whose unit in the last place is the
/// binary16 quantum 2^-24. From 2^982 up the encoding is not that power, and
/// `round_double_to_half_precision` gives the infinity.
#[inline]
pub fn double_half_precision_bias(bits: u64) -> u64 {
    ((bits & 0x7FF0_0000_0000_0000) + (42 << 52)).max(0x41B0_0000_0000_0000)
}

/// Rounds the bits of a binary64 value to the nearest binary16 value, ties
/// to even, and returns that value as binary32 bits, for the `mul_add` of a
/// binary16 block. It gives `round_double_to_half` widened to binary32 for
/// every value but a NaN, which gives a quiet NaN.
///
/// `rounded` is the encoding of `(magnitude + bias) - bias`, each rounded to
/// nearest even by the host in binary64, with the bias of
/// `double_half_precision_bias`. The sum rounds the magnitude to binary16
/// precision, and the difference is exact. A binary16 value has at most 11
/// bits and is normal in binary32, so the rebias of its binary64 encoding
/// and the shift by 29 bits give its binary32 encoding exactly. From 65520
/// up, the result is the infinity.
#[inline]
pub fn round_double_to_half_precision(bits: u64, rounded: u64) -> u32 {
    let magnitude = bits & 0x7FFF_FFFF_FFFF_FFFF;
    let single = if magnitude > 0x7FF0_0000_0000_0000 {
        0x7FC0_0000
    } else if magnitude >= 0x40EF_FE00_0000_0000 {
        0x7F80_0000
    } else if rounded == 0 {
        0
    } else {
        // The exponent field moves from the bias 1023 to the bias 127.
        let single = (rounded >> 29).wrapping_sub(896 << 23) & 0xFFFF_FFFF;
        u32::try_from(single).expect("the mask keeps 32 bits")
    };
    (u32::from(bits >> 63 == 1) << 31) | single
}

/// Rounds the bits of a binary32 value to bfloat16, to nearest even, with
/// integer instructions. x86-64 has no instruction for this rounding below
/// `AVX512_BF16`, and `VCVTNEPS2BF16` reads a subnormal input as zero.
///
/// A bfloat16 encoding is the high half of a binary32 encoding. The rounding
/// adds one less than half a unit of the high half, plus one when the high
/// half is odd, and drops the low half. A carry out of the fraction raises
/// the exponent, and a carry past the largest finite value gives the
/// infinity. A NaN gives a quiet NaN, which the caller sends to the engine.
#[inline]
pub fn round_to_bfloat(bits: u32) -> u16 {
    let high = bits >> 16;
    let rounded = if super::bits::nan_32(bits) {
        high | 0x0040
    } else {
        (bits + 0x7FFF + (high & 1)) >> 16
    };
    u16::try_from(rounded).expect("a shift by 16 leaves 16 bits")
}

#[cfg(test)]
mod tests {
    use crate::env::Env;
    use crate::float::{BF16, F32};

    /// The low halves of the test encodings: zero, a tie at each bit from 12
    /// to 15 with every higher low bit, and values just around the ties.
    const LOWS: [u32; 22] = [
        0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000, 0x8000, 0x9000, 0xA000,
        0xB000, 0xC000, 0xD000, 0xE000, 0xF000, 0x0001, 0x0FFF, 0x1001, 0x7FFF, 0x8001, 0xFFFF,
    ];

    /// Returns binary32 encodings with every high half and each low half of
    /// `LOWS`. They reach each rounding case of both roundings.
    fn encodings() -> impl Iterator<Item = u32> {
        (0..=u32::from(u16::MAX)).flat_map(|high| LOWS.map(|low| (high << 16) | low))
    }

    #[test]
    fn bfloat16_rounding_matches_the_engine() {
        for bits in encodings() {
            let ours = super::round_to_bfloat(bits);
            let (engine, _) = F32::from_bits(bits).convert_with::<BF16>(Env::IEEE);
            if super::super::bits::nan_32(bits) {
                assert!(super::super::bits::nan_bfloat(ours), "{bits:#010x}");
            } else {
                assert_eq!(ours, engine.to_bits(), "{bits:#010x}");
            }
        }
    }

    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            not(target_feature = "f16c")
        ),
        target_arch = "s390x"
    ))]
    #[test]
    fn binary16_widening_matches_the_engine() {
        use crate::float::F16;
        for bits in 0..=u16::MAX {
            let ours = super::widen_half_bits(bits);
            let (engine, _) = F16::from_bits(bits).convert_with::<F32>(Env::IEEE);
            if super::super::bits::nan_16(bits) {
                assert!(super::super::bits::nan_32(ours), "{bits:#06x}");
            } else {
                assert_eq!(ours, engine.to_bits(), "{bits:#06x}");
            }
        }
    }

    /// Returns binary64 encodings with every exponent field near the range
    /// of binary16 and below it, each pattern of the ten fraction bits that
    /// binary16 keeps, and low bits that make the dropped part zero, a tie,
    /// or just around one, of both signs, and the special values.
    fn doubles_near_half() -> impl Iterator<Item = u64> {
        let lows = [0, 1, (1 << 41) - 1, 1 << 41, (1 << 41) + 1, (1 << 42) - 1];
        let specials = [
            0_u64,
            0x7FF0_0000_0000_0000,
            0x7FF8_0000_0000_0001,
            0x7FF0_0000_0000_0001,
            0x0000_0000_0000_0001,
            0x40EF_FE00_0000_0000,
            0x40EF_FDFF_FFFF_FFFF,
        ];
        let tops = (980_u64..=1050).flat_map(|field| (0..1024).map(move |high| (field, high)));
        let near =
            tops.flat_map(move |(field, high)| lows.map(|low| (field << 52) | (high << 42) | low));
        specials
            .into_iter()
            .chain(near)
            .flat_map(|magnitude| [magnitude, magnitude | (1 << 63)])
    }

    /// Checks the rounding of binary64 to binary16 against the engine.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "s390x"))]
    #[test]
    fn binary64_to_binary16_rounding_matches_the_engine() {
        use crate::float::{F16, F64};
        for bits in doubles_near_half() {
            let ours = super::round_double_to_half(bits);
            let (engine, _) = F64::from_bits(bits).convert_with::<F16>(Env::IEEE);
            if super::super::bits::nan_64(bits) {
                assert!(super::super::bits::nan_16(ours), "{bits:#018x}");
            } else {
                assert_eq!(ours, engine.to_bits(), "{bits:#018x}");
            }
        }
    }

    /// Checks the rounding of binary64 to binary16 precision against the
    /// engine. The sum of each magnitude and its bias, and the difference of
    /// the sum and the bias, come from the engine, which the host gives in a
    /// block.
    #[test]
    fn binary64_to_binary16_precision_rounding_matches_the_engine() {
        use crate::float::{F16, F64};
        for bits in doubles_near_half() {
            let magnitude = F64::from_bits(bits & 0x7FFF_FFFF_FFFF_FFFF);
            let bias = F64::from_bits(super::double_half_precision_bias(bits));
            let (sum, _) = magnitude.add_with(bias, Env::IEEE);
            let (rounded, _) = sum.sub_with(bias, Env::IEEE);
            let ours = super::round_double_to_half_precision(bits, rounded.to_bits());
            let (half, _) = F64::from_bits(bits).convert_with::<F16>(Env::IEEE);
            let (engine, _) = half.convert_with::<F32>(Env::IEEE);
            if super::super::bits::nan_64(bits) {
                assert_eq!(ours & 0x7FC0_0000, 0x7FC0_0000, "{bits:#018x}");
            } else {
                assert_eq!(ours, engine.to_bits(), "{bits:#018x}");
            }
        }
    }

    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            not(target_feature = "f16c")
        ),
        target_arch = "s390x"
    ))]
    #[test]
    fn binary16_rounding_matches_the_engine() {
        use crate::float::F16;
        for bits in encodings() {
            let ours = super::round_to_half(bits);
            let (engine, _) = F32::from_bits(bits).convert_with::<F16>(Env::IEEE);
            if super::super::bits::nan_32(bits) {
                assert!(super::super::bits::nan_16(ours), "{bits:#010x}");
            } else {
                assert_eq!(ours, engine.to_bits(), "{bits:#010x}");
            }
        }
    }

    /// The sum of each magnitude and 0.5 comes from the engine, which the
    /// host sum gives in the packed paths.
    #[test]
    fn binary16_lane_rounding_matches_the_engine() {
        use crate::float::F16;
        let half = F32::from_bits(super::SUBNORMAL_BIAS);
        for bits in encodings() {
            let magnitude = F32::from_bits(bits & 0x7FFF_FFFF);
            let (sum, _) = magnitude.add_with(half, Env::IEEE);
            let ours = super::round_to_half_lanes(bits, sum.to_bits());
            let (engine, _) = F32::from_bits(bits).convert_with::<F16>(Env::IEEE);
            if super::super::bits::nan_32(bits) {
                assert!(super::super::bits::nan_16(ours), "{bits:#010x}");
            } else {
                assert_eq!(ours, engine.to_bits(), "{bits:#010x}");
            }
        }
    }

    /// The sum of each magnitude and its bias, and the difference of the sum
    /// and the bias, come from the engine, which the host gives in a block.
    #[test]
    fn binary16_precision_rounding_matches_the_engine() {
        use crate::float::F16;
        for bits in encodings() {
            let magnitude = F32::from_bits(bits & 0x7FFF_FFFF);
            let bias = F32::from_bits(super::half_precision_bias(bits));
            let (sum, _) = magnitude.add_with(bias, Env::IEEE);
            let (rounded, _) = sum.sub_with(bias, Env::IEEE);
            let ours = super::round_to_half_precision(bits, rounded.to_bits());
            let (rounded, _) = F32::from_bits(bits).convert_with::<F16>(Env::IEEE);
            let (engine, _) = rounded.convert_with::<F32>(Env::IEEE);
            if super::super::bits::nan_32(bits) {
                assert!(super::super::bits::nan_32(ours), "{bits:#010x}");
                assert_ne!(ours & 0x0040_0000, 0, "{bits:#010x}");
            } else {
                assert_eq!(ours, engine.to_bits(), "{bits:#010x}");
            }
        }
    }
}
