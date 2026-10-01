//! The field layouts of the binary formats, and test encodings at the field
//! boundaries of a layout and at the rounding edges.
//!
//! [`Layout`], [`boundary_encodings_u128`], [`sample_encodings_u128`],
//! [`rounding_edges`], and [`integer_edges`] build on every target. The functions on MPFR's integers
//! build only for x86-64, with MPFR.

use core::ops::RangeInclusive;

#[cfg(target_arch = "x86_64")]
use rug::Integer;

use crate::random::SplitMix64;

/// Whether a format stores its integer bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerBit {
    /// The integer bit is implicit, as in the IEEE 754 formats.
    Implicit,
    /// The integer bit is the top fraction bit, as in the x87 format.
    Explicit,
}

/// The field layout of a binary format: its width, the width of its exponent
/// field, and whether it stores its integer bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// The width in bits.
    pub width: u32,
    /// The width of the exponent field.
    pub exponent_bits: u32,
    /// Whether the format stores its integer bit.
    pub integer_bit: IntegerBit,
}

impl Layout {
    /// binary16.
    pub const BINARY16: Self = Self::ieee(16, 5);
    /// bfloat16.
    pub const BFLOAT16: Self = Self::ieee(16, 8);
    /// TF32, 19 bits wide.
    pub const TF32: Self = Self::ieee(19, 8);
    /// binary32.
    pub const BINARY32: Self = Self::ieee(32, 8);
    /// binary64.
    pub const BINARY64: Self = Self::ieee(64, 11);
    /// x87 extended precision, with its explicit integer bit.
    pub const X87_EXTENDED: Self = Self {
        width: 80,
        exponent_bits: 15,
        integer_bit: IntegerBit::Explicit,
    };
    /// binary128.
    pub const BINARY128: Self = Self::ieee(128, 15);
    /// binary160.
    pub const BINARY160: Self = Self::ieee(160, 16);
    /// binary192.
    pub const BINARY192: Self = Self::ieee(192, 17);
    /// binary224.
    pub const BINARY224: Self = Self::ieee(224, 18);
    /// binary256.
    pub const BINARY256: Self = Self::ieee(256, 19);
    /// binary288.
    pub const BINARY288: Self = Self::ieee(288, 20);
    /// binary320.
    pub const BINARY320: Self = Self::ieee(320, 20);
    /// binary352.
    pub const BINARY352: Self = Self::ieee(352, 21);
    /// binary384.
    pub const BINARY384: Self = Self::ieee(384, 21);
    /// binary416.
    pub const BINARY416: Self = Self::ieee(416, 22);
    /// binary448.
    pub const BINARY448: Self = Self::ieee(448, 22);
    /// binary480.
    pub const BINARY480: Self = Self::ieee(480, 23);
    /// binary512.
    pub const BINARY512: Self = Self::ieee(512, 23);

    /// Returns the layout of a format `width` bits wide with `exponent_bits`
    /// exponent bits and an implicit integer bit, as in the IEEE 754 formats.
    #[must_use]
    pub const fn ieee(width: u32, exponent_bits: u32) -> Self {
        Self {
            width,
            exponent_bits,
            integer_bit: IntegerBit::Implicit,
        }
    }

    /// Returns the number of bits below the exponent field, with a stored
    /// integer bit.
    #[must_use]
    pub const fn fraction_bits(self) -> u32 {
        self.width - 1 - self.exponent_bits
    }

    /// Returns the precision in bits.
    #[must_use]
    pub const fn precision(self) -> u32 {
        match self.integer_bit {
            IntegerBit::Implicit => self.fraction_bits() + 1,
            IntegerBit::Explicit => self.fraction_bits(),
        }
    }

    /// Returns the largest exponent field, all ones: the field of the
    /// infinities and the NaNs of an IEEE format.
    #[must_use]
    pub const fn largest_field(self) -> u32 {
        (1 << self.exponent_bits) - 1
    }

    /// Returns the exponent bias by the rule of IEEE 754, `2^(e - 1) - 1` for
    /// `e` exponent bits. The x87 format has the same bias. A format with
    /// another bias, such as a Fnuz format, does not use this method.
    #[must_use]
    pub const fn ieee_bias(self) -> i32 {
        (1 << (self.exponent_bits - 1)) - 1
    }

    /// Returns the encoding of a sign, an exponent field, and a fraction in a
    /// format of at most 128 bits. The bits of `fraction` above the fraction
    /// field do not change the encoding.
    ///
    /// # Panics
    ///
    /// Panics when the format has more than 128 bits.
    #[must_use]
    pub fn encode(self, negative: bool, field: u64, fraction: u128) -> u128 {
        assert!(self.width <= 128, "the format has at most 128 bits");
        let fraction_bits = self.fraction_bits();
        let fraction = fraction & ((1 << fraction_bits) - 1);
        (u128::from(negative) << (self.width - 1)) | (u128::from(field) << fraction_bits) | fraction
    }

    /// Returns the encoding of [`Layout::encode`] in a format of at most 64
    /// bits, as a `u64`.
    ///
    /// # Panics
    ///
    /// Panics when the encoding does not fit a `u64`.
    #[must_use]
    pub fn encode_u64(self, negative: bool, field: u64, fraction: u64) -> u64 {
        u64::try_from(self.encode(negative, field, u128::from(fraction)))
            .expect("the format has at most 64 bits")
    }
}

/// Returns the patterns of `dropped` low bits that decide a rounding: exact,
/// just below halfway, halfway, just above halfway, and all ones. With no
/// dropped bit, every pattern is zero.
#[must_use]
pub fn rounding_edges(dropped: u32) -> [u64; 5] {
    if dropped == 0 {
        return [0; 5];
    }
    let half = 1_u64 << (dropped - 1);
    let mask = (half << 1).wrapping_sub(1);
    [0, half - 1, half, half + 1, mask].map(|pattern| pattern & mask)
}

/// Returns encodings around the integers with the unbiased exponents
/// `exponents`, for both signs, in a format of at most 64 significand bits.
///
/// The significand bits below the binary point take each pattern of
/// [`rounding_edges`], and the bits above the binary point are all zeros,
/// all ones, or random. A value below one has every significand bit below
/// the binary point. A stored integer bit is set.
///
/// # Panics
///
/// Panics for a format with more than 64 significand bits, for an exponent
/// outside the normal range, and when an encoding does not fit `T`.
pub fn integer_edges<T: TryFrom<u128>>(
    layout: Layout,
    random: &mut SplitMix64,
    exponents: RangeInclusive<i32>,
) -> Vec<T> {
    let trailing = layout.precision() - 1;
    assert!(trailing < 64, "the format has at most 64 significand bits");
    let integer = match layout.integer_bit {
        IntegerBit::Implicit => 0,
        IntegerBit::Explicit => 1 << trailing,
    };
    let trailing_bits = i32::try_from(trailing).expect("at most 63 bits");
    let mut encodings = Vec::new();
    for exponent in exponents {
        let dropped = u32::try_from((trailing_bits - exponent).clamp(0, trailing_bits))
            .expect("between 0 and the trailing width");
        let all_ones = (1 << (trailing - dropped)) - 1;
        let field = u64::try_from(exponent + layout.ieee_bias()).expect("a normal exponent");
        for high in [0, all_ones, random.next_u64(), random.next_u64()] {
            for pattern in rounding_edges(dropped) {
                for negative in [false, true] {
                    let significand = integer | ((high & all_ones) << dropped) | pattern;
                    let bits = layout.encode(negative, field, u128::from(significand));
                    encodings.push(
                        T::try_from(bits)
                            .ok()
                            .expect("the encoding fits the storage"),
                    );
                }
            }
        }
    }
    encodings
}

/// Returns the encodings of `boundary_encodings` for a format of at most 128
/// bits, as `u128` values. The patterns are the same, and a test checks that
/// both functions give the same encodings.
///
/// # Panics
///
/// Panics when the format has more than 128 bits.
#[must_use]
pub fn boundary_encodings_u128(layout: Layout) -> Vec<u128> {
    let Layout {
        width, integer_bit, ..
    } = layout;
    assert!(width <= 128, "the format has at most 128 bits");
    let fraction_bits = layout.fraction_bits();
    let field_max = u128::from(layout.largest_field());
    let fraction_all = (1_u128 << fraction_bits) - 1;
    let top = 1_u128 << (fraction_bits - 1);
    let next = 1_u128 << (fraction_bits - 2);
    let mut fractions = vec![0, 1, top, top | 1, next, next | 1, fraction_all];
    if integer_bit == IntegerBit::Explicit {
        let quiet = next;
        let low_quiet = 1_u128 << (fraction_bits - 3);
        fractions.extend([
            quiet | 1,
            top | quiet,
            top | quiet | 1,
            top | low_quiet,
            fraction_all >> 1,
        ]);
    }
    let fields = [0, 1, 2, field_max >> 1, field_max - 1, field_max];
    let mut encodings = Vec::new();
    for sign in [0, 1_u128 << (width - 1)] {
        for &field in &fields {
            for &fraction in &fractions {
                encodings.push(sign | (field << fraction_bits) | fraction);
            }
        }
    }
    encodings
}

/// Returns the boundary encodings and `count` random encodings of a format
/// of at most 128 bits. One random encoding in 16 gets the exponent field of
/// the NaNs. Three x87 encodings in four get the integer bit of a canonical
/// encoding, and the fourth keeps a random integer bit, which gives unnormals
/// and pseudo-denormals.
///
/// # Panics
///
/// Panics when the format has more than 128 bits.
#[must_use]
pub fn sample_encodings_u128(layout: Layout, count: usize, random: &mut SplitMix64) -> Vec<u128> {
    let mask = u128::MAX >> (128 - layout.width);
    let fraction_bits = layout.fraction_bits();
    let nan_field = u128::from(layout.largest_field()) << fraction_bits;
    let mut encodings = boundary_encodings_u128(layout);
    encodings.extend((0..count).map(|index| {
        let mut bits = random.next_u128() & mask;
        if index % 16 == 0 {
            bits |= nan_field;
        }
        if layout.integer_bit == IntegerBit::Explicit && index % 4 != 0 {
            let integer_bit = 1 << (fraction_bits - 1);
            let field_is_zero = bits & (u128::from(layout.largest_field()) << fraction_bits) == 0;
            bits = if field_is_zero {
                bits & !integer_bit
            } else {
                bits | integer_bit
            };
        }
        bits
    }));
    encodings
}

/// Returns encodings that exercise every field boundary of a binary format
/// with the layout `layout`.
///
/// Each encoding combines a sign, an exponent field from both ends and the
/// middle of its range, and a fraction from a set of edge patterns.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn boundary_encodings(layout: Layout) -> Vec<Integer> {
    let Layout {
        width, integer_bit, ..
    } = layout;
    let fraction_bits = layout.fraction_bits();
    let field_max = Integer::from(layout.largest_field());
    let fraction_all = (Integer::from(1) << fraction_bits) - 1u32;
    let top = Integer::from(1) << (fraction_bits - 1);
    let next = Integer::from(1) << (fraction_bits - 2);
    let mut fractions = vec![
        Integer::ZERO,
        Integer::from(1),
        top.clone(),
        top.clone() | 1u32,
        next.clone(),
        next.clone() | 1u32,
        fraction_all.clone(),
    ];
    if integer_bit == IntegerBit::Explicit {
        let quiet = next.clone();
        let low_quiet = Integer::from(1) << (fraction_bits - 3);
        fractions.extend([
            quiet.clone() | 1u32,
            top.clone() | &quiet,
            top.clone() | &quiet | 1u32,
            top | low_quiet,
            fraction_all >> 1u32,
        ]);
    }
    let fields = [
        Integer::ZERO,
        Integer::from(1),
        Integer::from(2),
        field_max.clone() >> 1u32,
        field_max.clone() - 1u32,
        field_max,
    ];
    let mut encodings = Vec::new();
    for sign in [Integer::ZERO, Integer::from(1) << (width - 1)] {
        for field in &fields {
            for fraction in &fractions {
                encodings.push(sign.clone() | (field.clone() << fraction_bits) | fraction);
            }
        }
    }
    encodings
}

/// Converts an encoding to a `u128`.
///
/// # Panics
///
/// Panics when the encoding does not fit in 128 bits.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn to_u128(encoding: &Integer) -> u128 {
    encoding.to_u128().expect("the encoding fits in 128 bits")
}

/// Converts an encoding to `N` little-endian limbs.
///
/// # Panics
///
/// Panics when the encoding does not fit in `N` limbs.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn to_limbs<const N: usize>(encoding: &Integer) -> [u64; N] {
    let digits = encoding.to_digits::<u64>(rug::integer::Order::Lsf);
    assert!(digits.len() <= N, "the encoding fits in {N} limbs");
    let mut limbs = [0; N];
    limbs[..digits.len()].copy_from_slice(&digits);
    limbs
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::{Layout, boundary_encodings, boundary_encodings_u128, to_u128};

    #[test]
    fn both_forms_give_the_same_encodings() {
        let layouts = [
            Layout::ieee(8, 4),
            Layout::BINARY16,
            Layout::BFLOAT16,
            Layout::TF32,
            Layout::BINARY32,
            Layout::BINARY64,
            Layout::X87_EXTENDED,
            Layout::BINARY128,
        ];
        for layout in layouts {
            let shared: Vec<u128> = boundary_encodings(layout).iter().map(to_u128).collect();
            assert_eq!(boundary_encodings_u128(layout), shared, "{layout:?}");
        }
    }
}
