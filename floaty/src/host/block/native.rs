//! The host types of the lanes of a block, `f32` and `f64`, with the
//! encoding and the constants of their formats, and the instructions of
//! their fused multiply-add and square root.

use core::fmt::Debug;
use core::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Not, Shl, Sub};

use crate::host::environment::{
    block_mul_add_f32, block_mul_add_f64, block_sqrt_f32, block_sqrt_f64,
};

/// A host type of the lanes of a block: `f32` for binary32 and `f64` for
/// binary64, with the encoding and the constants of its format, and the
/// instructions of the fused multiply-add and the square root.
pub trait Native:
    Copy
    + Debug
    + PartialOrd
    + From<i16>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// The encoding: `u32` or `u64`.
    type Bits: Copy
        + Eq
        + Ord
        + From<u16>
        + Not<Output = Self::Bits>
        + BitAnd<Output = Self::Bits>
        + BitOr<Output = Self::Bits>
        + Add<Output = Self::Bits>
        + Sub<Output = Self::Bits>
        + Shl<u32, Output = Self::Bits>;

    /// The fraction bits of the encoding.
    const FRACTION_BITS: u8;
    /// The bias of the exponent field.
    const BIAS: i16;
    /// The sign bit of the encoding.
    const SIGN: Self::Bits;
    /// The bit that makes a NaN quiet.
    const QUIET: Self::Bits;
    /// A quiet NaN.
    const NAN: Self;
    /// Positive infinity.
    const INFINITY: Self;
    /// One half.
    const HALF: Self;
    /// `2^FRACTION_BITS`: from this value up, every value is integral, and a
    /// subnormal value times it is a normal value, exactly.
    const INTEGRAL: Self;

    /// Returns the encoding of the value.
    fn to_bits(self) -> Self::Bits;

    /// Returns the value of the encoding `bits`.
    fn from_bits(bits: Self::Bits) -> Self;

    /// Returns `true` when the value is a NaN, by a float compare.
    fn is_nan(self) -> bool;

    /// Returns the exponent field of the encoding `bits` of a magnitude.
    fn exponent_field(bits: Self::Bits) -> i16;

    /// Returns the square root, by the instruction of the format.
    fn block_sqrt(self) -> Self;

    /// Returns `self * multiplier + addend`, rounded once, by the fused
    /// multiply-add of the format.
    ///
    /// # Safety
    ///
    /// The instruction set of the caller must have the fused multiply-add.
    unsafe fn block_mul_add(self, multiplier: Self, addend: Self) -> Self;

    /// Returns `true` when the encoding `bits` is a NaN, by integer
    /// operations.
    #[inline]
    fn is_nan_bits(bits: Self::Bits) -> bool {
        bits & !Self::SIGN > Self::INFINITY.to_bits()
    }
}

impl Native for f32 {
    type Bits = u32;

    const FRACTION_BITS: u8 = 23;
    const BIAS: i16 = 127;
    const SIGN: u32 = 0x8000_0000;
    const QUIET: u32 = 0x0040_0000;
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    const HALF: Self = 0.5;
    const INTEGRAL: Self = 8_388_608.0;

    #[inline]
    fn to_bits(self) -> u32 {
        f32::to_bits(self)
    }

    #[inline]
    fn from_bits(bits: u32) -> Self {
        f32::from_bits(bits)
    }

    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }

    #[inline]
    fn exponent_field(bits: u32) -> i16 {
        i16::try_from(bits >> 23).expect("a magnitude holds eight bits above its fraction")
    }

    #[inline]
    fn block_sqrt(self) -> Self {
        block_sqrt_f32(self)
    }

    #[inline]
    unsafe fn block_mul_add(self, multiplier: Self, addend: Self) -> Self {
        // SAFETY: the caller guarantees the fused multiply-add.
        unsafe { block_mul_add_f32(self, multiplier, addend) }
    }
}

impl Native for f64 {
    type Bits = u64;

    const FRACTION_BITS: u8 = 52;
    const BIAS: i16 = 1023;
    const SIGN: u64 = 0x8000_0000_0000_0000;
    const QUIET: u64 = 0x0008_0000_0000_0000;
    const NAN: Self = f64::NAN;
    const INFINITY: Self = f64::INFINITY;
    const HALF: Self = 0.5;
    const INTEGRAL: Self = 4_503_599_627_370_496.0;

    #[inline]
    fn to_bits(self) -> u64 {
        f64::to_bits(self)
    }

    #[inline]
    fn from_bits(bits: u64) -> Self {
        f64::from_bits(bits)
    }

    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }

    #[inline]
    fn exponent_field(bits: u64) -> i16 {
        i16::try_from(bits >> 52).expect("a magnitude holds eleven bits above its fraction")
    }

    #[inline]
    fn block_sqrt(self) -> Self {
        block_sqrt_f64(self)
    }

    #[inline]
    unsafe fn block_mul_add(self, multiplier: Self, addend: Self) -> Self {
        // SAFETY: the caller guarantees the fused multiply-add.
        unsafe { block_mul_add_f64(self, multiplier, addend) }
    }
}
