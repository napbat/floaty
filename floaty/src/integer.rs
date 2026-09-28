//! Integers of every width from 1 to 512 bits, and the `Integer` trait of
//! the types that a float converts to and from. `ToInt` is the result of a
//! conversion to an integer.

use crate::format::internal::LimbConversion;
use crate::format::{Storage, Width};
use crate::limbs::Limbs;
use crate::sealed::Sealed;

/// An unsigned integer of `BITS` bits, for every `BITS` from 1 to 512.
///
/// The value uses the storage type of the width table, as a float of the same
/// width does. The storage bits above `BITS` are zero.
///
/// ```
/// use floaty::UInt;
///
/// let value = UInt::<24>::from_bits(0x0100_0005);
/// assert_eq!(value.to_bits(), 5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UInt<const BITS: usize>
where
    Width<BITS>: Storage,
{
    bits: <Width<BITS> as Storage>::Bits,
}

impl<const BITS: usize> UInt<BITS>
where
    Width<BITS>: Storage,
{
    /// The width in bits.
    pub const BITS: u32 = <Width<BITS> as Storage>::WIDTH;

    /// Makes a value from its bits. The storage bits above `BITS` are ignored.
    #[must_use]
    pub fn from_bits(bits: <Width<BITS> as Storage>::Bits) -> Self {
        Self {
            bits: mask::<BITS>(bits),
        }
    }

    /// Returns the bits. The storage bits above `BITS` are zero.
    #[must_use]
    pub fn to_bits(self) -> <Width<BITS> as Storage>::Bits {
        self.bits
    }
}

/// A signed two's complement integer of `BITS` bits, for every `BITS` from 1
/// to 512.
///
/// The value uses the storage type of the width table. Bit `BITS - 1` is the
/// sign bit, and the storage bits above `BITS` are zero.
///
/// ```
/// use floaty::Int;
///
/// // 0x80_0000 is the smallest 24-bit value, -2^23.
/// let smallest = Int::<24>::from_bits(0x80_0000);
/// assert!(smallest.is_negative());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Int<const BITS: usize>
where
    Width<BITS>: Storage,
{
    bits: <Width<BITS> as Storage>::Bits,
}

impl<const BITS: usize> Int<BITS>
where
    Width<BITS>: Storage,
{
    /// The width in bits.
    pub const BITS: u32 = <Width<BITS> as Storage>::WIDTH;

    /// Makes a value from its two's complement bits. The storage bits above
    /// `BITS` are ignored.
    #[must_use]
    pub fn from_bits(bits: <Width<BITS> as Storage>::Bits) -> Self {
        Self {
            bits: mask::<BITS>(bits),
        }
    }

    /// Returns the two's complement bits. The storage bits above `BITS` are
    /// zero.
    #[must_use]
    pub fn to_bits(self) -> <Width<BITS> as Storage>::Bits {
        self.bits
    }

    /// Returns `true` when the value is below zero.
    #[must_use]
    pub fn is_negative(self) -> bool {
        self.bits.to_limbs().bit(Self::BITS - 1)
    }
}

/// Clears the storage bits above `BITS`.
fn mask<const BITS: usize>(bits: <Width<BITS> as Storage>::Bits) -> <Width<BITS> as Storage>::Bits
where
    Width<BITS>: Storage,
{
    LimbConversion::from_limbs(bits.to_limbs().low_bits(<Width<BITS> as Storage>::WIDTH))
}

/// The sign and magnitude of an integer.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parts {
    /// `true` for a value below zero.
    pub negative: bool,
    /// The magnitude. It fits the integer type.
    pub magnitude: [u64; 8],
}

/// An integer type that a float converts to and from.
///
/// The types are the Rust primitive integers from 8 to 128 bits, [`Int`], and
/// [`UInt`]. `isize` and `usize` are not integer types here, because their
/// width depends on the host. The trait is sealed.
pub trait Integer: Sealed + Copy {
    /// The width in bits.
    const BITS: u32;

    /// `true` for a signed type.
    const SIGNED: bool;

    /// Returns the sign and the magnitude.
    #[doc(hidden)]
    fn to_parts(self) -> Parts;

    /// Makes a value from a sign and a magnitude that fit the type.
    #[doc(hidden)]
    fn from_parts(parts: Parts) -> Self;
}

/// Returns the magnitude of a `u128` as limbs.
fn u128_limbs(value: u128) -> [u64; 8] {
    let [low, high] = value.to_limbs();
    [low, high, 0, 0, 0, 0, 0, 0]
}

/// Returns a magnitude of at most 128 bits as a `u128`.
fn limbs_u128(magnitude: [u64; 8]) -> u128 {
    debug_assert!(
        magnitude[2..].iter().all(|&limb| limb == 0),
        "the magnitude fits 128 bits"
    );
    u128::from_limbs([magnitude[0], magnitude[1]])
}

/// Implements [`Integer`] for an unsigned primitive. The storage types
/// already implement `Sealed`.
macro_rules! unsigned {
    ($($integer:ty),*) => {
        $(
            impl Integer for $integer {
                const BITS: u32 = <$integer>::BITS;
                const SIGNED: bool = false;

                fn to_parts(self) -> Parts {
                    Parts {
                        negative: false,
                        magnitude: u128_limbs(u128::from(self)),
                    }
                }

                fn from_parts(parts: Parts) -> Self {
                    debug_assert!(!parts.negative, "an unsigned value is not negative");
                    Self::try_from(limbs_u128(parts.magnitude)).expect("the magnitude fits the type")
                }
            }
        )*
    };
}

/// Implements [`Integer`] for a signed primitive.
macro_rules! signed {
    ($($integer:ty),*) => {
        $(
            impl Sealed for $integer {}
            impl Integer for $integer {
                const BITS: u32 = <$integer>::BITS;
                const SIGNED: bool = true;

                fn to_parts(self) -> Parts {
                    Parts {
                        negative: self < 0,
                        magnitude: u128_limbs(u128::from(self.unsigned_abs())),
                    }
                }

                fn from_parts(parts: Parts) -> Self {
                    let magnitude = limbs_u128(parts.magnitude);
                    let value = if parts.negative {
                        0_i128.checked_sub_unsigned(magnitude)
                    } else {
                        i128::try_from(magnitude).ok()
                    };
                    value
                        .and_then(|value| Self::try_from(value).ok())
                        .expect("the magnitude fits the type")
                }
            }
        )*
    };
}

unsigned!(u8, u16, u32, u64, u128);
signed!(i8, i16, i32, i64, i128);

impl<const BITS: usize> Sealed for UInt<BITS> where Width<BITS>: Storage {}

impl<const BITS: usize> Integer for UInt<BITS>
where
    Width<BITS>: Storage,
{
    const BITS: u32 = <Width<BITS> as Storage>::WIDTH;
    const SIGNED: bool = false;

    fn to_parts(self) -> Parts {
        Parts {
            negative: false,
            magnitude: self.bits.to_limbs().resize(),
        }
    }

    fn from_parts(parts: Parts) -> Self {
        debug_assert!(!parts.negative, "an unsigned value is not negative");
        debug_assert!(
            parts.magnitude.bit_length() <= <Self as Integer>::BITS,
            "the magnitude fits the type"
        );
        Self {
            bits: LimbConversion::from_limbs(parts.magnitude.resize()),
        }
    }
}

impl<const BITS: usize> Sealed for Int<BITS> where Width<BITS>: Storage {}

impl<const BITS: usize> Integer for Int<BITS>
where
    Width<BITS>: Storage,
{
    const BITS: u32 = <Width<BITS> as Storage>::WIDTH;
    const SIGNED: bool = true;

    fn to_parts(self) -> Parts {
        // 2^BITS needs a ninth limb when BITS is 512.
        let bits: [u64; 9] = self.bits.to_limbs().resize();
        if !self.is_negative() {
            return Parts {
                negative: false,
                magnitude: bits.resize(),
            };
        }
        let modulus = <[u64; 9]>::ZERO.with_bit(<Self as Integer>::BITS);
        Parts {
            negative: true,
            magnitude: modulus.sub(bits).resize(),
        }
    }

    fn from_parts(parts: Parts) -> Self {
        let width = <Self as Integer>::BITS;
        let magnitude: [u64; 9] = parts.magnitude.resize();
        debug_assert!(
            magnitude.bit_length() < width
                || (parts.negative && magnitude == <[u64; 9]>::ZERO.with_bit(width - 1)),
            "the magnitude fits the type"
        );
        let bits = if parts.negative && !magnitude.is_zero() {
            <[u64; 9]>::ZERO.with_bit(width).sub(magnitude)
        } else {
            magnitude
        };
        Self {
            bits: LimbConversion::from_limbs(bits.low_bits(width).resize()),
        }
    }
}

/// The result of a conversion from a float to an integer.
///
/// Each instruction set maps an out-of-range value and a NaN to its own
/// integer. For example, x86 gives the integer indefinite for both, and ARM
/// saturates an out-of-range value and gives zero for a NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToInt<I> {
    /// The rounded value, which fits the integer type.
    Value(I),
    /// The rounded value, or an infinity, is outside the range of the integer
    /// type. `negative` is the sign of the float.
    OutOfRange {
        /// The sign of the float.
        negative: bool,
    },
    /// The float is a NaN or an unsupported encoding.
    Nan,
}

#[cfg(test)]
mod tests {
    use super::{Int, Integer, Parts, UInt};
    use crate::limbs::Limbs;

    fn parts(negative: bool, magnitude: u128) -> Parts {
        Parts {
            negative,
            magnitude: super::u128_limbs(magnitude),
        }
    }

    #[test]
    fn primitive_integers_round_trip_through_parts() {
        assert_eq!(i8::MIN.to_parts(), parts(true, 128));
        assert_eq!(i8::from_parts(parts(true, 128)), i8::MIN);
        assert_eq!(i128::from_parts(i128::MIN.to_parts()), i128::MIN);
        assert_eq!(u128::from_parts(u128::MAX.to_parts()), u128::MAX);
        assert_eq!((-5_i32).to_parts(), parts(true, 5));
    }

    #[test]
    fn wide_integers_use_twos_complement_at_their_width() {
        let smallest = Int::<24>::from_bits(0x80_0000);
        assert!(smallest.is_negative());
        assert_eq!(smallest.to_parts(), parts(true, 1 << 23));
        assert_eq!(Int::<24>::from_parts(parts(true, 1 << 23)), smallest);
        assert_eq!(Int::<24>::from_parts(parts(true, 1)).to_bits(), 0xFF_FFFF);
        assert_eq!(UInt::<24>::from_bits(0xFF12_3456).to_bits(), 0x12_3456);

        let low = Int::<512>::from_bits([0, 0, 0, 0, 0, 0, 0, 1 << 63]);
        let magnitude = low.to_parts().magnitude;
        assert_eq!(magnitude, [0_u64; 8].with_bit(511));
        assert_eq!(Int::<512>::from_parts(low.to_parts()), low);
        let minus_one = Int::<512>::from_parts(parts(true, 1));
        assert_eq!(minus_one.to_bits(), [u64::MAX; 8]);
        assert_eq!(Int::<1>::from_bits(1).to_parts(), parts(true, 1));
    }
}
