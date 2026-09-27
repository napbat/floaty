//! The value type, its classification, and the named formats.

use core::fmt::{self, Debug, Formatter};
use core::marker::PhantomData;

use crate::binary::Unpacked;
use crate::format::internal::LimbConversion;
use crate::format::{Binary, Fnuz, NoInf, Standard, X87};
use crate::limbs::Limbs;

/// A floating-point value of standard `S` at width `W`.
///
/// A value is a format and its bits. `S` and `W` select the storage type and
/// the engine at compile time. An unsupported pair does not compile.
///
/// ```
/// use floaty::{Class, F32};
///
/// let one = F32::from_bits(0x3F80_0000);
/// assert_eq!(one.classify(), Class::Normal);
/// assert_eq!(one.to_bits(), 0x3F80_0000);
/// ```
pub struct Float<S: Standard<W>, const W: usize> {
    bits: S::Bits,
    standard: PhantomData<S>,
}

impl<S: Standard<W>, const W: usize> Clone for Float<S, W> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Standard<W>, const W: usize> Copy for Float<S, W> {}

impl<S: Standard<W>, const W: usize> Float<S, W> {
    /// The precision in bits, including the leading bit.
    pub const PRECISION: u32 = S::PRECISION;

    /// The exponent of the largest finite value, `emax` in IEEE 754.
    pub const EMAX: i32 = S::EMAX;

    /// The exponent of the smallest normal value, `emin` in IEEE 754.
    pub const EMIN: i32 = S::EMIN;

    /// Makes a value from its encoding.
    ///
    /// The storage bits above `W` are ignored. For example, a 19-bit format in
    /// a `u32` ignores bits 19 to 31.
    #[must_use]
    pub fn from_bits(bits: S::Bits) -> Self {
        Self {
            bits: S::mask(bits),
            standard: PhantomData,
        }
    }

    /// Returns the encoding. The storage bits above `W` are zero.
    #[must_use]
    pub fn to_bits(self) -> S::Bits {
        self.bits
    }

    /// Returns the exact value of the encoding.
    ///
    /// `N` is the limb count of the significand and the payload. It must hold
    /// at least [`PRECISION`](Self::PRECISION) bits. A smaller `N` fails a
    /// compile-time assertion when the compiler generates code for the call.
    /// `cargo check` does not report that assertion.
    ///
    /// ```
    /// use floaty::{Decoded, F32};
    ///
    /// // 1.0 is 2^23 * 2^-23.
    /// let one = F32::from_bits(0x3F80_0000).decode::<1>();
    /// assert_eq!(one, Decoded::Finite { negative: false, exponent: -23, significand: [1 << 23] });
    /// ```
    #[must_use]
    pub fn decode<const N: usize>(self) -> Decoded<N> {
        const {
            // The cast widens a u32 to a usize and loses no bits. `From` is not
            // callable in a constant.
            assert!(
                N * 64 >= S::PRECISION as usize,
                "the limb count holds the significand"
            );
        }
        match S::unpack(self.bits) {
            Unpacked::Zero { negative } => Decoded::Zero { negative },
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => Decoded::Finite {
                negative,
                exponent,
                significand: significand.resize(),
            },
            Unpacked::Infinity { negative } => Decoded::Infinity { negative },
            Unpacked::Nan {
                negative,
                signaling,
                payload,
            } => Decoded::Nan {
                negative,
                signaling,
                payload: payload.resize(),
            },
            Unpacked::Unsupported => Decoded::Unsupported,
        }
    }

    /// Returns the class of the value.
    #[must_use]
    pub fn classify(self) -> Class {
        S::classify(self.bits)
    }

    /// Returns `true` when the sign bit is set, for every class of value.
    #[must_use]
    pub fn is_sign_negative(self) -> bool {
        S::is_sign_negative(self.bits)
    }

    /// Returns `true` when the sign bit is clear, for every class of value.
    #[must_use]
    pub fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    /// Returns `true` for a quiet or a signaling NaN.
    #[must_use]
    pub fn is_nan(self) -> bool {
        matches!(self.classify(), Class::QuietNan | Class::SignalingNan)
    }

    /// Returns `true` for a signaling NaN.
    #[must_use]
    pub fn is_signaling_nan(self) -> bool {
        self.classify() == Class::SignalingNan
    }

    /// Returns `true` for a positive or a negative infinity.
    #[must_use]
    pub fn is_infinite(self) -> bool {
        self.classify() == Class::Infinite
    }

    /// Returns `true` for a zero, a subnormal, or a normal value.
    #[must_use]
    pub fn is_finite(self) -> bool {
        matches!(
            self.classify(),
            Class::Zero | Class::Subnormal | Class::Normal
        )
    }

    /// Returns `true` for a positive or a negative zero.
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.classify() == Class::Zero
    }

    /// Returns `true` for a subnormal encoding.
    #[must_use]
    pub fn is_subnormal(self) -> bool {
        self.classify() == Class::Subnormal
    }

    /// Returns `true` for a normal value.
    #[must_use]
    pub fn is_normal(self) -> bool {
        self.classify() == Class::Normal
    }

    /// Returns `true` when the encoding is the canonical encoding of its
    /// value, as IEEE 754 `isCanonical` defines.
    ///
    /// An x87 pseudo-denormal and every unsupported encoding are not
    /// canonical. Every encoding of the other binary formats is canonical.
    #[must_use]
    pub fn is_canonical(self) -> bool {
        S::is_canonical(self.bits)
    }
}

impl<S: Standard<W>, const W: usize> Debug for Float<S, W> {
    /// Writes the encoding in hexadecimal, for example `Float(0x3f800000)`.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let limbs = self.bits.to_limbs();
        let digits = u32::try_from(W.div_ceil(4)).expect("a width is at most 512 bits");
        formatter.write_str("Float(0x")?;
        for digit in (0..digits).rev() {
            write!(formatter, "{:x}", limbs.field(digit * 4, 4))?;
        }
        formatter.write_str(")")
    }
}

/// The exact value of an encoding, as [`Float::decode`] returns it.
///
/// A finite value is `significand * 2^exponent`. `exponent` is the weight of
/// the lowest significand bit. The significand is below `2^PRECISION`, and a
/// normal value has bit `PRECISION - 1` set. Limb 0 of an array holds the
/// least significant 64 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Decoded<const N: usize> {
    /// A zero.
    Zero {
        /// The sign.
        negative: bool,
    },
    /// A nonzero finite value.
    Finite {
        /// The sign.
        negative: bool,
        /// The weight of the lowest significand bit.
        exponent: i32,
        /// The significand, an integer below `2^PRECISION`.
        significand: [u64; N],
    },
    /// An infinity.
    Infinity {
        /// The sign.
        negative: bool,
    },
    /// A NaN.
    Nan {
        /// The sign.
        negative: bool,
        /// `true` for a signaling NaN.
        signaling: bool,
        /// The fraction bits below the quiet bit. The `NoInf` and `Fnuz`
        /// encodings have no payload, so their payload is zero.
        payload: [u64; N],
    },
    /// An encoding that the format does not define as a value, such as an x87
    /// unnormal.
    Unsupported,
}

/// The class of a floating-point encoding.
///
/// The sign is not part of the class. Use
/// [`is_sign_negative`](Float::is_sign_negative) for the sign.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Class {
    /// A positive or a negative zero.
    Zero,
    /// A subnormal encoding. An x87 pseudo-denormal is subnormal, as the x87
    /// `FXAM` instruction reports it.
    Subnormal,
    /// A normal value.
    Normal,
    /// A positive or a negative infinity.
    Infinite,
    /// A quiet NaN. The NaN of a format without signaling NaNs is quiet.
    QuietNan,
    /// A signaling NaN.
    SignalingNan,
    /// An encoding that the format does not define as a value, such as an x87
    /// unnormal, pseudo-NaN, or pseudo-infinity. Arithmetic treats it as an
    /// invalid operand.
    Unsupported,
}

/// IEEE 754 binary16.
pub type F16 = Float<Binary<5>, 16>;
/// IEEE 754 binary32.
pub type F32 = Float<Binary<8>, 32>;
/// IEEE 754 binary64.
pub type F64 = Float<Binary<11>, 64>;
/// IEEE 754 binary128.
pub type F128 = Float<Binary<15>, 128>;
/// IEEE 754 binary160.
pub type F160 = Float<Binary<16>, 160>;
/// IEEE 754 binary192.
pub type F192 = Float<Binary<17>, 192>;
/// IEEE 754 binary224.
pub type F224 = Float<Binary<18>, 224>;
/// IEEE 754 binary256.
pub type F256 = Float<Binary<19>, 256>;
/// IEEE 754 binary288.
pub type F288 = Float<Binary<20>, 288>;
/// IEEE 754 binary320.
pub type F320 = Float<Binary<20>, 320>;
/// IEEE 754 binary352.
pub type F352 = Float<Binary<21>, 352>;
/// IEEE 754 binary384.
pub type F384 = Float<Binary<21>, 384>;
/// IEEE 754 binary416.
pub type F416 = Float<Binary<22>, 416>;
/// IEEE 754 binary448.
pub type F448 = Float<Binary<22>, 448>;
/// IEEE 754 binary480.
pub type F480 = Float<Binary<23>, 480>;
/// IEEE 754 binary512.
pub type F512 = Float<Binary<23>, 512>;
/// bfloat16: the exponent range of binary32 with 8 bits of precision.
pub type BF16 = Float<Binary<8>, 16>;
/// TF32: the exponent range of binary32 with the precision of binary16.
pub type TF32 = Float<Binary<8>, 19>;
/// OCP FP8 E4M3: no infinities, one NaN encoding for each sign.
pub type F8E4M3 = Float<Binary<4, NoInf>, 8>;
/// OCP FP8 E5M2: IEEE 754 special values.
pub type F8E5M2 = Float<Binary<5>, 8>;
/// FNUZ FP8 E4M3: no infinities, no negative zero, one NaN.
pub type F8E4M3Fnuz = Float<Binary<4, Fnuz>, 8>;
/// FNUZ FP8 E5M2: no infinities, no negative zero, one NaN.
pub type F8E5M2Fnuz = Float<Binary<5, Fnuz>, 8>;
/// x87 80-bit extended precision.
pub type F80 = Float<Binary<15, X87>, 80>;

#[cfg(test)]
mod tests {
    extern crate std;

    use std::format;

    use super::{
        BF16, Class, Decoded, F8E4M3, F8E4M3Fnuz, F8E5M2, F8E5M2Fnuz, F16, F32, F64, F80, F128,
        F256, F512, TF32,
    };

    #[test]
    fn from_bits_ignores_bits_above_the_width() {
        assert_eq!(TF32::from_bits(u32::MAX).to_bits(), 0x7_FFFF);
        assert_eq!(F32::from_bits(u32::MAX).to_bits(), u32::MAX);
    }

    #[test]
    fn debug_writes_every_digit_of_the_width() {
        assert_eq!(
            format!("{:?}", F32::from_bits(0x3F80_0000)),
            "Float(0x3f800000)"
        );
        assert_eq!(format!("{:?}", TF32::from_bits(0x1)), "Float(0x00001)");
        let text = format!("{:?}", F512::from_bits([1, 0, 0, 0, 0, 0, 0, 1 << 63]));
        assert_eq!(text.len(), "Float(0x)".len() + 128);
        assert!(text.starts_with("Float(0x8000") && text.ends_with("0001)"));
    }

    #[test]
    fn predicates_follow_the_class() {
        let quiet = F32::from_bits(0x7FC0_0000);
        assert!(quiet.is_nan() && !quiet.is_signaling_nan() && !quiet.is_finite());
        let signaling = F32::from_bits(0xFF80_0001);
        assert_eq!(signaling.classify(), Class::SignalingNan);
        assert!(signaling.is_signaling_nan() && signaling.is_sign_negative());
        let infinity = F32::from_bits(0x7F80_0000);
        assert!(infinity.is_infinite() && infinity.is_sign_positive());
        assert!(F32::from_bits(0x8000_0000).is_zero());
        assert!(F32::from_bits(0x0000_0001).is_subnormal());
        assert!(F32::from_bits(0x0080_0000).is_normal());
    }

    #[test]
    fn decode_resizes_to_the_requested_limbs() {
        let value = F80::from_bits(0xBFFF_C000_0000_0000_0001);
        let expected = Decoded::Finite {
            negative: true,
            exponent: -63,
            significand: [0xC000_0000_0000_0001, 0, 0],
        };
        assert_eq!(value.decode::<3>(), expected);
        let nan = F32::from_bits(0x7FA0_0001).decode::<1>();
        assert_eq!(
            nan,
            Decoded::Nan {
                negative: false,
                signaling: true,
                payload: [0x20_0001]
            }
        );
    }

    #[test]
    fn aliases_have_the_published_parameters() {
        assert_eq!((F16::PRECISION, F16::EMAX, F16::EMIN), (11, 15, -14));
        assert_eq!((F32::PRECISION, F32::EMAX, F32::EMIN), (24, 127, -126));
        assert_eq!((F64::PRECISION, F64::EMAX, F64::EMIN), (53, 1023, -1022));
        assert_eq!(
            (F128::PRECISION, F128::EMAX, F128::EMIN),
            (113, 16383, -16382)
        );
        assert_eq!(
            (F256::PRECISION, F256::EMAX, F256::EMIN),
            (237, 262_143, -262_142)
        );
        assert_eq!(
            (F512::PRECISION, F512::EMAX, F512::EMIN),
            (489, 4_194_303, -4_194_302)
        );
        assert_eq!((BF16::PRECISION, BF16::EMAX, BF16::EMIN), (8, 127, -126));
        assert_eq!((TF32::PRECISION, TF32::EMAX, TF32::EMIN), (11, 127, -126));
        assert_eq!((F8E4M3::PRECISION, F8E4M3::EMAX, F8E4M3::EMIN), (4, 8, -6));
        assert_eq!(
            (F8E5M2::PRECISION, F8E5M2::EMAX, F8E5M2::EMIN),
            (3, 15, -14)
        );
        assert_eq!(
            (F8E4M3Fnuz::PRECISION, F8E4M3Fnuz::EMAX, F8E4M3Fnuz::EMIN),
            (4, 7, -7)
        );
        assert_eq!(
            (F8E5M2Fnuz::PRECISION, F8E5M2Fnuz::EMAX, F8E5M2Fnuz::EMIN),
            (3, 15, -15)
        );
        assert_eq!((F80::PRECISION, F80::EMAX, F80::EMIN), (64, 16383, -16382));
    }
}
