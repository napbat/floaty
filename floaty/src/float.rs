//! The value type, its classification, and the named formats.

use core::fmt::{self, Debug, Formatter};
use core::marker::PhantomData;

use crate::binary::Unpacked;
use crate::env::{Env, Flags, Mode, Override, mode};
use crate::exact::{Exact, Unrounded};
use crate::format::internal::LimbConversion;
use crate::format::{Binary, Fnuz, NoInf, Standard, X87};
use crate::limbs::Limbs;
use crate::sealed::Sealed;

mod arithmetic;
mod compare;
mod integer;

/// A floating-point value of standard `S` at width `W`, with default mode `M`.
///
/// A value is a format and its bits. `S` and `W` select the storage type and
/// the engine at compile time. An unsupported pair does not compile. The mode
/// is the behavior that an operation uses when the call does not override it.
/// It does not change the value.
///
/// ```
/// use floaty::{Class, F32};
///
/// let one = F32::from_bits(0x3F80_0000);
/// assert_eq!(one.classify(), Class::Normal);
/// assert_eq!(one.to_bits(), 0x3F80_0000);
/// ```
pub struct Float<S: Standard<W>, const W: usize, M: Mode = mode::Ieee> {
    bits: S::Bits,
    marker: PhantomData<(S, M)>,
}

impl<S: Standard<W>, const W: usize, M: Mode> Clone for Float<S, W, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Standard<W>, const W: usize, M: Mode> Copy for Float<S, W, M> {}

impl<S: Standard<W>, const W: usize, M: Mode> Float<S, W, M> {
    /// The behavior of the default mode.
    pub const ENV: Env = M::ENV;

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
        Self::from_masked(S::mask(bits))
    }

    /// Makes a value from an encoding whose bits above `W` are zero.
    fn from_masked(bits: S::Bits) -> Self {
        Self {
            bits,
            marker: PhantomData,
        }
    }

    /// Returns the same value with another default mode. The bits do not
    /// change.
    #[must_use]
    pub fn with_mode<Other: Mode>(self) -> Float<S, W, Other> {
        Float::from_masked(self.bits)
    }

    /// Rounds an exact value to this format.
    ///
    /// `behavior` is a [`Rounding`](crate::Rounding) that overrides only the
    /// rounding direction, or an [`Env`] that replaces the whole behavior.
    /// Pass [`Self::ENV`] to use the default mode.
    #[must_use]
    pub fn round<const N: usize>(exact: Exact<N>, behavior: impl Override) -> (Self, Flags) {
        let env = behavior.apply(M::ENV);
        let value = Unrounded {
            negative: exact.negative,
            exponent: exact.exponent,
            significand: exact.significand,
            sticky: exact.sticky,
        };
        let (bits, flags) = S::round(&value, &env);
        (Self::from_masked(bits), flags)
    }

    /// Converts the value to another format, rounding with the mode of the
    /// destination type.
    ///
    /// ```
    /// use floaty::{BF16, F32};
    ///
    /// let pi = F32::from_bits(0x4049_0FDB);
    /// let rounded: BF16 = pi.convert();
    /// assert_eq!(rounded.to_bits(), 0x4049);
    /// ```
    #[must_use]
    pub fn convert<T: FloatType>(self) -> T {
        self.convert_with(T::DEFAULT_ENV).0
    }

    /// Converts the value to another format, with an override of the
    /// destination behavior. Returns the result and the flags.
    ///
    /// A subnormal input sets [`Flags::DENORMAL_INPUT`], and reads as a zero
    /// when the behavior has denormals-are-zero set. A signaling NaN input
    /// signals invalid. A destination without an infinity converts an infinity
    /// to a NaN, or to the largest finite value when the behavior saturates,
    /// and signals invalid.
    #[must_use]
    pub fn convert_with<T: FloatType>(self, behavior: impl Override) -> (T, Flags) {
        let env = behavior.apply(T::DEFAULT_ENV);
        let mut value = S::unpack(self.bits);
        let mut input = Flags::NONE;
        if S::classify(self.bits) == Class::Subnormal {
            input = Flags::DENORMAL_INPUT;
            if env.denormals_are_zero {
                value = Unpacked::Zero {
                    negative: self.is_sign_negative(),
                };
            }
        }
        let (result, flags) = T::convert_from(value, S::PAYLOAD_BITS, &env);
        (result, flags | input)
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

    /// Returns the value with a clear sign bit, as IEEE 754 `abs` does.
    ///
    /// The sign operations change only the sign bit, for every class of
    /// value, and signal nothing. The zero and the NaN of [`Fnuz`] each have
    /// one encoding, so they do not change.
    #[must_use]
    pub fn abs(self) -> Self {
        Self::from_masked(S::with_sign(self.bits, false))
    }

    /// Returns the value with the sign of `sign`, as IEEE 754 `copySign`
    /// does.
    #[must_use]
    pub fn copy_sign(self, sign: Self) -> Self {
        Self::from_masked(S::with_sign(self.bits, sign.is_sign_negative()))
    }
}

/// Negation as IEEE 754 `negate`: the sign bit flips, for every class of
/// value, and nothing signals.
impl<S: Standard<W>, const W: usize, M: Mode> core::ops::Neg for Float<S, W, M> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::from_masked(S::with_sign(self.bits, !self.is_sign_negative()))
    }
}

impl<S: Standard<W>, const W: usize, M: Mode> Debug for Float<S, W, M> {
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

/// A [`Float`] type of any standard, width, and mode, as a conversion
/// destination. The trait is sealed.
pub trait FloatType: Sealed + Copy {
    /// The behavior of the default mode of the type.
    #[doc(hidden)]
    const DEFAULT_ENV: Env;

    /// Converts a decoded value of another format, whose NaN payloads have
    /// `payload_bits` bits.
    #[doc(hidden)]
    fn convert_from<L: Limbs>(value: Unpacked<L>, payload_bits: u32, env: &Env) -> (Self, Flags);
}

impl<S: Standard<W>, const W: usize, M: Mode> Sealed for Float<S, W, M> {}

impl<S: Standard<W>, const W: usize, M: Mode> FloatType for Float<S, W, M> {
    const DEFAULT_ENV: Env = M::ENV;

    fn convert_from<L: Limbs>(value: Unpacked<L>, payload_bits: u32, env: &Env) -> (Self, Flags) {
        let (bits, flags) = S::convert_from(value, payload_bits, env);
        (Self::from_masked(bits), flags)
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
mod tests;
