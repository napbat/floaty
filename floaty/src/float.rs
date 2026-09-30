//! The value type, its classification, and the named formats.

use core::fmt::{self, Debug, Formatter};
use core::marker::PhantomData;

use crate::binary::Unpacked;
use crate::env::{Behavior, Env, Flags, Mode, Override, mode};
use crate::exact::{Exact, Unrounded};
use crate::format::internal::{Host, LimbConversion, Source};
use crate::format::{B11Fnuz, Bid, Binary, Decimal, Dpd, Finite, Fnuz, NoInf, Standard, X87};
use crate::host;
use crate::limbs::Limbs;
use crate::sealed::Sealed;

mod arithmetic;
mod compare;
mod decimal;
mod integer;

pub(crate) use self::integer::from_host_integer;

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
// The layout is that of the encoding, so the packed host paths of `Lanes`
// read an array of values as an array of encodings.
#[repr(transparent)]
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

    /// The radix: 2 for a binary format, 10 for a decimal format.
    pub const RADIX: u32 = S::RADIX;

    /// The precision in digits of the radix, including the leading digit.
    pub const PRECISION: u32 = S::PRECISION;

    /// The exponent of the largest finite value, `emax` in IEEE 754, as a
    /// power of the radix.
    pub const EMAX: i32 = S::EMAX;

    /// The exponent of the smallest normal value, `emin` in IEEE 754, as a
    /// power of the radix.
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
    /// rounding direction, or a behavior that replaces the whole behavior: an
    /// [`Env`], or a mode. Pass the mode of the type, `M::default()`, to use
    /// the default mode with its fields as constants.
    ///
    /// The exact value is in the radix of the format. A decimal result that
    /// is exact takes the member of its cohort whose exponent is nearest the
    /// exponent of `exact`. An inexact decimal result has the least possible
    /// exponent.
    #[must_use]
    pub fn round<const N: usize>(exact: Exact<N>, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        let value = Unrounded {
            negative: exact.negative,
            exponent: exact.exponent,
            significand: exact.significand,
            sticky: exact.sticky,
        };
        let (bits, flags) = S::round(&value, behavior);
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
    #[inline]
    pub fn convert<T: FloatType>(self) -> T {
        if !host::convertible(S::HOST, T::HOST) {
            return self.convert_with(T::Mode::default()).0;
        }
        let bits = self.bits.to_limbs().resize();
        match host::convert(S::HOST, T::HOST, bits, &<T::Mode as Mode>::ENV) {
            Some(bits) => T::from_host(bits),
            None => convert_in_engine(self),
        }
    }

    /// Converts the value to another format, with an override of the
    /// destination behavior. Returns the result and the flags.
    ///
    /// A subnormal input sets [`Flags::DENORMAL_INPUT`], and reads as a zero
    /// when the behavior has denormals-are-zero set. A signaling NaN input
    /// signals invalid. A destination without an infinity converts an infinity
    /// to a NaN, or to the largest finite value when the behavior saturates or
    /// the destination has no NaN, and signals invalid. An unsupported x87
    /// input signals invalid and gives the default NaN.
    ///
    /// A NaN converts by the NaN rule of the behavior.
    /// [`DefaultNan`](crate::env::NanPropagation::DefaultNan) gives the
    /// default NaN. The other rules keep the sign, make the NaN quiet, and
    /// keep the high-order payload bits, as x86 and ARM do. A narrowing
    /// conversion drops the low-order payload bits, and a format with one NaN
    /// encoding gives that NaN. A format without a NaN, such as
    /// [`F4E2M1Fn`](crate::F4E2M1Fn), gives `+0` and signals invalid. Between
    /// two decimal formats, the payload keeps its high-order digits. Between a
    /// binary and a decimal format, the payload bits align with the trailing
    /// significand field of the decimal format, and a decimal payload above
    /// `10^(PRECISION - 1) - 1` becomes zero, as the Intel decimal library
    /// does.
    ///
    /// A conversion between a binary and a decimal format can hold exact
    /// values of up to 16,384 bits on the stack, tens of KiB. A `no_std`
    /// target with a small stack must allow for this.
    #[must_use]
    pub fn convert_with<T: FloatType>(self, behavior: impl Override) -> (T, Flags) {
        let behavior = behavior.apply::<T::Mode>();
        let (value, input) = S::operand(self.bits, &behavior.env());
        let source = Source {
            radix: S::RADIX,
            payload_digits: S::PAYLOAD_DIGITS,
        };
        let (result, flags) = T::convert_from(value, source, behavior);
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
    /// the largest significand: [`PRECISION`](Self::PRECISION) bits for a
    /// binary format, and 24, 54, or 113 bits for decimal32, decimal64, or
    /// decimal128. A smaller `N` fails a
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
                N * 64 >= S::SIGNIFICAND_BITS as usize,
                "the limb count holds the significand"
            );
        }
        match S::unpack(self.bits) {
            Unpacked::Zero { negative, exponent } => Decoded::Zero { negative, exponent },
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
    /// canonical. Every encoding of the other binary formats is canonical. In
    /// a decimal format, each member of a cohort, such as 1.0 and 1.00, has
    /// its own canonical encoding. A BID coefficient above `10^p - 1`, a
    /// non-canonical DPD declet, and an infinity or NaN with extra bits are
    /// not canonical.
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

/// Implements the bit casts between a host float type and the format with its
/// encoding.
macro_rules! host_bit_cast {
    ($host:ty, $exponent:literal, $width:literal, $alias:literal) => {
        impl<M: Mode> From<$host> for Float<Binary<$exponent>, $width, M> {
            #[doc = concat!("Makes the ", $alias, " value with the bits of a host `", stringify!($host), "`.")]
            ///
            /// The conversion is a bit cast. It runs no floating-point
            /// instruction and keeps every encoding, a signaling NaN and its
            /// payload included. A target that moves an `f32` or `f64`
            /// through the x87 stack, as i686 can, can make a signaling NaN
            /// quiet when a call passes or returns the host value.
            #[inline]
            fn from(value: $host) -> Self {
                Self::from_bits(value.to_bits())
            }
        }

        impl<M: Mode> From<Float<Binary<$exponent>, $width, M>> for $host {
            #[doc = concat!("Makes the host `", stringify!($host), "` with the bits of an ", $alias, " value.")]
            ///
            /// The conversion is a bit cast. It runs no floating-point
            /// instruction and keeps every encoding, a signaling NaN and its
            /// payload included. A target that moves an `f32` or `f64`
            /// through the x87 stack, as i686 can, can make a signaling NaN
            /// quiet when a call passes or returns the host value.
            #[inline]
            fn from(value: Float<Binary<$exponent>, $width, M>) -> Self {
                <$host>::from_bits(value.to_bits())
            }
        }
    };
}

host_bit_cast!(f32, 8, 32, "`F32`");
host_bit_cast!(f64, 11, 64, "`F64`");

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

/// A [`Float`] type of any standard, width, and mode, or a
/// [`DoubleDouble`](crate::DoubleDouble) type, as a conversion destination.
/// The trait is sealed.
pub trait FloatType: Sealed + Copy {
    /// The default mode of the type.
    #[doc(hidden)]
    type Mode: Mode;

    /// The host format of the type, for the host paths of conversions.
    #[doc(hidden)]
    const HOST: Host;

    /// Returns the value of an encoding of the host format, in limbs from the
    /// low bits up. The type must have a host format.
    #[doc(hidden)]
    fn from_host(bits: [u64; 2]) -> Self;

    /// Converts a decoded value of another format, the source.
    #[doc(hidden)]
    fn convert_from<L: Limbs, B: Behavior>(
        value: Unpacked<L>,
        source: Source,
        behavior: B,
    ) -> (Self, Flags);
}

/// Runs a conversion in the engine, for a pair of formats whose host path
/// does not apply.
#[cold]
#[inline(never)]
fn convert_in_engine<S: Standard<W>, const W: usize, M: Mode, T: FloatType>(
    value: Float<S, W, M>,
) -> T {
    value.convert_with(T::Mode::default()).0
}

impl<S: Standard<W>, const W: usize, M: Mode> Sealed for Float<S, W, M> {}

impl<S: Standard<W>, const W: usize, M: Mode> FloatType for Float<S, W, M> {
    type Mode = M;

    const HOST: Host = S::HOST;

    #[inline]
    fn from_host(bits: [u64; 2]) -> Self {
        Self::from_masked(S::Bits::from_limbs(bits.resize()))
    }

    #[inline]
    fn convert_from<L: Limbs, B: Behavior>(
        value: Unpacked<L>,
        source: Source,
        behavior: B,
    ) -> (Self, Flags) {
        let (bits, flags) = S::convert_from(value, source, behavior);
        (Self::from_masked(bits), flags)
    }
}

/// The exact value of an encoding, as [`Float::decode`] returns it.
///
/// A finite value is `significand * RADIX^exponent`. `exponent` is the weight
/// of the lowest significand digit. The significand is below
/// `RADIX^PRECISION`. A normal binary value has bit `PRECISION - 1` set. A
/// decimal value keeps the coefficient and the exponent of its encoding, so
/// 1.0 and 1.00 decode apart. Limb 0 of an array holds the least significant
/// 64 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Decoded<const N: usize> {
    /// A zero.
    Zero {
        /// The sign.
        negative: bool,
        /// The exponent of a decimal zero, its quantum. A binary zero has
        /// exponent 0.
        exponent: i32,
    },
    /// A nonzero finite value.
    Finite {
        /// The sign.
        negative: bool,
        /// The weight of the lowest significand digit, as a power of the
        /// radix.
        exponent: i32,
        /// The significand, an integer below `RADIX^PRECISION`.
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
        /// The payload. A binary format gives the fraction bits below the
        /// quiet bit; the `NoInf` and `Fnuz` encodings have no payload, so
        /// their payload is zero. A decimal format gives the value of the
        /// trailing significand field, or zero for a non-canonical value.
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
/// OCP FP8 E4M3: no infinities, one NaN encoding for each sign. LLVM and
/// `ml_dtypes` call it E4M3FN, for finite values and a NaN.
pub type F8E4M3Fn = Float<Binary<4, NoInf>, 8>;
/// OCP FP8 E5M2: IEEE 754 special values.
pub type F8E5M2 = Float<Binary<5>, 8>;
/// FP8 E4M3 with IEEE 754 special values, as LLVM and `ml_dtypes` define
/// it. The largest finite value is 240.
pub type F8E4M3 = Float<Binary<4>, 8>;
/// FP8 E3M4 with IEEE 754 special values, as LLVM and `ml_dtypes` define
/// it. The largest finite value is 15.5.
pub type F8E3M4 = Float<Binary<3>, 8>;
/// FNUZ FP8 E4M3: no infinities, no negative zero, one NaN.
pub type F8E4M3Fnuz = Float<Binary<4, Fnuz>, 8>;
/// FNUZ FP8 E5M2: no infinities, no negative zero, one NaN.
pub type F8E5M2Fnuz = Float<Binary<5, Fnuz>, 8>;
/// FNUZ FP8 E4M3 with the exponent bias 11: no infinities, no negative
/// zero, one NaN. The largest finite value is 30.
pub type F8E4M3B11Fnuz = Float<Binary<4, B11Fnuz>, 8>;
/// OCP MX FP4 E2M1: no infinities and no NaNs. LLVM and `ml_dtypes` call it
/// E2M1FN.
pub type F4E2M1Fn = Float<Binary<2, Finite>, 4>;
/// OCP MX FP6 E2M3: no infinities and no NaNs.
pub type F6E2M3Fn = Float<Binary<2, Finite>, 6>;
/// OCP MX FP6 E3M2: no infinities and no NaNs.
pub type F6E3M2Fn = Float<Binary<3, Finite>, 6>;
/// x87 80-bit extended precision.
pub type F80 = Float<Binary<15, X87>, 80>;
/// IEEE 754 decimal32 in the BID encoding.
pub type D32Bid = Float<Decimal<Bid>, 32>;
/// IEEE 754 decimal64 in the BID encoding.
pub type D64Bid = Float<Decimal<Bid>, 64>;
/// IEEE 754 decimal128 in the BID encoding.
pub type D128Bid = Float<Decimal<Bid>, 128>;
/// IEEE 754 decimal32 in the DPD encoding.
pub type D32Dpd = Float<Decimal<Dpd>, 32>;
/// IEEE 754 decimal64 in the DPD encoding.
pub type D64Dpd = Float<Decimal<Dpd>, 64>;
/// IEEE 754 decimal128 in the DPD encoding.
pub type D128Dpd = Float<Decimal<Dpd>, 128>;

#[cfg(test)]
mod tests;
