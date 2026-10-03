//! The conversions into and out of a double-double, and the rounding of an
//! exact value to a pair by the rule of the documentation of `DoubleDouble`.
//!
//! The value first rounds to odd on the grid of [`Fixed`]. Then the high half
//! rounds to nearest even, the low half rounds the rest, and the sum of the
//! two halves splits again into its canonical pair.

use core::cmp::Ordering;

use super::{Algorithm, DoubleDouble, Pair};
use crate::env::{Behavior, Env, Flags, Mode, Override, Rounding};
use crate::exact::{self, Exact};
use crate::float::{Decoded, F64, FloatType};
use crate::format::internal::Source;
use crate::format::{Binary, Standard};
use crate::host::Host;
use crate::integer::Integer;
use crate::limbs::{self, Limbs};
use crate::radix;
use crate::sealed::Sealed;
use crate::unpacked::Unpacked;

/// `|x| * 2^1100`, rounded to odd: the integer part, with the lowest bit set
/// when a bit below it is set.
///
/// Every rounding of this module is at the binary64 quantum 2^-1074 or
/// above, 26 bits above the lowest bit. A value rounded to odd at least two
/// bits below a rounding position rounds as the exact value does (Boldo and
/// Melquiond, "Emulation of FMA and Correctly Rounded Sums: Proved
/// Algorithms Using Rounding to Odd", IEEE Transactions on Computers 57(4),
/// 2008, Theorem 1). A binary64 value is a multiple of 2^-1074, so it is
/// even here, and a sum with it keeps the rounding to odd. A value below
/// 2^1024 needs 2,124 bits.
type Fixed = [u64; 34];

/// The weight of the lowest bit of a [`Fixed`].
const FIXED_LOWEST: i32 = -1100;

/// A signed [`Fixed`]: `true` for a negative value, and the magnitude.
type Signed = (bool, Fixed);

/// Rounds a decoded value of another format, the source, to a pair. Returns
/// the pair and the flags.
///
/// A zero gives the zero of its sign with a `+0` low half. An infinity
/// gives the infinity with a `+0` low half, or with saturation the largest
/// finite pair, as binary64 saturates an infinity. A NaN converts to
/// binary64, by the rules of a conversion, and takes a `+0` low half.
fn round_value<L: Limbs, B: Behavior>(
    value: Unpacked<L>,
    source: Source,
    behavior: B,
) -> (Pair, Flags) {
    let env = behavior.env();
    match value {
        Unpacked::Zero { negative, .. } => ((zero(negative), zero(false)), Flags::NONE),
        Unpacked::Infinity { negative } if env.saturate => (largest(negative, &env), Flags::NONE),
        Unpacked::Infinity { negative } => ((infinity(negative), zero(false)), Flags::NONE),
        Unpacked::Nan { .. } | Unpacked::Unsupported => {
            let (high, flags) = <F64 as FloatType>::convert_from(value, source, env);
            ((high, zero(false)), flags)
        }
        Unpacked::Finite {
            negative,
            exponent,
            significand,
        } => {
            let fixed = match source {
                Source::Decimal { .. } => from_decimal(&significand, exponent),
                Source::Binary { .. } => from_binary(significand, exponent),
            };
            match fixed {
                Some(fixed) => round_fixed(&(negative, fixed), &env),
                None => overflow(negative, &env),
            }
        }
    }
}

/// Returns `significand * 2^exponent` as a [`Fixed`], or `None` when it is
/// at least 2^1024.
fn from_binary<L: Limbs>(significand: L, exponent: i32) -> Option<Fixed> {
    let top = i64::from(exponent) + i64::from(significand.bit_length()) - 1;
    if top >= 1024 {
        return None;
    }
    let shift = i64::from(exponent) - i64::from(FIXED_LOWEST);
    if shift >= 0 {
        let shift = u32::try_from(shift).expect("a value below 2^1024 fits the width");
        Some(significand.resize::<Fixed>().shl(shift))
    } else {
        let count = u32::try_from(-shift).unwrap_or(u32::MAX);
        Some(significand.shr_jam(count).resize())
    }
}

/// The limbs that hold 5^1100, 2,554 bits, for [`from_decimal`].
type DivisionLimbs = [u64; 48];

/// Returns `significand * 10^exponent` as a [`Fixed`], or `None` when it is
/// at least 2^1024. The significand has at most 113 bits, as a decimal128
/// coefficient does.
fn from_decimal<L: Limbs>(significand: &L, exponent: i32) -> Option<Fixed> {
    let power = exponent.unsigned_abs();
    if exponent >= 0 {
        // 10^309 is above 2^1026, and 10^34 * 5^308 fits 1,024 bits.
        if exponent > 308 {
            return None;
        }
        let product = radix::multiply(
            &radix::power_of_five::<{ radix::SMALL }>(power),
            significand,
        );
        return from_binary(product, exponent);
    }
    // `significand * 10^-power` is `significand * 2^-power / 5^power`. A
    // value below 10^34 * 10^-1101 is below 2^-1100, so its fixed form is the
    // sticky bit alone.
    let Some(shift) = 1100_u32.checked_sub(power) else {
        return Some(Fixed::ZERO.with_bit(0));
    };
    let numerator = significand.resize::<DivisionLimbs>().shl(shift);
    let (quotient, remainder) = limbs::divide(numerator, radix::power_of_five(power));
    let fixed: Fixed = quotient.resize();
    Some(if remainder.is_zero() {
        fixed
    } else {
        fixed.with_bit(0)
    })
}

/// Rounds a finite value to a pair, by the rule of the module.
fn round_fixed(value: &Signed, env: &Env) -> (Pair, Flags) {
    let negative = value.0;
    let nearest = Env {
        rounding: Rounding::TiesToEven,
        saturate: false,
        ..*env
    };
    let Some((high, rest)) = split(value, nearest) else {
        return overflow(negative, env);
    };
    let (low, flags) = if rest.1.is_zero() {
        (zero(false), Flags::NONE)
    } else {
        to_f64(&rest, *env)
    };
    // `ROUNDED_UP` of the low half says that it grew away from the rest. The
    // pair grows away from the value when the rest has the sign of the value.
    let grew =
        flags.contains(Flags::INEXACT) && flags.contains(Flags::ROUNDED_UP) == (rest.0 == negative);
    let mut flags = flags.difference(Flags::ROUNDED_UP);
    if grew {
        flags |= Flags::ROUNDED_UP;
    }
    let total = sum(&signed(high), &signed(low));
    if total.1.is_zero() {
        // An exact zero takes the sign of the value.
        return ((zero(negative), zero(false)), flags);
    }
    let Some((high, rest)) = split(&total, nearest) else {
        return overflow(negative, env);
    };
    if rest.1.is_zero() {
        return ((high, zero(false)), flags);
    }
    let (low, exact_flags) = to_f64(&rest, *env);
    debug_assert!(
        !exact_flags.contains(Flags::INEXACT),
        "the rest of the canonical split is exact"
    );
    ((high, low), flags)
}

/// Splits a finite value into its high half, rounded to nearest even, and the
/// exact rest. `None` means that the high half overflows.
fn split(value: &Signed, nearest: Env) -> Option<(F64, Signed)> {
    let (high, _) = to_f64(value, nearest);
    if high.is_infinite() {
        return None;
    }
    Some((high, sum(value, &negate(signed(high)))))
}

/// Rounds a signed [`Fixed`] to binary64.
fn to_f64(&(negative, magnitude): &Signed, env: Env) -> (F64, Flags) {
    let exact = Exact {
        negative,
        exponent: FIXED_LOWEST,
        significand: magnitude,
        sticky: false,
    };
    F64::round(exact, env)
}

/// Returns a finite binary64 value as a signed [`Fixed`].
fn signed(value: F64) -> Signed {
    match value.decode::<1>() {
        Decoded::Zero { negative, .. } => (negative, Fixed::ZERO),
        Decoded::Finite {
            negative,
            exponent,
            significand,
        } => {
            let shift = u32::try_from(exponent - FIXED_LOWEST)
                .expect("a binary64 exponent is above the lowest weight");
            (negative, significand.resize::<Fixed>().shl(shift))
        }
        Decoded::Infinity { .. } | Decoded::Nan { .. } | Decoded::Unsupported => {
            unreachable!("the caller passes a finite value")
        }
    }
}

/// Returns the negation of a signed [`Fixed`].
fn negate((negative, magnitude): Signed) -> Signed {
    (!negative, magnitude)
}

/// Returns the exact sum of two signed [`Fixed`] values. The sum fits the
/// width.
fn sum(left: &Signed, right: &Signed) -> Signed {
    if left.0 == right.0 {
        return (left.0, left.1.add(right.1));
    }
    match left.1.compare(&right.1) {
        Ordering::Less => (right.0, right.1.sub(left.1)),
        Ordering::Equal | Ordering::Greater => (left.0, left.1.sub(right.1)),
    }
}

/// Returns the result of an overflow of the sign `negative`: an infinity with
/// a `+0` low half, or the largest finite pair when the direction or
/// saturation gives the largest finite binary64 value.
fn overflow(negative: bool, env: &Env) -> (Pair, Flags) {
    let flags = Flags::OVERFLOW | Flags::INEXACT;
    if exact::overflow_is_infinite(env, negative) {
        return ((infinity(negative), zero(false)), flags | Flags::ROUNDED_UP);
    }
    (largest(negative, env), flags)
}

/// Returns the largest finite pair of the sign `negative` at the precision
/// of `env`.
///
/// The largest finite pair at precision `p` is the largest binary64 value
/// at `p` bits, plus the largest value at `p` bits below half of its ulp:
/// the largest canonical pair. A value below the overflow threshold rounds
/// to it, so every larger value must round to it too. At 53 bits, the pair
/// is `2^1024 - 2^970 - 2^917`. The `LDBL_MAX` of GCC is `2^1024 - 2^970 -
/// 2^918`, because GCC counts 106 bits in the IBM `long double`.
pub(super) fn largest(negative: bool, env: &Env) -> Pair {
    let precision = env.precision_within(<Binary<11> as Standard<64>>::PRECISION);
    let ones = [(1_u64 << precision) - 1];
    let bits = i32::try_from(precision).expect("a precision fits an i32");
    let part = |exponent: i32| {
        let exact = Exact {
            negative,
            exponent,
            significand: ones,
            sticky: false,
        };
        F64::round(exact, *env).0
    };
    (part(1024 - bits), part(1023 - 2 * bits))
}

/// Returns the zero of the sign `negative`. A pair without a rest has the
/// positive zero as its low half.
fn zero(negative: bool) -> F64 {
    F64::from_bits(u64::from(negative) << 63)
}

/// Returns the infinity of the sign `negative`.
fn infinity(negative: bool) -> F64 {
    F64::from_bits((u64::from(negative) << 63) | 0x7FF0_0000_0000_0000)
}

/// The source of a binary value, for an integer or an exact result. A
/// finite value reads no payload.
const BINARY: Source = <Binary<11> as Standard<64>>::SOURCE;

impl<Alg: Algorithm, M: Mode> DoubleDouble<Alg, M> {
    /// Converts the exact value to another format with the default mode of
    /// that format: once rounded to a [`Float`](crate::Float), or to a pair
    /// by the rule of the type documentation.
    #[must_use]
    pub fn convert<T: FloatType>(self) -> T {
        self.convert_with(T::Mode::default()).0
    }

    /// Converts the exact value to another format, with an override of the
    /// destination behavior: once rounded to a [`Float`](crate::Float), or to
    /// a pair by the rule of the type documentation. Returns the result and
    /// the flags. A NaN keeps the high-order bits of its payload.
    #[must_use]
    pub fn convert_with<T: FloatType>(self, behavior: impl Override) -> (T, Flags) {
        T::convert_from(self.exact(), BINARY, behavior.apply::<T::Mode>())
    }
}

impl<Alg: Algorithm, M: Mode> Sealed for DoubleDouble<Alg, M> {}

/// A double-double is a conversion destination, so
/// [`Float::convert`](crate::Float::convert) and
/// [`DoubleDouble::convert`] round to a pair. The rule of the type
/// documentation of [`DoubleDouble`] applies.
impl<Alg: Algorithm, M: Mode> FloatType for DoubleDouble<Alg, M> {
    type Mode = M;

    const HOST: Host = Host::None;

    fn from_host(_: [u64; 2]) -> Self {
        unreachable!("a double-double has no host format, so no host path converts to one")
    }

    fn convert_from<L: Limbs, B: Behavior>(
        value: Unpacked<L>,
        source: Source,
        behavior: B,
    ) -> (Self, Flags) {
        let ((hi, lo), flags) = round_value(value, source, behavior);
        (Self::new(hi, lo), flags)
    }
}

impl<Alg: Algorithm, M: Mode> DoubleDouble<Alg, M> {
    /// Converts an integer, rounding with the default mode.
    ///
    /// ```
    /// use floaty::{DoubleDouble, Qd};
    ///
    /// // 2^64 - 1 has 64 bits, and the pair holds it exactly.
    /// let value = DoubleDouble::<Qd>::from_int(u64::MAX);
    /// assert_eq!(value.hi().to_bits(), 0x43F0_0000_0000_0000); // 2^64
    /// assert_eq!(value.lo().to_bits(), 0xBFF0_0000_0000_0000); // -1
    /// ```
    #[must_use]
    pub fn from_int<I: Integer>(value: I) -> Self {
        Self::from_int_with(value, M::default()).0
    }

    /// Converts an integer, and returns the result and the flags.
    ///
    /// The integer rounds to a pair by the rule of the type documentation.
    /// Every integer of up to 106 bits is exact, and so is an integer whose
    /// set bits span at most 53 bits in each of two groups. A zero converts
    /// to `+0`.
    #[must_use]
    pub fn from_int_with<I: Integer>(value: I, behavior: impl Override) -> (Self, Flags) {
        let parts = value.to_parts();
        let value = if parts.magnitude.is_zero() {
            Unpacked::zero(false)
        } else {
            Unpacked::Finite {
                negative: parts.negative,
                exponent: 0,
                significand: parts.magnitude,
            }
        };
        Self::from_exact(value, behavior.apply::<M>())
    }

    /// Rounds a binary exact value to a pair by the rule of the type
    /// documentation. Returns the pair and the flags.
    pub(super) fn from_exact<L: Limbs, B: Behavior>(
        value: Unpacked<L>,
        behavior: B,
    ) -> (Self, Flags) {
        let ((hi, lo), flags) = round_value(value, BINARY, behavior);
        (Self::new(hi, lo), flags)
    }
}

#[cfg(test)]
mod tests {
    use crate::double_double::{DoubleDouble, Gcc, Qd};
    use crate::env::{Env, Flags, Rounding};
    use crate::float::F128;

    fn bits<Alg: crate::double_double::Algorithm>(value: DoubleDouble<Alg>) -> (u64, u64) {
        (value.hi().to_bits(), value.lo().to_bits())
    }

    #[test]
    fn an_overflow_gives_an_infinity_or_the_largest_pair() {
        // 2^1024 in binary128.
        let huge = F128::from_bits(0x43FF_0000_0000_0000_0000_0000_0000_0000);
        let (nearest, flags) = huge.convert_with::<DoubleDouble<Gcc>>(Env::IEEE);
        assert_eq!(bits(nearest), (0x7FF0_0000_0000_0000, 0));
        assert_eq!(flags, Flags::OVERFLOW | Flags::INEXACT | Flags::ROUNDED_UP);
        let (toward_zero, flags) = huge.convert_with::<DoubleDouble<Gcc>>(Rounding::TowardZero);
        // 2^1024 - 2^970 - 2^917, the largest canonical pair.
        assert_eq!(
            bits(toward_zero),
            (0x7FEF_FFFF_FFFF_FFFF, 0x7C8F_FFFF_FFFF_FFFF)
        );
        assert_eq!(flags, Flags::OVERFLOW | Flags::INEXACT);
    }

    #[test]
    fn a_low_half_that_rounds_to_a_tie_splits_again() {
        // 1 + 2^-52 + 2^-53 - 2^-110: the high half rounds to 1 + 2^-52, and the
        // rest rounds up to 2^-53. The sum is a tie, which the canonical split
        // gives to the even high half 1 + 2^-51.
        let value = F128::from_bits(0x3FFF_0000_0000_0000_17FF_FFFF_FFFF_FFFC);
        let (pair, flags) = value.convert_with::<DoubleDouble<Qd>>(Env::IEEE);
        assert_eq!(bits(pair), (0x3FF0_0000_0000_0002, 0xBCA0_0000_0000_0000));
        assert_eq!(flags, Flags::INEXACT | Flags::ROUNDED_UP);
        assert!(pair.is_canonical());
    }

    #[test]
    fn an_exact_value_gives_its_canonical_pair_in_every_direction() {
        // 2^53 + 3 is 2^53 + 4 - 1 in its canonical pair.
        let value = (1_u64 << 53) + 3;
        for rounding in [
            Rounding::TiesToEven,
            Rounding::TowardNegative,
            Rounding::TowardPositive,
            Rounding::ToOdd,
        ] {
            let (pair, flags) = DoubleDouble::<Gcc>::from_int_with(value, rounding);
            assert_eq!(bits(pair), (0x4340_0000_0000_0002, 0xBFF0_0000_0000_0000));
            assert_eq!(flags, Flags::NONE);
        }
    }
}
