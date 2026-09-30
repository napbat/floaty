//! Conversion to a decimal format: from another decimal format, and from a
//! binary format, correctly rounded.

use super::digits::{power_of_ten, power_of_ten_u128};
use super::{DecimalLayout, Wide};
use crate::env::{Env, Flags, NanPropagation};
use crate::exact::Unrounded;
use crate::format::internal::Source;
use crate::format::{DecimalEncoding, Storage, Width};
use crate::limbs::{self, Limbs, Widen};
use crate::nan::{self, default_nan};
use crate::radix;
use crate::unpacked::Unpacked;

impl<Enc: DecimalEncoding, const W: usize> DecimalLayout<Enc, W>
where
    Width<W>: Storage,
{
    /// Converts a decoded value of another format.
    ///
    /// A decimal source keeps its exponent where the format holds it, as
    /// IEEE 754 `convertFormat` prefers. A binary source prefers exponent 0,
    /// as the Intel library gives it. A NaN payload keeps its high-order
    /// digits between decimal formats, and its high-order bits between a
    /// binary payload and the trailing significand field, as the Intel library
    /// does. A payload above `10^(p - 1) - 1` becomes zero.
    pub fn convert_from<In: Limbs, Out: Widen>(
        value: Unpacked<In>,
        source: Source,
        env: &Env,
    ) -> (Out, Flags) {
        match value {
            Unpacked::Nan {
                negative,
                signaling,
                payload,
            } => {
                let flags = if signaling {
                    Flags::INVALID
                } else {
                    Flags::NONE
                };
                if env.nan.propagation == NanPropagation::DefaultNan {
                    return Self::exact(default_nan(env), flags);
                }
                let payload = Self::payload(payload, source);
                let nan = Unpacked::Nan {
                    negative,
                    signaling: false,
                    payload: limbs::from_u128(payload),
                };
                Self::exact(nan, flags)
            }
            Unpacked::Infinity { negative } => {
                Self::exact(Unpacked::Infinity { negative }, Flags::NONE)
            }
            Unpacked::Zero { negative, exponent } => {
                let exponent = if source.radix == 10 {
                    i64::from(exponent)
                } else {
                    0
                };
                Self::exact(Self::zero(negative, exponent), Flags::NONE)
            }
            Unpacked::Unsupported => Self::exact(default_nan(env), Flags::INVALID),
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => {
                if source.radix == 10 {
                    let value = Unrounded {
                        negative,
                        exponent,
                        significand: significand.resize::<Wide<Out>>(),
                        sticky: false,
                    };
                    Self::finish(&value, i64::from(exponent), *env, Flags::NONE)
                } else {
                    Self::from_binary(negative, exponent, &significand, env)
                }
            }
        }
    }

    /// Returns the payload of a NaN of another format, in this format.
    fn payload<In: Limbs>(payload: In, source: Source) -> u128 {
        let payload = if source.radix == 10 {
            let (from, to) = (source.payload_digits, Self::PRECISION - 1);
            let value = limbs::to_u128(&payload);
            if to >= from {
                value * power_of_ten_u128(to - from)
            } else {
                value / power_of_ten_u128(from - to)
            }
        } else {
            let field: [u64; 2] = nan::align_payload(payload, source.payload_bits, Self::TRAILING);
            limbs::to_u128(&field)
        };
        if payload > Self::LARGEST_PAYLOAD {
            0
        } else {
            payload
        }
    }

    /// Converts `significand * 2^exponent`, correctly rounded.
    ///
    /// The value divided by 10^s, for s four digits below the precision under
    /// the estimated leading digit, gives a coefficient of three to five
    /// digits more than the precision, and a sticky bit. The decimal rounding
    /// routine rounds it once.
    fn from_binary<In: Limbs, Out: Widen>(
        negative: bool,
        exponent: i32,
        significand: &In,
        env: &Env,
    ) -> (Out, Flags) {
        let precision = i64::from(Self::PRECISION);
        let lowest = i64::from(Self::EMIN) - precision + 1;
        let top = i64::from(exponent) + i64::from(significand.bit_length()) - 1;
        let digits = radix::digits_estimate(top);
        let unit: Wide<Out> = power_of_ten(Self::PRECISION + 1);
        // A value far outside the range rounds as a stand-in on the same
        // side, so that the powers of 5 stay small. A value of more than
        // EMAX + 2 digits gives a stand-in far above the largest value. A value
        // below a hundredth of the smallest subnormal value gives a stand-in
        // just above a thousandth of it. Each stand-in rounds as the value
        // does in every direction.
        if digits - 1 > i64::from(Self::EMAX) + 1 {
            let above = Unrounded {
                negative,
                exponent: Self::EMAX + 2,
                significand: unit,
                sticky: true,
            };
            return Self::finish(&above, 0, *env, Flags::NONE);
        }
        if digits < lowest - 2 {
            let below = Unrounded {
                negative,
                exponent: i32::try_from(lowest - precision - 4).expect("the exponent fits an i32"),
                significand: unit,
                sticky: true,
            };
            return Self::finish(&below, 0, *env, Flags::NONE);
        }
        let scale = digits - precision - 4;
        let magnitude = u32::try_from(scale.unsigned_abs()).expect("the scale is small");
        let bits = i64::from(significand.bit_length())
            + radix::power_of_five_bits(magnitude)
            + (i64::from(exponent) - scale).max(0)
            + 1;
        let (coefficient, sticky) = if radix::fits::<{ radix::NARROW }>(bits) {
            reduce::<{ radix::NARROW }, In, Wide<Out>>(significand, exponent, scale)
        } else if radix::fits::<{ radix::SMALL }>(bits) {
            reduce::<{ radix::SMALL }, In, Wide<Out>>(significand, exponent, scale)
        } else {
            reduce::<{ radix::LARGE }, In, Wide<Out>>(significand, exponent, scale)
        };
        let value = Unrounded {
            negative,
            exponent: i32::try_from(scale).expect("the scale fits an i32"),
            significand: coefficient,
            sticky,
        };
        Self::finish(&value, 0, *env, Flags::NONE)
    }
}

/// Returns `significand * 2^exponent / 10^scale`, rounded down, computed in
/// numbers of `N` limbs, and `true` when the division is not exact.
fn reduce<const N: usize, In: Limbs, Out: Limbs>(
    significand: &In,
    exponent: i32,
    scale: i64,
) -> (Out, bool) {
    let magnitude = u32::try_from(scale.unsigned_abs()).expect("the scale is small");
    let five = radix::power_of_five::<N>(magnitude);
    let shift = i64::from(exponent) - scale;
    let (coefficient, sticky) = if scale >= 0 {
        let (numerator, lost) = radix::shift(&significand.resize(), shift);
        let (quotient, remainder) = limbs::divide(numerator, five);
        (quotient, lost || !remainder.is_zero())
    } else {
        radix::shift(&radix::multiply(&five, significand), shift)
    };
    (coefficient.resize(), sticky)
}

#[cfg(test)]
mod tests {
    use crate::float::{D32Bid, D64Bid, D128Dpd, Decoded, F32, F64};

    #[test]
    fn binary_values_convert_correctly_rounded() {
        let parts = |value: D64Bid| match value.decode::<1>() {
            Decoded::Finite {
                exponent,
                significand: [coefficient],
                negative,
            } => (negative, coefficient, exponent),
            Decoded::Zero { exponent, .. } => (false, 0, exponent),
            other => panic!("{other:?}"),
        };
        assert_eq!(
            parts(F64::from_bits(0x3FF0_0000_0000_0000).convert()),
            (false, 1, 0)
        );
        assert_eq!(
            parts(F64::from_bits(0x3FE0_0000_0000_0000).convert()),
            (false, 5, -1)
        );
        assert_eq!(parts(F64::from_bits(0).convert()), (false, 0, 0));
        // 0.1 in binary64 is 0.1000000000000000055511151231257827...
        assert_eq!(
            parts(F64::from_bits(0x3FB9_9999_9999_999A).convert()),
            (false, 1_000_000_000_000_000, -16)
        );
        // The Intel test data: the smallest binary64 subnormal is inexact in
        // decimal64, and rounds to 4.940656458412465E-324.
        let tiny: D64Bid = F64::from_bits(1).convert();
        assert_eq!(parts(tiny), (false, 4_940_656_458_412_465, -339));
        let max: D32Bid = F32::from_bits(0x7F7F_FFFF).convert();
        assert!(max.is_finite());
        let wide: D128Dpd = F64::from_bits(0x7FEF_FFFF_FFFF_FFFF).convert();
        assert!(wide.is_finite());
    }

    #[test]
    fn decimal_values_keep_their_exponent() {
        let small = D64Bid::from_bits(0x31A0_0000_0000_0064); // 100E-2
        let wide: D128Dpd = small.convert();
        let back: D64Bid = wide.convert();
        assert_eq!(back.to_bits(), small.to_bits());
    }
}
