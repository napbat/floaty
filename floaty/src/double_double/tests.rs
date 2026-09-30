use core::cmp::Ordering;

use super::{DoubleDouble, Gcc, Qd};
use crate::env::{Env, Flags, Rounding};
use crate::float::{Decoded, F32, F64};

fn gcc(hi: u64, lo: u64) -> DoubleDouble<Gcc> {
    DoubleDouble::from_parts(F64::from_bits(hi), F64::from_bits(lo))
}

fn qd(hi: u64, lo: u64) -> DoubleDouble<Qd> {
    DoubleDouble::from_parts(F64::from_bits(hi), F64::from_bits(lo))
}

fn bits<Alg: super::Algorithm>(value: DoubleDouble<Alg>) -> (u64, u64) {
    (value.hi().to_bits(), value.lo().to_bits())
}

const ONE: u64 = 0x3FF0_0000_0000_0000;
const THREE: u64 = 0x4008_0000_0000_0000;

#[test]
fn a_small_addend_lands_in_the_low_half() {
    // 1 + 2^-60 needs the low half.
    let tiny = 0x3C30_0000_0000_0000;
    assert_eq!(bits(gcc(ONE, 0) + gcc(tiny, 0)), (ONE, tiny));
    assert_eq!(bits(qd(ONE, 0) + qd(tiny, 0)), (ONE, tiny));
    assert_eq!(bits(gcc(ONE, 0) - gcc(ONE, 0)), (0, 0));
}

#[test]
fn a_third_has_a_low_half() {
    let expected = (0x3FD5_5555_5555_5555, 0x3C75_5555_5555_5555);
    assert_eq!(bits(gcc(ONE, 0) / gcc(THREE, 0)), expected);
    assert_eq!(bits(qd(ONE, 0) / qd(THREE, 0)), expected);
    let (_, flags) = gcc(ONE, 0).div_with(gcc(THREE, 0), Env::IEEE);
    // Some step rounded up; the flags are the union of the steps.
    assert_eq!(flags, Flags::INEXACT | Flags::ROUNDED_UP);
    let product = gcc(0x3FD5_5555_5555_5555, 0x3C75_5555_5555_5555) * gcc(THREE, 0);
    assert_eq!(product.hi().to_bits(), ONE);
}

#[test]
fn a_zero_result_keeps_the_sign_of_the_high_part() {
    let negative_zero = 0x8000_0000_0000_0000;
    assert_eq!(
        bits(gcc(negative_zero, 0) * gcc(ONE, 0)),
        (negative_zero, 0)
    );
    assert_eq!(bits(qd(0, 0).sqrt()), (0, 0));
    assert_eq!(bits(qd(negative_zero, 0).sqrt()), (0, 0));
    let nan = 0x7FF8_0000_0000_0000;
    assert_eq!(bits(qd(0xBFF0_0000_0000_0000, 0).sqrt()), (nan, nan));
}

#[test]
fn saturation_leaves_the_arithmetic_unchanged() {
    // Each operation overflows in a step. `Gcc` gives an infinity, and `Qd`
    // gives a NaN in both halves.
    let (max, min) = (0x7FEF_FFFF_FFFF_FFFF, 0x0010_0000_0000_0000);
    let saturating = Env::IEEE.with_saturate(true);
    let outcome = |(value, flags): (DoubleDouble<Gcc>, Flags)| (bits(value), flags);
    let (x, y) = (gcc(max, 0), gcc(min, 0));
    let overflow = Flags::OVERFLOW | Flags::INEXACT | Flags::ROUNDED_UP;
    for result in [
        x.add_with(x, saturating),
        x.sub_with(-x, saturating),
        x.mul_with(x, saturating),
        x.div_with(y, saturating),
    ] {
        assert_eq!(outcome(result), ((0x7FF0_0000_0000_0000, 0), overflow));
    }
    let outcome = |(value, flags): (DoubleDouble<Qd>, Flags)| (bits(value), flags);
    let (x, y) = (qd(max, 0), qd(min, 0));
    let nan = 0x7FF8_0000_0000_0000;
    for result in [
        x.add_with(x, saturating),
        x.sub_with(-x, saturating),
        x.mul_with(x, saturating),
        x.div_with(y, saturating),
    ] {
        assert_eq!(outcome(result), ((nan, nan), overflow | Flags::INVALID));
    }
    // A conversion saturates as its destination format does.
    let (value, flags) = x.convert_with::<F32>(saturating);
    assert_eq!(
        (value.to_bits(), flags),
        (0x7F7F_FFFF, Flags::OVERFLOW | Flags::INEXACT)
    );
}

#[test]
fn the_exact_value_decodes_converts_and_compares() {
    let value = gcc(ONE, 0x3C30_0000_0000_0000);
    assert_eq!(
        value.decode(),
        Decoded::Finite {
            negative: false,
            exponent: -60,
            significand: {
                let mut limbs = [0; 33];
                limbs[0] = (1 << 60) + 1;
                limbs
            },
        }
    );
    let single: F32 = value.convert();
    assert_eq!(single.to_bits(), 0x3F80_0000);
    let (up, flags) = value.convert_with::<F32>(Rounding::TowardPositive);
    assert_eq!(
        (up.to_bits(), flags),
        (0x3F80_0001, Flags::INEXACT | Flags::ROUNDED_UP)
    );
    assert!(value > gcc(ONE, 0));
    assert_eq!(
        gcc(ONE, 0).partial_cmp(&gcc(ONE, 0x8000_0000_0000_0000)),
        Some(Ordering::Equal)
    );
    // (1, -1) is zero, with the sign of the high half.
    let cancelled = gcc(ONE, 0xBFF0_0000_0000_0000);
    assert_eq!(
        cancelled.decode(),
        Decoded::Zero {
            negative: false,
            exponent: 0
        }
    );
    assert_eq!(bits(-cancelled), (0xBFF0_0000_0000_0000, ONE));
}

#[test]
fn abs_reads_the_sign_of_the_exact_value() {
    // (-0, 1) is 1, so it keeps its halves; (+0, -1) is -1.
    let negative_zero = 0x8000_0000_0000_0000;
    assert_eq!(bits(gcc(negative_zero, ONE).abs()), (negative_zero, ONE));
    assert_eq!(
        bits(gcc(0, ONE | negative_zero).abs()),
        (negative_zero, ONE)
    );
    assert_eq!(bits(gcc(negative_zero, 0).abs()), (0, negative_zero));
}

#[test]
fn an_overflow_gives_an_infinity_or_the_largest_pair() {
    // 2^1024 in binary128.
    let huge = crate::float::F128::from_bits(0x43FF_0000_0000_0000_0000_0000_0000_0000);
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
    let value = crate::float::F128::from_bits(0x3FFF_0000_0000_0000_17FF_FFFF_FFFF_FFFC);
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

#[test]
fn the_class_follows_the_exact_value() {
    use crate::float::Class;
    // 2^-969 is the smallest normal value, and 2^-969 - 2^-1074 is below it.
    assert_eq!(gcc(0x0360_0000_0000_0000, 0).classify(), Class::Normal);
    assert_eq!(
        gcc(0x0360_0000_0000_0000, 0x8000_0000_0000_0001).classify(),
        Class::Subnormal
    );
    // The halves of (1, -1) cancel.
    assert!(gcc(ONE, 0xBFF0_0000_0000_0000).is_zero());
    // A NaN low half makes the value a NaN, and an infinite high half hides it.
    assert!(gcc(ONE, 0x7FF0_0000_0000_0001).is_signaling_nan());
    assert!(gcc(0xFFF0_0000_0000_0000, 0x7FF8_0000_0000_0000).is_infinite());
    assert!(gcc(0xFFF0_0000_0000_0000, 0x7FF8_0000_0000_0000).is_sign_negative());
}

#[test]
fn a_canonical_low_half_is_at_most_half_an_ulp() {
    let half_ulp = 0x3CA0_0000_0000_0000; // 2^-53
    assert!(gcc(ONE, half_ulp).is_canonical());
    assert!(!gcc(0x3FF0_0000_0000_0001, half_ulp).is_canonical());
    assert!(!gcc(ONE, 0x3CB0_0000_0000_0000).is_canonical());
    assert!(gcc(ONE, 0x3C9F_FFFF_FFFF_FFFF).is_canonical());
    assert!(!gcc(0x7FF0_0000_0000_0000, 1).is_canonical());
    assert!(gcc(0x7FF8_0000_0000_0000, 1).is_canonical());
    assert!(!gcc(0, 1).is_canonical());
}

#[test]
fn pairs_of_one_value_order_by_their_halves() {
    // (1, 0) and (2, -1) hold 1.
    let (one, twin) = (
        gcc(ONE, 0),
        gcc(0x4000_0000_0000_0000, 0xBFF0_0000_0000_0000),
    );
    assert_eq!(one.total_cmp(twin), Ordering::Less);
    assert_eq!(twin.total_cmp(one), Ordering::Greater);
    assert_eq!(bits(one.minimum(twin)), bits(one));
    assert_eq!(bits(one.maximum(twin)), bits(twin));
    // -0 orders below +0.
    let (negative, positive) = (gcc(1 << 63, 0), gcc(0, 0));
    assert_eq!(bits(positive.minimum(negative)), bits(negative));
}

#[test]
fn compound_assignment_follows_the_operators() {
    let (x, y) = (gcc(ONE, 0), gcc(THREE, 0));
    let mut value = x;
    value += y;
    assert_eq!(bits(value), bits(x + y));
    value -= y;
    assert_eq!(bits(value), bits(x + y - y));
    value *= y;
    assert_eq!(bits(value), bits((x + y - y) * y));
    value /= y;
    assert_eq!(bits(value), bits((x + y - y) * y / y));
    // 7 % 3 truncates the quotient to 2, for both references.
    let seven = 0x401C_0000_0000_0000;
    assert_eq!(bits(gcc(seven, 0) % y), (0x3FF0_0000_0000_0000, 0));
    assert_eq!(
        bits(qd(seven, 0) % qd(THREE, 0)),
        (0x3FF0_0000_0000_0000, 0)
    );
    let mut value = gcc(seven, 0);
    value %= y;
    assert_eq!(bits(value), bits(gcc(seven, 0).truncated_remainder(y)));
}
