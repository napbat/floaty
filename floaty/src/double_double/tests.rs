use super::{DoubleDouble, Gcc, Qd};
use crate::env::{Env, Flags};
use crate::float::{F32, F64};

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
