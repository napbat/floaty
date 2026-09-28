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
