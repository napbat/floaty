//! Decimal base-two exact cases and directed MPFR rounding checks.
#![cfg(all(target_arch = "x86_64", feature = "alloc"))]

use floaty::format::Standard;
use floaty::{D32Bid, D32Dpd, D64Bid, D64Dpd, D128Bid, D128Dpd, Decoded, Flags, Float};
use floaty_verify::intel_decimal::{Bid32, Bid64, Bid128, Format as IntelFormat};
use floaty_verify::mpfr::decimal::{self, DecimalFormat, DecimalValue, TO_DECIMAL_BEHAVIORS};
use rug::float::Round;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer, Rational};

#[derive(Clone, Copy, Debug)]
enum Function {
    Exp2,
    Log2,
}

fn bid(format: DecimalFormat, negative: bool, coefficient: u128, exponent: i64) -> u128 {
    let trailing = format.trailing_bits();
    let width = 16 * (trailing + 10) / 15;
    let biased = u128::try_from(exponent - format.exponents().0).unwrap();
    let bits = if coefficient >> (trailing + 3) == 0 {
        (biased << (trailing + 3)) | coefficient
    } else {
        (0b11 << (width - 3))
            | (biased << (trailing + 1))
            | (coefficient & ((1 << (trailing + 1)) - 1))
    };
    bits | (u128::from(negative) << (width - 1))
}

fn check<S: Standard<W>, const W: usize>(
    x: Float<S, W>,
    format: DecimalFormat,
    function: Function,
) {
    let argument = match x.decode::<2>() {
        Decoded::Finite {
            negative,
            exponent,
            significand,
        } => {
            let mut value = Rational::from(Integer::from_digits(&significand, Order::Lsf));
            let power = Integer::from(Integer::u_pow_u(10, exponent.unsigned_abs()));
            if exponent >= 0 {
                value *= power;
            } else {
                value /= power;
            }
            if negative { -value } else { value }
        }
        Decoded::Zero { .. } => Rational::from(0),
        _ => panic!("this finite-value oracle does not accept specials"),
    };
    for env in &TO_DECIMAL_BEHAVIORS {
        if x.is_subnormal() && env.denormals_are_zero {
            continue;
        }
        let mut working = 8 * format.precision + 64;
        let (expected, mut expected_flags) = loop {
            let evaluate = |round| {
                let bound = BigFloat::with_val_round(working, &argument, round).0;
                match function {
                    Function::Exp2 => BigFloat::with_val_round(working, bound.exp2_ref(), round).0,
                    Function::Log2 => BigFloat::with_val_round(working, bound.log2_ref(), round).0,
                }
            };
            let convert = |value: &BigFloat| {
                if value.is_zero() {
                    (
                        DecimalValue::Zero {
                            negative: value.is_sign_negative(),
                            exponent: 0,
                        },
                        Flags::NONE,
                    )
                } else {
                    decimal::to_decimal(value, format, env)
                }
            };
            let low = convert(&evaluate(Round::Down));
            let high = convert(&evaluate(Round::Up));
            // Directed bounds must agree in result, exactness, and whether
            // the magnitude rounded up; this excludes an exact grid point
            // or a rounding boundary between the endpoints.
            if low == high {
                break low;
            }
            working = working.checked_mul(2).unwrap();
            assert!(
                working < 1 << 20,
                "MPFR bounds eventually certify decimal rounding"
            );
        };
        if x.is_subnormal() {
            expected_flags |= Flags::DENORMAL_INPUT;
        }
        let (result, flags) = match function {
            Function::Exp2 => x.exp2_with(*env),
            Function::Log2 => x.log2_with(*env),
        };
        assert_eq!(
            (DecimalValue::from_decoded(result.decode::<2>()), flags),
            (expected, expected_flags),
            "{function:?} {x:?} {env:?}",
        );
    }
}

fn samples(format: DecimalFormat) -> Vec<(u128, Function)> {
    let p = format.precision;
    let ten_p = 10_u128.pow(p);
    let mut samples = Vec::new();
    for n in [0_u128, 1, 3, 17, 32, 63, 127, 257] {
        for negative in [false, true] {
            samples.push((bid(format, negative, n, 0), Function::Exp2));
        }
    }
    for (coefficient, exponent) in [(5, -1), (125, -2), (375, -2), (1_234_567, -4)] {
        for negative in [false, true] {
            samples.push((bid(format, negative, coefficient, exponent), Function::Exp2));
        }
    }
    // Neighbors of the overflow, normal/subnormal, least-subnormal, and
    // halfway-to-zero thresholds, computed independently in MPFR.
    let working = 8 * p + 64;
    let log2_10 = BigFloat::with_val(working, 10).log2();
    let threshold = |exponent: i64| BigFloat::with_val(working, &log2_10 * Integer::from(exponent));
    for value in [
        threshold(i64::from(format.emax) + 1),
        threshold(i64::from(1 - format.emax)),
        threshold(format.exponents().0),
        threshold(format.exponents().0) - 1u32,
    ] {
        let (negative, digits, exponent) =
            value.to_sign_string_exp_round(10, Some(p as usize), Round::Nearest);
        let coefficient = Integer::from_str_radix(&digits, 10)
            .unwrap()
            .to_u128()
            .unwrap();
        let exponent = i64::from(exponent.unwrap()) - i64::from(p);
        for step in -2_i32..=2 {
            let coefficient = coefficient.checked_add_signed(i128::from(step)).unwrap();
            samples.push((bid(format, negative, coefficient, exponent), Function::Exp2));
        }
    }
    for (coefficient, exponent) in [
        (1, 0),
        (2, 0),
        (8, 0),
        (125, -3),
        (5, -1),
        (25, -2),
        (3, 0),
        (5, 0),
        (ten_p / 10 + 1, 1 - i64::from(p)),
        (ten_p - 1, -i64::from(p)),
        (1, format.exponents().0),
        (ten_p / 10 - 1, format.exponents().0),
        (ten_p - 1, format.exponents().1),
    ] {
        samples.push((bid(format, false, coefficient, exponent), Function::Log2));
    }
    // Every cohort of one and one half must give an exact logarithm.
    for zeros in 0..p {
        samples.push((
            bid(format, false, 10_u128.pow(zeros), -i64::from(zeros)),
            Function::Log2,
        ));
        if zeros > 0 {
            samples.push((
                bid(format, false, 5 * 10_u128.pow(zeros - 1), -i64::from(zeros)),
                Function::Log2,
            ));
        }
    }
    samples
}

macro_rules! check_format {
    ($name:ident, $intel:ty, $bid:ty, $dpd:ty, $bits:ty) => {
        #[test]
        fn $name() {
            let format = DecimalFormat::of::<$bid>();
            for (bits, function) in samples(format) {
                let bits = <$bits>::try_from(bits).unwrap();
                check(<$bid>::from_bits(bits), format, function);
                let dpd = <$intel as IntelFormat>::to_dpd(bits);
                check(<$dpd>::from_bits(dpd), format, function);
            }
        }
    };
}

check_format!(decimal32_bid_and_dpd, Bid32, D32Bid, D32Dpd, u32);
check_format!(decimal64_bid_and_dpd, Bid64, D64Bid, D64Dpd, u64);
check_format!(decimal128_bid_and_dpd, Bid128, D128Bid, D128Dpd, u128);
