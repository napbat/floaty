//! Directed MPFR checks of quadrant reduction, large arguments, and tiny results.
#![cfg(all(target_arch = "x86_64", feature = "alloc"))]

use floaty::{D64Bid, D128Bid, Env, F64, Flags};
use floaty_verify::mpfr::decimal::{self, DecimalFormat, DecimalValue, TO_DECIMAL_BEHAVIORS};
use floaty_verify::mpfr::{self, Format, Input, Specials};
use floaty_verify::operations::{BEHAVIORS, Outcome, outcome};
use rug::float::Round;
use rug::{Float as BigFloat, Integer, Rational};

fn bounds(x: &Rational, bits: u32, cosine: bool) -> (BigFloat, BigFloat) {
    let low = BigFloat::with_val_round(bits, x, Round::Down).0;
    let high = BigFloat::with_val_round(bits, x, Round::Up).0;
    let error = BigFloat::with_val_round(bits, &high - &low, Round::Up).0;
    // Both functions are globally 1-Lipschitz. Unlike an endpoint-only
    // oracle this remains valid across extrema and at arbitrary quadrants.
    let center = BigFloat::with_val(bits, x);
    let (below, above) = if cosine {
        (
            BigFloat::with_val_round(bits, center.cos_ref(), Round::Down).0,
            BigFloat::with_val_round(bits, center.cos_ref(), Round::Up).0,
        )
    } else {
        (
            BigFloat::with_val_round(bits, center.sin_ref(), Round::Down).0,
            BigFloat::with_val_round(bits, center.sin_ref(), Round::Up).0,
        )
    };
    (
        BigFloat::with_val_round(bits, below - &error, Round::Down).0,
        BigFloat::with_val_round(bits, above + &error, Round::Up).0,
    )
}

fn binary_expected(x: &Rational, cosine: bool, env: &Env) -> (Outcome, Flags) {
    let format = Format::of::<F64>(Specials::Ieee);
    let mut bits = 128;
    loop {
        let (low, high) = bounds(x, bits, cosine);
        if !low.is_zero() && low.is_sign_negative() == high.is_sign_negative() {
            let negative = low.is_sign_negative();
            let (small, large) = if negative { (-high, -low) } else { (low, high) };
            let lowest =
                small.get_exp().unwrap() - 1 - (i32::try_from(format.precision).unwrap() + 2);
            let below = (small >> lowest).floor().to_integer().unwrap();
            let above = (large >> lowest).floor().to_integer().unwrap();
            if below == above {
                let input = Input {
                    negative,
                    exponent: lowest,
                    significand: below,
                    sticky: true,
                };
                let (value, flags) = mpfr::round(&input, &format, env);
                let value = match value {
                    mpfr::Value::Zero { negative } => Outcome::Zero { negative },
                    mpfr::Value::Finite(value) => Outcome::Finite(value),
                    _ => unreachable!("sine and cosine have finite results"),
                };
                return (value, flags);
            }
        }
        bits *= 2;
        assert!(bits <= 1 << 20, "MPFR certifies the trigonometric result");
    }
}

#[test]
fn binary_quadrants_full_range_and_underflow() {
    let mut encodings = vec![
        1,
        2,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x3c00_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x4000_0000_0000_0000,
        0x4340_0000_0000_0000,
        0x5fe0_1234_5678_9abc,
        0x7fe0_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
    ];
    // Adjacent representable arguments to half-pi multiples stress both
    // quadrant boundaries and cancellation in the reduced argument.
    for multiple in [1_u64, 2, 3, 4, 17, 1 << 20, 1 << 50] {
        let pi = BigFloat::with_val(256, rug::float::Constant::Pi);
        let center = (pi * multiple / 2_u32).to_f64().to_bits();
        encodings.extend([center - 1, center, center + 1]);
    }
    for encoding in encodings {
        for sign in [0, 1_u64 << 63] {
            let x = F64::from_bits(encoding | sign);
            let Outcome::Finite(exact) = outcome(x) else {
                panic!("finite test argument")
            };
            let exact = exact.to_rational().unwrap();
            for env in &BEHAVIORS {
                for cosine in [false, true] {
                    let (result, flags) = if cosine {
                        x.cos_with(*env)
                    } else {
                        x.sin_with(*env)
                    };
                    let mut expected = if x.is_subnormal() && env.denormals_are_zero {
                        let value = if cosine {
                            F64::from_bits(0x3ff0_0000_0000_0000)
                        } else {
                            F64::from_bits(sign)
                        };
                        (outcome(value), Flags::NONE)
                    } else {
                        binary_expected(&exact, cosine, env)
                    };
                    if x.is_subnormal() {
                        expected.1 |= Flags::DENORMAL_INPUT;
                    }
                    assert_eq!(
                        (outcome(result), flags),
                        expected,
                        "x={x:?}, cos={cosine}, env={env:?}"
                    );
                }
            }
        }
    }
}

fn decimal_expected(
    x: &Rational,
    cosine: bool,
    format: DecimalFormat,
    env: &Env,
) -> (DecimalValue, Flags) {
    let mut bits = 256;
    loop {
        let (low, high) = bounds(x, bits, cosine);
        let below = decimal::to_decimal(&low, format, env);
        let above = decimal::to_decimal(&high, format, env);
        if below == above {
            return below;
        }
        bits *= 2;
        assert!(bits <= 1 << 20, "MPFR certifies the decimal result");
    }
}

#[test]
fn decimal_large_exact_arguments() {
    // BID finite numbers coefficient * 10^exponent; these powers require
    // thousands of bits of pi, not a binary64-narrowed input.
    for exponent in [0_u32, 1, 20, 100, 300] {
        let bits = (u64::from(exponent) + 398) << 53 | 1;
        let x = D64Bid::from_bits(bits);
        let rational = Rational::from(Integer::from(Integer::u_pow_u(10, exponent)));
        let format = DecimalFormat::of::<D64Bid>();
        for env in &TO_DECIMAL_BEHAVIORS {
            for cosine in [false, true] {
                let (result, flags) = if cosine {
                    x.cos_with(*env)
                } else {
                    x.sin_with(*env)
                };
                assert_eq!(
                    (DecimalValue::from_decoded(result.decode()), flags),
                    decimal_expected(&rational, cosine, format, env)
                );
            }
        }
    }
    for exponent in [1000_u32, 6000] {
        let x = D128Bid::from_bits((u128::from(exponent) + 6176) << 113 | 1);
        let rational = Rational::from(Integer::from(Integer::u_pow_u(10, exponent)));
        let format = DecimalFormat::of::<D128Bid>();
        for env in &TO_DECIMAL_BEHAVIORS {
            for cosine in [false, true] {
                let (result, flags) = if cosine {
                    x.cos_with(*env)
                } else {
                    x.sin_with(*env)
                };
                assert_eq!(
                    (DecimalValue::from_decoded(result.decode()), flags),
                    decimal_expected(&rational, cosine, format, env)
                );
            }
        }
    }
}

#[test]
fn zeros_infinities_and_signaling_nans() {
    for sign in [0, 1_u64 << 63] {
        let zero = F64::from_bits(sign);
        let (sine, flags) = zero.sin_with(Env::IEEE);
        assert_eq!(sine.to_bits(), sign);
        assert_eq!(flags, Flags::NONE);
        assert_eq!(
            zero.cos_with(Env::IEEE),
            (F64::from_bits(0x3ff0_0000_0000_0000), Flags::NONE)
        );
        for bits in [0x7ff0_0000_0000_0000, 0x7ff0_0000_0000_0001] {
            let value = F64::from_bits(bits | sign);
            for cosine in [false, true] {
                let (result, flags) = if cosine {
                    value.cos_with(Env::IEEE)
                } else {
                    value.sin_with(Env::IEEE)
                };
                assert!(result.is_nan());
                assert!(flags.contains(Flags::INVALID));
            }
        }
        let quiet = F64::from_bits(0x7ff8_0000_0000_1234 | sign);
        for cosine in [false, true] {
            let (result, flags) = if cosine {
                quiet.cos_with(Env::IEEE)
            } else {
                quiet.sin_with(Env::IEEE)
            };
            assert_eq!(result.to_bits(), quiet.to_bits());
            assert_eq!(flags, Flags::NONE);
        }
    }
}
