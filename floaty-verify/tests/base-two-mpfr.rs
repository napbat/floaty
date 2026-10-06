//! Exact powers and base-two rounding boundaries against directed MPFR.
#![cfg(all(target_arch = "x86_64", feature = "alloc"))]

use floaty::{Env, F64, Flags};
use floaty_verify::mpfr::{self, Format, Input, Specials, Value};
use floaty_verify::operations::BEHAVIORS;
use rug::Float as BigFloat;
use rug::float::Round;

#[derive(Clone, Copy, Debug)]
enum Function {
    Exp2,
    Log2,
}

/// MPFR bounds either agree exactly, or certify a common truncation with
/// three guard bits. Never round the center of an undecided interval.
fn reference(function: Function, x: f64, precision: u32) -> Input {
    let x = BigFloat::with_val(53, x);
    let mut working = precision + 64;
    loop {
        let evaluate = |round| match function {
            Function::Exp2 => BigFloat::with_val_round(working, x.exp2_ref(), round).0,
            Function::Log2 => BigFloat::with_val_round(working, x.log2_ref(), round).0,
        };
        let (low, high) = (evaluate(Round::Down), evaluate(Round::Up));
        if low == high {
            let negative = low.is_sign_negative();
            let (significand, exponent) = low.abs().to_integer_exp().expect("finite MPFR value");
            return Input {
                negative,
                exponent,
                significand,
                sticky: false,
            };
        }
        if low.is_sign_negative() == high.is_sign_negative() && !low.is_zero() && !high.is_zero() {
            let negative = low.is_sign_negative();
            let (small, large) = if negative {
                (high.abs(), low.abs())
            } else {
                (low, high)
            };
            let lowest = small.get_exp().expect("nonzero") - i32::try_from(precision + 3).unwrap();
            let below = (small >> lowest).floor().to_integer().unwrap();
            let above = (large >> lowest).ceil().to_integer().unwrap() - 1u32;
            if below == above {
                return Input {
                    negative,
                    exponent: lowest,
                    significand: below,
                    sticky: true,
                };
            }
        }
        working = working.checked_mul(2).unwrap();
        assert!(
            working < 1 << 20,
            "MPFR bounds must eventually certify the result"
        );
    }
}

fn check(function: Function, x: f64) {
    let format = Format::of::<F64>(Specials::Ieee);
    let input = reference(function, x, format.precision);
    for env in &BEHAVIORS {
        let value = F64::from_bits(x.to_bits());
        let (result, flags) = match function {
            Function::Exp2 => value.exp2_with(*env),
            Function::Log2 => value.log2_with(*env),
        };
        // The reference above reads the finite operand directly. The
        // operand behavior supplies DENORMAL_INPUT and the DAZ replacement.
        if x.is_subnormal() && env.denormals_are_zero {
            continue;
        }
        let (expected, mut expected_flags) = mpfr::round(&input, &format, env);
        if x.is_subnormal() {
            expected_flags |= Flags::DENORMAL_INPUT;
        }
        assert_eq!(
            (Value::from_decoded(result.decode::<8>()), flags),
            (expected, expected_flags),
            "{function:?} {x:?} {env:?}",
        );
    }
}

#[test]
fn binary64_exact_powers_and_neighbors() {
    for exponent in [
        -1075.0_f64,
        -1074.0,
        -1022.0,
        -1.0,
        0.0,
        1.0,
        1023.0,
        1024.0,
    ] {
        check(Function::Exp2, exponent);
        for bits in [
            exponent.to_bits().wrapping_sub(1),
            exponent.to_bits().wrapping_add(1),
        ] {
            let neighbor = f64::from_bits(bits);
            if neighbor.is_finite() {
                check(Function::Exp2, neighbor);
            }
        }
    }
    for bits in [
        1,
        2,
        3,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x3fef_ffff_ffff_ffff,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0001,
        0x3fff_ffff_ffff_ffff,
        0x4000_0000_0000_0000,
        0x4000_0000_0000_0001,
        0x7fef_ffff_ffff_ffff,
    ] {
        check(Function::Log2, f64::from_bits(bits));
    }
}

#[test]
fn binary64_nonintegral_arguments() {
    for x in [-37.25, -2.75, -0.5, 0.5, 1.25, 12.75, 511.5] {
        check(Function::Exp2, x);
    }
    for x in [0.3, 0.75, 1.1, 3.0, 5.0, 17.0, 1023.0] {
        check(Function::Log2, x);
    }
}

#[test]
fn binary_domain_specials_and_nan_payloads() {
    for zero in [0_u64, 1 << 63] {
        let x = F64::from_bits(zero);
        let (exponential, flags) = x.exp2_with(Env::IEEE);
        assert_eq!(
            (exponential.to_bits(), flags),
            (1.0_f64.to_bits(), Flags::NONE)
        );
        let (logarithm, flags) = x.log2_with(Env::IEEE);
        assert_eq!(
            (logarithm.to_bits(), flags),
            (f64::NEG_INFINITY.to_bits(), Flags::DIVIDE_BY_ZERO)
        );
    }
    for x in [f64::NEG_INFINITY, -1.0] {
        let (value, flags) = F64::from_bits(x.to_bits()).log2_with(Env::IEEE);
        assert!(value.is_nan());
        assert_eq!(flags, Flags::INVALID);
    }
    let (value, flags) = F64::from_bits(f64::NEG_INFINITY.to_bits()).exp2_with(Env::IEEE);
    assert_eq!((value.to_bits(), flags), (0, Flags::NONE));
    for bits in [0x7ff8_0000_0000_0321_u64, 0xfff8_0000_0000_0321] {
        let x = F64::from_bits(bits);
        for (value, flags) in [x.exp2_with(Env::IEEE), x.log2_with(Env::IEEE)] {
            assert_eq!((value.to_bits(), flags), (bits, Flags::NONE));
        }
    }
    let x = F64::from_bits(0x7ff0_0000_0000_0321);
    for (value, flags) in [x.exp2_with(Env::IEEE), x.log2_with(Env::IEEE)] {
        assert_eq!(
            (value.to_bits(), flags),
            (0x7ff8_0000_0000_0321, Flags::INVALID)
        );
    }
}
