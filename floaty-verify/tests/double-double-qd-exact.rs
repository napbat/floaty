//! Compares `DoubleDouble<Qd>::mul_add` and `next_up` and `next_down` with
//! oracles on the exact values `hi + lo`.
//!
//! QD has none of these functions, so `Qd` follows IEEE 754 on the exact
//! value. Exact rationals and MPFR evaluate the rules:
//!
//! - `mul_add` rounds the exact `a * b + c` once, by floaty's pair rule in
//!   [`round_pair`]. An exact zero takes the sign of IEEE 754-2019 section
//!   6.3. A NaN, an infinity, or an invalid case follows the MPFR arithmetic
//!   oracle of binary64, which reads the class of each exact value.
//! - `next_up` gives the least value of a canonical pair above the exact
//!   value. The oracle searches the candidates with the high half of the
//!   value and with the next binary64 value, and keeps the least canonical
//!   one. `next_down` mirrors `next_up`.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::{Decoded, DoubleDouble, Env, F64, Flags, Qd, Rounding};
use floaty_verify::arithmetic::{self, Operation};
use floaty_verify::double_double::reference::Pair;
use floaty_verify::double_double::{
    BINARY64, Exact, Rounded, behaviors, exact, infinity, input, operand, pair, round_pair,
};
use floaty_verify::mpfr::{self, Operand, Value};
use floaty_verify::random::SplitMix64;
use rug::Rational;

fn value(pair: Pair) -> DoubleDouble<Qd> {
    DoubleDouble::from_parts(F64::from_bits(pair.hi), F64::from_bits(pair.lo))
}

/// Returns floaty's pair and flags as the oracle writes them.
fn rounded((value, flags): (DoubleDouble<Qd>, Flags)) -> Rounded {
    Rounded {
        hi: Value::from_decoded(value.hi().decode::<1>()),
        lo: Value::from_decoded(value.lo().decode::<1>()),
        flags,
    }
}

/// Returns the finite exact value as a rational, or `None` for a NaN or an
/// infinity.
fn rational(value: &Exact) -> Option<Rational> {
    match value {
        Exact::Number(number) => Some(number.to_rational().expect("the value is finite")),
        Exact::Zero { .. } => Some(Rational::new()),
        Exact::Nan(_) | Exact::Infinity { .. } => None,
    }
}

/// Returns a binary64 operand with the class and sign of an exact value: the
/// NaN half itself, an infinity, a zero, or ±1 for a finite value.
fn class_operand(value: &Exact) -> Operand<1> {
    let decoded = match value {
        Exact::Nan(bits) => Decoded::Nan {
            negative: bits >> 63 == 1,
            signaling: bits & (1 << 51) == 0,
            payload: [bits & ((1 << 51) - 1)],
        },
        Exact::Infinity { negative } => Decoded::Infinity {
            negative: *negative,
        },
        Exact::Zero { negative } => Decoded::Zero {
            negative: *negative,
            exponent: 0,
        },
        Exact::Number(number) => Decoded::Finite {
            negative: number.is_sign_negative(),
            exponent: 0,
            significand: [1],
        },
    };
    Operand {
        decoded,
        subnormal: false,
    }
}

/// The positive zero of a low half without a rest.
const NO_REST: Value = Value::Zero { negative: false };

/// Returns the expected result of `mul_add`, with the payload of a NaN high
/// half.
fn expected_mul_add(a: &Exact, b: &Exact, c: &Exact, env: &Env) -> (Rounded, u64) {
    if let (Some(x), Some(y), Some(z)) = (rational(a), rational(b), rational(c)) {
        let sum = x * y + z;
        let product_negative = a.negative() != b.negative();
        let negative = if sum == 0 {
            // IEEE 754-2019 section 6.3: an exact zero sum of operands of
            // one sign has that sign, and otherwise is +0, or -0 toward
            // negative.
            if c.negative() == product_negative {
                product_negative
            } else {
                env.rounding == Rounding::TowardNegative
            }
        } else {
            sum < 0
        };
        return (round_pair(&sum, negative, env), 0);
    }
    let operands = [a, b, c].map(class_operand);
    let expected = arithmetic::compute(
        Operation::MulAdd,
        &operands,
        &BINARY64,
        &env.with_saturate(false),
    );
    let rounded = match expected.value {
        Value::Infinity { negative } => Rounded {
            flags: expected.flags,
            ..infinity(negative, env)
        },
        value => Rounded {
            hi: value,
            lo: NO_REST,
            flags: expected.flags,
        },
    };
    (rounded, expected.payload[0])
}

/// Returns a pair whose value is near `-(a * b)`, so that the sum cancels.
fn cancelling(a: Pair, b: Pair) -> Pair {
    let product = value(a) * value(b);
    let negated = -product;
    Pair::new(negated.hi().to_bits(), negated.lo().to_bits())
}

#[test]
fn qd_mul_add_rounds_the_exact_value_once() {
    let mut random = SplitMix64::new(0x0DD5_FADD);
    let mut failures = Vec::new();
    let mut count = 0;
    for index in 0..6_000 {
        let first = operand(&mut random);
        let second = operand(&mut random);
        let third = match index % 4 {
            0 => cancelling(first, second),
            1 => pair(&mut random),
            _ => operand(&mut random),
        };
        let (left, right, addend) = (value(first), value(second), value(third));
        let exacts = [first, second, third].map(exact);
        for env in behaviors() {
            let (want, payload) = expected_mul_add(&exacts[0], &exacts[1], &exacts[2], &env);
            let (ours, flags) = left.mul_add_with(right, addend, env);
            count += 1;
            let ours_payload = if ours.hi().is_nan() {
                ours.hi().to_bits() & ((1 << 51) - 1)
            } else {
                0
            };
            if (rounded((ours, flags)), ours_payload) != (want.clone(), payload)
                && failures.len() < 40
            {
                failures.push(format!(
                    "{first:?} * {second:?} + {third:?} {env:?}: floaty {:?} payload \
                     {ours_payload:#x}, expected {want:?} payload {payload:#x}",
                    rounded((ours, flags))
                ));
            }
        }
        let default = left.mul_add(right, addend);
        let (with, _) = left.mul_add_with(right, addend, DoubleDouble::<Qd>::ENV);
        assert_eq!(
            (default.hi().to_bits(), default.lo().to_bits()),
            (with.hi().to_bits(), with.lo().to_bits()),
            "{first:?} {second:?} {third:?} default mode"
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
    assert_eq!(count, 6_000 * 8);
}

/// Rounds a rational to binary64 in `rounding`, or returns `None` when it
/// overflows.
fn to_binary64(value: &Rational, rounding: Rounding) -> Option<f64> {
    if *value == 0 {
        return Some(0.0);
    }
    let env = Env::IEEE.with_rounding(rounding);
    match mpfr::round(&input(value), &BINARY64, &env).0 {
        Value::Finite(number) => Some(number.to_f64()),
        Value::Zero { negative } => Some(if negative { -0.0 } else { 0.0 }),
        Value::Infinity { .. } | Value::Nan { .. } => None,
    }
}

/// Returns `true` when a canonical pair holds `value`: the high half rounds
/// to nearest even without an overflow, and the rest is a binary64 value.
fn canonical(value: &Rational) -> bool {
    let Some(high) = to_binary64(value, Rounding::TiesToEven) else {
        return false;
    };
    let rest = value - Rational::from_f64(high).expect("finite");
    to_binary64(&rest, Rounding::TiesToEven)
        .is_some_and(|low| Rational::from_f64(low).expect("finite") == rest)
}

/// Returns the least value of a canonical pair above the finite value
/// `value`, or `None` above the largest finite pair.
fn successor(value: &Rational) -> Option<Rational> {
    let exact_f64 = |x: f64| Rational::from_f64(x).expect("the value is finite");
    let Some(high) = to_binary64(value, Rounding::TiesToEven) else {
        // A non-canonical pair at or above the overflow threshold. The value
        // above a negative one is the largest negative pair.
        return (*value < 0).then(|| largest(true));
    };
    let rest = value - exact_f64(high);
    let rest = rest.to_f64();
    let mut candidates = vec![exact_f64(high) + exact_f64(rest.next_up())];
    let above = high.next_up();
    if above.is_finite() {
        // With the next high half: the least binary64 value above the rest
        // from that half, and the half itself.
        let low = value - exact_f64(above);
        let up = to_binary64(&low, Rounding::TowardPositive).expect("the rest is small");
        let up = if exact_f64(up) == low {
            up.next_up()
        } else {
            up
        };
        candidates.push(exact_f64(above) + exact_f64(up));
        candidates.push(exact_f64(above));
    }
    candidates
        .into_iter()
        .filter(|candidate| candidate > value && canonical(candidate))
        .min()
}

/// Returns the value of the largest finite pair of the sign `negative`.
fn largest(negative: bool) -> Rational {
    let max = Rational::from_f64(f64::MAX).expect("finite");
    let low = Rational::from_f64(2.0_f64.powi(970) - 2.0_f64.powi(917)).expect("finite");
    let value = max + low;
    if negative { -value } else { value }
}

/// Returns the canonical pair of a value that a canonical pair holds. The
/// step does not round, so it reports no flags, not even `TINY` for a
/// subnormal low half. A zero is the negative zero, which only the value
/// above the negative subnormal quantum gives.
fn exact_pair(value: &Rational) -> Rounded {
    let negative = *value <= 0;
    let pair = round_pair(value, negative, &Env::IEEE);
    assert!(
        !pair.flags.contains(Flags::INEXACT),
        "a canonical pair holds {value}"
    );
    Rounded {
        flags: Flags::NONE,
        ..pair
    }
}

/// Returns the negation of an expected pair. A low half without a rest stays
/// `+0`.
fn negate(rounded: Rounded) -> Rounded {
    let flip = |value: Value| match value {
        Value::Zero { negative } => Value::Zero {
            negative: !negative,
        },
        Value::Finite(number) => Value::Finite(-number),
        Value::Infinity { negative } => Value::Infinity {
            negative: !negative,
        },
        Value::Nan { negative } => Value::Nan { negative },
    };
    let lo = match rounded.lo {
        Value::Zero { .. } => NO_REST,
        other => flip(other),
    };
    Rounded {
        hi: flip(rounded.hi),
        lo,
        flags: rounded.flags,
    }
}

/// Returns the expected `next_up` of an exact value that is not a NaN.
fn expected_next_up(value: &Exact, env: &Env) -> Rounded {
    match value {
        Exact::Infinity { negative: false } => infinity(false, env),
        Exact::Infinity { negative: true } => exact_pair(&largest(true)),
        Exact::Zero { .. } => exact_pair(&Rational::from_f64(f64::from_bits(1)).expect("finite")),
        Exact::Number(_) => {
            let rational = rational(value).expect("the value is finite");
            successor(&rational).map_or_else(|| infinity(false, env), |next| exact_pair(&next))
        }
        Exact::Nan(_) => unreachable!("the caller handles every NaN"),
    }
}

/// Returns the exact value with the opposite sign.
fn negated(value: &Exact) -> Exact {
    match value {
        Exact::Nan(bits) => Exact::Nan(*bits),
        Exact::Infinity { negative } => Exact::Infinity {
            negative: !negative,
        },
        Exact::Zero { negative } => Exact::Zero {
            negative: !negative,
        },
        Exact::Number(number) => Exact::Number(-number.clone()),
    }
}

/// Returns the expected result of a step of a NaN: its conversion to
/// binary64, with a `+0` low half.
fn expected_nan(bits: u64, env: &Env) -> Rounded {
    let (hi, flags) = mpfr::convert(&class_operand(&Exact::Nan(bits)), &BINARY64, env);
    Rounded {
        hi,
        lo: NO_REST,
        flags,
    }
}

/// Returns pairs around the ties and the binade boundaries of the high half,
/// and at the edges of the range.
fn neighbor_edges() -> Vec<Pair> {
    let highs = [
        0.0,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        1.0,
        1.0 + f64::EPSILON,
        2.0,
        3.0,
        1e300,
        f64::MAX,
    ];
    let mut pairs = Vec::new();
    for high in highs.into_iter().flat_map(|high| [high, -high]) {
        let up = high.next_up() - high;
        let down = high - high.next_down();
        let lows = [
            0.0,
            up / 2.0,
            down / 2.0,
            (up / 2.0).next_down(),
            (up / 2.0).next_up(),
            (down / 2.0).next_down(),
            f64::from_bits(1),
            2.0_f64.powi(970) - 2.0_f64.powi(917),
        ];
        for low in lows.into_iter().flat_map(|low| [low, -low]) {
            if low.is_finite() {
                pairs.push(Pair::new(high.to_bits(), low.to_bits()));
            }
        }
    }
    pairs
}

#[test]
fn qd_next_up_and_next_down_give_the_next_canonical_pair() {
    let mut random = SplitMix64::new(0x0DD5_0E27);
    let mut pairs = neighbor_edges();
    pairs.extend((0..10_000).map(|_| operand(&mut random)));
    pairs.extend((0..10_000).map(|_| pair(&mut random)));
    let mut failures = Vec::new();
    for &pair in &pairs {
        let (x, value_exact) = (value(pair), exact(pair));
        for env in behaviors() {
            let (up, down) = match value_exact {
                Exact::Nan(bits) => (expected_nan(bits, &env), expected_nan(bits, &env)),
                _ => (
                    expected_next_up(&value_exact, &env),
                    negate(expected_next_up(&negated(&value_exact), &env)),
                ),
            };
            let checks = [
                ("next_up", rounded(x.next_up_with(env)), up),
                ("next_down", rounded(x.next_down_with(env)), down),
            ];
            for (name, ours, want) in checks {
                if ours != want && failures.len() < 40 {
                    failures.push(format!(
                        "{name} {pair:?} {env:?}: floaty {ours:?}, expected {want:?}"
                    ));
                }
            }
        }
        let bits = |value: DoubleDouble<Qd>| (value.hi().to_bits(), value.lo().to_bits());
        let env = DoubleDouble::<Qd>::ENV;
        assert_eq!(bits(x.next_up()), bits(x.next_up_with(env).0), "{pair:?}");
        assert_eq!(
            bits(x.next_down()),
            bits(x.next_down_with(env).0),
            "{pair:?}"
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
