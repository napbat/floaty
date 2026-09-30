//! Compares the classification, the canonical test, the total order, and the
//! minimum and maximum operations of double-double pairs with their rules,
//! evaluated on exact values with MPFR.
//!
//! The rules are those of the documentation of `DoubleDouble`. The class and
//! the order follow the exact value `hi + lo`. The canonical test follows
//! `iscanonicall` of glibc 2.43: a nonzero low half is below half an ulp of
//! a finite high half, or exactly half an ulp with an even high half.
//! `double-double-functions.rs` also runs `iscanonicall` itself under QEMU.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::cmp::Ordering;

use floaty::{Class, DoubleDouble, Env, F64, Flags, Gcc, Qd};
use floaty_verify::double_double::{Exact, behaviors, exact, is_infinite, is_nan, pair};
use floaty_verify::random::SplitMix64;
use rug::{Float as BigFloat, Integer, Rational};

fn value(hi: u64, lo: u64) -> DoubleDouble<Gcc> {
    DoubleDouble::from_parts(F64::from_bits(hi), F64::from_bits(lo))
}

/// Returns the expected class of an exact value. A finite value below
/// `2^-969` in magnitude is subnormal.
fn expected_class(value: &Exact) -> Class {
    match value {
        Exact::Nan(bits) if bits & (1 << 51) == 0 => Class::SignalingNan,
        Exact::Nan(_) => Class::QuietNan,
        Exact::Infinity { .. } => Class::Infinite,
        Exact::Zero { .. } => Class::Zero,
        Exact::Number(number) => {
            let smallest_normal = BigFloat::with_val(2, 1) >> 969;
            if number.clone().abs() < smallest_normal {
                Class::Subnormal
            } else {
                Class::Normal
            }
        }
    }
}

#[test]
fn pairs_classify_by_their_exact_values() {
    let mut random = SplitMix64::new(0xDD_C1A5);
    // The magnitudes around 2^-969, with low halves of both signs.
    let edges = [
        (0x0360_0000_0000_0000_u64, 0),
        (0x0360_0000_0000_0000, 0x8000_0000_0000_0001),
        (0x0360_0000_0000_0001, 0x8000_0000_0000_0001),
        (0x035F_FFFF_FFFF_FFFF, 0x0000_0000_0000_0001),
        (0x3FF0_0000_0000_0000, 0xBFF0_0000_0000_0000),
    ];
    let pairs = (0..50_000).map(|_| pair(&mut random)).chain(edges);
    for (hi, lo) in pairs {
        let (ours, exact_value) = (value(hi, lo), exact(hi, lo));
        let class = expected_class(&exact_value);
        let context = format!("{hi:#x} {lo:#x}");
        assert_eq!(ours.classify(), class, "{context}");
        assert_eq!(ours.is_sign_negative(), exact_value.negative(), "{context}");
        assert_eq!(
            ours.is_sign_positive(),
            !exact_value.negative(),
            "{context}"
        );
        let predicates = [
            (
                ours.is_nan(),
                matches!(class, Class::QuietNan | Class::SignalingNan),
            ),
            (ours.is_signaling_nan(), class == Class::SignalingNan),
            (ours.is_infinite(), class == Class::Infinite),
            (
                ours.is_finite(),
                matches!(class, Class::Zero | Class::Subnormal | Class::Normal),
            ),
            (ours.is_zero(), class == Class::Zero),
            (ours.is_subnormal(), class == Class::Subnormal),
            (ours.is_normal(), class == Class::Normal),
        ];
        for (index, (ours, expected)) in predicates.into_iter().enumerate() {
            assert_eq!(ours, expected, "{context}: predicate {index}");
        }
    }
}

/// Returns whether a pair is canonical by the rule of glibc's `iscanonicall`,
/// on exact values.
fn expected_canonical(hi: u64, lo: u64) -> bool {
    let magnitude = |bits: u64| bits & !(1 << 63);
    if magnitude(lo) == 0 || is_nan(hi) {
        return true;
    }
    if is_infinite(hi) {
        return false;
    }
    let field = magnitude(hi) >> 52;
    if field == 0 || is_nan(lo) || is_infinite(lo) {
        // A nonzero low half is at least the ulp of a zero or subnormal high
        // half, and a special low half is above every ulp.
        return false;
    }
    let exponent = i32::try_from(field).expect("11 bits fit an i32") - 1075;
    let half_ulp = power_of_two(exponent - 1);
    let low = BigFloat::with_val(53, f64::from_bits(magnitude(lo)))
        .to_rational()
        .expect("a finite low half");
    match low.cmp(&half_ulp) {
        Ordering::Less => true,
        Ordering::Equal => hi & 1 == 0,
        Ordering::Greater => false,
    }
}

/// Returns `2^exponent` as a rational.
fn power_of_two(exponent: i32) -> Rational {
    let power = Integer::from(1) << exponent.unsigned_abs();
    if exponent >= 0 {
        Rational::from(power)
    } else {
        Rational::from((1, power))
    }
}

#[test]
fn canonical_pairs_follow_the_rule_of_glibc() {
    let mut random = SplitMix64::new(0xDD_CA70);
    let mut pairs: Vec<(u64, u64)> = (0..100_000).map(|_| pair(&mut random)).collect();
    // Low halves at half an ulp of an even and of an odd high half, and just
    // below and above it, with subnormal and normal low halves.
    for hi in [
        0x3FF0_0000_0000_0000_u64,
        0x3FF0_0000_0000_0001,
        0x0370_0000_0000_0000,
        0x0350_0000_0000_0001,
        0x7FEF_FFFF_FFFF_FFFF,
    ] {
        let half = (((hi >> 52) - 53) << 52) & 0x7FF0_0000_0000_0000;
        let half = if hi >> 52 <= 53 {
            1 << ((hi >> 52) - 2)
        } else {
            half
        };
        for lo in [half, half - 1, half + 1, half | 1 << 63] {
            pairs.push((hi, lo));
        }
    }
    for (hi, lo) in pairs {
        assert_eq!(
            value(hi, lo).is_canonical(),
            expected_canonical(hi, lo),
            "{hi:#x} {lo:#x}"
        );
    }
}

/// Returns the place of an exact value in the total order of the
/// magnitudes, with the magnitude or the payload that orders values of one
/// place.
fn rank(value: &Exact) -> (u8, Rational) {
    match value {
        Exact::Zero { .. } => (0, Rational::new()),
        Exact::Number(number) => (
            1,
            number
                .clone()
                .abs()
                .to_rational()
                .expect("the value is finite"),
        ),
        Exact::Infinity { .. } => (2, Rational::new()),
        Exact::Nan(bits) => {
            let payload = Rational::from(bits & ((1 << 51) - 1));
            (if bits & (1 << 51) == 0 { 3 } else { 4 }, payload)
        }
    }
}

/// Returns the expected total order of two pairs: the IEEE 754 `totalOrder`
/// of their exact values, then the binary64 total order of the high halves
/// and of the low halves.
fn expected_order(a: (u64, u64), b: (u64, u64)) -> Ordering {
    let (first, second) = (exact(a.0, a.1), exact(b.0, b.1));
    let by_value = match (first.negative(), second.negative()) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (negative, _) => {
            let order = rank(&first).cmp(&rank(&second));
            if negative { order.reverse() } else { order }
        }
    };
    let halves = |x: u64, y: u64| F64::from_bits(x).total_cmp(F64::from_bits(y));
    by_value
        .then_with(|| halves(a.0, b.0))
        .then_with(|| halves(a.1, b.1))
}

/// Returns a pair and a second pair: random, or with the high half of the
/// first, or with the value of the first in another pair.
fn two_pairs(random: &mut SplitMix64) -> ((u64, u64), (u64, u64)) {
    let a = pair(random);
    let b = match random.next_u64() % 4 {
        0 => (a.0, pair(random).1),
        // (hi, lo) and (hi + lo rounded, rest) often hold one value.
        1 => {
            let sum = F64::from_bits(a.0) + F64::from_bits(a.1);
            let rest = (value(a.0, a.1) - value(sum.to_bits(), 0)).hi();
            (sum.to_bits(), rest.to_bits())
        }
        2 => (a.0 ^ 1 << 63, a.1 ^ 1 << 63),
        _ => pair(random),
    };
    (a, b)
}

#[test]
fn the_total_order_orders_values_then_halves() {
    let mut random = SplitMix64::new(0xDD_707A);
    for _ in 0..100_000 {
        let (a, b) = two_pairs(&mut random);
        let expected = expected_order(a, b);
        let context = format!("{a:x?} {b:x?}");
        assert_eq!(
            value(a.0, a.1).total_cmp(value(b.0, b.1)),
            expected,
            "{context}"
        );
        let qd = |pair: (u64, u64)| {
            DoubleDouble::<Qd>::from_parts(F64::from_bits(pair.0), F64::from_bits(pair.1))
        };
        assert_eq!(qd(b).total_cmp(qd(a)), expected.reverse(), "{context}: Qd");
    }
}

/// The half that holds the NaN of a NaN pair, or a zero for a number.
fn nan_half((hi, lo): (u64, u64)) -> F64 {
    if is_nan(hi) {
        F64::from_bits(hi)
    } else if !is_infinite(hi) && is_nan(lo) {
        F64::from_bits(lo)
    } else {
        F64::from_bits(0)
    }
}

/// Checks one minimum or maximum operation of two pairs against its rule.
macro_rules! check_min_max {
    ($a:expr, $b:expr, $env:expr, $with:ident, $minimum:expr) => {{
        let (a, b, env): ((u64, u64), (u64, u64), Env) = ($a, $b, $env);
        let (ours, flags) = value(a.0, a.1).$with(value(b.0, b.1), env);
        let (first, second) = (exact(a.0, a.1), exact(b.0, b.1));
        let is_nan = |value: &Exact| matches!(value, Exact::Nan(_));
        let expected = if is_nan(&first) || is_nan(&second) {
            let (half, flags) = nan_half(a).$with(nan_half(b), env);
            if half.is_nan() {
                ((half.to_bits(), 0), flags)
            } else if is_nan(&first) {
                (b, flags)
            } else {
                (a, flags)
            }
        } else {
            let smaller_first = expected_order(a, b) != Ordering::Greater;
            let result = if smaller_first == $minimum { a } else { b };
            (result, Flags::NONE)
        };
        let context = format!("{} {a:x?} {b:x?} {env:?}", stringify!($with));
        let ours = ((ours.hi().to_bits(), ours.lo().to_bits()), flags);
        assert_eq!(ours, expected, "{context}");
    }};
}

#[test]
fn minimum_and_maximum_select_by_the_exact_value() {
    let mut random = SplitMix64::new(0xDD_313A);
    for _ in 0..30_000 {
        let (a, b) = two_pairs(&mut random);
        for env in behaviors() {
            check_min_max!(a, b, env, minimum_with, true);
            check_min_max!(a, b, env, maximum_with, false);
            check_min_max!(a, b, env, minimum_number_with, true);
            check_min_max!(a, b, env, maximum_number_with, false);
            check_min_max!(a, b, env, min_num_with, true);
            check_min_max!(a, b, env, max_num_with, false);
        }
        let (x, y) = (value(a.0, a.1), value(b.0, b.1));
        let same = |left: DoubleDouble<Gcc>, right: DoubleDouble<Gcc>| {
            (left.hi().to_bits(), left.lo().to_bits())
                == (right.hi().to_bits(), right.lo().to_bits())
        };
        let default = DoubleDouble::<Gcc>::ENV;
        assert!(
            same(x.minimum(y), x.minimum_with(y, default).0),
            "{a:x?} {b:x?}"
        );
        assert!(
            same(x.maximum_number(y), x.maximum_number_with(y, default).0),
            "{a:x?} {b:x?}"
        );
        assert!(
            same(x.min_num(y), x.min_num_with(y, default).0),
            "{a:x?} {b:x?}"
        );
    }
}
