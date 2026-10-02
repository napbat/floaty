//! Compares the operations that round an exact value to a double-double pair
//! with floaty's rule, evaluated with exact rationals and MPFR in
//! [`floaty_verify::double_double::round_pair`]: the conversions into a pair
//! from binary, decimal, and double-double values and from integers,
//! `scale_b`, `log_b`, rounding to an integral value, and the conversion to
//! integers. It also checks `copy_sign`, which reads the sign of the exact
//! value.
//!
//! No reference library rounds to a pair by one documented rule, so the
//! oracle evaluates the rule of `DoubleDouble`. Each test runs both
//! algorithms, because the rule does not depend on the algorithm.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::{
    Class, D64Bid, D128Bid, D128Dpd, Decoded, DoubleDouble, Env, F32, F64, F80, F128, F256, Flags,
    Gcc, Int, Qd, Rounding, ToInt, UInt,
};
use floaty_verify::double_double::reference::Pair;
use floaty_verify::double_double::{
    BINARY64, Exact, Rounded, behaviors, exact, infinity, pair, rational, round_pair,
};
use floaty_verify::mpfr::{self, Operand, Value};
use floaty_verify::random::SplitMix64;
use rug::ops::Pow;
use rug::{Integer, Rational};

/// The top exponents of the random values: around the subnormal range of
/// binary64 and of the low half, around 1, around the precision of a pair,
/// and around the overflow threshold.
const TOPS: [i32; 38] = [
    -1200, -1100, -1080, -1076, -1075, -1074, -1073, -1060, -1030, -1023, -1022, -1021, -1000,
    -970, -969, -968, -900, -500, -106, -53, -1, 0, 1, 52, 53, 105, 106, 500, 900, 969, 970, 1000,
    1020, 1022, 1023, 1024, 1025, 1100,
];

/// Returns a random top exponent: one of [`TOPS`], moved by up to 3.
fn top(random: &mut SplitMix64) -> i32 {
    let index = usize::try_from(random.below(38)).expect("an index fits a usize");
    let offset = i32::try_from(random.below(7)).expect("below 7") - 3;
    TOPS[index] + offset
}

/// Returns floaty's pair and flags as the oracle writes them.
fn rounded<Alg: floaty::Algorithm>((value, flags): (DoubleDouble<Alg>, Flags)) -> Rounded {
    Rounded {
        hi: Value::from_decoded(value.hi().decode::<1>()),
        lo: Value::from_decoded(value.lo().decode::<1>()),
        flags,
    }
}

/// Returns the bits of both halves.
fn bits<Alg: floaty::Algorithm>(value: DoubleDouble<Alg>) -> (u64, u64) {
    (value.hi().to_bits(), value.lo().to_bits())
}

/// Returns the expected rounding of a decoded source of radix `radix`.
fn expected<const N: usize>(decoded: &Decoded<N>, radix: u32, env: &Env) -> Rounded {
    let no_rest = Value::Zero { negative: false };
    match *decoded {
        Decoded::Zero { negative, .. } | Decoded::Finite { negative, .. } => round_pair(
            &rational(decoded, radix).expect("the value is finite"),
            negative,
            env,
        ),
        Decoded::Infinity { negative } => infinity(negative, env),
        Decoded::Nan {
            negative,
            signaling,
            ..
        } => {
            let nan = Operand {
                decoded: Decoded::<1>::Nan {
                    negative,
                    signaling,
                    payload: [0],
                },
                subnormal: false,
            };
            let (hi, flags) = mpfr::convert(&nan, &BINARY64, env);
            Rounded {
                hi,
                lo: no_rest,
                flags,
            }
        }
        Decoded::Unsupported => panic!("the sources have no unsupported encoding"),
    }
}

/// Checks the conversion of one source value to both algorithms.
macro_rules! check_source {
    ($value:expr, $limbs:literal, $radix:literal, $env:expr) => {{
        let value = $value;
        let env = $env;
        let (ours, flags) = value.convert_with::<DoubleDouble<Gcc>>(env);
        let mut want = expected(&value.decode::<$limbs>(), $radix, &env);
        if value.classify() == Class::Subnormal {
            want.flags |= Flags::DENORMAL_INPUT;
        }
        let context = format!("{value:?} {env:?}");
        assert_eq!(rounded((ours, flags)), want, "{context}");
        // The canonical form is that of 53-bit halves.
        if env.precision.is_none() {
            assert!(ours.is_canonical(), "{context}: canonical");
        }
        if value.is_nan() {
            // The high half is the binary64 conversion of the NaN.
            let (nan, _) = value.convert_with::<F64>(env);
            assert_eq!(ours.hi().to_bits(), nan.to_bits(), "{context}: payload");
        }
        let (qd, qd_flags) = value.convert_with::<DoubleDouble<Qd>>(env);
        assert_eq!((bits(qd), qd_flags), (bits(ours), flags), "{context}: Qd");
    }};
}

/// Returns a random binary value of precision `precision` as an exact value:
/// a significand with its top bit set and a random top exponent, or random
/// bits.
fn binary_exact(random: &mut SplitMix64, precision: u32) -> (bool, i32, Integer) {
    let negative = random.coin_flip();
    let mut significand = Integer::from(1) << (precision - 1);
    for word in 0..precision.div_ceil(64) {
        significand |= Integer::from(random.next_u64()) << (word * 64);
    }
    significand.keep_bits_mut(precision);
    significand.set_bit(precision - 1, true);
    // A third of the values end in a run of zeros, so the low half ties.
    if random.below(3) == 0 {
        let zeros = u32::try_from(random.below(u64::from(precision))).expect("fits");
        significand >>= zeros;
        significand <<= zeros;
    }
    let exponent = top(random) - i32::try_from(precision - 1).expect("fits");
    (negative, exponent, significand)
}

/// Returns the limbs of an integer, least significant first.
fn limbs<const N: usize>(value: &Integer) -> [u64; N] {
    let mut limbs = [0; N];
    for (slot, digit) in limbs
        .iter_mut()
        .zip(value.to_digits::<u64>(rug::integer::Order::Lsf))
    {
        *slot = digit;
    }
    limbs
}

/// Checks random values of one binary format.
macro_rules! binary_format {
    ($alias:ty, $limbs:literal, $random:expr, $count:literal) => {{
        let random: &mut SplitMix64 = $random;
        for _ in 0..$count {
            let (negative, exponent, significand) = binary_exact(random, <$alias>::PRECISION);
            let exact = floaty::Exact::<$limbs> {
                negative,
                exponent,
                significand: limbs(&significand),
                sticky: false,
            };
            let (value, _) = <$alias>::round(exact, Env::IEEE);
            for env in behaviors() {
                check_source!(value, $limbs, 2, env);
            }
        }
    }};
}

#[test]
fn binary_values_round_to_pairs() {
    let mut random = SplitMix64::new(0xDD_B1A5);
    binary_format!(F32, 1, &mut random, 500);
    binary_format!(F64, 1, &mut random, 500);
    binary_format!(F80, 2, &mut random, 3_000);
    binary_format!(F128, 2, &mut random, 6_000);
    binary_format!(F256, 4, &mut random, 3_000);
    let specials = [
        0x7FF0_0000_0000_0000_u64,
        0xFFF0_0000_0000_0000,
        0x7FF8_0000_0000_0123,
        0xFFF4_0000_0000_0005,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
    ];
    for bits in specials {
        for env in behaviors() {
            check_source!(F64::from_bits(bits), 1, 2, env);
        }
    }
}

#[test]
fn unsupported_x87_encodings_convert_as_to_binary64() {
    // An unnormal, a pseudo-infinity, and a pseudo-NaN.
    let unsupported = [
        0x3FFF_0000_0000_0000_0001_u128,
        0x7FFF_0000_0000_0000_0000,
        0xFFFF_4000_0000_0000_0001,
    ];
    for bits in unsupported {
        let value = F80::from_bits(bits);
        assert_eq!(value.classify(), Class::Unsupported, "{bits:#x}");
        for env in behaviors() {
            let (ours, flags) = value.convert_with::<DoubleDouble<Gcc>>(env);
            let (nan, nan_flags) = value.convert_with::<F64>(env);
            assert!(
                nan.is_nan() && nan_flags.contains(Flags::INVALID),
                "{bits:#x}"
            );
            assert_eq!(
                (bits_of(ours), flags),
                ((nan.to_bits(), 0), nan_flags),
                "{bits:#x} {env:?}"
            );
        }
    }
}

/// Returns the bits of both halves of a `Gcc` pair.
fn bits_of(value: DoubleDouble<Gcc>) -> (u64, u64) {
    (value.hi().to_bits(), value.lo().to_bits())
}

/// Checks random values of one decimal format.
macro_rules! decimal_format {
    ($alias:ty, $digits:literal, $random:expr, $count:literal, $bits:expr) => {{
        let random: &mut SplitMix64 = $random;
        let bound = Integer::from(Integer::from(10).pow($digits));
        for _ in 0..$count {
            let coefficient = Integer::from(random.next_u128()) % &bound;
            let digits = i32::try_from(coefficient.to_string().len()).expect("fits");
            // 2^top is about 10^(top * 0.30103).
            let scaled = i64::from(top(random)) * 30_103 / 100_000;
            let exponent = i32::try_from(scaled).expect("fits") - digits + 1;
            let exact = floaty::Exact::<2> {
                negative: random.coin_flip(),
                exponent,
                significand: limbs(&coefficient),
                sticky: false,
            };
            let (value, _) = <$alias>::round(exact, Env::IEEE);
            for env in behaviors() {
                check_source!(value, 2, 10, env);
            }
        }
        for _ in 0..$count / 10 {
            let value = <$alias>::from_bits($bits(&mut *random));
            for env in behaviors() {
                check_source!(value, 2, 10, env);
            }
        }
    }};
}

#[test]
fn decimal_values_at_the_limits_round_to_pairs() {
    // Around the largest binary64 value, 1.7976931348623157E308, and around
    // the limits of the sticky bit, 10^-1100 and 10^-1101.
    let limits: [(u128, i32); 10] = [
        (1, 308),
        (17, 307),
        (17_976_931_348_623_157, 292),
        (17_976_931_348_623_158, 292),
        (18, 307),
        (1, 309),
        (1, -1100),
        (1, -1101),
        (5, -324),
        (9_999_999_999_999_999_999_999_999_999_999, -1133),
    ];
    for (coefficient, exponent) in limits {
        for negative in [false, true] {
            let exact = floaty::Exact::<2> {
                negative,
                exponent,
                significand: limbs(&Integer::from(coefficient)),
                sticky: false,
            };
            let (value, _) = D128Bid::round(exact, Env::IEEE);
            for env in behaviors() {
                check_source!(value, 2, 10, env);
            }
        }
    }
}

#[test]
fn decimal_values_round_to_pairs() {
    let mut random = SplitMix64::new(0xDD_DEC1);
    decimal_format!(D64Bid, 16, &mut random, 3_000, SplitMix64::next_u64);
    decimal_format!(D128Bid, 34, &mut random, 3_000, SplitMix64::next_u128);
    decimal_format!(D128Dpd, 34, &mut random, 1_000, SplitMix64::next_u128);
}

#[test]
fn pairs_convert_between_the_algorithms() {
    let mut random = SplitMix64::new(0xDD_A160);
    for _ in 0..30_000 {
        let Pair { hi, lo } = pair(&mut random);
        let value = DoubleDouble::<Gcc>::from_parts(F64::from_bits(hi), F64::from_bits(lo));
        for env in behaviors() {
            let (ours, flags) = value.convert_with::<DoubleDouble<Qd>>(env);
            let want = match exact(Pair::new(hi, lo)) {
                Exact::Number(number) => round_pair(
                    &number.to_rational().expect("the value is finite"),
                    number.is_sign_negative(),
                    &env,
                ),
                Exact::Zero { negative } => round_pair(&Rational::new(), negative, &env),
                special => expected(&special_decoded(&special), 2, &env),
            };
            let context = format!("{hi:#x} {lo:#x} {env:?}");
            assert_eq!(rounded((ours, flags)), want, "{context}");
        }
    }
}

/// Returns a special exact value as a decoded value.
fn special_decoded(value: &Exact) -> Decoded<1> {
    match *value {
        Exact::Nan(bits) => Decoded::Nan {
            negative: bits >> 63 == 1,
            signaling: bits & (1 << 51) == 0,
            payload: [bits & ((1 << 51) - 1)],
        },
        Exact::Infinity { negative } => Decoded::Infinity { negative },
        Exact::Zero { negative } => Decoded::Zero {
            negative,
            exponent: 0,
        },
        Exact::Number(_) => unreachable!("the caller passes a special value"),
    }
}

/// Checks the conversion of one integer.
macro_rules! check_integer {
    ($value:expr) => {{
        let value = $value;
        let exact = Integer::from(value);
        for env in behaviors() {
            let want = round_pair(&Rational::from(exact.clone()), false, &env);
            let (ours, flags) = DoubleDouble::<Gcc>::from_int_with(value, env);
            assert_eq!(rounded((ours, flags)), want, "{value} {env:?}");
            let (qd, qd_flags) = DoubleDouble::<Qd>::from_int_with(value, env);
            assert_eq!(
                (bits(qd), qd_flags),
                (bits(ours), flags),
                "{value} {env:?}: Qd"
            );
        }
    }};
}

#[test]
fn integers_round_to_pairs() {
    let mut random = SplitMix64::new(0xDD_1A7E);
    for value in [0, 1, -1, i64::MAX, i64::MIN, (1 << 53) + 1, -(1 << 60) - 3] {
        check_integer!(value);
    }
    for value in [u128::MAX, 1 << 127 | 1, (1 << 106) - 1, (1 << 107) - 1] {
        check_integer!(value);
    }
    for _ in 0..2_000 {
        let shift = u32::try_from(random.below(128)).expect("below 128");
        check_integer!(random.next_u128() >> shift);
        let signed = i128::from_ne_bytes(random.next_u128().to_ne_bytes());
        check_integer!(signed >> shift);
    }
    // Integers of 512 bits, whose values need both halves and round.
    for _ in 0..500 {
        let words: [u64; 8] = core::array::from_fn(|_| random.next_u64());
        let shift = u32::try_from(random.below(512)).expect("below 512");
        let magnitude = Integer::from_digits(&words, rug::integer::Order::Lsf) >> shift;
        let value = UInt::<512>::from_bits(limbs(&magnitude));
        let signed = Int::<512>::from_bits(limbs(&magnitude));
        for env in behaviors() {
            let want = round_pair(&Rational::from(magnitude.clone()), false, &env);
            let (ours, flags) = DoubleDouble::<Gcc>::from_int_with(value, env);
            assert_eq!(rounded((ours, flags)), want, "{magnitude} {env:?}");
            let top_set = magnitude.get_bit(511);
            let signed_value = if top_set {
                magnitude.clone() - (Integer::from(1) << 512)
            } else {
                magnitude.clone()
            };
            let want = round_pair(&Rational::from(signed_value), top_set, &env);
            let (ours, flags) = DoubleDouble::<Gcc>::from_int_with(signed, env);
            assert_eq!(rounded((ours, flags)), want, "signed {magnitude} {env:?}");
        }
    }
}

/// Returns the exact value of a pair as a rational, or `None` for a special
/// value.
fn pair_value(hi: u64, lo: u64) -> Option<(Rational, bool)> {
    match exact(Pair::new(hi, lo)) {
        Exact::Number(number) => Some((
            number.to_rational().expect("the value is finite"),
            number.is_sign_negative(),
        )),
        Exact::Zero { negative } => Some((Rational::new(), negative)),
        Exact::Nan(_) | Exact::Infinity { .. } => None,
    }
}

/// The largest scale magnitude that the oracle applies. Every finite pair
/// scaled by 2^4000 overflows, and by 2^-4000 falls below 2^-1076, as it
/// does for every larger magnitude.
const ORACLE_SCALE: i32 = 4000;

#[test]
fn scale_b_rounds_the_scaled_value() {
    let mut random = SplitMix64::new(0xDD_5CA1);
    let fixed = [
        0,
        1,
        -1,
        53,
        -53,
        969,
        -969,
        1074,
        -1074,
        2098,
        -2098,
        1 << 30,
        i32::MIN,
        i32::MAX,
    ];
    for _ in 0..20_000 {
        let Pair { hi, lo } = pair(&mut random);
        let value = DoubleDouble::<Gcc>::from_parts(F64::from_bits(hi), F64::from_bits(lo));
        let scale = if random.below(4) == 0 {
            fixed[usize::try_from(random.below(14)).expect("below 14")]
        } else {
            i32::try_from(random.below(4401)).expect("fits") - 2200
        };
        for env in behaviors() {
            let (ours, flags) = value.scale_b_with(scale, env);
            let want = match pair_value(hi, lo) {
                Some((number, negative)) => {
                    let scale = scale.clamp(-ORACLE_SCALE, ORACLE_SCALE);
                    let power = Rational::from(Integer::from(1) << scale.unsigned_abs());
                    let scaled = if scale >= 0 {
                        number * power
                    } else {
                        number / power
                    };
                    round_pair(&scaled, negative, &env)
                }
                None => expected(&special_decoded(&exact(Pair::new(hi, lo))), 2, &env),
            };
            let context = format!("{hi:#x} {lo:#x} {scale} {env:?}");
            assert_eq!(rounded((ours, flags)), want, "{context}");
            assert!(
                env.precision.is_some() || ours.is_canonical(),
                "{context}: canonical"
            );
            let qd = DoubleDouble::<Qd>::from_parts(F64::from_bits(hi), F64::from_bits(lo));
            let (qd, qd_flags) = qd.scale_b_with(scale, env);
            assert_eq!((bits(qd), qd_flags), (bits(ours), flags), "{context}: Qd");
        }
    }
}

/// Returns the exponent `e` of a nonzero rational, with
/// `2^e <= |value| < 2^(e + 1)`.
fn exponent(value: &Rational) -> i64 {
    let magnitude = value.clone().abs();
    let bits = |integer: &Integer| i64::from(integer.significant_bits());
    let estimate = bits(magnitude.numer()) - bits(magnitude.denom());
    let power = |exponent: i64| {
        let shift = u32::try_from(exponent.unsigned_abs()).expect("an exponent fits a u32");
        let power = Rational::from(Integer::from(1) << shift);
        if exponent >= 0 { power } else { power.recip() }
    };
    if power(estimate) > magnitude {
        estimate - 1
    } else {
        estimate
    }
}

#[test]
fn log_b_gives_the_exponent_of_the_exact_value() {
    let mut random = SplitMix64::new(0xDD_10B8);
    for _ in 0..20_000 {
        let Pair { hi, lo } = pair(&mut random);
        let value = DoubleDouble::<Gcc>::from_parts(F64::from_bits(hi), F64::from_bits(lo));
        for env in behaviors() {
            let (ours, flags) = value.log_b_with(env);
            let want = match exact(Pair::new(hi, lo)) {
                Exact::Number(number) => {
                    let e = exponent(&number.to_rational().expect("the value is finite"));
                    round_pair(&Rational::from(e), e < 0, &env)
                }
                Exact::Zero { .. } => {
                    let mut infinite = infinity(true, &env);
                    infinite.flags |= Flags::DIVIDE_BY_ZERO;
                    infinite
                }
                Exact::Infinity { .. } => infinity(false, &env),
                nan @ Exact::Nan(_) => expected(&special_decoded(&nan), 2, &env),
            };
            let context = format!("{hi:#x} {lo:#x} {env:?}");
            assert_eq!(rounded((ours, flags)), want, "{context}");
            let qd = DoubleDouble::<Qd>::from_parts(F64::from_bits(hi), F64::from_bits(lo));
            let (qd, qd_flags) = qd.log_b_with(env);
            assert_eq!((bits(qd), qd_flags), (bits(ours), flags), "{context}: Qd");
        }
    }
}

/// Returns the sign of an exact value: the sign of the number, of the zero,
/// of the infinity, or of the NaN half.
fn exact_sign(value: &Exact) -> bool {
    match value {
        Exact::Number(number) => number.is_sign_negative(),
        Exact::Zero { negative } | Exact::Infinity { negative } => *negative,
        Exact::Nan(bits) => bits >> 63 == 1,
    }
}

#[test]
fn copy_sign_negates_both_halves_when_the_signs_differ() {
    let mut random = SplitMix64::new(0xDD_C0B5);
    for _ in 0..20_000 {
        let (x, y) = (pair(&mut random), pair(&mut random));
        let negate = exact_sign(&exact(x)) != exact_sign(&exact(y));
        let sign = if negate { 1 << 63 } else { 0 };
        let want = (x.hi ^ sign, x.lo ^ sign);
        let value = |pair: Pair| (F64::from_bits(pair.hi), F64::from_bits(pair.lo));
        let ((xh, xl), (yh, yl)) = (value(x), value(y));
        let ours = DoubleDouble::<Gcc>::from_parts(xh, xl)
            .copy_sign(DoubleDouble::<Gcc>::from_parts(yh, yl));
        assert_eq!(bits(ours), want, "{x:?} {y:?}");
        let qd = DoubleDouble::<Qd>::from_parts(xh, xl)
            .copy_sign(DoubleDouble::<Qd>::from_parts(yh, yl));
        assert_eq!(bits(qd), want, "{x:?} {y:?}: Qd");
    }
}

/// Returns the integer that a rational rounds to in the direction
/// `rounding`, by the definitions of IEEE 754 section 4.3 and of round to
/// odd.
fn round_integer(value: &Rational, rounding: Rounding) -> Integer {
    let floor = Integer::from(value.floor_ref());
    let fraction = Rational::from(value - &floor);
    if fraction == 0 {
        return floor;
    }
    let ceil = Integer::from(&floor + 1u32);
    let negative = *value < 0;
    let (toward_zero, away) = if negative {
        (ceil.clone(), floor.clone())
    } else {
        (floor.clone(), ceil.clone())
    };
    let half = Rational::from((1, 2));
    let tie = |rule: Rounding| match fraction.cmp(&half) {
        core::cmp::Ordering::Less => floor.clone(),
        core::cmp::Ordering::Greater => ceil.clone(),
        core::cmp::Ordering::Equal => match rule {
            Rounding::TiesToEven if floor.is_even() => floor.clone(),
            Rounding::TiesToEven => ceil.clone(),
            Rounding::TiesToAway => away.clone(),
            _ => toward_zero.clone(),
        },
    };
    match rounding {
        Rounding::TowardZero => toward_zero,
        Rounding::AwayFromZero => away,
        Rounding::TowardNegative => floor,
        Rounding::ToOdd if floor.is_odd() => floor,
        Rounding::TowardPositive | Rounding::ToOdd => ceil,
        other => tie(other),
    }
}

/// Returns a random pair near an integer: a high half of up to 110 bits and
/// a low half with a fraction, or a random pair.
fn near_integer(random: &mut SplitMix64) -> (u64, u64) {
    if random.below(3) == 0 {
        let Pair { hi, lo } = pair(random);
        return (hi, lo);
    }
    let magnitude = i32::try_from(random.below(110)).expect("below 110");
    let hi = F64::from_bits(random.next_u64() >> 12 | 0x3FF0_0000_0000_0000).scale_b(magnitude);
    let fractions = [
        0x3FE0_0000_0000_0000_u64,
        0x3FD0_0000_0000_0000,
        0x3FE8_0000_0000_0000,
    ];
    let lo = match random.below(4) {
        0 => fractions[usize::try_from(random.below(3)).expect("below 3")],
        1 => 0,
        _ => random.next_u64() >> 12 | 0x3FC0_0000_0000_0000,
    } | (random.next_u64() & (1 << 63));
    let sign = random.next_u64() & (1 << 63);
    (hi.to_bits() | sign, lo)
}

/// Returns the flags of a rounding from `value` to the integer `integer`.
fn integral_flags(value: &Rational, integer: &Integer) -> Flags {
    let mut flags = Flags::NONE;
    if *value != *integer {
        flags |= Flags::INEXACT;
        if Rational::from(integer.clone()).abs() > value.clone().abs() {
            flags |= Flags::ROUNDED_UP;
        }
    }
    flags
}

#[test]
fn round_to_integral_gives_the_canonical_pair_of_the_integer() {
    let mut random = SplitMix64::new(0xDD_1A7A);
    for _ in 0..20_000 {
        let (hi, lo) = near_integer(&mut random);
        let value = DoubleDouble::<Gcc>::from_parts(F64::from_bits(hi), F64::from_bits(lo));
        for env in behaviors() {
            let (ours, flags) = value.round_to_integral_with(env);
            let want = match pair_value(hi, lo) {
                Some((number, negative)) => {
                    let integer = round_integer(&number, env.rounding);
                    let exact_env = env.with_precision(None).with_flush_to_zero(false);
                    // The integer of a pair above the largest finite pair
                    // overflows the canonical split.
                    let mut want =
                        round_pair(&Rational::from(integer.clone()), negative, &exact_env);
                    want.flags |= integral_flags(&number, &integer);
                    want
                }
                None => expected(&special_decoded(&exact(Pair::new(hi, lo))), 2, &env),
            };
            let context = format!("{hi:#x} {lo:#x} {env:?}");
            assert_eq!(rounded((ours, flags)), want, "{context}");
            assert!(ours.is_canonical(), "{context}: canonical");
            let qd = DoubleDouble::<Qd>::from_parts(F64::from_bits(hi), F64::from_bits(lo));
            let (qd, qd_flags) = qd.round_to_integral_with(env);
            assert_eq!((bits(qd), qd_flags), (bits(ours), flags), "{context}: Qd");
        }
    }
}

/// Checks the conversion of one pair to the integer type `$integer`.
macro_rules! check_to_int {
    ($value:expr, $hi:expr, $lo:expr, $env:expr, $integer:ty) => {{
        let (ours, flags) = $value.to_int_with::<$integer>($env);
        let want = match exact(Pair::new($hi, $lo)) {
            Exact::Nan(_) => (ToInt::Nan, Flags::INVALID),
            Exact::Infinity { negative } => (ToInt::OutOfRange { negative }, Flags::INVALID),
            Exact::Zero { .. } => (ToInt::Value(0), Flags::NONE),
            Exact::Number(number) => {
                let number = number.to_rational().expect("the value is finite");
                let integer = round_integer(&number, $env.rounding);
                match integer
                    .to_i128()
                    .and_then(|value| <$integer>::try_from(value).ok())
                {
                    Some(value) => (ToInt::Value(value), integral_flags(&number, &integer)),
                    None => {
                        let negative = number < 0;
                        (ToInt::OutOfRange { negative }, Flags::INVALID)
                    }
                }
            }
        };
        let context = format!("{:#x} {:#x} {:?} {}", $hi, $lo, $env, stringify!($integer));
        assert_eq!((ours, flags), want, "{context}");
    }};
}

#[test]
fn to_int_rounds_the_exact_value() {
    let mut random = SplitMix64::new(0xDD_7017);
    for _ in 0..20_000 {
        let (hi, lo) = near_integer(&mut random);
        let value = DoubleDouble::<Gcc>::from_parts(F64::from_bits(hi), F64::from_bits(lo));
        for env in behaviors() {
            check_to_int!(value, hi, lo, env, i64);
            check_to_int!(value, hi, lo, env, u64);
            check_to_int!(value, hi, lo, env, i32);
            check_to_int!(value, hi, lo, env, u8);
            check_to_int!(value, hi, lo, env, i128);
        }
    }
}
