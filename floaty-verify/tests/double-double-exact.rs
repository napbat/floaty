//! Compares the operations on the exact value of a double-double pair with
//! MPFR: `decode`, conversion to binary and decimal formats, and the
//! comparisons.
//!
//! MPFR adds the two halves exactly. The special values follow floaty's rule:
//! a NaN or an infinite high half gives that value; with a finite high half,
//! a NaN or an infinite low half gives that value; and a zero sum takes the
//! sign of the high half. The conversion rules for the special values are the
//! rules that the MPFR conversion oracles already check.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::cmp::Ordering;
use core::num::NonZeroU32;

use floaty::env::Tininess;
use floaty::{
    BF16, D64Bid, D128Dpd, Decoded, DoubleDouble, Env, F16, F32, F64, F80, F128, Flags, Gcc,
    Rounding,
};
use floaty_verify::mpfr::decimal::{DecimalFormat, DecimalValue, decimal_payload, to_decimal};
use floaty_verify::mpfr::{self, Format, Input, Specials, Value};
use floaty_verify::random::SplitMix64;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer};

/// The precision of the exact sum: every finite pair spans at most 2,099
/// bits.
const EXACT_BITS: u32 = 2_200;

/// The exact value of a pair, computed from the bits of its halves.
enum Exact {
    /// The NaN of a half: its bits.
    Nan(u64),
    /// An infinity.
    Infinity { negative: bool },
    /// A zero, with the sign of the high half.
    Zero { negative: bool },
    /// A nonzero finite value.
    Number(BigFloat),
}

fn is_nan(bits: u64) -> bool {
    bits & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000 && bits & ((1 << 52) - 1) != 0
}

fn is_infinite(bits: u64) -> bool {
    bits & 0x7FFF_FFFF_FFFF_FFFF == 0x7FF0_0000_0000_0000
}

fn exact(hi: u64, lo: u64) -> Exact {
    let special = |bits: u64| {
        if is_nan(bits) {
            Some(Exact::Nan(bits))
        } else if is_infinite(bits) {
            Some(Exact::Infinity {
                negative: bits >> 63 == 1,
            })
        } else {
            None
        }
    };
    if let Some(value) = special(hi).or_else(|| special(lo)) {
        return value;
    }
    let half = |bits: u64| BigFloat::with_val(53, f64::from_bits(bits));
    let sum = BigFloat::with_val(EXACT_BITS, &half(hi) + &half(lo));
    if sum.is_zero() {
        Exact::Zero {
            negative: hi >> 63 == 1,
        }
    } else {
        Exact::Number(sum)
    }
}

/// Returns the integer and the exponent of a nonzero finite value, with an
/// odd integer.
fn odd_parts(value: &BigFloat) -> (bool, Integer, i32) {
    let (integer, exponent) = value.to_integer_exp().expect("the value is finite");
    let zeros = integer.find_one(0).expect("the integer is not zero");
    let magnitude = integer.abs() >> zeros;
    let exponent = exponent + i32::try_from(zeros).expect("a bit index fits an i32");
    (value.is_sign_negative(), magnitude, exponent)
}

/// Returns the NaN of a binary64 half as a decoded value.
fn nan_half<const N: usize>(bits: u64) -> Decoded<N> {
    let mut payload = [0_u64; N];
    payload[0] = bits & ((1 << 51) - 1);
    Decoded::Nan {
        negative: bits >> 63 == 1,
        signaling: bits & (1 << 51) == 0,
        payload,
    }
}

fn expected_decode(value: &Exact) -> Decoded<33> {
    match value {
        Exact::Nan(bits) => nan_half(*bits),
        Exact::Infinity { negative } => Decoded::Infinity {
            negative: *negative,
        },
        Exact::Zero { negative } => Decoded::Zero {
            negative: *negative,
            exponent: 0,
        },
        Exact::Number(number) => {
            let (negative, magnitude, exponent) = odd_parts(number);
            let mut significand = [0_u64; 33];
            for (slot, limb) in significand
                .iter_mut()
                .zip(magnitude.to_digits::<u64>(Order::Lsf))
            {
                *slot = limb;
            }
            Decoded::Finite {
                negative,
                exponent,
                significand,
            }
        }
    }
}

/// Returns the expected result of a conversion to a binary format.
fn to_binary(value: &Exact, format: &Format, env: &Env) -> (Value, Flags) {
    match value {
        Exact::Number(number) => {
            let (negative, magnitude, exponent) = odd_parts(number);
            let input = Input {
                negative,
                exponent,
                significand: magnitude,
                sticky: false,
            };
            mpfr::round(&input, format, env)
        }
        Exact::Nan(bits) => mpfr::convert(&nan_half::<1>(*bits), false, format, env),
        Exact::Infinity { negative } => mpfr::convert(
            &Decoded::<1>::Infinity {
                negative: *negative,
            },
            false,
            format,
            env,
        ),
        Exact::Zero { negative } => mpfr::convert(
            &Decoded::<1>::Zero {
                negative: *negative,
                exponent: 0,
            },
            false,
            format,
            env,
        ),
    }
}

/// Returns the expected result of a conversion to a decimal format.
fn to_decimal_format(value: &Exact, format: DecimalFormat, env: &Env) -> (DecimalValue, Flags) {
    match value {
        Exact::Number(number) => to_decimal(number, format, env),
        Exact::Nan(bits) => {
            let payload = Integer::from(bits & ((1 << 51) - 1));
            let signaling = bits & (1 << 51) == 0;
            let value = DecimalValue::Nan {
                negative: bits >> 63 == 1,
                payload: decimal_payload(payload, 51, format),
            };
            (
                value,
                if signaling {
                    Flags::INVALID
                } else {
                    Flags::NONE
                },
            )
        }
        Exact::Infinity { negative } => (
            DecimalValue::Infinity {
                negative: *negative,
            },
            Flags::NONE,
        ),
        Exact::Zero { negative } => (
            DecimalValue::Zero {
                negative: *negative,
                exponent: 0,
            },
            Flags::NONE,
        ),
    }
}

/// Returns the expected order of two exact values, and whether a value is a
/// signaling NaN.
fn expected_order(first: &Exact, second: &Exact) -> (Option<Ordering>, bool) {
    let signaling = |value: &Exact| matches!(value, Exact::Nan(bits) if bits & (1 << 51) == 0);
    if matches!(first, Exact::Nan(_)) || matches!(second, Exact::Nan(_)) {
        return (None, signaling(first) || signaling(second));
    }
    let number = |value: &Exact| match value {
        Exact::Number(number) => number.clone(),
        Exact::Zero { .. } => BigFloat::with_val(53, 0),
        Exact::Infinity { negative: true } => BigFloat::with_val(53, f64::NEG_INFINITY),
        Exact::Infinity { negative: false } => BigFloat::with_val(53, f64::INFINITY),
        Exact::Nan(_) => unreachable!("the NaNs return above"),
    };
    (number(first).partial_cmp(&number(second)), false)
}

/// Returns a random pair: well-formed, with any finite halves, with special
/// halves, or with random bits.
fn pair(random: &mut SplitMix64) -> (u64, u64) {
    const EDGES: [u64; 8] = [
        0,
        1,
        0x0010_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0x7FEF_FFFF_FFFF_FFFF,
        0x7FF0_0000_0000_0000,
        0x7FF8_0000_0000_0005,
        0x7FF0_0000_0000_0003,
    ];
    let edge = |random: &mut SplitMix64| {
        let index = usize::try_from(random.next_u64() % 8).expect("an index fits a usize");
        EDGES[index] | (random.next_u64() & (1 << 63))
    };
    let finite = |random: &mut SplitMix64| {
        let bits = random.next_u64();
        if is_nan(bits) || is_infinite(bits) {
            bits & !(1 << 62)
        } else {
            bits
        }
    };
    match random.next_u64() % 6 {
        0 | 1 => {
            let hi = finite(random);
            let field = (hi >> 52) & 0x7FF;
            if field < 60 {
                return (hi, 0);
            }
            let low_field = field - 54 - random.next_u64() % 6;
            (
                hi,
                (random.next_u64() & 0x800F_FFFF_FFFF_FFFF) | (low_field << 52),
            )
        }
        2 => (finite(random), finite(random)),
        3 => (edge(random), edge(random)),
        4 => (finite(random), edge(random)),
        _ => (random.next_u64(), random.next_u64()),
    }
}

/// The behaviors of the conversions. Saturation applies to the rounding into
/// a binary destination format.
fn behaviors() -> [Env; 8] {
    [
        Env::IEEE,
        Env::IEEE.with_rounding(Rounding::TowardZero),
        Env::IEEE
            .with_rounding(Rounding::TowardPositive)
            .with_flush_to_zero(true)
            .with_tininess(Tininess::BeforeRounding),
        Env::IEEE.with_rounding(Rounding::ToOdd),
        Env::IEEE
            .with_rounding(Rounding::TiesToAway)
            .with_precision(NonZeroU32::new(5)),
        Env::IEEE.with_rounding(Rounding::TiesTowardZero),
        Env::IEEE.with_rounding(Rounding::AwayFromZero),
        Env::IEEE.with_saturate(true),
    ]
}

/// Compares one conversion to a binary format.
macro_rules! check_binary {
    ($value:expr, $exact:expr, $env:expr, $($target:ty),+) => {$(
        let format = Format {
            precision: <$target>::PRECISION,
            emin: <$target>::EMIN,
            emax: <$target>::EMAX,
            specials: Specials::Ieee,
        };
        let (ours, flags) = $value.convert_with::<$target>($env);
        let ours = (Value::from_decoded(ours.decode::<8>()), flags);
        assert_eq!(ours, to_binary($exact, &format, &$env), "{:?} to {} {:?}", $value, stringify!($target), $env);
    )+};
}

/// Compares one conversion to a decimal format.
macro_rules! check_decimal {
    ($value:expr, $exact:expr, $env:expr, $($target:ty),+) => {$(
        let format = DecimalFormat {
            precision: <$target>::PRECISION,
            emax: <$target>::EMAX,
        };
        let (ours, flags) = $value.convert_with::<$target>($env);
        let ours = (DecimalValue::from_decoded(ours.decode::<2>()), flags);
        assert_eq!(ours, to_decimal_format($exact, format, &$env), "{:?} to {} {:?}", $value, stringify!($target), $env);
    )+};
}

fn value(hi: u64, lo: u64) -> DoubleDouble<Gcc> {
    DoubleDouble::from_parts(F64::from_bits(hi), F64::from_bits(lo))
}

#[test]
fn the_exact_value_decodes_and_converts() {
    let mut random = SplitMix64::new(0xDD_E4AC);
    for _ in 0..20_000 {
        let (hi, lo) = pair(&mut random);
        let (dd, exact_value) = (value(hi, lo), exact(hi, lo));
        assert_eq!(
            dd.decode(),
            expected_decode(&exact_value),
            "{hi:#x} {lo:#x}"
        );
        for env in behaviors() {
            check_binary!(dd, &exact_value, env, F16, BF16, F32, F64, F80, F128);
            check_decimal!(dd, &exact_value, env, D64Bid, D128Dpd);
        }
    }
}

#[test]
fn the_exact_values_compare() {
    let mut random = SplitMix64::new(0xDD_C0DE);
    for _ in 0..50_000 {
        let (a, b) = (pair(&mut random), pair(&mut random));
        // Half the second operands share the first high half, so that the low
        // halves decide.
        let b = if random.next_u64() & 1 == 0 {
            (a.0, b.1)
        } else {
            b
        };
        let (x, y) = (value(a.0, a.1), value(b.0, b.1));
        let (order, signaling) = expected_order(&exact(a.0, a.1), &exact(b.0, b.1));
        let quiet = if signaling {
            Flags::INVALID
        } else {
            Flags::NONE
        };
        let signaled = if order.is_none() {
            Flags::INVALID
        } else {
            Flags::NONE
        };
        assert_eq!(x.compare_quiet(y), (order, quiet), "{a:x?} {b:x?}");
        assert_eq!(x.compare_signaling(y), (order, signaled), "{a:x?} {b:x?}");
        assert_eq!(x.partial_cmp(&y), order, "{a:x?} {b:x?}");
    }
}

/// Returns the sign of an exact value.
fn negative(value: &Exact) -> bool {
    match value {
        Exact::Nan(bits) => bits >> 63 == 1,
        Exact::Infinity { negative } | Exact::Zero { negative } => *negative,
        Exact::Number(number) => number.is_sign_negative(),
    }
}

#[test]
fn abs_clears_the_sign_of_the_exact_value() {
    let mut random = SplitMix64::new(0xDD_AB5);
    let sign = 1_u64 << 63;
    for _ in 0..50_000 {
        let (hi, lo) = pair(&mut random);
        let result = value(hi, lo).abs();
        let expected = if negative(&exact(hi, lo)) {
            (hi ^ sign, lo ^ sign)
        } else {
            (hi, lo)
        };
        assert_eq!(
            (result.hi().to_bits(), result.lo().to_bits()),
            expected,
            "abs {hi:#x} {lo:#x}"
        );
    }
}
