//! The exact value of a double-double pair, random pairs, and floaty's rule
//! for the rounding of an exact value to a pair, evaluated with exact
//! rationals and MPFR.
//!
//! floaty documents the rule on `DoubleDouble`. The high half is the value
//! rounded to binary64 to nearest even, and the low half is the rest rounded
//! in the direction of the behavior. The sum of the halves then splits into
//! its canonical pair. No reference library rounds to a pair this way, so
//! [`round_pair`] evaluates the documented rule: each step is an exact
//! rational operation or a binary64 rounding of [`mpfr::round`].
//!
//! The submodule [`reference`](mod@reference) holds the types that the
//! double-double references share.

pub mod reference;

use core::num::NonZeroU32;

use floaty::env::Tininess;
use floaty::{Decoded, Env, Flags, Rounding};
use rug::float::Round;
use rug::ops::Pow;
use rug::{Float as BigFloat, Integer, Rational};

use crate::encodings::Layout;
use crate::mpfr::{self, Format, Input, Specials, Value};
use crate::random::SplitMix64;
use reference::Pair;

/// The precision of the exact sum of two halves: every finite pair spans at
/// most 2,099 bits.
pub const EXACT_BITS: u32 = 2_200;

/// binary64, as the oracle rounds to it.
pub const BINARY64: Format = Format::of::<floaty::F64>(Specials::Ieee);

/// The precision of the rounding of a rational to an [`Input`]. A value below
/// 2^1025 keeps every bit above 2^-1975, far below the binary64 quantum
/// 2^-1074, and the sticky bit stands for the rest.
const INPUT_BITS: u32 = 3000;

/// The exact value of a pair, computed from the bits of its halves.
#[derive(Clone, Debug)]
pub enum Exact {
    /// The NaN of a half: its bits.
    Nan(u64),
    /// An infinity.
    Infinity {
        /// The sign.
        negative: bool,
    },
    /// A zero, with the sign of the high half.
    Zero {
        /// The sign.
        negative: bool,
    },
    /// A nonzero finite value.
    Number(BigFloat),
}

impl Exact {
    /// Returns the sign of the value.
    #[must_use]
    pub fn negative(&self) -> bool {
        match self {
            Self::Nan(bits) => bits >> 63 == 1,
            Self::Infinity { negative } | Self::Zero { negative } => *negative,
            Self::Number(number) => number.is_sign_negative(),
        }
    }
}

/// Returns `true` for the encoding of a binary64 NaN.
#[must_use]
pub fn is_nan(bits: u64) -> bool {
    bits & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000 && bits & ((1 << 52) - 1) != 0
}

/// Returns `true` for the encoding of a binary64 infinity.
#[must_use]
pub fn is_infinite(bits: u64) -> bool {
    bits & 0x7FFF_FFFF_FFFF_FFFF == 0x7FF0_0000_0000_0000
}

/// Returns the exact value of a pair by floaty's rule: a NaN or an infinite
/// high half gives that value; with a finite high half, a NaN or an infinite
/// low half gives that value; and a zero sum takes the sign of the high half.
#[must_use]
pub fn exact(Pair { hi, lo }: Pair) -> Exact {
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

/// Returns a random pair: well-formed, with any finite halves, with special
/// halves, or with random bits.
///
/// # Panics
///
/// Never: an index below 8 fits a `usize`.
pub fn pair(random: &mut SplitMix64) -> Pair {
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
        let index = usize::try_from(random.below(8)).expect("an index fits a usize");
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
    // The halves draw in order: the high half, then the low half.
    let (hi, lo) = match random.below(6) {
        0 | 1 => {
            let hi = finite(random);
            let field = (hi >> 52) & 0x7FF;
            if field < 60 {
                return Pair::new(hi, 0);
            }
            let low_field = field - 54 - random.below(6);
            (
                hi,
                (random.next_u64() & 0x800F_FFFF_FFFF_FFFF) | (low_field << 52),
            )
        }
        2 => (finite(random), finite(random)),
        3 => (edge(random), edge(random)),
        4 => (finite(random), edge(random)),
        _ => (random.next_u64(), random.next_u64()),
    };
    Pair::new(hi, lo)
}

/// Returns a random pair for a comparison with a reference: a canonical pair
/// of any magnitude, a pair near the edges of the range, or a pair of
/// [`pair`], which holds special and random halves.
pub fn operand(random: &mut SplitMix64) -> Pair {
    let negative = random.coin_flip();
    match random.below(4) {
        0 | 1 => {
            let field = match random.below(4) {
                0 => random.below(120),
                1 => 2046 - random.below(60),
                _ => 1 + random.below(2046),
            };
            let hi = Layout::BINARY64.encode_u64(negative, field, random.next_u64());
            if field < 56 || random.below(6) == 0 {
                return Pair::new(hi, 0);
            }
            let low_field = field - 54 - random.below((field - 54).min(70));
            let lo = Layout::BINARY64.encode_u64(random.coin_flip(), low_field, random.next_u64());
            Pair::new(hi, lo)
        }
        _ => pair(random),
    }
}

/// Returns a random pair of `generate` that is canonical: the functions of
/// glibc that read the halves and the operations of floaty that read the
/// exact value agree for a canonical pair.
pub fn canonical(random: &mut SplitMix64, generate: impl Fn(&mut SplitMix64) -> Pair) -> Pair {
    loop {
        let candidate = generate(random);
        let value = floaty::DoubleDouble::<floaty::Gcc>::from_parts(
            floaty::F64::from_bits(candidate.hi),
            floaty::F64::from_bits(candidate.lo),
        );
        if value.is_canonical() {
            return candidate;
        }
    }
}

/// Behaviors that use each rounding direction, flush-to-zero with each
/// tininess rule, a precision limit, and saturation.
#[must_use]
pub fn behaviors() -> [Env; 8] {
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

/// Returns the exact value of a decoded finite value of radix `radix`, or
/// `None` for another class.
#[must_use]
pub fn rational<const N: usize>(decoded: &Decoded<N>, radix: u32) -> Option<Rational> {
    match *decoded {
        Decoded::Zero { .. } => Some(Rational::new()),
        Decoded::Finite {
            negative,
            exponent,
            ref significand,
        } => {
            let magnitude =
                Rational::from(Integer::from_digits(significand, rug::integer::Order::Lsf));
            let power = Integer::from(radix).pow(exponent.unsigned_abs());
            let scale = if exponent >= 0 {
                Rational::from(power)
            } else {
                Rational::from((1, power))
            };
            let value = magnitude * scale;
            Some(if negative { -value } else { value })
        }
        _ => None,
    }
}

/// Returns a nonzero rational as an [`Input`]: its magnitude rounded down to
/// 3,000 bits, with the sticky bit of the rest.
///
/// # Panics
///
/// Panics for a zero, which has no finite exponent in MPFR.
#[must_use]
pub fn input(value: &Rational) -> Input {
    let (magnitude, order) = BigFloat::with_val_round(INPUT_BITS, value.clone().abs(), Round::Down);
    let (significand, exponent) = magnitude.to_integer_exp().expect("the magnitude is finite");
    Input {
        negative: *value < 0,
        exponent,
        significand,
        sticky: order != core::cmp::Ordering::Equal,
    }
}

/// Returns the exact value of a finite rounded value.
fn value_of(value: &Value) -> Option<Rational> {
    match value {
        Value::Zero { .. } => Some(Rational::new()),
        Value::Finite(number) => number.to_rational(),
        Value::Infinity { .. } | Value::Nan { .. } => None,
    }
}

/// The expected result of a rounding to a pair.
#[derive(Clone, Debug, PartialEq)]
pub struct Rounded {
    /// The high half.
    pub hi: Value,
    /// The low half.
    pub lo: Value,
    /// The flags.
    pub flags: Flags,
}

/// The positive zero of a low half without a rest.
const NO_REST: Value = Value::Zero { negative: false };

/// Returns the expected result of the rounding of a finite rational to a
/// pair, by floaty's rule. A zero takes the sign `negative`.
///
/// # Panics
///
/// Panics when a step of the rule breaks an invariant: an infinite low half,
/// or an inexact canonical split.
#[must_use]
pub fn round_pair(value: &Rational, negative: bool, env: &Env) -> Rounded {
    if *value == 0 {
        return Rounded {
            hi: Value::Zero { negative },
            lo: NO_REST,
            flags: Flags::NONE,
        };
    }
    let nearest = env.with_rounding(Rounding::TiesToEven).with_saturate(false);
    let (high, _) = mpfr::round(&input(value), &BINARY64, &nearest);
    let Some(high) = value_of(&high) else {
        return overflow(negative, env);
    };
    let rest = Rational::from(value - &high);
    let (low, low_flags) = if rest == 0 {
        (NO_REST, Flags::NONE)
    } else {
        mpfr::round(&input(&rest), &BINARY64, env)
    };
    let total = high + value_of(&low).expect("a low half is finite");
    let mut flags = low_flags.difference(Flags::ROUNDED_UP);
    if total.clone().abs() > value.clone().abs() {
        flags |= Flags::ROUNDED_UP;
    }
    if total == 0 {
        return Rounded {
            hi: Value::Zero { negative },
            lo: NO_REST,
            flags,
        };
    }
    let (hi, _) = mpfr::round(&input(&total), &BINARY64, &nearest);
    let Some(hi_value) = value_of(&hi) else {
        return overflow(negative, env);
    };
    let rest = total - hi_value;
    let lo = if rest == 0 {
        NO_REST
    } else {
        let (lo, split_flags) = mpfr::round(&input(&rest), &BINARY64, env);
        assert!(
            !split_flags.contains(Flags::INEXACT),
            "the canonical split is exact"
        );
        lo
    };
    Rounded { hi, lo, flags }
}

/// Returns the expected result of an overflow of the sign `negative`: the
/// binary64 overflow result of the behavior as the high half, with a `+0`
/// low half after an infinity, or the largest value at the precision below
/// half an ulp of the largest high half.
/// Returns the pair of an infinite source: the infinity with a `+0` low
/// half, or with saturation the largest finite pair. Neither signals.
#[must_use]
pub fn infinity(negative: bool, env: &Env) -> Rounded {
    if !env.saturate {
        return Rounded {
            hi: Value::Infinity { negative },
            lo: NO_REST,
            flags: Flags::NONE,
        };
    }
    Rounded {
        flags: Flags::NONE,
        ..overflow(negative, env)
    }
}

fn overflow(negative: bool, env: &Env) -> Rounded {
    let huge = Input {
        negative,
        exponent: 1100,
        significand: Integer::from(1),
        sticky: false,
    };
    let (hi, flags) = mpfr::round(&huge, &BINARY64, env);
    let lo = match &hi {
        Value::Finite(largest) => {
            let precision = BINARY64.precision_in(env);
            Value::Finite(largest.clone() >> (precision + 1))
        }
        _ => NO_REST,
    };
    Rounded { hi, lo, flags }
}
