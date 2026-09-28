//! Compares operations of build step 4 with `rustc_apfloat`, the Rust port of
//! LLVM APFloat: `next_up` and `next_down`, the IEEE remainder, rounding to an
//! integral value, conversion to and from integers of several widths, and
//! scaling. Every FP8 operand pair of the formats that `rustc_apfloat` has
//! runs, with every binary16 and bfloat16 operand, and boundary and random
//! operands of the wider formats.
//!
//! `rustc_apfloat` has five rounding directions and no round to odd, and it
//! reports the five IEEE 754 flags only. The tests compare floaty's flags
//! restricted to those five. It selects the first NaN operand, made quiet,
//! and its default NaN is positive, so the tests use that NaN rule.
//!
//! `rustc_apfloat` differs from IEEE 754 and `DESIGN.md` in these cases, which
//! the tests exclude:
//!
//! - `next_up` of a signaling NaN gives the default NaN with the sign of the
//!   operand. IEEE 754-2019 section 6.2 recommends the operand NaN made
//!   quiet, as floaty gives. The tests check that both results are quiet NaNs
//!   with the sign of the operand, and signal invalid.
//! - `scalbn` reports no flags, so the tests compare only its value.
//! - A rounding to the largest finite value reports inexact and not
//!   overflow: `overflow_result` in `src/ieee.rs` follows LLVM
//!   `handleOverflow`. IEEE 754-2019 section 7.4 signals overflow in every
//!   rounding direction, as floaty does. The tests remove the overflow flag
//!   of floaty's finite results.
//! - For a format without an infinity, `normalize` calls `overflow_result`
//!   without the negated direction that its documentation requires for a
//!   negative value. A negative value whose magnitude rounds past the largest
//!   finite value, below `2^(emax + 1)`, then gives the NaN toward positive
//!   and the largest finite value toward negative, the reverse of IEEE 754.
//!   The tests skip those `scale_b` results; the MPFR tests check them.
//! - It follows LLVM, not the processor, for the non-canonical x87
//!   encodings (`docs/anomalies/x87-noncanonical-rustc-apfloat.md`), so the
//!   x87 operands are canonical.

use floaty::env::{NanPropagation, NanRule};
use floaty::format::Standard;
use floaty::{
    Binary, Class, Decoded, Env, F80, Flags, Float, Int, NoInf, Rounding, ToInt, UInt, X87,
};
use floaty_verify::apfloat::Tf32;
use floaty_verify::encodings::{IntegerBit, boundary_encodings, to_u128};
use floaty_verify::operations::integral::{IntegerValue, to_int_value};
use floaty_verify::random::SplitMix64;
use rug::Integer;
use rug::integer::Order;
use rustc_apfloat::ieee::{
    BFloat, Double, Float8E4M3FN, Float8E5M2, Half, Quad, Single, X87DoubleExtended,
};
use rustc_apfloat::{Round, Status, StatusAnd};

/// The directions that both implementations have.
const ROUNDINGS: [(Rounding, Round); 5] = [
    (Rounding::NearestEven, Round::NearestTiesToEven),
    (Rounding::NearestAway, Round::NearestTiesToAway),
    (Rounding::TowardPositive, Round::TowardPositive),
    (Rounding::TowardNegative, Round::TowardNegative),
    (Rounding::TowardZero, Round::TowardZero),
];

/// The NaN rule of `rustc_apfloat`: the first NaN operand, made quiet, and a
/// positive default NaN.
const APFLOAT: Env = Env::IEEE.with_nan(NanRule::new(NanPropagation::FirstOperand));

/// Returns the `rustc_apfloat` status of the five IEEE 754 flags.
fn status(flags: Flags) -> Status {
    status_of(flags, false)
}

/// Returns the `rustc_apfloat` status of the five IEEE 754 flags of a
/// rounded result. `rustc_apfloat` does not report overflow when the result
/// is the largest finite value; see
/// `docs/anomalies/rustc-apfloat-directed-overflow-flag.md`.
fn rounded_status(flags: Flags, largest_finite: bool) -> Status {
    status_of(flags, largest_finite)
}

fn status_of(flags: Flags, largest_finite: bool) -> Status {
    [
        (Flags::INVALID, Status::INVALID_OP),
        (Flags::DIVIDE_BY_ZERO, Status::DIV_BY_ZERO),
        (Flags::OVERFLOW, Status::OVERFLOW),
        (Flags::UNDERFLOW, Status::UNDERFLOW),
        (Flags::INEXACT, Status::INEXACT),
    ]
    .into_iter()
    .filter(|&(flag, _)| flags.contains(flag) && !(largest_finite && flag == Flags::OVERFLOW))
    .fold(Status::OK, |status, (_, bit)| status | bit)
}

/// A floaty format and the `rustc_apfloat` type of the same format.
struct Formats<S: Standard<W>, const W: usize, A>(core::marker::PhantomData<(S, A)>);

/// Returns the floaty value of an encoding.
fn ours<S: Standard<W>, const W: usize>(bits: u128) -> Float<S, W>
where
    S::Bits: TryFrom<u128>,
{
    Float::from_bits(
        S::Bits::try_from(bits)
            .ok()
            .expect("the encoding fits the storage"),
    )
}

/// Returns the encoding of a floaty value.
fn bits<S: Standard<W>, const W: usize>(value: Float<S, W>) -> u128
where
    S::Bits: Into<u128>,
{
    value.to_bits().into()
}

impl<S: Standard<W>, const W: usize, A: rustc_apfloat::Float> Formats<S, W, A>
where
    S::Bits: TryFrom<u128> + Into<u128>,
{
    /// Checks `next_up` and `next_down` of one encoding.
    fn next(encoding: u128) {
        let (x, a) = (ours::<S, W>(encoding), A::from_bits(encoding));
        let results = [
            (x.next_up_with(APFLOAT), a.next_up()),
            (x.next_down_with(APFLOAT), a.next_down()),
        ];
        for (
            (result, flags),
            StatusAnd {
                status: theirs,
                value,
            },
        ) in results
        {
            let context = format!("next {x:?}: {result:?} {:#x}", value.to_bits());
            assert_eq!(status(flags), theirs, "{context}");
            if a.is_signaling() {
                assert_eq!(result.classify(), Class::QuietNan, "{context}");
                assert!(value.is_nan() && !value.is_signaling(), "{context}");
                assert_eq!(result.is_sign_negative(), a.is_negative(), "{context}");
                assert_eq!(value.is_negative(), a.is_negative(), "{context}");
            } else {
                assert_eq!(bits(result), value.to_bits(), "{context}");
            }
        }
    }

    /// Checks rounding to an integral value and scaling of one encoding.
    fn round_and_scale(encoding: u128, scales: &[i32]) {
        let (x, a) = (ours::<S, W>(encoding), A::from_bits(encoding));
        for (rounding, round) in ROUNDINGS {
            let env = APFLOAT.with_rounding(rounding);
            let (result, flags) = x.round_to_integral_with(env);
            let StatusAnd {
                status: theirs,
                value,
            } = a.round_to_integral(round);
            assert_eq!(
                (bits(result), status(flags)),
                (value.to_bits(), theirs),
                "round_to_integral {x:?} {rounding:?}"
            );
            for &scale in scales {
                // `scalbn` reports no flags, so only the values compare.
                let result = x.scale_b_with(scale, env).0;
                if Self::negative_overflow_without_infinity(x, scale, rounding) {
                    continue;
                }
                assert_eq!(
                    bits(result),
                    a.scalbn_r(scale, round).to_bits(),
                    "scale_b {x:?} {scale} {rounding:?}"
                );
            }
        }
    }

    /// Returns `true` for a `scale_b` result that the `rustc_apfloat` sign
    /// error of a format without an infinity changes
    /// (`docs/anomalies/rustc-apfloat-no-infinity-overflow-sign.md`): a
    /// negative value that scales exactly to the magnitude of the NaN
    /// encoding, in a directed rounding. The format parameters decide the
    /// case, not the result of floaty.
    fn negative_overflow_without_infinity(x: Float<S, W>, scale: i32, rounding: Rounding) -> bool {
        let Decoded::Finite {
            negative: true,
            exponent,
            significand,
        } = x.decode::<2>()
        else {
            return false;
        };
        let significand = Integer::from_digits(&significand, Order::Lsf);
        let width = significand.significant_bits();
        let precision = Float::<S, W>::PRECISION;
        let top = i64::from(exponent) + i64::from(width) - 1 + i64::from(scale);
        let all_ones =
            (significand << (precision - width)) == (Integer::from(1) << precision) - 1u32;
        A::INFINITY.is_nan()
            && matches!(
                rounding,
                Rounding::TowardPositive | Rounding::TowardNegative
            )
            && top == i64::from(Float::<S, W>::EMAX)
            && all_ones
    }

    /// Returns `true` for the largest finite value of either sign, from the
    /// format parameters. In a format without an infinity, the all-ones
    /// significand at `emax` is the NaN.
    fn is_largest_finite(value: Float<S, W>) -> bool {
        let Decoded::Finite {
            exponent,
            significand,
            ..
        } = value.decode::<2>()
        else {
            return false;
        };
        let precision = Float::<S, W>::PRECISION;
        let ones = (Integer::from(1) << precision) - 1u32;
        let largest = if A::INFINITY.is_nan() {
            ones - 1u32
        } else {
            ones
        };
        let lowest = Float::<S, W>::EMAX
            - i32::try_from(precision - 1).expect("a precision of at most 128 bits fits an i32");
        exponent == lowest && Integer::from_digits(&significand, Order::Lsf) == largest
    }

    /// Checks the IEEE remainder of a pair.
    fn remainder(first: u128, second: u128) {
        let (x, y) = (ours::<S, W>(first), ours::<S, W>(second));
        let (result, flags) = x.remainder_with(y, APFLOAT);
        let StatusAnd {
            status: theirs,
            value,
        } = A::from_bits(first).ieee_rem(A::from_bits(second));
        assert_eq!(
            (bits(result), status(flags)),
            (value.to_bits(), theirs),
            "remainder {x:?} {y:?}"
        );
    }

    /// Checks the conversion of one encoding to the integer type `I`, whose
    /// width `rustc_apfloat` takes as an argument. `rustc_apfloat` signals
    /// invalid for a NaN, an infinity, and an out-of-range value, and returns
    /// zero or a limit. The sign of the float tells an out-of-range value
    /// apart.
    fn to_int<I: IntegerValue>(encoding: u128) {
        let (x, a) = (ours::<S, W>(encoding), A::from_bits(encoding));
        let width = usize::try_from(I::width()).expect("an integer width fits a usize");
        for (rounding, round) in ROUNDINGS {
            let (result, flags) = x.to_int_with::<I>(APFLOAT.with_rounding(rounding));
            let mut exact = false;
            let (value, theirs) = if I::signed() {
                let StatusAnd { status, value } = a.to_i128_r(width, round, &mut exact);
                (Integer::from(value), status)
            } else {
                let StatusAnd { status, value } = a.to_u128_r(width, round, &mut exact);
                (Integer::from(value), status)
            };
            let expected = if !theirs.contains(Status::INVALID_OP) {
                ToInt::Value(value)
            } else if a.is_nan() {
                ToInt::Nan
            } else {
                ToInt::OutOfRange {
                    negative: a.is_negative(),
                }
            };
            assert_eq!(
                (to_int_value(result), status(flags)),
                (expected, theirs),
                "to_int into {} {x:?} {rounding:?}",
                core::any::type_name::<I>()
            );
        }
    }

    /// Checks the conversion of every integer in `values` of the type `I`.
    fn from_int<I: IntegerValue>(values: &[Integer]) {
        for value in values {
            let integer = I::from_integer(value);
            for (rounding, round) in ROUNDINGS {
                let (result, flags) = Float::<S, W>::from_int_with(integer, rounding);
                let StatusAnd {
                    status: theirs,
                    value: float,
                } = if I::signed() {
                    A::from_i128_r(value.to_i128().expect("the value fits"), round)
                } else {
                    A::from_u128_r(value.to_u128().expect("the value fits"), round)
                };
                assert_eq!(
                    (
                        bits(result),
                        rounded_status(flags, Self::is_largest_finite(result))
                    ),
                    (float.to_bits(), theirs),
                    "from_int {value} {rounding:?}"
                );
            }
        }
    }

    /// Checks every operation on one encoding: every integer width, and the
    /// scales that cross the ends of the range of the value.
    fn one_operand(encoding: u128) {
        Self::next(encoding);
        let mut scales = vec![0, 1, -1, 5, -5, i32::MIN, i32::MAX];
        if let Decoded::Finite {
            exponent,
            significand,
            ..
        } = ours::<S, W>(encoding).decode::<2>()
        {
            let width = Integer::from_digits(&significand, Order::Lsf).significant_bits();
            let top = exponent + i32::try_from(width).expect("a width fits an i32") - 1;
            let p = i32::try_from(Float::<S, W>::PRECISION).expect("a precision fits an i32");
            let (high, low) = (Float::<S, W>::EMAX - top, Float::<S, W>::EMIN - top);
            scales.extend([
                high,
                high + 1,
                low,
                low - 1,
                low - p + 1,
                low - p,
                low - p - 1,
            ]);
        }
        Self::round_and_scale(encoding, &scales);
        Self::to_int::<Int<7>>(encoding);
        Self::to_int::<UInt<7>>(encoding);
        Self::to_int::<Int<24>>(encoding);
        Self::to_int::<UInt<24>>(encoding);
        Self::to_int::<Int<53>>(encoding);
        Self::to_int::<UInt<53>>(encoding);
        Self::to_int::<i64>(encoding);
        Self::to_int::<u64>(encoding);
        Self::to_int::<Int<100>>(encoding);
        Self::to_int::<UInt<100>>(encoding);
        Self::to_int::<i128>(encoding);
        Self::to_int::<u128>(encoding);
    }

    /// Checks the conversion of edge and random integers of every width.
    fn integers(random: &mut SplitMix64) {
        Self::from_int::<Int<7>>(&integers(7, true, random));
        Self::from_int::<UInt<7>>(&integers(7, false, random));
        Self::from_int::<Int<24>>(&integers(24, true, random));
        Self::from_int::<UInt<24>>(&integers(24, false, random));
        Self::from_int::<Int<53>>(&integers(53, true, random));
        Self::from_int::<UInt<53>>(&integers(53, false, random));
        Self::from_int::<i64>(&integers(64, true, random));
        Self::from_int::<u64>(&integers(64, false, random));
        Self::from_int::<Int<100>>(&integers(100, true, random));
        Self::from_int::<UInt<100>>(&integers(100, false, random));
        Self::from_int::<i128>(&integers(128, true, random));
        Self::from_int::<u128>(&integers(128, false, random));
    }

    /// Checks every operation on the encodings and the remainder pairs.
    fn check(encodings: &[u128], pairs: impl Iterator<Item = (u128, u128)>, seed: u64) {
        for &encoding in encodings {
            Self::one_operand(encoding);
        }
        for (first, second) in pairs {
            Self::remainder(first, second);
        }
        Self::integers(&mut SplitMix64::new(seed));
    }
}

/// Returns the limits, the powers of two near them and near each precision,
/// and random integers of random widths that fit `bits` bits.
fn integers(bits: u32, signed: bool, random: &mut SplitMix64) -> Vec<Integer> {
    let (low, high) = if signed {
        let half = Integer::from(1) << (bits - 1);
        (-half.clone(), half - 1u32)
    } else {
        (Integer::ZERO, (Integer::from(1) << bits) - 1u32)
    };
    let mut values = vec![Integer::ZERO, Integer::from(1), low.clone(), high.clone()];
    for power in [3_u32, 4, 8, 11, 24, 53, 64, 113, bits - 1] {
        let two: Integer = Integer::from(1) << power;
        for delta in [-1_i32, 0, 1] {
            let near: Integer = two.clone() + delta;
            values.push(-near.clone());
            values.push(near);
        }
    }
    for _ in 0..300 {
        let width = u32::try_from(random.next_u64() % u64::from(bits))
            .expect("a width below the integer width fits a u32")
            + 1;
        let digits = [random.next_u64(), random.next_u64()];
        let mut value = Integer::from_digits(&digits, Order::Lsf).keep_bits(width);
        if signed && random.next_u64() % 2 == 0 {
            value = -value;
        }
        values.push(value);
    }
    values.retain(|value| *value >= low && *value <= high);
    values
}

/// Returns every pair of encodings of a format with `width` bits.
fn every_pair(width: u32) -> impl Iterator<Item = (u128, u128)> {
    (0..1_u128 << width)
        .flat_map(move |first| (0..1_u128 << width).map(move |second| (first, second)))
}

#[test]
fn every_fp8_operand_and_pair() {
    let every: Vec<u128> = (0..256).collect();
    Formats::<Binary<4, NoInf>, 8, Float8E4M3FN>::check(&every, every_pair(8), 1);
    Formats::<Binary<5>, 8, Float8E5M2>::check(&every, every_pair(8), 2);
}

/// Returns the boundary encodings, random encodings, and pairs of a format:
/// every pair of boundary encodings, and random pairs.
fn samples(
    width: u32,
    exponent_bits: u32,
    count: usize,
    seed: u64,
    canonical: &dyn Fn(u128) -> bool,
) -> (Vec<u128>, Vec<(u128, u128)>) {
    let mask = u128::MAX >> (128 - width);
    let mut random = SplitMix64::new(seed);
    let integer_bit = if width == 80 {
        IntegerBit::Explicit
    } else {
        IntegerBit::Implicit
    };
    let boundaries: Vec<u128> = boundary_encodings(width, exponent_bits, integer_bit)
        .iter()
        .map(to_u128)
        .filter(|&bits| canonical(bits))
        .collect();
    let mut encodings = boundaries.clone();
    encodings.extend(
        (0..count)
            .map(|_| random.next_u128() & mask)
            .filter(|&bits| canonical(bits)),
    );
    let mut pairs: Vec<(u128, u128)> = boundaries
        .iter()
        .flat_map(|&x| boundaries.iter().map(move |&y| (x, y)))
        .collect();
    pairs.extend(encodings.chunks_exact(2).map(|pair| (pair[0], pair[1])));
    (encodings, pairs)
}

#[test]
fn every_binary16_and_bfloat16_operand() {
    let every: Vec<u128> = (0..1 << 16).collect();
    let (_, pairs) = samples(16, 5, 20_000, 16, &|_| true);
    Formats::<Binary<5>, 16, Half>::check(&every, pairs.into_iter(), 16);
    let (_, pairs) = samples(16, 8, 20_000, 17, &|_| true);
    Formats::<Binary<8>, 16, BFloat>::check(&every, pairs.into_iter(), 17);
}

#[test]
fn tf32_binary32_and_binary64() {
    let (encodings, pairs) = samples(19, 8, 20_000, 19, &|_| true);
    Formats::<Binary<8>, 19, Tf32>::check(&encodings, pairs.into_iter(), 19);
    let (encodings, pairs) = samples(32, 8, 20_000, 32, &|_| true);
    Formats::<Binary<8>, 32, Single>::check(&encodings, pairs.into_iter(), 32);
    let (encodings, pairs) = samples(64, 11, 20_000, 64, &|_| true);
    Formats::<Binary<11>, 64, Double>::check(&encodings, pairs.into_iter(), 64);
}

#[test]
fn binary128_and_canonical_x87() {
    let (encodings, pairs) = samples(128, 15, 10_000, 128, &|_| true);
    Formats::<Binary<15>, 128, Quad>::check(&encodings, pairs.into_iter(), 128);
    let canonical = |bits: u128| F80::from_bits(bits).is_canonical();
    let (encodings, pairs) = samples(80, 15, 10_000, 80, &canonical);
    Formats::<Binary<15, X87>, 80, X87DoubleExtended>::check(&encodings, pairs.into_iter(), 80);
}
