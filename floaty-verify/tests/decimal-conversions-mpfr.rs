//! Compares the conversions between the decimal formats and the binary
//! formats that the Intel decimal library lacks with MPFR: binary16,
//! bfloat16, TF32, the FP8 formats, binary256, and binary512.
//!
//! - To decimal: MPFR gives the leading digits of the exact binary value,
//!   truncated and rounded away from zero, and the digit after them. The
//!   rounding directions follow from those digits by their IEEE 754
//!   definitions, and IBM's round to prepare for shorter precision for
//!   `ToOdd`. The exponent follows the IEEE 754 preferred exponent rules,
//!   with preferred exponent 0.
//! - To binary: GMP reduces `C * 10^q` to an integer of at least `p + 3`
//!   bits and a sticky bit, and the MPFR oracle rounds that input.
//!
//! The special values follow the conversion rules in `DESIGN.md`. A NaN
//! result must be a quiet NaN with the sign of the source and the payload of
//! the rule in `DESIGN.md`, in both directions: the payload bits align with
//! the trailing significand field of the decimal format and keep their
//! high-order bits, and a decimal payload above `10^(p - 1) - 1` becomes
//! zero. The Intel library confirms that rule for binary32, binary64, x87
//! extended, and binary128. The test computes the payload with GMP from the
//! payload of the source, which floaty's `decode` gives. The binary decoding
//! tests check that payload for the binary formats. For the decimal formats,
//! every decimal operation reads its NaN operands as `decode` does, and the
//! decTest and Intel tests check the payloads of those results bit for bit.

use core::num::NonZeroU32;

use floaty::env::Tininess;
use floaty::{
    BF16, Class, D32Bid, D32Dpd, D64Bid, D64Dpd, D128Bid, D128Dpd, Decoded, Env, Exact, F8E4M3,
    F8E4M3Fnuz, F8E5M2, F8E5M2Fnuz, F16, F256, F512, Flags, Rounding, TF32,
};
use floaty_verify::encodings::{IntegerBit, boundary_encodings, to_limbs, to_u128};
use floaty_verify::mpfr::{self, Format, Input, Specials, Value};
use floaty_verify::random::SplitMix64;
use rug::float::Round;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer};

/// The parameters of a decimal format.
#[derive(Clone, Copy, Debug)]
struct DecimalFormat {
    /// The precision in digits.
    precision: u32,
    /// The largest adjusted exponent.
    emax: i32,
}

impl DecimalFormat {
    /// Returns the width `t` of the trailing significand field, by IEEE
    /// 754-2019 Table 3.6: 20, 50, or 110 bits.
    fn trailing_bits(self) -> u32 {
        match self.precision {
            7 => 20,
            16 => 50,
            34 => 110,
            _ => panic!("a decimal interchange format has 7, 16, or 34 digits"),
        }
    }

    /// Returns the largest canonical NaN payload, `10^(p - 1) - 1`.
    fn largest_payload(self) -> Integer {
        Integer::from(Integer::u_pow_u(10, self.precision - 1)) - 1u32
    }

    /// Returns the smallest and the largest exponent of a full coefficient.
    fn exponents(self) -> (i64, i64) {
        let digits = i64::from(self.precision);
        (
            i64::from(1 - self.emax) - digits + 1,
            i64::from(self.emax) - digits + 1,
        )
    }
}

/// A decimal result in a form that compares across implementations.
#[derive(Clone, Debug, PartialEq, Eq)]
enum DecimalValue {
    /// A zero with its exponent.
    Zero { negative: bool, exponent: i64 },
    /// A nonzero finite value, `coefficient * 10^exponent`.
    Finite {
        negative: bool,
        coefficient: Integer,
        exponent: i64,
    },
    /// An infinity.
    Infinity { negative: bool },
    /// A quiet NaN with its payload.
    Nan { negative: bool, payload: Integer },
}

impl DecimalValue {
    /// Converts a decoded floaty decimal value.
    fn from_decoded(decoded: Decoded<2>) -> Self {
        match decoded {
            Decoded::Zero { negative, exponent } => Self::Zero {
                negative,
                exponent: i64::from(exponent),
            },
            Decoded::Finite {
                negative,
                exponent,
                significand,
            } => Self::Finite {
                negative,
                coefficient: Integer::from_digits(&significand, Order::Lsf),
                exponent: i64::from(exponent),
            },
            Decoded::Infinity { negative } => Self::Infinity { negative },
            Decoded::Nan {
                negative,
                signaling,
                payload,
            } => {
                assert!(!signaling, "a conversion quiets a NaN");
                Self::Nan {
                    negative,
                    payload: Integer::from_digits(&payload, Order::Lsf),
                }
            }
            Decoded::Unsupported => panic!("a decimal format has no unsupported encoding"),
        }
    }
}

/// Where the dropped digits lie against half a unit of the last kept digit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dropped {
    Nothing,
    BelowHalf,
    Half,
    AboveHalf,
}

/// Returns the `digits` leading digits of `value` as an integer, truncated,
/// and `true` when they hold `value` exactly. MPFR writes `|value|` as
/// `0.d1 d2 ... * 10^exponent`; the function also returns that exponent.
fn leading_digits(value: &BigFloat, digits: usize) -> (Integer, i64, bool) {
    let (_, toward, exponent) = value.to_sign_string_exp_round(10, Some(digits), Round::Zero);
    let (_, away, away_exponent) =
        value.to_sign_string_exp_round(10, Some(digits), Round::AwayZero);
    let exponent = exponent.expect("a nonzero finite value has an exponent");
    let integer = Integer::from_str_radix(&toward, 10).expect("MPFR writes decimal digits");
    let exact = toward == away && Some(exponent) == away_exponent;
    (integer, i64::from(exponent), exact)
}

/// Returns the leading `kept_digits` digits of `exact`, truncated, and where
/// the dropped digits lie against half a unit of the last kept digit.
fn cut(exact: &BigFloat, kept_digits: i64) -> (Integer, Dropped) {
    if kept_digits < 0 {
        return (Integer::ZERO, Dropped::BelowHalf);
    }
    if kept_digits == 0 {
        let (first, _, exact_first) = leading_digits(exact, 1);
        let dropped = match (first.to_u32().expect("a digit fits a u32"), exact_first) {
            (5, true) => Dropped::Half,
            (0..=4, _) => Dropped::BelowHalf,
            _ => Dropped::AboveHalf,
        };
        return (Integer::ZERO, dropped);
    }
    let count = usize::try_from(kept_digits).expect("a digit count fits a usize");
    let (kept, _, exact_digits) = leading_digits(exact, count);
    let (longer, _, exact_longer) = leading_digits(exact, count + 1);
    let next = (longer % 10u32).to_u32().expect("a digit fits a u32");
    let dropped = match (exact_digits, next, exact_longer) {
        (true, _, _) => Dropped::Nothing,
        (false, 5, true) => Dropped::Half,
        (false, 0..=4, _) => Dropped::BelowHalf,
        _ => Dropped::AboveHalf,
    };
    (kept, dropped)
}

/// Returns `true` when the rounding direction adds one unit to the kept
/// digits, by the definition of the direction.
fn rounds_up(rounding: Rounding, negative: bool, dropped: Dropped, kept: &Integer) -> bool {
    let inexact = dropped != Dropped::Nothing;
    let last = (kept.clone() % 10u32).to_u32().expect("a digit fits a u32");
    match rounding {
        Rounding::NearestEven => {
            dropped == Dropped::AboveHalf || (dropped == Dropped::Half && last % 2 == 1)
        }
        Rounding::NearestAway => matches!(dropped, Dropped::Half | Dropped::AboveHalf),
        Rounding::TowardPositive => inexact && !negative,
        Rounding::TowardNegative => inexact && negative,
        Rounding::TowardZero => false,
        // IBM's round to prepare for shorter precision.
        Rounding::ToOdd => inexact && (last == 0 || last == 5),
        _ => panic!("the oracle knows every rounding direction"),
    }
}

/// Returns the expected decimal result of converting a nonzero finite
/// binary value, `exact`, and the flags.
fn to_decimal(exact: &BigFloat, format: DecimalFormat, env: &Env) -> (DecimalValue, Flags) {
    let negative = exact.is_sign_negative();
    let precision = env
        .precision
        .map_or(format.precision, |limit| limit.get().min(format.precision));
    let emin = i64::from(1 - format.emax);
    let (full_lowest, full_highest) = format.exponents();
    let (_, top_plus_one, _) = leading_digits(exact, 1);
    let top = top_plus_one - 1;
    let position = (top - i64::from(precision) + 1).max(emin - i64::from(precision) + 1);
    let (kept, dropped) = cut(exact, top - position + 1);
    let inexact = dropped != Dropped::Nothing;
    let step = rounds_up(env.rounding, negative, dropped, &kept);
    let (mut coefficient, mut exponent) = (kept + u32::from(step), position);
    if coefficient == Integer::from(Integer::u_pow_u(10, precision)) {
        coefficient /= 10u32;
        exponent += 1;
    }
    let tiny = top < emin;
    let mut flags = Flags::NONE;
    let zero = DecimalValue::Zero {
        negative,
        exponent: full_lowest,
    };
    if tiny {
        flags |= Flags::TINY;
        if env.flush_to_zero {
            return (zero, flags | Flags::UNDERFLOW | Flags::INEXACT);
        }
    }
    if inexact {
        flags |= Flags::INEXACT;
        if tiny {
            flags |= Flags::UNDERFLOW;
        }
    }
    if step {
        flags |= Flags::ROUNDED_UP;
    }
    if coefficient == 0 {
        return (zero, flags);
    }
    let digits = |value: &Integer| i64::try_from(value.to_string().len()).expect("a count fits");
    if exponent + digits(&coefficient) - 1 > i64::from(format.emax) {
        return overflow(negative, precision, format, env);
    }
    let full = i64::from(format.precision);
    if inexact {
        // An inexact result has the least possible exponent.
        while exponent > full_lowest && digits(&coefficient) < full {
            coefficient *= 10u32;
            exponent -= 1;
        }
    } else {
        // An exact result has the exponent nearest the preferred exponent 0.
        while exponent < 0 && coefficient.is_divisible_u(10) {
            coefficient /= 10u32;
            exponent += 1;
        }
        while exponent > 0 && digits(&coefficient) < full {
            coefficient *= 10u32;
            exponent -= 1;
        }
    }
    assert!(
        exponent <= full_highest,
        "the value is below the overflow bound"
    );
    let finite = DecimalValue::Finite {
        negative,
        coefficient,
        exponent,
    };
    (finite, flags)
}

/// Returns the result of an overflow: an infinity, or the largest value of
/// `precision` digits in the directions that round toward zero.
fn overflow(
    negative: bool,
    precision: u32,
    format: DecimalFormat,
    env: &Env,
) -> (DecimalValue, Flags) {
    let flags = Flags::OVERFLOW | Flags::INEXACT;
    let to_infinity = match env.rounding {
        Rounding::NearestEven | Rounding::NearestAway => true,
        Rounding::TowardPositive => !negative,
        Rounding::TowardNegative => negative,
        _ => false,
    };
    if to_infinity {
        return (
            DecimalValue::Infinity { negative },
            flags | Flags::ROUNDED_UP,
        );
    }
    let full = format.precision;
    let coefficient = Integer::from(Integer::u_pow_u(10, full))
        - Integer::from(Integer::u_pow_u(10, full - precision));
    let (_, highest) = format.exponents();
    let largest = DecimalValue::Finite {
        negative,
        coefficient,
        exponent: highest,
    };
    (largest, flags)
}

/// Aligns the high-order bits of a field of `from` bits with a field of `to`
/// bits: a wider field gains low-order zeros, and a narrower field drops the
/// low-order bits.
fn align(payload: Integer, from: u32, to: u32) -> Integer {
    if to >= from {
        payload << (to - from)
    } else {
        payload >> (from - to)
    }
}

/// Returns the payload of a decimal NaN converted from a binary NaN with
/// `payload`, a field of `bits` bits, by the rule in `DESIGN.md`.
fn decimal_payload(payload: Integer, bits: u32, format: DecimalFormat) -> Integer {
    let aligned = align(payload, bits, format.trailing_bits());
    if aligned > format.largest_payload() {
        Integer::ZERO
    } else {
        aligned
    }
}

/// Returns the payload of a binary NaN converted from a decimal value, by
/// the rule in `DESIGN.md`. The payload value of a decimal NaN is a field of
/// `trailing` bits. The binary payload is the fraction below the quiet bit.
/// A format with one NaN encoding has no payload, and neither has a NaN that
/// a finite value or an infinity gives.
fn binary_payload(source: &Decoded<2>, trailing: u32, format: &Format) -> Integer {
    let Decoded::Nan { payload, .. } = *source else {
        return Integer::ZERO;
    };
    match format.specials {
        Specials::Ieee => align(
            Integer::from_digits(&payload, Order::Lsf),
            trailing,
            format.precision - 2,
        ),
        Specials::NoInf | Specials::Fnuz => Integer::ZERO,
    }
}

/// Returns the expected result of converting a decoded binary value to a
/// decimal format, by the conversion rules in `DESIGN.md`. A NaN payload of
/// the source is a field of `payload_bits` bits.
fn binary_to_decimal<const N: usize>(
    source: &Decoded<N>,
    subnormal: bool,
    payload_bits: u32,
    format: DecimalFormat,
    env: &Env,
) -> (DecimalValue, Flags) {
    let input = if subnormal {
        Flags::DENORMAL_INPUT
    } else {
        Flags::NONE
    };
    match *source {
        Decoded::Zero { negative, .. } => (
            DecimalValue::Zero {
                negative,
                exponent: 0,
            },
            Flags::NONE,
        ),
        Decoded::Finite { negative, .. } if subnormal && env.denormals_are_zero => (
            DecimalValue::Zero {
                negative,
                exponent: 0,
            },
            input,
        ),
        Decoded::Finite {
            negative,
            exponent,
            significand,
        } => {
            let integer = Integer::from_digits(&significand, Order::Lsf);
            let bits = integer.significant_bits();
            let mut exact = BigFloat::with_val(bits, integer) << exponent;
            if negative {
                exact = -exact;
            }
            let (value, flags) = to_decimal(&exact, format, env);
            (value, flags | input)
        }
        Decoded::Infinity { negative } => (DecimalValue::Infinity { negative }, Flags::NONE),
        Decoded::Nan {
            negative,
            signaling,
            payload,
        } => {
            let flags = if signaling {
                Flags::INVALID
            } else {
                Flags::NONE
            };
            let payload = Integer::from_digits(&payload, Order::Lsf);
            let payload = decimal_payload(payload, payload_bits, format);
            (DecimalValue::Nan { negative, payload }, flags)
        }
        Decoded::Unsupported => panic!("the sources have no unsupported encoding"),
    }
}

/// Returns the expected result of converting a decoded decimal value to a
/// binary format.
fn decimal_to_binary(
    source: &Decoded<2>,
    subnormal: bool,
    format: &Format,
    env: &Env,
) -> (Value, Flags) {
    let Decoded::Finite {
        negative,
        exponent,
        significand,
    } = *source
    else {
        return mpfr::convert(source, subnormal, format, env);
    };
    if subnormal && env.denormals_are_zero {
        return mpfr::convert(source, subnormal, format, env);
    }
    let coefficient = Integer::from_digits(&significand, Order::Lsf);
    let input = if exponent >= 0 {
        Input {
            negative,
            exponent,
            significand: coefficient * Integer::from(Integer::u_pow_u(5, exponent.unsigned_abs())),
            sticky: false,
        }
    } else {
        // C * 10^-k = (C * 2^t / 5^k) * 2^(-k - t), with a quotient of at
        // least p + 3 bits.
        let divisor = Integer::from(Integer::u_pow_u(5, exponent.unsigned_abs()));
        let room = format.precision + 3 + divisor.significant_bits();
        let shift = room.saturating_sub(coefficient.significant_bits());
        let (quotient, remainder) = (coefficient << shift).div_rem(divisor);
        Input {
            negative,
            exponent: exponent - i32::try_from(shift).expect("a shift fits an i32"),
            significand: quotient,
            sticky: remainder != 0,
        }
    };
    let (value, flags) = mpfr::round(&input, format, env);
    let flags = if subnormal {
        flags | Flags::DENORMAL_INPUT
    } else {
        flags
    };
    (value, flags)
}

/// The behaviors of a conversion to a decimal format. The tininess rule and
/// `saturate` do not apply to a decimal destination.
fn decimal_behaviors() -> [Env; 7] {
    [
        Env::IEEE,
        Env::IEEE.with_rounding(Rounding::TowardZero),
        Env::IEEE
            .with_rounding(Rounding::TowardPositive)
            .with_flush_to_zero(true),
        Env::IEEE.with_rounding(Rounding::ToOdd),
        Env::IEEE
            .with_rounding(Rounding::NearestAway)
            .with_denormals_are_zero(true),
        Env::IEEE
            .with_rounding(Rounding::TowardNegative)
            .with_precision(NonZeroU32::new(3)),
        Env::IEEE.with_precision(NonZeroU32::new(1)),
    ]
}

/// The behaviors of a conversion to a binary format.
fn binary_behaviors() -> [Env; 7] {
    [
        Env::IEEE,
        Env::IEEE
            .with_rounding(Rounding::TowardZero)
            .with_tininess(Tininess::BeforeRounding),
        Env::IEEE
            .with_rounding(Rounding::TowardPositive)
            .with_flush_to_zero(true),
        Env::IEEE.with_rounding(Rounding::ToOdd).with_saturate(true),
        Env::IEEE
            .with_rounding(Rounding::NearestAway)
            .with_denormals_are_zero(true),
        Env::IEEE
            .with_rounding(Rounding::TowardNegative)
            .with_precision(NonZeroU32::new(3)),
        Env::IEEE
            .with_saturate(true)
            .with_precision(NonZeroU32::new(2)),
    ]
}

/// Converts every binary source value to every decimal destination in every
/// behavior.
macro_rules! check_to_decimal {
    ($source:ty, $values:expr => $($destination:ty),+) => {{
        // The fraction bits below the quiet bit. A format with one NaN
        // encoding decodes a zero payload, so its width does not matter.
        let payload_bits = <$source>::PRECISION - 2;
        for env in decimal_behaviors() {
            for &bits in &$values {
                let source = <$source>::from_bits(bits);
                let decoded = source.decode::<8>();
                let subnormal = source.classify() == Class::Subnormal;
                $(
                    let format = DecimalFormat {
                        precision: <$destination>::PRECISION,
                        emax: <$destination>::EMAX,
                    };
                    let (ours, flags): ($destination, _) = source.convert_with(env);
                    assert!(ours.is_canonical(), "{ours:?} is canonical");
                    let ours = (DecimalValue::from_decoded(ours.decode::<2>()), flags);
                    let expected =
                        binary_to_decimal(&decoded, subnormal, payload_bits, format, &env);
                    let context = format!(
                        "{} {bits:x?} to {} {env:?}",
                        stringify!($source),
                        stringify!($destination)
                    );
                    assert_eq!(ours, expected, "{context}");
                )+
            }
        }
    }};
}

/// The oracle parameters of a binary destination.
macro_rules! oracle_format {
    ($alias:ty, $specials:expr) => {
        Format {
            precision: <$alias>::PRECISION,
            emin: <$alias>::EMIN,
            emax: <$alias>::EMAX,
            specials: $specials,
        }
    };
}

/// Converts every decimal source value to every binary destination in every
/// behavior. A NaN result must be quiet, with the payload of
/// [`binary_payload`].
macro_rules! check_to_binary {
    ($source:ty, $values:expr => $($destination:ty: $specials:expr),+) => {{
        let trailing = DecimalFormat {
            precision: <$source>::PRECISION,
            emax: <$source>::EMAX,
        }
        .trailing_bits();
        for env in binary_behaviors() {
            for source in &$values {
                let source: $source = *source;
                let decoded = source.decode::<2>();
                let subnormal = source.classify() == Class::Subnormal;
                $(
                    let format = oracle_format!($destination, $specials);
                    let (ours, flags): ($destination, _) = source.convert_with(env);
                    assert!(ours.is_canonical(), "{ours:?} is canonical");
                    let result = ours.decode::<8>();
                    let ours = (Value::from_decoded(result), flags);
                    let expected = decimal_to_binary(&decoded, subnormal, &format, &env);
                    let context = format!(
                        "{} {:x?} to {} {env:?}",
                        stringify!($source),
                        source.to_bits(),
                        stringify!($destination)
                    );
                    assert_eq!(ours, expected, "{context}");
                    if let Decoded::Nan { signaling, payload, .. } = result {
                        assert!(!signaling, "{context}: a conversion quiets a NaN");
                        assert_eq!(
                            Integer::from_digits(&payload, Order::Lsf),
                            binary_payload(&decoded, trailing, &format),
                            "{context}: the NaN payload"
                        );
                    }
                )+
            }
        }
    }};
}

/// Returns decimal values at the edges of the coefficient range, at every
/// `stride`-th exponent and at both ends of the exponent range, random
/// values, and the special encodings.
macro_rules! decimal_values {
    ($alias:ty, $stride:expr, $random:expr, $seed:expr, $specials:expr) => {{
        let precision = <$alias>::PRECISION;
        let emax = i64::from(<$alias>::EMAX);
        let digits = i64::from(precision);
        let (lowest, highest) = (2 - emax - digits, emax - digits + 1);
        let largest = Integer::from(Integer::u_pow_u(10, precision)) - 1u32;
        let mut random = SplitMix64::new($seed);
        let mut coefficients: Vec<Integer> = vec![
            Integer::from(1),
            Integer::from(5),
            Integer::from(Integer::u_pow_u(10, precision - 1)),
            largest.clone(),
            largest.clone() / 2u32,
        ];
        coefficients.extend((0..3).map(|_| Integer::from(random.next_u128()) % &largest));
        let mut exponents: Vec<i64> = (lowest..=highest).step_by($stride).collect();
        exponents.extend([lowest, lowest + 1, highest - 1, highest, -1, 0, 1]);
        let make = |negative: bool, coefficient: &Integer, exponent: i64| {
            let mut significand = [0_u64; 2];
            for (slot, limb) in significand
                .iter_mut()
                .zip(coefficient.to_digits::<u64>(Order::Lsf))
            {
                *slot = limb;
            }
            let exact = Exact {
                negative,
                exponent: i32::try_from(exponent).expect("an exponent of the format fits an i32"),
                significand,
                sticky: false,
            };
            <$alias>::round(exact, Env::IEEE).0
        };
        let mut values = Vec::new();
        for &exponent in &exponents {
            for coefficient in &coefficients {
                values.push(make(false, coefficient, exponent));
                values.push(make(true, coefficient, exponent));
            }
        }
        for _ in 0..$random {
            let coefficient = Integer::from(random.next_u128()) % &largest;
            let span = u64::try_from(highest - lowest + 1).expect("the range is positive");
            let exponent =
                lowest + i64::try_from(random.next_u64() % span).expect("the offset fits");
            values.push(make(random.next_u64() & 1 == 1, &coefficient, exponent));
        }
        values.extend($specials.map(<$alias>::from_bits));
        values
    }};
}

/// The infinities, and quiet and signaling NaNs of both signs. A NaN has a
/// zero or small payload, the top bit of the trailing field, the largest
/// canonical BID payload `10^(p - 1) - 1`, the non-canonical BID payload
/// `10^(p - 1)`, alternate bits, or the bits between the signaling bit and
/// the trailing field. A DPD encoding reads the trailing field as declets,
/// so the same bits have another payload value there.
const DECIMAL32_SPECIALS: [u32; 9] = [
    0x7800_0000,
    0xF800_0000,
    0x7C00_0000,
    0xFE00_0005,
    0x7C08_0000,
    0xFE0F_423F,
    0x7C0F_4240,
    0x7DF0_0001,
    0xFE05_5555,
];
const DECIMAL64_SPECIALS: [u64; 9] = [
    0x7800_0000_0000_0000,
    0xF800_0000_0000_0000,
    0x7C00_0000_0000_0000,
    0xFE00_0000_0000_0005,
    0x7C02_0000_0000_0000,
    0xFE03_8D7E_A4C6_7FFF,
    0x7C03_8D7E_A4C6_8000,
    0x7DFC_0000_0000_0001,
    0xFE01_5555_5555_5555,
];
const DECIMAL128_SPECIALS: [u128; 9] = [
    0x7800 << 112,
    0xF800 << 112,
    0x7C00 << 112,
    (0xFE00 << 112) | 5,
    (0x7C00 << 112) | (1 << 109),
    (0xFE00 << 112) | (10_u128.pow(33) - 1),
    (0x7C00 << 112) | 10_u128.pow(33),
    (0x7C00 << 112) | (0x7FF << 110) | 1,
    (0xFE00 << 112) | ((u128::MAX >> 18) / 3),
];

#[test]
fn from_the_fp8_formats_to_decimal() {
    let every: Vec<u8> = (0..=u8::MAX).collect();
    check_to_decimal!(F8E4M3, every => D32Bid, D64Dpd, D128Bid);
    check_to_decimal!(F8E5M2, every => D32Dpd, D64Bid, D128Dpd);
    check_to_decimal!(F8E4M3Fnuz, every => D32Bid, D64Bid);
    check_to_decimal!(F8E5M2Fnuz, every => D32Dpd, D128Bid);
}

#[test]
fn from_binary16_and_bfloat16_to_decimal() {
    let every: Vec<u16> = (0..=u16::MAX).collect();
    check_to_decimal!(F16, every => D32Bid, D64Dpd);
    check_to_decimal!(BF16, every => D32Dpd, D64Bid);
}

#[test]
fn from_the_wide_formats_to_decimal() {
    let mut random = SplitMix64::new(0xDEC_0512);
    let tf32: Vec<u32> = boundary_encodings(19, 8, IntegerBit::Implicit)
        .iter()
        .map(|encoding| u32::try_from(to_u128(encoding)).expect("19 bits"))
        .chain((0..5_000).map(|_| u32::try_from(random.next_u64() >> 45).expect("19 bits")))
        .collect();
    check_to_decimal!(TF32, tf32 => D32Bid, D128Dpd);
    let mut wide: Vec<[u64; 8]> = boundary_encodings(512, 23, IntegerBit::Implicit)
        .iter()
        .map(to_limbs::<8>)
        .collect();
    wide.extend((0..1_000).map(|_| core::array::from_fn(|_| random.next_u64())));
    check_to_decimal!(F512, wide => D32Dpd, D64Bid, D128Bid);
    let mut narrow: Vec<[u64; 4]> = boundary_encodings(256, 19, IntegerBit::Implicit)
        .iter()
        .map(to_limbs::<4>)
        .collect();
    narrow.extend((0..1_000).map(|_| core::array::from_fn(|_| random.next_u64())));
    check_to_decimal!(F256, narrow => D32Bid, D64Dpd, D128Dpd);
}

/// Returns `count` random encodings of a binary interchange format of
/// `width` bits with `exponent_bits` exponent bits, in `N` limbs. Each has a
/// random sign and random bits in the whole fraction. Its exponent is in
/// the range of `format`, from the smallest subnormal value to the largest
/// value, or up to 16 binades beyond either end.
fn in_decimal_range<const N: usize>(
    random: &mut SplitMix64,
    width: u32,
    exponent_bits: u32,
    format: DecimalFormat,
    count: usize,
) -> Vec<[u64; N]> {
    // 10^q is about 2^(q * log2(10)), and log2(10) is about 3.3219281.
    let binary = |q: i64| (q * 33_219_281).div_euclid(10_000_000);
    let (lowest, _) = format.exponents();
    let low = binary(lowest) - 16;
    let span = binary(i64::from(format.emax) + 1) + 16 - low + 1;
    let fraction_bits = width - 1 - exponent_bits;
    let bias = (1_i64 << (exponent_bits - 1)) - 1;
    (0..count)
        .map(|_| {
            let offset = random.next_u64() % span.unsigned_abs();
            let exponent = low + i64::try_from(offset).expect("the offset fits an i64");
            let limbs: [u64; N] = core::array::from_fn(|_| random.next_u64());
            let fraction = Integer::from_digits(&limbs, Order::Lsf).keep_bits(fraction_bits);
            let sign = Integer::from(random.next_u64() & 1) << (width - 1);
            let field = Integer::from(bias + exponent) << fraction_bits;
            to_limbs::<N>(&(sign | field | fraction))
        })
        .collect()
}

/// The parameters of the decimal format `$alias`.
macro_rules! decimal_format {
    ($alias:ty) => {
        DecimalFormat {
            precision: <$alias>::PRECISION,
            emax: <$alias>::EMAX,
        }
    };
}

/// The number of random binary256 and binary512 values in the range of each
/// decimal format.
const IN_RANGE: usize = 3_000;

#[test]
fn from_the_wide_formats_in_the_decimal_ranges_to_decimal() {
    let mut random = SplitMix64::new(0xDEC_0256);
    let (wide, narrow) = (
        in_decimal_range::<8>(&mut random, 512, 23, decimal_format!(D32Bid), IN_RANGE),
        in_decimal_range::<4>(&mut random, 256, 19, decimal_format!(D32Bid), IN_RANGE),
    );
    check_to_decimal!(F512, wide => D32Bid, D32Dpd);
    check_to_decimal!(F256, narrow => D32Dpd, D32Bid);
    let (wide, narrow) = (
        in_decimal_range::<8>(&mut random, 512, 23, decimal_format!(D64Bid), IN_RANGE),
        in_decimal_range::<4>(&mut random, 256, 19, decimal_format!(D64Bid), IN_RANGE),
    );
    check_to_decimal!(F512, wide => D64Dpd, D64Bid);
    check_to_decimal!(F256, narrow => D64Bid, D64Dpd);
    let (wide, narrow) = (
        in_decimal_range::<8>(&mut random, 512, 23, decimal_format!(D128Bid), IN_RANGE),
        in_decimal_range::<4>(&mut random, 256, 19, decimal_format!(D128Bid), IN_RANGE),
    );
    check_to_decimal!(F512, wide => D128Bid, D128Dpd);
    check_to_decimal!(F256, narrow => D128Dpd, D128Bid);
}

/// Returns the decimal values of format `$alias` nearest the random
/// binary256 and binary512 values in the range of that format, as floaty
/// converts them. The oracle checks the conversions back to binary, so floaty
/// only picks the inputs here.
macro_rules! near_wide_values {
    ($alias:ty, $random:expr) => {{
        let format = decimal_format!($alias);
        let wide = in_decimal_range::<8>($random, 512, 23, format, IN_RANGE);
        let narrow = in_decimal_range::<4>($random, 256, 19, format, IN_RANGE);
        let from_wide = wide
            .into_iter()
            .map(|bits| F512::from_bits(bits).convert::<$alias>());
        let from_narrow = narrow
            .into_iter()
            .map(|bits| F256::from_bits(bits).convert::<$alias>());
        from_wide.chain(from_narrow).collect::<Vec<$alias>>()
    }};
}

#[test]
fn from_decimal_values_near_wide_binary_values() {
    let mut random = SplitMix64::new(0xDEC_0257);
    let values = near_wide_values!(D32Dpd, &mut random);
    check_to_binary!(D32Dpd, values => F256: Specials::Ieee, F512: Specials::Ieee);
    let values = near_wide_values!(D64Bid, &mut random);
    check_to_binary!(D64Bid, values => F256: Specials::Ieee, F512: Specials::Ieee);
    let values = near_wide_values!(D128Dpd, &mut random);
    check_to_binary!(D128Dpd, values => F256: Specials::Ieee, F512: Specials::Ieee);
}

#[test]
fn from_decimal32_to_binary() {
    let values: Vec<D32Bid> = decimal_values!(D32Bid, 1, 2_000, 0xD32, DECIMAL32_SPECIALS);
    check_to_binary!(D32Bid, values =>
        F16: Specials::Ieee, BF16: Specials::Ieee, TF32: Specials::Ieee,
        F8E4M3: Specials::NoInf, F8E5M2: Specials::Ieee,
        F8E4M3Fnuz: Specials::Fnuz, F8E5M2Fnuz: Specials::Fnuz,
        F256: Specials::Ieee, F512: Specials::Ieee);
}

#[test]
fn from_decimal64_to_binary() {
    let values: Vec<D64Dpd> = decimal_values!(D64Dpd, 5, 2_000, 0xD64, DECIMAL64_SPECIALS);
    check_to_binary!(D64Dpd, values =>
        F16: Specials::Ieee, BF16: Specials::Ieee, F8E4M3: Specials::NoInf,
        F8E5M2Fnuz: Specials::Fnuz, F256: Specials::Ieee, F512: Specials::Ieee);
}

#[test]
fn from_decimal128_to_binary() {
    let values: Vec<D128Bid> = decimal_values!(D128Bid, 97, 500, 0xD128, DECIMAL128_SPECIALS);
    check_to_binary!(D128Bid, values =>
        F16: Specials::Ieee, BF16: Specials::Ieee, F8E5M2: Specials::Ieee,
        F256: Specials::Ieee, F512: Specials::Ieee);
    let dpd: Vec<D128Dpd> = decimal_values!(D128Dpd, 389, 100, 0xD128D, DECIMAL128_SPECIALS);
    check_to_binary!(D128Dpd, dpd => F8E4M3Fnuz: Specials::Fnuz, TF32: Specials::Ieee);
}
