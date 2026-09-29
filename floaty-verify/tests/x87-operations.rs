//! Compares the compares, rounding to an integral value, the conversions
//! between floats and integers, the partial remainder, scaling, and the sign
//! operations with the x87 unit of the host processor. The test runs only on
//! x86-64 hosts.
//!
//! Each test maps a floaty result to the instruction as a consumer does. A
//! mapping cites the Intel SDM Volume 1, revision 253665-093US, or states
//! what the hardware shows.

#![cfg(target_arch = "x86_64")]

use core::num::NonZeroU32;
use core::ops::RangeInclusive;

use floaty::{Env, F80, Flags, Rounding, ToInt};
use floaty_verify::encodings::{IntegerBit, boundary_encodings, to_u128};
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::{
    self, X87_C1, X87_MASKED, X87_PRECISIONS, X87_ROUNDINGS, X87_STATUS_FLAGS,
    x87_arithmetic_status, x87_env, x87_status,
};

/// The control word after `FNINIT`: every exception masked, rounding to
/// nearest, and 64-bit precision.
const DEFAULT_CONTROL: u16 = 0x037F;

/// The status bits that the comparisons check: IE, DE, ZE, OE, UE, PE, and C1.
const CHECKED: u16 = X87_STATUS_FLAGS;
const C1: u16 = X87_C1;
const C2: u16 = 1 << 10;

/// The bits of an 80-bit encoding.
const MASK: u128 = (1 << 80) - 1;

/// One x87 control setting of the tests.
#[derive(Clone, Copy, Debug)]
struct Setting {
    /// The control word.
    control: u16,
    /// The direction of the rounding-control field.
    rounding: Rounding,
    /// The limit of the precision-control field.
    precision: Option<NonZeroU32>,
}

impl Setting {
    /// Returns the behavior with the limit of the precision-control field.
    fn env(self) -> Env {
        x87_env(self.rounding).with_precision(self.precision)
    }

    /// Returns the behavior of an instruction that precision control does
    /// not affect. Precision control affects only the add, subtract,
    /// multiply, divide, and square root instructions (Intel SDM Volume 1,
    /// section 8.1.5.2 on page 8-8).
    fn full_precision(self) -> Env {
        x87_env(self.rounding)
    }
}

/// Every x87 control setting: each rounding direction at each precision.
fn settings() -> Vec<Setting> {
    let mut settings = Vec::new();
    for (rounding, rounding_field) in X87_ROUNDINGS {
        for (precision, precision_field) in X87_PRECISIONS {
            settings.push(Setting {
                control: X87_MASKED | rounding_field | precision_field,
                rounding,
                precision: NonZeroU32::new(precision),
            });
        }
    }
    settings
}

/// The x87 special operands: zeros, infinities, quiet and signaling NaNs of
/// both signs, the real indefinite, unsupported encodings, pseudo-denormals,
/// denormals, the normal boundaries, values near one and near halfway, and
/// the powers of two at the edge of the 64-bit integers.
const SPECIALS: [u128; 30] = [
    0x0000_0000_0000_0000_0000,
    0x8000_0000_0000_0000_0000,
    0x7FFF_8000_0000_0000_0000,
    0xFFFF_8000_0000_0000_0000,
    0x7FFF_C000_0000_0000_0001,
    0xFFFF_C000_0000_0000_0009,
    0xFFFF_C000_0000_0000_0000,
    0x7FFF_8000_0000_0000_0002,
    0xFFFF_A000_0000_0000_0000,
    // A pseudo-NaN, a pseudo-infinity, and two unnormals: unsupported.
    0x7FFF_4000_0000_0000_0000,
    0xFFFF_0000_0000_0000_0000,
    0x3FFF_0000_0000_0000_0001,
    0x4000_0000_0000_0000_0000,
    // Pseudo-denormals.
    0x0000_8000_0000_0000_0003,
    0x8000_8000_0000_0000_0000,
    0x0000_0000_0000_0000_0001,
    0x8000_7FFF_FFFF_FFFF_FFFF,
    0x0001_8000_0000_0000_0000,
    0x7FFE_FFFF_FFFF_FFFF_FFFF,
    0xFFFE_FFFF_FFFF_FFFF_FFFF,
    0x3FFF_8000_0000_0000_0000,
    0xBFFF_8000_0000_0000_0000,
    0x3FFF_8000_0000_0000_0001,
    0x3FFE_FFFF_FFFF_FFFF_FFFF,
    0x3FFE_8000_0000_0000_0000,
    0xBFFE_C000_0000_0000_0000,
    0x4000_A000_0000_0000_0000,
    0xC000_C000_0000_0000_0000,
    0x403E_8000_0000_0000_0000,
    0xC03E_8000_0000_0000_0000,
];

/// The exponent bias of the x87 format.
const BIAS: i32 = 16383;

/// Returns an encoding from its sign, biased exponent, and significand.
fn encode(negative: bool, biased: u64, significand: u64) -> u128 {
    (u128::from(negative) << 79) | (u128::from(biased) << 64) | u128::from(significand)
}

/// Returns the biased exponent of the unbiased exponent `exponent`.
fn biased(exponent: i32) -> u64 {
    u64::try_from(exponent + BIAS)
        .expect("an exponent in the x87 range has a biased field that is not negative")
}

/// Returns `count` random encodings: a quarter across the whole range, a
/// quarter near the smallest normal value, a quarter with unbiased exponents
/// in `near`, and a quarter across the whole range with a random integer
/// bit, which gives unsupported encodings and pseudo-denormals.
fn random_operands(random: &mut SplitMix64, count: usize, near: RangeInclusive<i32>) -> Vec<u128> {
    let near_start = biased(*near.start());
    let near_span = biased(*near.end()) - near_start + 1;
    (0..count)
        .map(|index| {
            let negative = random.next_u64() & 1 == 1;
            let fraction = random.next_u64() >> 1;
            let (exponent, integer) = match index % 4 {
                0 => (random.next_u64() % 0x8000, 1 << 63),
                1 => (random.next_u64() % 80, 1 << 63),
                2 => (near_start + random.next_u64() % near_span, 1 << 63),
                _ => (random.next_u64() % 0x8000, random.next_u64() & (1 << 63)),
            };
            encode(negative, exponent, integer | fraction)
        })
        .collect()
}

/// Returns the patterns of `dropped` low bits that decide a rounding: exact,
/// just below halfway, halfway, just above halfway, and all ones.
fn edge_patterns(dropped: u32) -> [u64; 5] {
    if dropped == 0 {
        return [0; 5];
    }
    let half = 1_u64 << (dropped - 1);
    let mask = (half << 1).wrapping_sub(1);
    [0, half - 1, half, half + 1, mask].map(|pattern| pattern & mask)
}

/// Returns encodings around the integers with the unbiased exponents
/// `exponents`, for both signs.
///
/// The fraction bits below the binary point take each edge pattern, and the
/// bits above the binary point are all zeros, all ones, or random. A value
/// below one has every fraction bit below the binary point.
fn integer_edges(random: &mut SplitMix64, exponents: RangeInclusive<i32>) -> Vec<u128> {
    let mut encodings = Vec::new();
    for exponent in exponents {
        let dropped = u32::try_from((63 - exponent).clamp(0, 63))
            .expect("a shift clamped to 0..=63 fits a u32");
        let all_ones = (1_u64 << (63 - dropped)) - 1;
        for high in [0, all_ones, random.next_u64(), random.next_u64()] {
            for pattern in edge_patterns(dropped) {
                for negative in [false, true] {
                    let significand = (1 << 63) | ((high & all_ones) << dropped) | pattern;
                    encodings.push(encode(negative, biased(exponent), significand));
                }
            }
        }
    }
    encodings
}

/// Returns the specials, the edges around the integers up to 2^66, and
/// random values.
fn integral_operands(random: &mut SplitMix64) -> Vec<u128> {
    let mut operands = SPECIALS.to_vec();
    operands.extend(integer_edges(random, -3..=66));
    operands.extend(random_operands(random, 20_000, -3..=66));
    operands
}

/// Compares two operands with the four compare instructions and floaty.
///
/// `FUCOMIP` and `FUCOMPP` are the quiet compares, and `FCOMIP` and `FCOMPP`
/// the signaling ones (Intel SDM Volume 1, section 8.3.6 on page 8-19). C1 is
/// clear after a compare (Table 8-1 on page 8-5). C3, C2, and C0 are
/// undefined after `FUCOMIP` and `FCOMIP`, so the harness reads the order
/// from EFLAGS there.
fn check_compare(a: u128, b: u128) {
    let (x, y) = (F80::from_bits(a), F80::from_bits(b));
    let env = x87_env(Rounding::TiesToEven);
    let nan_operand = x.is_nan() || y.is_nan();
    let quiet = x.compare_quiet_with(y, env);
    let signaling = x.compare_signaling_with(y, env);
    for (instruction, (order, flags), (expected, status)) in [
        ("fucomip", quiet, x86::fucomip(a, b, DEFAULT_CONTROL)),
        ("fcomip", signaling, x86::fcomip(a, b, DEFAULT_CONTROL)),
        ("fucompp", quiet, x86::fucompp(a, b, DEFAULT_CONTROL)),
        ("fcompp", signaling, x86::fcompp(a, b, DEFAULT_CONTROL)),
    ] {
        let context = format!("{instruction} {a:#x} {b:#x} {flags:?} status {status:#06x}");
        assert_eq!(order, expected, "{context}: order");
        // The hardware shows the DE precedence of the arithmetic
        // instructions for the compares too.
        assert_eq!(
            x87_arithmetic_status(flags, nan_operand),
            status & CHECKED,
            "{context}: flags"
        );
    }
}

#[test]
fn compares_match_the_quiet_and_signaling_compares() {
    let mut random = SplitMix64::new(0x0087_C0A5);
    for &a in &SPECIALS {
        for &b in &SPECIALS {
            check_compare(a, b);
        }
    }
    let operands = random_operands(&mut random, 40_000, -2..=2);
    for pair in operands.chunks_exact(2) {
        let a = pair[0];
        for b in [pair[1], a, a ^ (1 << 79), a ^ 1] {
            check_compare(a, b);
        }
        // A pseudo-denormal equals the normal encoding with exponent field 1
        // (Intel SDM Volume 1, section 8.2.2 on page 8-14).
        let pseudo = a & !(0x7FFF << 64) | (1 << 63);
        check_compare(pseudo, pseudo | (1 << 64));
    }
}

/// `FRNDINT` rounds in the rounding-control direction and ignores precision
/// control, as floaty ignores the precision limit. So the test passes the
/// limit of each precision-control setting to floaty too.
#[test]
fn frndint_matches_rounding_to_an_integral_value() {
    let mut random = SplitMix64::new(0x0087_12D1);
    let operands = integral_operands(&mut random);
    for setting in settings() {
        for &input in &operands {
            let (expected, status) = x86::frndint(input, setting.control);
            let value = F80::from_bits(input);
            let (ours, flags) = value.round_to_integral_with(setting.env());
            let context = format!("frndint {input:#x} {setting:?} {flags:?} status {status:#06x}");
            assert_eq!(ours.to_bits(), expected & MASK, "{context}: result");
            assert_eq!(
                x87_arithmetic_status(flags, value.is_nan()),
                status & CHECKED,
                "{context}: flags"
            );
        }
    }
}

/// Returns the integer that `FISTP` or `FISTTP` stores: the value, or the
/// integer indefinite for a value out of range or a NaN (Intel SDM Volume 1,
/// Table 8-10 on page 8-27). The indefinite is the smallest integer, which
/// has only the sign bit set.
fn stored<I: Copy>(result: ToInt<I>, indefinite: I) -> I {
    match result {
        ToInt::Value(value) => value,
        ToInt::OutOfRange { .. } | ToInt::Nan => indefinite,
    }
}

/// Returns the status bits of `FISTP` or `FISTTP`: IE, PE, and C1 for a
/// rounding that grew the magnitude. The stores never report DE (Intel SDM
/// Volume 1, Table C-2 on page C-2, and the hardware shows it), so the
/// consumer drops `DENORMAL_INPUT`.
fn integer_store_status(flags: Flags) -> u16 {
    x87_status(flags.difference(Flags::DENORMAL_INPUT))
}

/// Compares an integer store that rounds in the rounding-control direction,
/// and one that truncates, with floaty in every setting.
macro_rules! integer_store {
    ($operands:expr, $integer:ty, $rounded:path, $truncated:path) => {
        for setting in settings() {
            for &input in &$operands {
                let value = F80::from_bits(input);
                for (instruction, rounding, (expected, status)) in [
                    (
                        stringify!($rounded),
                        setting.rounding,
                        $rounded(input, setting.control),
                    ),
                    (
                        stringify!($truncated),
                        Rounding::TowardZero,
                        $truncated(input, setting.control),
                    ),
                ] {
                    let env = setting.env().with_rounding(rounding);
                    let (ours, flags) = value.to_int_with::<$integer>(env);
                    let context = format!(
                        "{instruction} {input:#x} {setting:?} {ours:?} {flags:?} status {status:#06x}"
                    );
                    assert_eq!(stored(ours, <$integer>::MIN), expected, "{context}: result");
                    assert_eq!(
                        integer_store_status(flags),
                        status & CHECKED,
                        "{context}: flags"
                    );
                }
            }
        }
    };
}

#[test]
fn fistp_and_fisttp_match_conversions_to_integers() {
    assert!(
        std::arch::is_x86_feature_detected!("sse3"),
        "the FISTTP test needs a host with SSE3"
    );
    let mut random = SplitMix64::new(0x0087_F157);
    let operands = integral_operands(&mut random);
    integer_store!(operands, i16, x86::fistp_m16, x86::fisttp_m16);
    integer_store!(operands, i32, x86::fistp_m32, x86::fisttp_m32);
    integer_store!(operands, i64, x86::fistp_m64, x86::fisttp_m64);
}

/// Returns 64-bit integers of every bit length up to `bits`, for both signs,
/// with random bits below the top bit, the extremes, and random values.
fn integer_operands(random: &mut SplitMix64, bits: u32) -> Vec<i64> {
    let largest = i64::MAX >> (64 - bits);
    let mut operands = vec![0, 1, -1, largest, -largest, -largest - 1];
    for length in 1..bits {
        for _ in 0..8 {
            let below = (random.next_u64() >> 1) >> (64 - length);
            let magnitude = (1 << (length - 1)) | below;
            let magnitude = i64::try_from(magnitude).expect("a magnitude below 2^63 fits an i64");
            operands.extend([magnitude, -magnitude]);
        }
    }
    for _ in 0..20_000 {
        let value = random.next_u64() >> (random.next_u64() % 64);
        operands.push(i64::from_le_bytes(value.to_le_bytes()) >> (64 - bits));
    }
    operands
}

/// Compares an integer load with floaty in every setting.
///
/// `FILD` is exact and ignores precision control, so floaty converts without
/// the precision limit, which it would apply otherwise. `FILD` reports no
/// flag (Intel SDM Volume 1, Table C-2 on page C-2), and C1 is clear.
macro_rules! integer_load {
    ($operands:expr, $instruction:path) => {
        for setting in settings() {
            let env = setting.full_precision();
            for &input in &$operands {
                let (expected, status) = $instruction(input, setting.control);
                let (ours, flags) = F80::from_int_with(input, env);
                let context = format!(
                    "{} {input} {setting:?} {flags:?} status {status:#06x}",
                    stringify!($instruction)
                );
                assert_eq!(ours.to_bits(), expected, "{context}: result");
                assert_eq!(x87_status(flags), status & CHECKED, "{context}: flags");
            }
        }
    };
}

#[test]
fn fild_matches_exact_conversions_from_integers() {
    let mut random = SplitMix64::new(0x0087_F11D);
    let words: Vec<i16> = (i16::MIN..=i16::MAX).collect();
    integer_load!(words, x86::fild_m16);
    let doublewords: Vec<i32> = integer_operands(&mut random, 32)
        .into_iter()
        .map(|value| i32::try_from(value).expect("the value fits an i32"))
        .collect();
    integer_load!(doublewords, x86::fild_m32);
    let quadwords = integer_operands(&mut random, 64);
    integer_load!(quadwords, x86::fild_m64);
}

/// Returns dividend and divisor pairs for the remainder: every pair of
/// specials, random pairs with a moderate exponent difference, pairs near the
/// subnormal range, pairs with a large exponent difference, and pairs whose
/// quotient is halfway between two integers.
fn remainder_pairs(random: &mut SplitMix64) -> Vec<(u128, u128)> {
    let mut pairs = Vec::new();
    for &a in &SPECIALS {
        for &b in &SPECIALS {
            pairs.push((a, b));
        }
    }
    let operands = random_operands(random, 20_000, -2..=2);
    for &dividend in &operands {
        let field =
            u64::try_from((dividend >> 64) & 0x7FFF).expect("the x87 exponent field has 15 bits");
        let difference = random.next_u64() % 140;
        let divisor_field = (field + 8).saturating_sub(difference).min(0x7FFE);
        let significand = random.next_u64() | (1 << 63);
        let negative = random.next_u64() & 1 == 1;
        pairs.push((dividend, encode(negative, divisor_field, significand)));
    }
    for _ in 0..4_000 {
        let [dividend, divisor] = [0; 2].map(|_| {
            let significand = random.next_u64() | (random.next_u64() & (1 << 63));
            encode(
                random.next_u64() & 1 == 1,
                random.next_u64() % 70,
                significand,
            )
        });
        pairs.push((dividend, divisor));
    }
    for _ in 0..300 {
        let dividend = encode(
            false,
            0x4000 + random.next_u64() % 0x3FFF,
            random.next_u64() | (1 << 63),
        );
        let divisor = encode(
            true,
            random.next_u64() % 0x100,
            random.next_u64() | (1 << 63),
        );
        pairs.push((dividend, divisor));
    }
    for _ in 0..2_000 {
        // The divisor m * 2^e and the dividend m * q * 2^(e - 1) for an odd q:
        // the quotient is q / 2, halfway between two integers. m * q has at
        // most 62 bits, so both values are exact.
        let multiple = (random.next_u64() >> 17) | (1 << 46);
        let odd = (random.next_u64() >> 49) | 1;
        let exponent =
            i32::try_from(random.next_u64() % 400).expect("an offset below 400 fits an i32") - 200;
        let divisor = F80::from_int(multiple).scale_b(exponent);
        let dividend = F80::from_int(multiple * odd).scale_b(exponent - 1);
        pairs.push((dividend.to_bits(), divisor.to_bits()));
    }
    pairs
}

/// `FPREM1` is the IEEE 754 remainder; the harness repeats it until C2 is
/// clear. The remainder is exact, so the rounding direction and precision
/// control do not apply, and PE, OE, and ZE never appear.
///
/// A tiny remainder is exact. With the underflow exception masked, the
/// processor reports UE only for a tiny inexact result (Intel SDM Volume 1,
/// section 4.9.1.5 on page 4-23), and the hardware shows no UE for a tiny
/// remainder. floaty reports `TINY` for it, which `x87_status` does not map:
/// a consumer maps it to UE only when the exception is unmasked. C1 holds a
/// quotient bit after `FPREM1` (Table 8-1 on page 8-5), so the test does not
/// compare it.
#[test]
fn fprem1_matches_the_ieee_remainder() {
    let mut random = SplitMix64::new(0x0087_2E31);
    let pairs = remainder_pairs(&mut random);
    for setting in settings() {
        for &(a, b) in &pairs {
            let (expected, status) = x86::fprem1(a, b, setting.control);
            let (x, y) = (F80::from_bits(a), F80::from_bits(b));
            let (ours, flags) = x.remainder_with(y, setting.env());
            let context =
                format!("fprem1 {a:#x} {b:#x} {setting:?} {flags:?} status {status:#06x}");
            assert_eq!(status & C2, 0, "{context}: the reduction is complete");
            assert_eq!(ours.to_bits(), expected & MASK, "{context}: result");
            assert_eq!(
                x87_arithmetic_status(flags, x.is_nan() || y.is_nan()),
                status & CHECKED & !C1,
                "{context}: flags"
            );
        }
    }
}

/// Returns value and scale pairs for `FSCALE`: every special with small and
/// large scales, and random values with a scale that lands near the subnormal
/// range, near the overflow threshold, or anywhere.
fn scale_pairs(random: &mut SplitMix64) -> Vec<(u128, i32)> {
    let scales = [
        0, 1, -1, 2, -2, 63, -63, 64, -64, 16_383, -16_445, 40_000, -40_000,
    ];
    let mut pairs = Vec::new();
    for &value in &SPECIALS {
        pairs.extend(scales.map(|scale| (value, scale)));
    }
    for value in random_operands(random, 40_000, -2..=2) {
        let field =
            i32::try_from((value >> 64) & 0x7FFF).expect("the x87 exponent field has 15 bits");
        let exponent = field.max(1) - BIAS;
        let offset = i32::try_from(random.next_u64() % 80).expect("an offset below 80 fits an i32");
        let scale = match random.next_u64() % 4 {
            0 => -16_382 - 70 + offset - exponent,
            1 => 16_383 - 8 + offset / 8 - exponent,
            2 => {
                i32::try_from(random.next_u64() % 401).expect("an offset below 401 fits an i32")
                    - 200
            }
            _ => {
                i32::try_from(random.next_u64() % 140_001)
                    .expect("an offset below 140,001 fits an i32")
                    - 70_000
            }
        };
        pairs.push((value, scale));
    }
    pairs
}

/// `FSCALE` ignores precision control (Intel SDM Volume 1, section 8.1.5.2
/// on page 8-8), and the hardware confirms it: `FSCALE` keeps 64 bits under
/// every precision-control setting. floaty's `scale_b_with` applies the
/// precision limit of the behavior, so a consumer that emulates `FSCALE`
/// passes the behavior without it.
#[test]
fn fscale_matches_scale_b_without_precision_control() {
    let mut random = SplitMix64::new(0x0087_5CA1);
    let pairs = scale_pairs(&mut random);
    for setting in settings() {
        let env = setting.full_precision();
        for &(input, scale) in &pairs {
            let (expected, status) = x86::fscale(input, scale, setting.control);
            let value = F80::from_bits(input);
            let (ours, flags) = value.scale_b_with(scale, env);
            let context =
                format!("fscale {input:#x} {scale} {setting:?} {flags:?} status {status:#06x}");
            assert_eq!(ours.to_bits(), expected & MASK, "{context}: result");
            assert_eq!(
                x87_arithmetic_status(flags, value.is_nan()),
                status & CHECKED,
                "{context}: flags"
            );
        }
    }
}

/// `FCHS` and `FABS` change only the sign bit, for every encoding including
/// the unsupported ones, and report no flag (Intel SDM Volume 1, Table C-2
/// on pages C-1 and C-2). C1 is clear after both.
#[test]
fn fchs_and_fabs_change_only_the_sign_of_every_encoding() {
    let mut random = SplitMix64::new(0x0087_5165);
    let mut encodings: Vec<u128> = boundary_encodings(80, 15, IntegerBit::Explicit)
        .iter()
        .map(to_u128)
        .collect();
    encodings.extend(SPECIALS);
    encodings.extend((0..100_000).map(|_| random.next_u128() & MASK));
    for &input in &encodings {
        let value = F80::from_bits(input);
        for (instruction, ours, (expected, status)) in [
            ("fchs", -value, x86::fchs(input, DEFAULT_CONTROL)),
            ("fabs", value.abs(), x86::fabs(input, DEFAULT_CONTROL)),
        ] {
            let context = format!("{instruction} {input:#x} status {status:#06x}");
            assert_eq!(ours.to_bits(), expected, "{context}: result");
            assert_eq!(status & CHECKED, 0, "{context}: flags");
        }
    }
}
