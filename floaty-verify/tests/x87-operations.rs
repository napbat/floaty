//! Compares the compares, rounding to an integral value, the conversions
//! between floats and integers, the partial remainder, scaling, the sign
//! operations, `FXTRACT`, the packed BCD loads and stores, and the constant
//! loads with the x87 unit of the host processor. The test runs only on
//! x86-64 hosts.
//!
//! Each test maps a floaty result to the instruction as a consumer does. A
//! mapping cites the Intel SDM Volume 1, revision 253665-093US, or states
//! what the hardware shows.

#![cfg(target_arch = "x86_64")]

use core::ops::RangeInclusive;

use floaty::{Env, F80, F128, Flags, QuotientBits, Rounding, ToInt};
use floaty_verify::encodings::{Layout, boundary_encodings, integer_edges, to_u128};
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::{
    self, X87_C1, X87_STATUS_FLAGS, stored, x87_arithmetic_status, x87_env, x87_status,
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

/// Returns an encoding from its sign, biased exponent, and significand.
fn encode(negative: bool, biased: u64, significand: u64) -> u128 {
    Layout::X87_EXTENDED.encode(negative, biased, u128::from(significand))
}

/// Returns the biased exponent of the unbiased exponent `exponent`.
fn biased(exponent: i32) -> u64 {
    u64::try_from(exponent + Layout::X87_EXTENDED.ieee_bias())
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
            let negative = random.coin_flip();
            let fraction = random.next_u64() >> 1;
            let (exponent, integer) = match index % 4 {
                0 => (random.below(0x8000), 1 << 63),
                1 => (random.below(80), 1 << 63),
                2 => (near_start + random.below(near_span), 1 << 63),
                _ => (random.below(0x8000), random.next_u64() & (1 << 63)),
            };
            encode(negative, exponent, integer | fraction)
        })
        .collect()
}

/// Returns the specials, the edges around the integers up to 2^66, and
/// random values.
fn integral_operands(random: &mut SplitMix64) -> Vec<u128> {
    let mut operands = SPECIALS.to_vec();
    operands.extend(integer_edges::<u128>(Layout::X87_EXTENDED, random, -3..=66));
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
    for &[a, second] in operands.as_chunks::<2>().0 {
        for b in [second, a, a ^ (1 << 79), a ^ 1] {
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
    for setting in x86::x87_settings() {
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
        for setting in x86::x87_settings() {
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
        let value = random.next_u64() >> random.below(64);
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
        for setting in x86::x87_settings() {
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
        let difference = random.below(140);
        let divisor_field = (field + 8).saturating_sub(difference).min(0x7FFE);
        let significand = random.next_u64() | (1 << 63);
        let negative = random.coin_flip();
        pairs.push((dividend, encode(negative, divisor_field, significand)));
    }
    for _ in 0..4_000 {
        let [dividend, divisor] = [0; 2].map(|_| {
            let significand = random.next_u64() | (random.next_u64() & (1 << 63));
            encode(random.coin_flip(), random.below(70), significand)
        });
        pairs.push((dividend, divisor));
    }
    for _ in 0..300 {
        let dividend = encode(
            false,
            0x4000 + random.below(0x3FFF),
            random.next_u64() | (1 << 63),
        );
        let divisor = encode(true, random.below(0x100), random.next_u64() | (1 << 63));
        pairs.push((dividend, divisor));
    }
    for _ in 0..2_000 {
        // The divisor m * 2^e and the dividend m * q * 2^(e - 1) for an odd q:
        // the quotient is q / 2, halfway between two integers. m * q has at
        // most 62 bits, so both values are exact.
        let multiple = (random.next_u64() >> 17) | (1 << 46);
        let odd = (random.next_u64() >> 49) | 1;
        let exponent =
            i32::try_from(random.below(400)).expect("an offset below 400 fits an i32") - 200;
        let divisor = F80::from_int(multiple).scale_b(exponent);
        let dividend = F80::from_int(multiple * odd).scale_b(exponent - 1);
        pairs.push((dividend.to_bits(), divisor.to_bits()));
    }
    pairs
}

/// The status bits that hold the low three bits of the quotient after
/// `FPREM1` and `FPREM`: C0 holds bit 2, C3 bit 1, and C1 bit 0 (Intel SDM
/// Volume 1, Table 8-1 on page 8-5).
const QUOTIENT: u16 = C0 | C3 | C1;
const C0: u16 = 1 << 8;
const C3: u16 = 1 << 14;

/// Returns C0, C3, and C1 for the low bits of a quotient.
fn quotient_status(quotient: QuotientBits) -> u16 {
    let bits = quotient.low_bits();
    [(4, C0), (2, C3), (1, C1)]
        .into_iter()
        .filter(|&(bit, _)| bits & bit != 0)
        .fold(0, |status, (_, flag)| status | flag)
}

/// The values of C0 and C3 before a remainder instruction: neither, and both.
const PRESETS: [u16; 2] = [0, C0 | C3];

/// A remainder of floaty with the low bits of its quotient, and the flags.
type Recipe = fn(F80, F80, Env) -> (F80, QuotientBits, Flags);

/// A remainder instruction of the harness, which runs `FPREM1` or `FPREM`
/// until C2 is clear or for a number of steps.
type Instruction = fn(u128, u128, u16, u16, u64) -> (u128, u16);

/// Compares `instruction`, repeated until C2 is clear, with `recipe` on every
/// pair in every setting, from each preset of C0 and C3. A NaN result keeps
/// the preset, as [`fprem1_matches_the_ieee_remainder`] states.
fn check_remainders(name: &str, instruction: Instruction, recipe: Recipe, seed: u64) {
    let mut random = SplitMix64::new(seed);
    let pairs = remainder_pairs(&mut random);
    for setting in x86::x87_settings() {
        for &(a, b) in &pairs {
            let (x, y) = (F80::from_bits(a), F80::from_bits(b));
            let (ours, quotient, flags) = recipe(x, y, setting.env());
            for preset in PRESETS {
                let (expected, status) =
                    instruction(a, b, setting.control, preset, x86::PARTIAL_REMAINDER_STEPS);
                let context = format!(
                    "{name} {a:#x} {b:#x} {setting:?} {flags:?} preset {preset:#06x} status {status:#06x}"
                );
                assert_eq!(status & C2, 0, "{context}: the reduction is complete");
                assert_eq!(ours.to_bits(), expected & MASK, "{context}: result");
                assert_eq!(
                    x87_arithmetic_status(flags, x.is_nan() || y.is_nan()),
                    status & CHECKED & !C1,
                    "{context}: flags"
                );
                let bits = if ours.is_nan() {
                    preset
                } else {
                    quotient_status(quotient)
                };
                assert_eq!(status & QUOTIENT, bits, "{context}: quotient");
            }
        }
    }
}

/// `FPREM1` is the IEEE 754 remainder; the harness repeats it until C2 is
/// clear. The remainder is exact, so the rounding direction and precision
/// control do not apply, and PE, OE, and ZE never appear. C0, C3, and C1
/// then hold the low bits of the whole quotient, as `remainder_quotient_with`
/// gives them, also after partial steps. A zero dividend and an infinite
/// divisor give the bits 0.
///
/// Hardware rule: a NaN result, from a NaN operand or an invalid operation,
/// keeps C0 and C3 as earlier instructions left them, and C1 is clear. The
/// SDM Volume 2A, revision 325383-084US, at `FPREM1` and `FPREM`, sets C0,
/// C3, and C1 from the quotient, which such an operation does not have.
/// floaty gives the bits 0 for a NaN result, so a consumer keeps C0 and C3.
/// The harness sets C0 and C3 before the instruction, and the test requires
/// that they keep those values.
///
/// A tiny remainder is exact. With the underflow exception masked, the
/// processor reports UE only for a tiny inexact result (Intel SDM Volume 1,
/// section 4.9.1.5 on page 4-23), and the hardware shows no UE for a tiny
/// remainder. floaty reports `TINY` for it, which `x87_status` does not map:
/// a consumer maps it to UE only when the exception is unmasked.
#[test]
fn fprem1_matches_the_ieee_remainder() {
    let recipe: Recipe = |x, y, env| x.remainder_quotient_with(y, env);
    check_remainders("fprem1", x86::fprem1, recipe, 0x0087_2E31);
}

/// `FPREM` is the remainder of the quotient truncated toward zero, the
/// remainder of C `fmod`; the harness repeats it until C2 is clear. The
/// flags and the quotient bits follow the rules of
/// [`fprem1_matches_the_ieee_remainder`].
#[test]
fn fprem_matches_the_truncated_remainder() {
    let recipe: Recipe = |x, y, env| x.truncated_remainder_quotient_with(y, env);
    check_remainders("fprem", x86::fprem, recipe, 0x0087_2E30);
}

/// Returns the exponent field of an 80-bit encoding.
fn exponent_field(bits: u128) -> i32 {
    i32::try_from((bits >> 64) & 0x7FFF).expect("the x87 exponent field has 15 bits")
}

/// Returns pairs of normal values whose exponents differ by less than 200,
/// by 60 to 69, and by anything in the range.
fn step_pairs(random: &mut SplitMix64) -> Vec<(u128, u128)> {
    (0..6_000)
        .map(|index| {
            let field = 0x0100 + random.below(0x7E00);
            let difference = match index % 3 {
                0 => random.below(200),
                1 => 60 + random.below(10),
                _ => random.below(0x7E00),
            };
            let divisor_field = field.saturating_sub(difference).max(1);
            let mut normal =
                |field| encode(random.coin_flip(), field, random.next_u64() | (1 << 63));
            (normal(field), normal(divisor_field))
        })
        .collect()
}

/// One step of `FPREM1` or `FPREM` reduces completely when the exponents
/// differ by D < 64 (Intel SDM Volume 2A, `FPREM1` and `FPREM`), as
/// [`fprem1_matches_the_ieee_remainder`] checks. Otherwise the step is
/// partial for both instructions: it leaves `x - y * q * 2^(D - N)`, with
/// `q` the integer part of `x / (y * 2^(D - N))`. It sets C2 and clears C0,
/// C3, and C1. A consumer builds the partial step as the truncated
/// remainder by the divisor scaled with `scale_b`, which is exact. Precision
/// control does not apply, and the step reports no flag for normal
/// operands.
///
/// The SDM leaves N to the implementation, from 32 to 63. The hardware uses
/// N = 32 + (D mod 32), as Bochs 3.0 does (`bochs/cpu/fpu/fprem.cc`, line
/// 125). So a partial step lowers the exponent difference by a multiple of
/// 32.
#[test]
fn one_partial_remainder_step_reduces_by_a_multiple_of_32() {
    let mut random = SplitMix64::new(0x0087_2E32);
    let pairs = step_pairs(&mut random);
    let nearest: Recipe = |x, y, env| x.remainder_quotient_with(y, env);
    let truncated: Recipe = |x, y, env| x.truncated_remainder_quotient_with(y, env);
    let instructions: [(&str, Instruction, Recipe); 2] = [
        ("fprem1", x86::fprem1, nearest),
        ("fprem", x86::fprem, truncated),
    ];
    for setting in x86::x87_settings() {
        let env = setting.env();
        for &(a, b) in &pairs {
            let (x, y) = (F80::from_bits(a), F80::from_bits(b));
            let difference = exponent_field(a) - exponent_field(b);
            for (name, instruction, complete) in instructions {
                let (ours, bits) = if difference < 64 {
                    let (remainder, quotient, _) = complete(x, y, env);
                    (remainder, quotient_status(quotient))
                } else {
                    let scale = difference - (32 + difference % 32);
                    (x.truncated_remainder_with(y.scale_b(scale), env).0, C2)
                };
                for preset in PRESETS {
                    let (result, status) = instruction(a, b, setting.control, preset, 1);
                    let context = format!(
                        "{name} {a:#x} {b:#x} {setting:?} preset {preset:#06x} status {status:#06x}"
                    );
                    assert_eq!(ours.to_bits(), result & MASK, "{context}: result");
                    assert_eq!(status & (QUOTIENT | C2), bits, "{context}: C3 to C0");
                    assert_eq!(status & CHECKED & !C1, 0, "{context}: flags");
                }
            }
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
        let exponent = field.max(1) - Layout::X87_EXTENDED.ieee_bias();
        let offset = i32::try_from(random.below(80)).expect("an offset below 80 fits an i32");
        let scale = match random.below(4) {
            0 => -16_382 - 70 + offset - exponent,
            1 => 16_383 - 8 + offset / 8 - exponent,
            2 => i32::try_from(random.below(401)).expect("an offset below 401 fits an i32") - 200,
            _ => {
                i32::try_from(random.below(140_001)).expect("an offset below 140,001 fits an i32")
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
    for setting in x86::x87_settings() {
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
    let mut encodings: Vec<u128> = boundary_encodings(Layout::X87_EXTENDED)
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

/// `FXTRACT` leaves the exponent in ST(1) and the significand in ST(0)
/// (Intel SDM Volume 2A, `FXTRACT`). A consumer builds the exponent with
/// `log_b_with`, and the significand with `scale_b_with` by the negated
/// exponent, or by zero when the exponent is not finite. So a zero gives an
/// exponent of negative infinity with ZE and keeps its sign, and a denormal
/// gives the exponent of its normalized value. Precision control does not
/// affect `FXTRACT`, and the hardware confirms it.
#[test]
fn fxtract_matches_log_b_and_the_scaled_significand() {
    let mut random = SplitMix64::new(0x0087_E7AC);
    let mut operands: Vec<u128> = boundary_encodings(Layout::X87_EXTENDED)
        .iter()
        .map(to_u128)
        .collect();
    operands.extend(SPECIALS);
    operands.extend(random_operands(&mut random, 20_000, -16_382..=-16_300));
    operands.extend((0..2_000).map(|_| {
        let significand = (random.next_u64() >> 1) >> random.below(63);
        encode(random.coin_flip(), 0, significand)
    }));
    for setting in x86::x87_settings() {
        let env = setting.full_precision();
        for &input in &operands {
            let (significand, exponent, status) = x86::fxtract(input, setting.control);
            let value = F80::from_bits(input);
            let (ours_exponent, exponent_flags) = value.log_b_with(env);
            let scale = match ours_exponent.to_int_with::<i32>(env).0 {
                ToInt::Value(exponent) => -exponent,
                ToInt::OutOfRange { .. } | ToInt::Nan => 0,
            };
            let (ours_significand, significand_flags) = value.scale_b_with(scale, env);
            let flags = exponent_flags | significand_flags;
            let context = format!("fxtract {input:#x} {setting:?} {flags:?} status {status:#06x}");
            assert_eq!(ours_exponent.to_bits(), exponent, "{context}: exponent");
            assert_eq!(
                ours_significand.to_bits(),
                significand,
                "{context}: significand"
            );
            assert_eq!(
                x87_arithmetic_status(flags, value.is_nan()),
                status & CHECKED,
                "{context}: flags"
            );
        }
    }
}

/// The largest magnitude of a packed BCD integer: 18 nines.
const BCD_LARGEST: u64 = 999_999_999_999_999_999;

/// The packed BCD indefinite that `FBSTP` stores for an invalid operation
/// (Intel SDM Volume 1, Table 4-4 on page 4-12).
const BCD_INDEFINITE: u128 = 0xFFFF_C000_0000_0000_0000;

/// Returns the packed BCD encoding of `magnitude`, at most [`BCD_LARGEST`]:
/// two digits in each of the low nine bytes, the low digit first, and the
/// sign in bit 79.
fn bcd(negative: bool, magnitude: u64) -> u128 {
    let (digits, _) = (0..18).fold((0_u128, magnitude), |(digits, rest), index| {
        (digits | (u128::from(rest % 10) << (4 * index)), rest / 10)
    });
    digits | (u128::from(negative) << 79)
}

/// `FBLD` converts a packed BCD integer exactly, keeps the sign of a zero,
/// and reports no flag (Intel SDM Volume 2A, `FBLD`). A consumer reads the
/// digits and converts the magnitude with `from_int_with`, which is exact
/// below 10^18 under every setting.
#[test]
fn fbld_matches_exact_conversions_from_bcd_integers() {
    let mut random = SplitMix64::new(0x0087_FB1D);
    let mut magnitudes: Vec<u64> = integer_operands(&mut random, 64)
        .into_iter()
        .map(|value| value.unsigned_abs() % (BCD_LARGEST + 1))
        .collect();
    magnitudes.extend([
        0,
        1,
        9,
        10,
        99,
        BCD_LARGEST,
        BCD_LARGEST - 1,
        BCD_LARGEST / 9,
    ]);
    for setting in x86::x87_settings() {
        let env = setting.full_precision();
        for &magnitude in &magnitudes {
            for negative in [false, true] {
                let input = bcd(negative, magnitude);
                let (expected, status) = x86::fbld(input, setting.control);
                let (ours, flags) = F80::from_int_with(magnitude, env);
                let ours = if negative { -ours } else { ours };
                let context = format!("fbld {input:#x} {setting:?} {flags:?} status {status:#06x}");
                assert_eq!(ours.to_bits(), expected, "{context}: result");
                assert_eq!(x87_status(flags), status & CHECKED, "{context}: flags");
            }
        }
    }
}

/// Returns the operands of `FBSTP`: the integral operands, and the integers
/// near the largest packed BCD magnitude with the halfway values beside
/// them, for both signs.
fn bcd_store_operands(random: &mut SplitMix64) -> Vec<u128> {
    let mut operands = integral_operands(random);
    for magnitude in [BCD_LARGEST - 1, BCD_LARGEST, BCD_LARGEST + 1, 1, 0] {
        let twice = i64::try_from(2 * magnitude).expect("twice the magnitude fits an i64");
        for doubled in [twice - 1, twice, twice + 1] {
            for sign in [1, -1] {
                operands.push(F80::from_int(sign * doubled).scale_b(-1).to_bits());
            }
        }
    }
    operands
}

/// `FBSTP` rounds as the rounding-control field selects and stores the
/// packed BCD integer with the sign of the operand (Intel SDM Volume 2A,
/// `FBSTP`). A consumer rounds with `to_int_with` and takes a magnitude
/// above [`BCD_LARGEST`] as out of range. A NaN, an infinity, an unsupported
/// encoding, and an operand out of range store [`BCD_INDEFINITE`] with IE
/// alone. The flags follow `FISTP`: no DE, and C1 for a rounding that grew
/// the magnitude.
#[test]
fn fbstp_matches_conversions_to_bcd_integers() {
    let mut random = SplitMix64::new(0x0087_FB57);
    let operands = bcd_store_operands(&mut random);
    for setting in x86::x87_settings() {
        let env = setting.full_precision();
        for &input in &operands {
            let (expected, status) = x86::fbstp(input, setting.control);
            let value = F80::from_bits(input);
            let (rounded, flags) = value.to_int_with::<i64>(env);
            let (ours, flags) = match rounded {
                ToInt::Value(integer) if integer.unsigned_abs() <= BCD_LARGEST => {
                    (bcd(value.is_sign_negative(), integer.unsigned_abs()), flags)
                }
                _ => (BCD_INDEFINITE, Flags::INVALID),
            };
            let context =
                format!("fbstp {input:#x} {setting:?} {rounded:?} {flags:?} status {status:#06x}");
            assert_eq!(ours, expected, "{context}: result");
            assert_eq!(
                integer_store_status(flags),
                status & CHECKED,
                "{context}: flags"
            );
        }
    }
}

/// Returns `constant` rounded to nearest at 113 bits, as a binary128 value.
fn binary128_constant(constant: &rug::Float) -> F128 {
    let (significand, exponent) = rug::Float::with_val(113, constant)
        .to_integer_exp()
        .expect("a constant is finite");
    let significand = significand
        .to_u128()
        .expect("a 113-bit significand fits a u128");
    F128::from_int(significand).scale_b(exponent)
}

/// An instruction that pushes a constant under a control word.
type Load = fn(u16) -> (u128, u16);

/// The constant instructions round the constant as the rounding-control
/// field selects, and report neither PE nor C1 (Intel SDM Volume 2A,
/// `FLD1/FLDL2T/FLDL2E/FLDPI/FLDLG2/FLDLN2/FLDZ`). A consumer holds each
/// constant in binary128 and converts it with `convert_with`, without
/// precision control, which the instructions ignore, and drops the flags.
/// The constants are computed by MPFR at 256 bits. Bits 65 to 113 of none
/// of them are a tie or exact, so the binary128 value rounds to 64 bits as
/// the true constant does.
///
/// Conflict: the Intel SDM Volume 2A, revision 325383-084US, at the same
/// instruction, states that each instruction rounds an internal 66-bit
/// constant. log10 2 rounded to nearest at 66 bits is a tie at 64 bits:
/// its bits 65 and 66 are 10, and the next bits are 0011111. Ties to even
/// gives the significand `9A20_9A84_FBCF_F798`, but the hardware gives
/// `9A20_9A84_FBCF_F799`, the true constant rounded to nearest. The same
/// page states that round to nearest gives the constants of the 8087.
/// Resolution: the test rounds the true constant, and the hardware
/// confirms the rule for each constant in each direction.
#[test]
fn constants_round_the_true_constant() {
    use rug::float::Constant;
    let wide = |value| rug::Float::with_val(256, value);
    let constants: [(&str, Load, F128); 7] = [
        ("fld1", x86::fld1, F128::from_int(1)),
        ("fldz", x86::fldz, F128::from_int(0)),
        ("fldpi", x86::fldpi, binary128_constant(&wide(Constant::Pi))),
        (
            "fldl2t",
            x86::fldl2t,
            binary128_constant(&rug::Float::with_val(256, 10).log2()),
        ),
        (
            "fldl2e",
            x86::fldl2e,
            binary128_constant(&wide(Constant::Log2).recip()),
        ),
        (
            "fldlg2",
            x86::fldlg2,
            binary128_constant(&rug::Float::with_val(256, 2).log10()),
        ),
        (
            "fldln2",
            x86::fldln2,
            binary128_constant(&wide(Constant::Log2)),
        ),
    ];
    for setting in x86::x87_settings() {
        for (instruction, load, constant) in constants {
            let (expected, status) = load(setting.control);
            let (ours, _): (F80, _) = constant.convert_with(setting.full_precision());
            let context = format!("{instruction} {setting:?} status {status:#06x}");
            assert_eq!(ours.to_bits(), expected, "{context}: result");
            assert_eq!(status & CHECKED, 0, "{context}: flags");
        }
    }
}
