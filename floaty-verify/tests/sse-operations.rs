//! Compares the compares, the conversions between floats and integers, and the
//! rounding to an integral value with the SSE unit of the host processor, in
//! every MXCSR setting. The test runs only on x86-64 hosts.
//!
//! Each test maps a floaty result to the instruction as a consumer does. A
//! mapping cites the Intel SDM Volume 1, revision 253665-093US, or states
//! what the hardware shows.

#![cfg(target_arch = "x86_64")]

use core::ops::{Range, RangeInclusive};

use floaty::{F32, F64, Flags, Rounding};
use floaty_verify::encodings::Layout;
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::{
    self, MXCSR_ROUNDINGS, ROUND_USE_MXCSR, mxcsr_flags, mxcsr_round_flags, stored,
};

/// Returns the largest biased exponent of binary32 or binary64: the field of
/// the infinities and NaNs.
fn field_max(layout: Layout) -> u64 {
    (1 << layout.exponent_bits) - 1
}

/// Returns a binary32 or binary64 encoding from its sign, biased exponent,
/// and fraction.
fn encode(layout: Layout, negative: bool, biased: u64, fraction: u64) -> u64 {
    u64::try_from(layout.encode(negative, biased, u128::from(fraction)))
        .expect("a binary32 or binary64 encoding fits a u64")
}

/// Returns a random encoding with a biased exponent in `biased` and a random
/// sign.
fn random_encoding(layout: Layout, random: &mut SplitMix64, biased: Range<u64>) -> u64 {
    let exponent = biased.start + random.next_u64() % (biased.end - biased.start);
    let negative = random.next_u64() & 1 == 1;
    encode(layout, negative, exponent, random.next_u64())
}

/// Returns the biased exponent of the unbiased exponent `exponent`.
fn biased(layout: Layout, exponent: i32) -> u64 {
    u64::try_from(exponent + layout.ieee_bias()).expect("a normal exponent")
}

/// Returns encodings around the integers with the unbiased exponents
/// `exponents`, for both signs.
///
/// The fraction bits below the binary point take each edge pattern, and the
/// bits above the binary point are all zeros, all ones, or random. A value
/// below one has every fraction bit below the binary point.
fn integer_edges(
    layout: Layout,
    random: &mut SplitMix64,
    exponents: RangeInclusive<i32>,
) -> Vec<u64> {
    let mut encodings = Vec::new();
    for exponent in exponents {
        let fraction_bits = i32::try_from(layout.fraction_bits()).expect("at most 52");
        let dropped = u32::try_from((fraction_bits - exponent).clamp(0, fraction_bits))
            .expect("between 0 and the fraction width");
        let kept = layout.fraction_bits() - dropped;
        let all_ones = (1 << kept) - 1;
        for high in [0, all_ones, random.next_u64(), random.next_u64()] {
            for pattern in edge_patterns(dropped) {
                for negative in [false, true] {
                    let fraction = ((high & all_ones) << dropped) | pattern;
                    encodings.push(encode(layout, negative, biased(layout, exponent), fraction));
                }
            }
        }
    }
    encodings
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

/// The binary32 special operands: zeros, infinities, quiet and signaling
/// NaNs of both signs, the subnormal and normal boundaries, and values near
/// one.
const SINGLE_SPECIALS: [u32; 22] = [
    0x0000_0000,
    0x8000_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0001,
    0xFFC0_0002,
    0x7F80_0003,
    0xFFA0_0004,
    0x0000_0001,
    0x8000_0001,
    0x007F_FFFF,
    0x807F_FFFF,
    0x0080_0000,
    0x8080_0000,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x3F80_0000,
    0xBF80_0000,
    0x3F80_0001,
    0x3F7F_FFFF,
    0x3F00_0000,
    0xBFC0_0000,
];

/// The binary64 special operands, as for binary32.
const DOUBLE_SPECIALS: [u64; 22] = [
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x7FF0_0000_0000_0000,
    0xFFF0_0000_0000_0000,
    0x7FF8_0000_0000_0001,
    0xFFF8_0000_0000_0002,
    0x7FF0_0000_0000_0003,
    0xFFF4_0000_0000_0004,
    0x0000_0000_0000_0001,
    0x8000_0000_0000_0001,
    0x000F_FFFF_FFFF_FFFF,
    0x800F_FFFF_FFFF_FFFF,
    0x0010_0000_0000_0000,
    0x8010_0000_0000_0000,
    0x7FEF_FFFF_FFFF_FFFF,
    0xFFEF_FFFF_FFFF_FFFF,
    0x3FF0_0000_0000_0000,
    0xBFF0_0000_0000_0000,
    0x3FF0_0000_0000_0001,
    0x3FEF_FFFF_FFFF_FFFF,
    0x3FE0_0000_0000_0000,
    0xBFF8_0000_0000_0000,
];

/// Returns `count` random encodings: a third across the whole range, a third
/// near the subnormal range, and a third with exponents in `near`, unbiased.
fn random_operands(
    layout: Layout,
    random: &mut SplitMix64,
    count: usize,
    near: RangeInclusive<i32>,
) -> Vec<u64> {
    let near = biased(layout, *near.start())..biased(layout, *near.end()) + 1;
    (0..count)
        .map(|index| match index % 3 {
            0 => random_encoding(layout, random, 0..field_max(layout) + 1),
            1 => random_encoding(layout, random, 0..3),
            _ => random_encoding(layout, random, near.clone()),
        })
        .collect()
}

/// Returns binary32 encodings from wider bits.
fn singles(encodings: impl IntoIterator<Item = u64>) -> Vec<u32> {
    encodings
        .into_iter()
        .map(|bits| u32::try_from(bits).expect("a binary32 encoding"))
        .collect()
}

/// Returns operand pairs: every pair of specials, and for each random
/// operand a random partner, itself, its negation, and its neighbor.
fn compare_pairs<T: Copy + core::ops::BitXor<Output = T>>(
    specials: &[T],
    operands: &[T],
    sign: T,
    low: T,
) -> Vec<(T, T)> {
    let mut pairs = Vec::new();
    for &a in specials {
        for &b in specials {
            pairs.push((a, b));
        }
    }
    for pair in operands.chunks_exact(2) {
        let a = pair[0];
        pairs.extend([(a, pair[1]), (a, a), (a, a ^ sign), (a, a ^ low)]);
    }
    pairs
}

/// Compares a quiet and a signaling compare instruction with floaty in every
/// setting.
///
/// The flags follow the precedence of the Intel SDM Volume 1, section 4.9.2
/// on page 4-24: a NaN operand, or an invalid operation, hides DE. Under DAZ
/// the processor reads a subnormal operand as zero and does not report DE
/// (section 10.2.3.4 on page 10-5). `mxcsr_flags` maps both rules. The
/// hardware shows both for `UCOMISS` and `COMISS` with a NaN and a subnormal
/// operand.
macro_rules! compare {
    ($pairs:expr, $alias:ty, $quiet:path, $signaling:path) => {
        for setting in x86::sse_settings() {
            let env = setting.env();
            for &(a, b) in &$pairs {
                let (x, y) = (<$alias>::from_bits(a), <$alias>::from_bits(b));
                let nan_operand = x.is_nan() || y.is_nan();
                for (instruction, (order, flags), (expected, expected_flags)) in [
                    (
                        stringify!($quiet),
                        x.compare_quiet_with(y, env),
                        $quiet(a, b, setting.control),
                    ),
                    (
                        stringify!($signaling),
                        x.compare_signaling_with(y, env),
                        $signaling(a, b, setting.control),
                    ),
                ] {
                    let context = format!("{instruction} {a:#x} {b:#x} {setting:?} {flags:?}");
                    assert_eq!(order, expected, "{context}: order");
                    assert_eq!(
                        mxcsr_flags(flags, setting.daz, nan_operand),
                        expected_flags,
                        "{context}: flags"
                    );
                }
            }
        }
    };
}

#[test]
fn ucomis_and_comis_match_the_quiet_and_signaling_compares() {
    let mut random = SplitMix64::new(0x0C0A_5E00);
    let single = singles(random_operands(
        Layout::BINARY32,
        &mut random,
        40_000,
        -2..=2,
    ));
    let pairs = compare_pairs(&SINGLE_SPECIALS, &single, 1 << 31, 1);
    compare!(pairs, F32, x86::ucomiss, x86::comiss);
    let double = random_operands(Layout::BINARY64, &mut random, 40_000, -2..=2);
    let pairs = compare_pairs(&DOUBLE_SPECIALS, &double, 1 << 63, 1);
    compare!(pairs, F64, x86::ucomisd, x86::comisd);
}

/// Returns the MXCSR flags of a conversion between a float and an integer.
///
/// The conversions never signal DE (Intel SDM Volume 1, section 11.5.2.2 on
/// page 11-15), so the consumer drops `DENORMAL_INPUT`. The hardware shows
/// that DAZ still reads a subnormal operand as zero.
fn conversion_flags(flags: Flags, daz: bool) -> u32 {
    mxcsr_flags(flags.difference(Flags::DENORMAL_INPUT), daz, false)
}

/// Returns operands for a conversion to an integer: the specials, every edge
/// around the integers up to 2^66, and random values.
fn to_int_operands(layout: Layout, random: &mut SplitMix64, specials: &[u64]) -> Vec<u64> {
    let mut operands = specials.to_vec();
    operands.extend(integer_edges(layout, random, -3..=66));
    operands.extend(random_operands(layout, random, 30_000, -3..=66));
    operands
}

/// Compares a conversion to an integer that rounds as MXCSR selects, and one
/// that truncates, with floaty in every setting. `CVTT*` rounds toward zero
/// (Intel SDM Volume 1, section 4.8.4.2 on page 4-19).
macro_rules! to_int {
    ($operands:expr, $alias:ty, $integer:ty, $rounded:path, $truncated:path) => {
        for setting in x86::sse_settings() {
            for &input in &$operands {
                let value = <$alias>::from_bits(input);
                for (instruction, rounding, (expected, expected_flags)) in [
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
                    let (ours, flags) =
                        value.to_int_with::<$integer>(setting.env().with_rounding(rounding));
                    let context =
                        format!("{instruction} {input:#x} {setting:?} {ours:?} {flags:?}");
                    assert_eq!(stored(ours, <$integer>::MIN), expected, "{context}: result");
                    assert_eq!(
                        conversion_flags(flags, setting.daz),
                        expected_flags,
                        "{context}: flags"
                    );
                }
            }
        }
    };
}

#[test]
fn cvtss2si_and_cvttss2si_match_conversions_to_integers() {
    let mut random = SplitMix64::new(0x0C07_5500);
    let specials = SINGLE_SPECIALS.map(u64::from);
    let operands = singles(to_int_operands(Layout::BINARY32, &mut random, &specials));
    to_int!(operands, F32, i32, x86::cvtss2si_r32, x86::cvttss2si_r32);
    to_int!(operands, F32, i64, x86::cvtss2si_r64, x86::cvttss2si_r64);
}

#[test]
fn cvtsd2si_and_cvttsd2si_match_conversions_to_integers() {
    let mut random = SplitMix64::new(0x0C07_5D00);
    let operands = to_int_operands(Layout::BINARY64, &mut random, &DOUBLE_SPECIALS);
    to_int!(operands, F64, i32, x86::cvtsd2si_r32, x86::cvttsd2si_r32);
    to_int!(operands, F64, i64, x86::cvtsd2si_r64, x86::cvttsd2si_r64);
}

/// Returns 64-bit integers of every bit length up to `bits`, for both signs:
/// the low bits that a rounding to `precision` bits drops take each edge
/// pattern, and the other bits are random. Also returns the extremes and
/// random values.
fn integer_operands(random: &mut SplitMix64, bits: u32, precision: u32) -> Vec<i64> {
    let largest = i64::MAX >> (64 - bits);
    let mut operands = vec![0, 1, -1, largest, -largest, -largest - 1];
    for length in 1..bits {
        let dropped = length.saturating_sub(precision);
        for pattern in edge_patterns(dropped) {
            for _ in 0..4 {
                let high = (random.next_u64() >> (64 - length)) >> dropped << dropped;
                let magnitude = (1 << (length - 1)) | high | pattern;
                let magnitude = i64::try_from(magnitude).expect("below 2^63");
                operands.extend([magnitude, -magnitude]);
            }
        }
    }
    for _ in 0..20_000 {
        let value = random.next_u64() >> (random.next_u64() % 64);
        let value = i64::from_le_bytes(value.to_le_bytes()) >> (64 - bits);
        operands.push(value);
    }
    operands
}

/// Compares a conversion from an integer with floaty in every setting.
///
/// The conversions signal at most PE (Intel SDM Volume 1, Table C-3 on page
/// C-4). FTZ and DAZ do not apply, because the operand is an integer and the
/// result cannot be tiny.
macro_rules! from_int {
    ($operands:expr, $alias:ty, $instruction:path) => {
        for setting in x86::sse_settings() {
            let env = setting.env();
            for &input in &$operands {
                let (expected, expected_flags) = $instruction(input, setting.control);
                let (ours, flags) = <$alias>::from_int_with(input, env);
                let context = format!("{} {input} {setting:?} {flags:?}", stringify!($instruction));
                assert_eq!(ours.to_bits(), expected, "{context}: result");
                assert_eq!(
                    mxcsr_flags(flags, setting.daz, false),
                    expected_flags,
                    "{context}: flags"
                );
            }
        }
    };
}

#[test]
fn cvtsi2ss_and_cvtsi2sd_match_conversions_from_integers() {
    let mut random = SplitMix64::new(0x0C75_1255);
    let narrow = |operands: Vec<i64>| -> Vec<i32> {
        operands
            .into_iter()
            .map(|value| i32::try_from(value).expect("a 32-bit integer"))
            .collect()
    };
    let operands = narrow(integer_operands(&mut random, 32, 24));
    from_int!(operands, F32, x86::cvtsi2ss_r32);
    let operands = integer_operands(&mut random, 64, 24);
    from_int!(operands, F32, x86::cvtsi2ss_r64);
    let operands = narrow(integer_operands(&mut random, 32, 53));
    from_int!(operands, F64, x86::cvtsi2sd_r32);
    let operands = integer_operands(&mut random, 64, 53);
    from_int!(operands, F64, x86::cvtsi2sd_r64);
}

/// Returns the rounding direction of a `ROUNDSS` or `ROUNDSD` immediate.
///
/// Bits 1 and 0 use the encoding of the rounding-control field (Intel SDM
/// Volume 1, Table 4-9 on page 4-19). Bit 2 selects the MXCSR direction
/// instead (Volume 2B, `ROUNDSS`; the test confirms both on hardware).
fn immediate_rounding(immediate: u8, mxcsr: Rounding) -> Rounding {
    if immediate & ROUND_USE_MXCSR == 0 {
        MXCSR_ROUNDINGS[usize::from(immediate & 3)].0
    } else {
        mxcsr
    }
}

/// Compares `ROUNDSS` or `ROUNDSD` with floaty for each immediate in every
/// setting: the four directions, the MXCSR direction, and each of those with
/// the precision exception suppressed.
macro_rules! round {
    ($operands:expr, $alias:ty, $instruction:ident) => {
        round!($operands, $alias, $instruction, [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]);
    };
    ($operands:expr, $alias:ty, $instruction:ident, [$($immediate:literal),+]) => {
        $(
            for setting in x86::sse_settings() {
                let env = setting
                    .env()
                    .with_rounding(immediate_rounding($immediate, setting.rounding));
                for &input in &$operands {
                    let (expected, expected_flags) =
                        x86::$instruction::<$immediate>(input, setting.control);
                    let (ours, flags) = <$alias>::from_bits(input).round_to_integral_with(env);
                    let context = format!(
                        "{} {:#x} {input:#x} {setting:?} {flags:?}",
                        stringify!($instruction),
                        $immediate
                    );
                    assert_eq!(ours.to_bits(), expected, "{context}: result");
                    assert_eq!(
                        mxcsr_round_flags(flags, $immediate, setting.daz),
                        expected_flags,
                        "{context}: flags"
                    );
                }
            }
        )+
    };
}

#[test]
fn roundss_and_roundsd_match_rounding_to_an_integral_value() {
    assert!(
        std::arch::is_x86_feature_detected!("sse4.1"),
        "the round test needs a host with SSE4.1"
    );
    let mut random = SplitMix64::new(0x0400_5D00);
    let mut single = SINGLE_SPECIALS.map(u64::from).to_vec();
    single.extend(integer_edges(Layout::BINARY32, &mut random, -3..=25));
    single.extend(random_operands(
        Layout::BINARY32,
        &mut random,
        15_000,
        -3..=25,
    ));
    let single = singles(single);
    round!(single, F32, roundss);
    let mut double = DOUBLE_SPECIALS.to_vec();
    double.extend(integer_edges(Layout::BINARY64, &mut random, -3..=54));
    double.extend(random_operands(
        Layout::BINARY64,
        &mut random,
        15_000,
        -3..=54,
    ));
    round!(double, F64, roundsd);
}
