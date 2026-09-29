//! Compares x87 classification, stores, and arithmetic with the x87 unit of
//! the host processor. The test runs only on x86-64 hosts.

#![cfg(target_arch = "x86_64")]

use core::num::NonZeroU32;

use floaty::{Class, Env, F32, F64, F80, ToInt};
use floaty_verify::encodings::{IntegerBit, boundary_encodings, to_u128};
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::{
    self, X87_DE, X87_MASKED, X87_PRECISIONS, X87_ROUNDINGS, X87_STATUS_FLAGS,
    x87_arithmetic_status, x87_env, x87_status,
};

/// The classes that `FXAM` reports in condition codes C3, C2, and C0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fxam {
    Unsupported,
    Nan,
    Normal,
    Infinity,
    Zero,
    Denormal,
}

/// Returns the class and the sign, which `FXAM` reports in C1.
fn fxam(bits: u128) -> (Fxam, bool) {
    let status = x86::fxam(bits);
    let flag = |bit: u16| (status >> bit) & 1 == 1;
    let class = match (flag(14), flag(10), flag(8)) {
        (false, false, false) => Fxam::Unsupported,
        (false, false, true) => Fxam::Nan,
        (false, true, false) => Fxam::Normal,
        (false, true, true) => Fxam::Infinity,
        (true, false, false) => Fxam::Zero,
        (true, true, false) => Fxam::Denormal,
        other => panic!("FXAM reported an empty register or an unknown class: {other:?}"),
    };
    (class, flag(9))
}

fn expected(class: Class) -> Fxam {
    match class {
        Class::Unsupported => Fxam::Unsupported,
        Class::QuietNan | Class::SignalingNan => Fxam::Nan,
        Class::Normal => Fxam::Normal,
        Class::Infinite => Fxam::Infinity,
        Class::Zero => Fxam::Zero,
        Class::Subnormal => Fxam::Denormal,
        other => panic!("F80 has no {other:?} class"),
    }
}

#[test]
fn every_boundary_and_random_encoding_matches_fxam() {
    let mask = (1_u128 << 80) - 1;
    let mut random = SplitMix64::new(0xF8A4);
    let boundaries: Vec<u128> = boundary_encodings(80, 15, IntegerBit::Explicit)
        .iter()
        .map(to_u128)
        .collect();
    let samples = (0..200_000).map(|_| random.next_u128() & mask);
    let mut seen = Vec::new();
    for bits in boundaries.into_iter().chain(samples) {
        let ours = F80::from_bits(bits);
        let (class, negative) = fxam(bits);
        assert_eq!(expected(ours.classify()), class, "F80 {bits:#x}");
        assert_eq!(ours.is_sign_negative(), negative, "F80 {bits:#x}");
        if !seen.contains(&class) {
            seen.push(class);
        }
    }
    assert_eq!(seen.len(), 6, "the inputs reach every FXAM class: {seen:?}");
}

/// A pseudo-denormal has the value of the normal encoding with exponent field
/// 1 (Intel SDM Volume 1, section 8.2.2). Multiplying by 1.0 makes the
/// processor store that normal encoding, which gives an independent value.
#[test]
fn pseudo_denormals_have_the_value_that_the_processor_computes() {
    // Round to nearest with 64-bit precision, every exception masked.
    assert_eq!(
        x86::control_word(),
        0x037F,
        "the thread has the default x87 control word"
    );
    let mut random = SplitMix64::new(0x0D0D);
    for _ in 0..10_000 {
        let fraction = random.next_u64() | (1 << 63);
        let sign = u128::from(random.next_u64() & 1) << 79;
        let pseudo = sign | u128::from(fraction);
        let (normal, _) = x86::times_one(pseudo, 0x037F);
        let ours = F80::from_bits(pseudo);
        let theirs = F80::from_bits(normal);
        assert_eq!(theirs.classify(), Class::Normal, "{normal:#x} is normal");
        assert!(theirs.is_canonical(), "{normal:#x} is canonical");
        assert_eq!(ours.decode::<2>(), theirs.decode::<2>(), "F80 {pseudo:#x}");
    }
}

/// The status bits that the comparisons check: IE, DE, ZE, OE, UE, PE, and C1.
const CHECKED: u16 = X87_STATUS_FLAGS;
const DE: u16 = X87_DE;

/// Returns the low-bit patterns of `dropped` discarded bits that decide a
/// rounding: exact, just below halfway, halfway, just above halfway, and all
/// ones.
fn edge_patterns(dropped: u32) -> [u64; 5] {
    let half = 1_u64 << (dropped - 1);
    [0, half - 1, half, half + 1, (half << 1) - 1]
}

/// Returns boundary and random 80-bit encodings, with exponents biased toward
/// the binary32 and binary64 ranges, and every edge pattern at the binary32
/// and binary64 boundaries.
fn store_inputs() -> Vec<u128> {
    let mut random = SplitMix64::new(0x0087_5708);
    let mut inputs: Vec<u128> = boundary_encodings(80, 15, IntegerBit::Explicit)
        .iter()
        .map(to_u128)
        .collect();
    // (precision, emin, emax) of binary32 and binary64.
    for (precision, emin, emax) in [(24, -126, 127), (53, -1022, 1023)] {
        let edges = (emin - precision - 2..=emin + 1).chain(emax - 1..=emax + 2);
        for exponent in edges {
            let dropped = (64 - precision + (emin - exponent).max(0)).min(63);
            let biased = u128::try_from(16383 + exponent).expect("a normal x87 exponent");
            for pattern in edge_patterns(u32::try_from(dropped).expect("at most 63")) {
                for _ in 0..8 {
                    let high = random.next_u64() & !((1 << dropped) - 1);
                    let significand = u128::from((1 << 63) | high | pattern);
                    let sign = u128::from(random.next_u64() & 1) << 79;
                    inputs.push(sign | (biased << 64) | significand);
                }
            }
        }
    }
    for _ in 0..100_000 {
        let exponent = u128::from(16383 - 1100 + random.next_u64() % 2300);
        let sign = u128::from(random.next_u64() & 1) << 79;
        let integer = if random.next_u64() % 16 == 0 {
            0
        } else {
            1 << 63
        };
        let significand = u128::from(random.next_u64() | integer);
        inputs.push(sign | (exponent << 64) | significand);
    }
    inputs
}

/// `FSTP` does not report DE for a denormal operand, so the store test checks
/// that the processor never sets it for a store.
#[test]
fn stores_to_binary64_and_binary32_match_the_processor() {
    let inputs = store_inputs();
    for (rounding, field) in X87_ROUNDINGS {
        let control = 0x037F | field;
        let env = x87_env(rounding);
        for &input in &inputs {
            let value = F80::from_bits(input);
            let (expected, status) = x86::store_double(input, control);
            let (ours, flags): (F64, _) = value.convert_with(env);
            let context =
                format!("F80 {input:#x} to F64 {rounding:?} {flags:?} status {status:#06x}");
            assert_eq!(ours.to_bits(), expected, "{context}: result");
            assert_eq!(
                x87_status(flags) & CHECKED & !DE,
                status & CHECKED & !DE,
                "{context}: flags"
            );
            assert_eq!(status & DE, 0, "{context}: a store never reports DE");

            let (expected, status) = x86::store_single(input, control);
            let (ours, flags): (F32, _) = value.convert_with(env);
            let context =
                format!("F80 {input:#x} to F32 {rounding:?} {flags:?} status {status:#06x}");
            assert_eq!(ours.to_bits(), expected, "{context}: result");
            assert_eq!(
                x87_status(flags) & CHECKED & !DE,
                status & CHECKED & !DE,
                "{context}: flags"
            );
            assert_eq!(status & DE, 0, "{context}: a store never reports DE");
        }
    }
}

/// The x87 special operands: zeros, infinities, NaNs of both kinds and signs,
/// quiet and signaling pairs of NaNs with equal significands and different
/// signs, the real
/// indefinite, unsupported encodings, a pseudo-denormal, denormals, the normal
/// boundaries, and ones.
const SPECIALS: [u128; 24] = [
    0x7FFF_C000_0000_0000_0005,
    0xFFFF_C000_0000_0000_0005,
    0x7FFF_A000_0000_0000_0007,
    0xFFFF_A000_0000_0000_0007,
    0x0000_0000_0000_0000_0000,
    0x8000_0000_0000_0000_0000,
    0x7FFF_8000_0000_0000_0000,
    0xFFFF_8000_0000_0000_0000,
    0x7FFF_C000_0000_0000_0001,
    0xFFFF_C000_0000_0000_0009,
    0x7FFF_8000_0000_0000_0002,
    0xFFFF_A000_0000_0000_0000,
    0xFFFF_C000_0000_0000_0000,
    0x7FFF_4000_0000_0000_0000,
    0x3FFF_0000_0000_0000_0001,
    0x0000_8000_0000_0000_0003,
    0x0000_0000_0000_0000_0001,
    0x8000_7FFF_FFFF_FFFF_FFFF,
    0x0001_8000_0000_0000_0000,
    0x7FFE_FFFF_FFFF_FFFF_FFFF,
    0x3FFF_8000_0000_0000_0000,
    0xBFFF_8000_0000_0000_0000,
    0x4000_8000_0000_0000_0000,
    0x3FFF_8000_0000_0000_0001,
];

/// Returns x87 operands: the specials, and random values across the range,
/// near the smallest normal value, and near the largest value.
fn operands(random: &mut SplitMix64, count: usize) -> Vec<u128> {
    let mut operands = SPECIALS.to_vec();
    for index in 0..count {
        let exponent = u128::from(match index % 3 {
            0 => random.next_u64() % 0x7FFF,
            1 => random.next_u64() % 80,
            _ => 0x7FFF - 80 + random.next_u64() % 80,
        });
        let integer = if random.next_u64() % 32 == 0 {
            0
        } else {
            1 << 63
        };
        let significand = u128::from(random.next_u64() | integer);
        let sign = u128::from(random.next_u64() & 1) << 79;
        operands.push(sign | (exponent << 64) | significand);
    }
    operands
}

/// Every x87 control setting: each rounding direction at each precision.
fn settings() -> Vec<(u16, Env)> {
    let mut settings = Vec::new();
    for (rounding, rounding_field) in X87_ROUNDINGS {
        for (precision, precision_field) in X87_PRECISIONS {
            let control = X87_MASKED | rounding_field | precision_field;
            let limit = NonZeroU32::new(precision);
            settings.push((control, x87_env(rounding).with_precision(limit)));
        }
    }
    settings
}

/// Returns the precision limit of an x87 setting in bits.
fn precision(env: Env) -> u32 {
    env.precision.map_or(64, NonZeroU32::get)
}

/// Compares one x87 arithmetic instruction with floaty in every setting: with
/// the behavior as an `Env` at run time, and with the static mode of the
/// setting, which `with_x87_mode!` selects as an emulator would.
macro_rules! arithmetic {
    ($pairs:expr, $instruction:path, $method:ident) => {
        for (control, env) in settings() {
            for &(a, b) in &$pairs {
                let (expected, status) = $instruction(a, b, control);
                let (x, y) = (F80::from_bits(a), F80::from_bits(b));
                let (ours, flags) = x.$method(y, env);
                let context = format!(
                    "{} {a:#x} {b:#x} {env:?} {flags:?} status {status:#06x}",
                    stringify!($instruction)
                );
                assert_eq!(
                    ours.to_bits(),
                    expected & ((1 << 80) - 1),
                    "{context}: result"
                );
                let nan_operand = x.is_nan() || y.is_nan();
                assert_eq!(
                    x87_arithmetic_status(flags, nan_operand),
                    status & CHECKED,
                    "{context}: flags"
                );
                let (fixed, fixed_flags) =
                    floaty_verify::with_x87_mode!(env.rounding, precision(env), Mode => {
                        assert_eq!(<Mode as floaty::env::Mode>::ENV, env, "{context}: mode");
                        let (x, y) = (x.with_mode::<Mode>(), y.with_mode::<Mode>());
                        let (value, flags) = x.$method(y, Mode::default());
                        (value.to_bits(), flags)
                    });
                assert_eq!(
                    (fixed, x87_arithmetic_status(fixed_flags, nan_operand)),
                    (expected & ((1 << 80) - 1), status & CHECKED),
                    "{context}: static mode"
                );
            }
        }
    };
}

#[test]
fn arithmetic_matches_at_every_rounding_and_precision() {
    let mut random = SplitMix64::new(0x0087_A817);
    let operands = operands(&mut random, 30_000);
    let mut pairs = Vec::new();
    for &a in &SPECIALS {
        for &b in &SPECIALS {
            pairs.push((a, b));
        }
    }
    for pair in operands[SPECIALS.len()..].chunks_exact(2) {
        pairs.push((pair[0], pair[1]));
        // The same magnitude with the other sign and a changed low bit.
        pairs.push((pair[0], pair[0] ^ (1 << 79) ^ 1));
    }
    arithmetic!(pairs, x86::fadd, add_with);
    arithmetic!(pairs, x86::fsub, sub_with);
    arithmetic!(pairs, x86::fmul, mul_with);
    arithmetic!(pairs, x86::fdiv, div_with);
    for (control, env) in settings() {
        for &a in &operands {
            let (expected, status) = x86::fsqrt(a, control);
            let value = F80::from_bits(a);
            let (ours, flags) = value.sqrt_with(env);
            let context = format!("fsqrt {a:#x} {env:?} {flags:?} status {status:#06x}");
            assert_eq!(
                ours.to_bits(),
                expected & ((1 << 80) - 1),
                "{context}: result"
            );
            assert_eq!(
                x87_arithmetic_status(flags, value.is_nan()),
                status & CHECKED,
                "{context}: flags"
            );
            let (fixed, fixed_flags) = floaty_verify::with_x87_mode!(env.rounding, precision(env), Mode => {
                assert_eq!(<Mode as floaty::env::Mode>::ENV, env, "{context}: mode");
                let (root, flags) = value.with_mode::<Mode>().sqrt_with(Mode::default());
                (root.to_bits(), flags)
            });
            assert_eq!(
                (fixed, x87_arithmetic_status(fixed_flags, value.is_nan())),
                (expected & ((1 << 80) - 1), status & CHECKED),
                "{context}: static mode"
            );
        }
    }
}

/// The results of the x87 extended entry points for one pair of operands.
#[derive(Debug, PartialEq)]
struct Results {
    /// The operators, `sqrt`, `round_to_integral`, `from_int`, and the
    /// conversions to and from binary32 and binary64, as encodings.
    encodings: [u128; 11],
    /// `to_int` to `i64`.
    signed: ToInt<i64>,
    /// `to_int` to `u32`.
    unsigned: ToInt<u32>,
}

/// Returns the other operands of the entry points for a pair: an `i64` from
/// the low 64 bits of `a`, a binary64 value from bits 16 to 79 of `b`, and a
/// binary32 value from the low 32 bits of `b`.
fn other_operands(a: u128, b: u128) -> (i64, F64, F32) {
    let integer = i64::from_le_bytes(
        a.to_le_bytes()[..8]
            .try_into()
            .expect("the slice holds 8 bytes"),
    );
    let double = u64::try_from(b >> 16).expect("an 80-bit encoding holds 64 bits above bit 15");
    let single = u32::from_le_bytes(
        b.to_le_bytes()[..4]
            .try_into()
            .expect("the slice holds 4 bytes"),
    );
    (integer, F64::from_bits(double), F32::from_bits(single))
}

/// Returns the results of the entry points without flags, which run on the
/// x87 unit where the build has a host path.
fn entry_points(a: u128, b: u128) -> Results {
    let (x, y) = (F80::from_bits(a), F80::from_bits(b));
    let (integer, double, single) = other_operands(a, b);
    Results {
        encodings: [
            (x + y).to_bits(),
            (x - y).to_bits(),
            (x * y).to_bits(),
            (x / y).to_bits(),
            x.sqrt().to_bits(),
            x.round_to_integral().to_bits(),
            F80::from_int(integer).to_bits(),
            u128::from(x.convert::<F64>().to_bits()),
            u128::from(x.convert::<F32>().to_bits()),
            double.convert::<F80>().to_bits(),
            single.convert::<F80>().to_bits(),
        ],
        signed: x.to_int(),
        unsigned: x.to_int(),
    }
}

/// Returns the results of the `_with` methods under the default modes, which
/// always run the engine.
fn engine_results(a: u128, b: u128) -> Results {
    let (x, y) = (F80::from_bits(a), F80::from_bits(b));
    let (integer, double, single) = other_operands(a, b);
    let env = F80::ENV;
    let to_double: F64 = x.convert_with(F64::ENV).0;
    let to_single: F32 = x.convert_with(F32::ENV).0;
    let from_double: F80 = double.convert_with(env).0;
    let from_single: F80 = single.convert_with(env).0;
    Results {
        encodings: [
            x.add_with(y, env).0.to_bits(),
            x.sub_with(y, env).0.to_bits(),
            x.mul_with(y, env).0.to_bits(),
            x.div_with(y, env).0.to_bits(),
            x.sqrt_with(env).0.to_bits(),
            x.round_to_integral_with(env).0.to_bits(),
            F80::from_int_with(integer, env).0.to_bits(),
            u128::from(to_double.to_bits()),
            u128::from(to_single.to_bits()),
            from_double.to_bits(),
            from_single.to_bits(),
        ],
        signed: x.to_int_with(env).0,
        unsigned: x.to_int_with(env).0,
    }
}

#[test]
fn entry_points_read_the_control_word_before_the_x87_unit() {
    // Each directed rounding and each precision below 64 bits changes the
    // x87 results, and an unmasked exception traps. Under each, the entry
    // points still give the engine results of the default mode. The masks
    // IM, DM, ZM, OM, UM, and PM are bits 0 to 5.
    let full = X87_MASKED | (3 << 8);
    let directed = X87_ROUNDINGS.into_iter().map(|(_, field)| full | field);
    let precisions = X87_PRECISIONS
        .into_iter()
        .map(|(_, field)| X87_MASKED | field);
    let unmasked = (0..=5).map(|mask| full & !(1 << mask));
    let controls: Vec<u16> = directed.chain(precisions).chain(unmasked).collect();
    let mut random = SplitMix64::new(0x0087_C0DE);
    let operands = operands(&mut random, 4_000);
    let pairs: Vec<(u128, u128)> = operands
        .iter()
        .zip(operands.iter().rev())
        .map(|(&a, &b)| (a, b))
        .collect();
    for control in controls {
        for &(a, b) in &pairs {
            let ours = x86::with_control_word(control, || entry_points(a, b));
            assert_eq!(
                ours,
                engine_results(a, b),
                "{a:#x} {b:#x} under {control:#06x}"
            );
        }
    }
}
