//! Compares x87 classification with the `FXAM` instruction of the host
//! processor, which the Intel SDM Volume 2A describes. The test runs only on
//! x86-64 hosts.

#![cfg(target_arch = "x86_64")]

use core::arch::asm;

use floaty::env::{NanPropagation, NanRule};
use floaty::{Class, Env, F32, F64, F80, Flags, Rounding};
use floaty_verify::encodings::{IntegerBit, boundary_encodings, to_u128};
use floaty_verify::random::SplitMix64;

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

/// Loads an 80-bit encoding onto the x87 stack and examines it with `FXAM`.
/// Returns the class and the sign, which `FXAM` reports in C1.
fn fxam(bits: u128) -> (Fxam, bool) {
    let bytes = bits.to_le_bytes();
    let status: u16;
    // SAFETY: the code reads 10 bytes from `bytes`, which holds 16. It pushes
    // one value onto the x87 stack and pops it, so the stack is empty on exit,
    // as the ABI requires. `FLD` of an 80-bit operand raises no exception.
    unsafe {
        asm!(
            "fld tbyte ptr [{source}]",
            "fxam",
            "fnstsw ax",
            "fstp st(0)",
            source = in(reg) bytes.as_ptr(),
            out("ax") status,
            out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
            out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
            options(nostack, readonly),
        );
    }
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

/// Reads the x87 control word.
fn control_word() -> u16 {
    let mut word = 0_u16;
    // SAFETY: `FNSTCW` writes two bytes to `word` and changes no other state.
    unsafe {
        asm!(
            "fnstcw word ptr [{word}]",
            word = in(reg) &raw mut word,
            options(nostack),
        );
    }
    word
}

/// Multiplies an 80-bit encoding by 1.0 on the x87 unit and returns the
/// 80-bit result that `FSTP` stores.
fn times_one(bits: u128) -> u128 {
    let input = bits.to_le_bytes();
    let mut output = [0_u8; 16];
    // SAFETY: the code reads 10 bytes from `input` and writes 10 bytes to
    // `output`, and both hold 16. It pushes two values and pops both, so the
    // x87 stack is empty on exit, as the ABI requires.
    unsafe {
        asm!(
            "fld tbyte ptr [{source}]",
            "fld1",
            "fmulp",
            "fstp tbyte ptr [{target}]",
            "fnclex",
            source = in(reg) input.as_ptr(),
            target = in(reg) output.as_mut_ptr(),
            out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
            out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
            options(nostack),
        );
    }
    u128::from_le_bytes(output) & ((1 << 80) - 1)
}

/// A pseudo-denormal has the value of the normal encoding with exponent field
/// 1 (Intel SDM Volume 1, section 8.2.2). Multiplying by 1.0 makes the
/// processor store that normal encoding, which gives an independent value.
#[test]
fn pseudo_denormals_have_the_value_that_the_processor_computes() {
    // Round to nearest with 64-bit precision, every exception masked.
    assert_eq!(
        control_word(),
        0x037F,
        "the thread has the default x87 control word"
    );
    let mut random = SplitMix64::new(0x0D0D);
    for _ in 0..10_000 {
        let fraction = random.next_u64() | (1 << 63);
        let sign = u128::from(random.next_u64() & 1) << 79;
        let pseudo = sign | u128::from(fraction);
        let normal = times_one(pseudo);
        let ours = F80::from_bits(pseudo);
        let theirs = F80::from_bits(normal);
        assert_eq!(theirs.classify(), Class::Normal, "{normal:#x} is normal");
        assert!(theirs.is_canonical(), "{normal:#x} is canonical");
        assert_eq!(ours.decode::<2>(), theirs.decode::<2>(), "F80 {pseudo:#x}");
    }
}

/// The rounding directions and their x87 rounding-control field.
const ROUNDINGS: [(Rounding, u16); 4] = [
    (Rounding::NearestEven, 0),
    (Rounding::TowardNegative, 1 << 10),
    (Rounding::TowardPositive, 2 << 10),
    (Rounding::TowardZero, 3 << 10),
];

/// Loads an 80-bit encoding with `FLD` and stores it with `FSTP` to 64 bits,
/// or to 32 bits when `single` is set, under the control word `control`.
/// Returns the stored bits and the status word.
fn store(bits: u128, control: u16, single: bool) -> (u64, u16) {
    let input = bits.to_le_bytes();
    let mut output = 0_u64;
    let mut saved = 0_u16;
    let status: u16;
    // SAFETY: the code saves the control word, loads `control`, clears the
    // exception flags, pushes one value and pops it, clears the flags again,
    // and restores the control word. It reads 10 bytes of `input`, which holds 16, and writes at most 8
    // bytes to `output`. The x87 stack is empty on exit.
    unsafe {
        if single {
            asm!(
                "fnstcw word ptr [{saved}]",
                "fldcw word ptr [{control}]",
                "fnclex",
                "fld tbyte ptr [{source}]",
                "fstp dword ptr [{target}]",
                "fnstsw ax",
                "fnclex",
                "fldcw word ptr [{saved}]",
                saved = in(reg) &raw mut saved,
                control = in(reg) &raw const control,
                source = in(reg) input.as_ptr(),
                target = in(reg) &raw mut output,
                out("ax") status,
                out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
                out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
                options(nostack),
            );
        } else {
            asm!(
                "fnstcw word ptr [{saved}]",
                "fldcw word ptr [{control}]",
                "fnclex",
                "fld tbyte ptr [{source}]",
                "fstp qword ptr [{target}]",
                "fnstsw ax",
                "fnclex",
                "fldcw word ptr [{saved}]",
                saved = in(reg) &raw mut saved,
                control = in(reg) &raw const control,
                source = in(reg) input.as_ptr(),
                target = in(reg) &raw mut output,
                out("ax") status,
                out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
                out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
                options(nostack),
            );
        }
    }
    (output, status)
}

/// Returns the x87 status bits that a floaty result reports: IE, OE, UE, PE,
/// and C1 for a rounding that grew the magnitude.
fn status_bits(flags: Flags) -> u16 {
    let mut bits = 0;
    for (flag, bit) in [
        (Flags::INVALID, 0),
        (Flags::OVERFLOW, 3),
        (Flags::UNDERFLOW, 4),
        (Flags::INEXACT, 5),
        (Flags::ROUNDED_UP, 9),
    ] {
        if flags.contains(flag) {
            bits |= 1 << bit;
        }
    }
    bits
}

/// The status bits that the comparison checks: IE, OE, UE, PE, and C1.
///
/// `FSTP` does not report DE for a denormal operand, so the test checks DE
/// separately: the processor never sets it for a store. Mapping
/// `DENORMAL_INPUT` to the DE flag of each instruction belongs to a consumer.
const CHECKED: u16 = 0b10_0011_1001;
const DE: u16 = 1 << 1;

/// The environment of an x87 store: the x87 NaN rule with its negative
/// default NaN, tininess after rounding, and no precision control, which
/// stores ignore.
fn store_env(rounding: Rounding) -> Env {
    Env::IEEE.with_rounding(rounding).with_nan(NanRule {
        propagation: NanPropagation::X87,
        default_negative: true,
    })
}

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
        let significand = u128::from(
            random.next_u64()
                | if random.next_u64() % 16 == 0 {
                    0
                } else {
                    1 << 63
                },
        );
        inputs.push(sign | (exponent << 64) | significand);
    }
    inputs
}

#[test]
fn stores_to_binary64_and_binary32_match_the_processor() {
    let inputs = store_inputs();
    for (rounding, rc) in ROUNDINGS {
        let control = 0x037F | rc;
        let env = store_env(rounding);
        for &input in &inputs {
            let value = F80::from_bits(input);
            let (expected, status) = store(input, control, false);
            let (ours, flags): (F64, _) = value.convert_with(env);
            let context =
                format!("F80 {input:#x} to F64 {rounding:?} {flags:?} status {status:#06x}");
            assert_eq!(ours.to_bits(), expected, "{context}: result");
            assert_eq!(status_bits(flags), status & CHECKED, "{context}: flags");
            assert_eq!(status & DE, 0, "{context}: a store never reports DE");

            let (expected, status) = store(input, control, true);
            let (ours, flags): (F32, _) = value.convert_with(env);
            let context =
                format!("F80 {input:#x} to F32 {rounding:?} {flags:?} status {status:#06x}");
            assert_eq!(u64::from(ours.to_bits()), expected, "{context}: result");
            assert_eq!(status_bits(flags), status & CHECKED, "{context}: flags");
            assert_eq!(status & DE, 0, "{context}: a store never reports DE");
        }
    }
}
