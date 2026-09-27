//! Compares x87 classification with the `FXAM` instruction of the host
//! processor, which the Intel SDM Volume 2A describes. The test runs only on
//! x86-64 hosts.

#![cfg(target_arch = "x86_64")]

use core::arch::asm;

use floaty::{Class, F80};
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
