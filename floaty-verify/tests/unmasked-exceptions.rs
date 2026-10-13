//! Compares the response of the processor to unmasked exceptions with the
//! recipes that the README states, which build the response from floaty
//! flags. The test runs only on x86-64 hosts, where an SSE trap arrives as
//! SIGFPE.
//!
//! SSE (Intel SDM Volume 1, revision 253665-077US, section 11.5.3.2): the
//! instruction first looks for IE, ZE, and DE. When one of them is unmasked,
//! it sets those flags and traps, with no result and no OE, UE, or PE. Then
//! it looks for OE, UE, and PE, and an unmasked one traps with no result. An
//! unmasked UE reports every tiny result, exact or not, and flush-to-zero
//! does not apply (section 11.5.2.5). The hardware shows a rule that the
//! section does not state: with an unmasked OE or UE, PE reports whether the
//! result rounded with an unbounded exponent is inexact, not whether the
//! masked result is.
//!
//! x87 (sections 8.5.1 to 8.5.5): an unmasked IE, ZE, or DE leaves the
//! operands and TOP unchanged. An unmasked OE or UE stores the result
//! rounded with an unbounded exponent, divided or multiplied by 2^24576,
//! with C1 for a rounding up and PE for an inexact one. An unmasked exception
//! sets ES and B and waits for the next waiting instruction.

#![cfg(target_arch = "x86_64")]

use floaty::{Binary, Env, F32, F64, F80, Flags, Float};
use floaty_verify::encodings::{Layout, boundary_encodings, to_u128};
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::trap::{self, Unmasked};
use floaty_verify::x86::{
    self, X87Setting, mxcsr_flags, sample_pairs, samples, x87_arithmetic_status,
};

/// binary32 precision with 12 exponent bits: no operation of the test
/// overflows or underflows it, so it rounds with an unbounded exponent.
type Unbounded32 = Float<Binary<12>, 36>;
/// binary64 precision with 15 exponent bits.
type Unbounded64 = Float<Binary<15>, 68>;
/// x87 precision with 17 exponent bits.
type Unbounded80 = Float<Binary<17>, 81>;

/// The MXCSR and x87 exception flags: IE, DE, ZE, OE, UE, and PE.
const IE: u32 = 1;
const DE: u32 = 1 << 1;
const ZE: u32 = 1 << 2;
const OE: u32 = 1 << 3;
const UE: u32 = 1 << 4;
const PE: u32 = 1 << 5;
/// The flags of the exceptions that the processor detects before it computes.
const BEFORE: u32 = IE | DE | ZE;
/// The flags of the exceptions that the processor detects after it computes.
const AFTER: u32 = OE | UE | PE;

/// The sets of unmasked exceptions of the test: each one alone, overflow and
/// underflow, invalid and inexact, and all six.
const UNMASKED: [u32; 9] = [IE, DE, ZE, OE, UE, PE, OE | UE, IE | PE, BEFORE | AFTER];

/// The upper bits of an SSE destination, which no instruction of the test
/// changes.
const SENTINEL: u128 = 0x1234_5678_9ABC_DEF0_0F1E_2D3C_7FC1_2345;

/// What the SSE recipe needs from floaty for one instruction.
struct Sse {
    /// The result bits under the masked behavior.
    result: u64,
    /// The flags under the masked behavior.
    flags: Flags,
    /// The flags without flush-to-zero.
    unflushed: Flags,
    /// The flags of the result rounded with an unbounded exponent.
    unbounded: Flags,
    /// Whether an operand is a NaN.
    nan_operand: bool,
}

/// Returns whether an SSE instruction traps and its MXCSR flags, for the
/// unmasked exceptions `unmasked`.
fn sse_response(computation: &Sse, daz: bool, unmasked: u32) -> (bool, u32) {
    let masked = mxcsr_flags(computation.flags, daz, computation.nan_operand);
    let before = masked & BEFORE;
    if before & unmasked != 0 {
        return (true, before);
    }
    let overflow = computation.flags.contains(Flags::OVERFLOW) && unmasked & OE != 0;
    let underflow = computation.unflushed.contains(Flags::TINY) && unmasked & UE != 0;
    if overflow || underflow {
        let inexact = if computation.unbounded.contains(Flags::INEXACT) {
            PE
        } else {
            0
        };
        return (true, before | if overflow { OE } else { UE } | inexact);
    }
    (masked & unmasked != 0, masked)
}

/// Defines the computation of an SSE instruction on two binary32 operands.
macro_rules! single {
    ($operation:ident) => {
        |a: u64, b: u64, env: Env| {
            let (x, y) = (binary32(a), binary32(b));
            let unbounded = |value: F32| value.convert_with::<Unbounded32>(env).0;
            let (result, flags) = x.$operation(y, env);
            Sse {
                result: u64::from(result.to_bits()),
                flags,
                unflushed: x.$operation(y, env.with_flush_to_zero(false)).1,
                unbounded: unbounded(x).$operation(unbounded(y), env).1,
                nan_operand: x.is_nan() || y.is_nan(),
            }
        }
    };
}

/// Defines the computation of an SSE instruction on two binary64 operands.
macro_rules! double {
    ($operation:ident) => {
        |a: u64, b: u64, env: Env| {
            let (x, y) = (F64::from_bits(a), F64::from_bits(b));
            let unbounded = |value: F64| value.convert_with::<Unbounded64>(env).0;
            let (result, flags) = x.$operation(y, env);
            Sse {
                result: result.to_bits(),
                flags,
                unflushed: x.$operation(y, env.with_flush_to_zero(false)).1,
                unbounded: unbounded(x).$operation(unbounded(y), env).1,
                nan_operand: x.is_nan() || y.is_nan(),
            }
        }
    };
}

/// Returns the binary32 value of the low 32 bits of a sample.
fn binary32(bits: u64) -> F32 {
    F32::from_bits(u32::try_from(bits).expect("a binary32 sample has 32 bits"))
}

/// An SSE instruction of [`trap`]: a destination, a source, and MXCSR.
type SseInstruction = fn(u128, u128, u32) -> Unmasked<u128>;

/// The computation of an SSE instruction from the destination and source
/// bits.
type SseComputation = fn(u64, u64, Env) -> Sse;

/// The SSE instructions of the test: the name, the width of the operands and
/// of the result, the instruction, and its computation.
fn sse_instructions() -> [(&'static str, u32, u32, SseInstruction, SseComputation); 11] {
    [
        ("addss", 32, 32, trap::addss, single!(add_with)),
        ("subss", 32, 32, trap::subss, single!(sub_with)),
        ("mulss", 32, 32, trap::mulss, single!(mul_with)),
        ("divss", 32, 32, trap::divss, single!(div_with)),
        ("sqrtss", 32, 32, trap::sqrtss, |_, b, env| {
            let y = binary32(b);
            let (result, flags) = y.sqrt_with(env);
            Sse {
                result: u64::from(result.to_bits()),
                flags,
                unflushed: y.sqrt_with(env.with_flush_to_zero(false)).1,
                unbounded: y.convert_with::<Unbounded32>(env).0.sqrt_with(env).1,
                nan_operand: y.is_nan(),
            }
        }),
        ("addsd", 64, 64, trap::addsd, double!(add_with)),
        ("subsd", 64, 64, trap::subsd, double!(sub_with)),
        ("mulsd", 64, 64, trap::mulsd, double!(mul_with)),
        ("divsd", 64, 64, trap::divsd, double!(div_with)),
        ("sqrtsd", 64, 64, trap::sqrtsd, |_, b, env| {
            let y = F64::from_bits(b);
            let (result, flags) = y.sqrt_with(env);
            Sse {
                result: result.to_bits(),
                flags,
                unflushed: y.sqrt_with(env.with_flush_to_zero(false)).1,
                unbounded: y.convert_with::<Unbounded64>(env).0.sqrt_with(env).1,
                nan_operand: y.is_nan(),
            }
        }),
        ("cvtsd2ss", 64, 32, trap::cvtsd2ss, |_, b, env| {
            let y = F64::from_bits(b);
            let (result, flags): (F32, _) = y.convert_with(env);
            Sse {
                result: u64::from(result.to_bits()),
                flags,
                unflushed: y.convert_with::<F32>(env.with_flush_to_zero(false)).1,
                unbounded: y.convert_with::<Unbounded32>(env).1,
                nan_operand: y.is_nan(),
            }
        }),
    ]
}

/// Every SSE instruction of the test, in every MXCSR setting, with each set
/// of unmasked exceptions. A trap leaves the destination unchanged.
#[test]
fn sse_exceptions_trap_before_or_after_the_computation() {
    let pairs = [
        sample_pairs(&samples(Layout::BINARY32, 300, 0x5E01), 60, 10_000, 0x5E02),
        sample_pairs(&samples(Layout::BINARY64, 300, 0x5E03), 60, 10_000, 0x5E04),
    ];
    for setting in x86::sse_settings() {
        let env = setting.env();
        for unmasked in UNMASKED {
            let control = setting.control & !(unmasked << 7);
            for (name, width, result_width, instruction, computation) in sse_instructions() {
                let operands = if width == 32 { &pairs[0] } else { &pairs[1] };
                for &(a, b) in operands {
                    let destination = (SENTINEL & !((1 << width) - 1)) | u128::from(a);
                    let theirs = instruction(destination, u128::from(b), control);
                    let ours = computation(a, b, env);
                    let (trapped, flags) = sse_response(&ours, setting.daz, unmasked);
                    let context = format!(
                        "{name} {a:#x} {b:#x} {setting:?} unmasked {unmasked:#04x} {:?} {theirs:x?}",
                        ours.flags
                    );
                    assert_eq!(theirs.trapped, trapped, "{context}: trap");
                    assert_eq!(theirs.flags, flags, "{context}: flags");
                    let expected = if trapped {
                        destination
                    } else {
                        (destination & !((1 << result_width) - 1)) | u128::from(ours.result)
                    };
                    assert_eq!(theirs.value, expected, "{context}: destination");
                }
            }
        }
    }
}

/// The ES and B bits of the x87 status word, and C1.
const ES: u32 = 1 << 7;
const B: u32 = 1 << 15;
const C1: u32 = 1 << 9;
/// The TOP field of the x87 status word.
const TOP: u32 = 0b111 << 11;
/// The bias that an unmasked overflow subtracts from the exponent, and an
/// unmasked underflow adds.
const REBIAS: i32 = 24_576;

/// An x87 operation of the test.
#[derive(Clone, Copy)]
struct X87Case {
    /// The mnemonic, for the messages.
    name: &'static str,
    /// Whether the instruction pops the stack when it completes.
    pops: bool,
    /// The instruction of [`trap`]: `a`, then `b` in ST(0), and the control
    /// word.
    instruction: fn(u128, u128, u16) -> Unmasked<u128>,
    /// The operation of floaty on the x87 values.
    operation: fn(F80, F80, Env) -> (F80, Flags),
    /// The same operation on [`Unbounded80`] values.
    unbounded: fn(Unbounded80, Unbounded80, Env) -> (Unbounded80, Flags),
}

/// The x87 operations of the test.
const X87_CASES: [X87Case; 5] = [
    X87Case {
        name: "fadd",
        pops: true,
        instruction: trap::fadd_unmasked,
        operation: |x, y, env| x.add_with(y, env),
        unbounded: |x, y, env| x.add_with(y, env),
    },
    X87Case {
        name: "fsub",
        pops: true,
        instruction: trap::fsub_unmasked,
        operation: |x, y, env| x.sub_with(y, env),
        unbounded: |x, y, env| x.sub_with(y, env),
    },
    X87Case {
        name: "fmul",
        pops: true,
        instruction: trap::fmul_unmasked,
        operation: |x, y, env| x.mul_with(y, env),
        unbounded: |x, y, env| x.mul_with(y, env),
    },
    X87Case {
        name: "fdiv",
        pops: true,
        instruction: trap::fdiv_unmasked,
        operation: |x, y, env| x.div_with(y, env),
        unbounded: |x, y, env| x.div_with(y, env),
    },
    X87Case {
        name: "fsqrt",
        pops: false,
        instruction: trap::fsqrt_unmasked,
        operation: |_, y, env| y.sqrt_with(env),
        unbounded: |_, y, env| y.sqrt_with(env),
    },
];

/// The expected response of an x87 operation: ST(0) afterward, the status
/// bits that the test compares, and whether the instruction popped.
fn x87_response(
    case: X87Case,
    (a, b): (u128, u128),
    setting: X87Setting,
    unmasked: u32,
) -> (u128, u32, bool) {
    let (x, y) = (F80::from_bits(a), F80::from_bits(b));
    let nan_operand = (case.pops && x.is_nan()) || y.is_nan();
    let (result, flags) = (case.operation)(x, y, setting.env());
    let masked = u32::from(x87_arithmetic_status(flags, nan_operand));
    let before = masked & BEFORE;
    if before & unmasked != 0 {
        return (b, before | ES | B, false);
    }
    let overflow = flags.contains(Flags::OVERFLOW) && unmasked & OE != 0;
    let underflow = flags.contains(Flags::TINY) && unmasked & UE != 0;
    if !(overflow || underflow) {
        let pending = if masked & unmasked != 0 { ES | B } else { 0 };
        return (result.to_bits(), masked | pending, case.pops);
    }
    let full = setting.full_precision();
    let (wide, wide_flags) = (case.unbounded)(
        x.convert_with(full).0,
        y.convert_with(full).0,
        setting.env(),
    );
    let scale = if overflow { -REBIAS } else { REBIAS };
    let rebiased: F80 = wide.scale_b(scale).convert_with(full).0;
    let mut status = before | if overflow { OE } else { UE } | ES | B;
    if wide_flags.contains(Flags::INEXACT) {
        status |= PE;
    }
    if wide_flags.contains(Flags::ROUNDED_UP) {
        status |= C1;
    }
    (rebiased.to_bits(), status, case.pops)
}

/// Returns x87 operands: the boundary encodings, and normal values near the
/// ends of the exponent range and anywhere.
fn x87_operands(random: &mut SplitMix64) -> Vec<u128> {
    let mut operands: Vec<u128> = boundary_encodings(Layout::X87_EXTENDED)
        .iter()
        .map(to_u128)
        .collect();
    operands.extend((0..3_000).map(|index| {
        let field = match index % 3 {
            0 => random.below(0x80),
            1 => 0x7F80 + random.below(0x7F),
            _ => random.below(0x7FFF),
        };
        let significand = u128::from(random.next_u64() | (1 << 63));
        Layout::X87_EXTENDED.encode(random.coin_flip(), field, significand)
    }));
    operands
}

/// Every x87 operation of the test, in every control setting, with each set
/// of unmasked exceptions. The value is ST(0) after the operation, and TOP
/// shows whether it popped.
#[test]
fn x87_overflow_and_underflow_rebias_the_exponent() {
    let mut random = SplitMix64::new(0x0087_0E01);
    let operands = x87_operands(&mut random);
    let length = u64::try_from(operands.len()).expect("a length fits a u64");
    let pairs: Vec<(u128, u128)> = (0..20_000)
        .map(|_| {
            let mut pick =
                || operands[usize::try_from(random.below(length)).expect("an index fits")];
            (pick(), pick())
        })
        .collect();
    for setting in x86::x87_settings() {
        for unmasked in UNMASKED {
            let control =
                setting.control & !u16::try_from(unmasked).expect("the masks are bits 0 to 5");
            for case in X87_CASES {
                for &(a, b) in &pairs {
                    let theirs = (case.instruction)(a, b, control);
                    let (value, status, popped) = x87_response(case, (a, b), setting, unmasked);
                    let context = format!(
                        "{} {a:#x} {b:#x} {setting:?} unmasked {unmasked:#04x} {theirs:x?}",
                        case.name
                    );
                    assert_eq!(theirs.value, value, "{context}: ST(0)");
                    assert_eq!(
                        theirs.flags & (BEFORE | AFTER | ES | C1 | B),
                        status,
                        "{context}: status"
                    );
                    let top = if popped { 7 } else { 6 };
                    assert_eq!((theirs.flags & TOP) >> 11, top, "{context}: TOP");
                }
            }
        }
    }
}
