//! Compares the x87 transcendental instructions `FSIN`, `FCOS`, `FSINCOS`,
//! `FPTAN`, `FPATAN`, and `F2XM1` with floaty's correctly rounded functions.
//! The test runs only on x86-64 hosts.
//!
//! The Intel SDM Volume 1, revision 253665-077US, section 8.3.10, bounds the
//! error of these instructions but does not state their bits. So a test
//! compares bits where the SDM defines them: special operands, arguments out
//! of range, and the registers that each instruction leaves. Elsewhere it
//! requires the result to lie within the bound of the SDM from floaty's
//! correctly rounded result. A consumer that needs the bits of one processor
//! confirms them on that processor.

#![cfg(target_arch = "x86_64")]

use floaty::{Env, F80, F256, Flags, Rounding, ToInt};
use floaty_verify::encodings::Layout;
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::{
    self, StackTop, X87_C1, X87_MASKED, X87_PRECISIONS, X87_ROUNDINGS, X87_STATUS_FLAGS,
    x87_arithmetic_status, x87_env,
};

/// The exception flags IE, DE, ZE, OE, UE, and PE.
const EXCEPTIONS: u16 = X87_STATUS_FLAGS & !X87_C1;
/// C2, which reports an argument out of range.
const C2: u16 = 1 << 10;
/// 1.5, the value below the operand, which shows whether an instruction
/// pushed or popped.
const MARKER: u128 = 0x3FFF_C000_0000_0000_0000;
/// 1.0.
const ONE: u128 = 0x3FFF_8000_0000_0000_0000;
/// The real indefinite, which an empty register stores.
const INDEFINITE: u128 = 0xFFFF_C000_0000_0000_0000;

/// An x87 transcendental instruction of the harness.
type Instruction = fn(u128, u128, u16) -> StackTop;

/// The control words of every rounding direction and precision, with every
/// exception masked.
fn controls(rounding_field: u16) -> impl Iterator<Item = u16> {
    X87_PRECISIONS
        .into_iter()
        .map(move |(_, precision_field)| X87_MASKED | rounding_field | precision_field)
}

/// Returns the distance in units of the last place between two finite
/// encodings of the same sign class: 0 for equal encodings.
fn distance(left: u128, right: u128) -> i128 {
    let ordinal = |bits: u128| {
        let field = i128::try_from((bits >> 64) & 0x7FFF).expect("15 bits fit an i128");
        let low = i128::try_from(bits & 0x7FFF_FFFF_FFFF_FFFF).expect("63 bits fit an i128");
        let magnitude = (field << 63) | low;
        if bits >> 79 == 1 {
            -magnitude
        } else {
            magnitude
        }
    };
    (ordinal(left) - ordinal(right)).abs()
}

/// The largest distance between the processor and floaty's correctly rounded
/// result that the SDM allows. The processor errs by less than 1 ulp to
/// nearest and by less than 1.5 ulps in a directed rounding (section
/// 8.3.10), and floaty by at most 0.5 ulp and less than 1 ulp, so the
/// results differ by at most 1 and 2 units of the last place.
fn tolerance(rounding: Rounding) -> i128 {
    if rounding == Rounding::TiesToEven {
        1
    } else {
        2
    }
}

/// The special operands: zeros, infinities, quiet and signaling NaNs, the
/// real indefinite, and the unsupported encodings.
const SPECIALS: [u128; 10] = [
    0,
    1 << 79,
    0x7FFF_8000_0000_0000_0000,
    0xFFFF_8000_0000_0000_0000,
    0x7FFF_C000_0000_0000_0001,
    0x7FFF_A000_0000_0000_0000,
    INDEFINITE,
    0x7FFF_4000_0000_0000_0000,
    0xFFFF_0000_0000_0000_0000,
    0x3FFF_0000_0000_0000_0001,
];

/// The arguments at or above 2^63 in magnitude, which `FSIN`, `FCOS`,
/// `FSINCOS`, and `FPTAN` do not reduce: 2^63, the next value, a random
/// value, and the largest finite value, of both signs.
const OUT_OF_RANGE: [u128; 8] = [
    0x403E_8000_0000_0000_0000,
    0xC03E_8000_0000_0000_0000,
    0x403E_8000_0000_0000_0001,
    0xC03E_8000_0000_0000_0001,
    0x4123_9ABC_DEF0_1234_5678,
    0xC123_9ABC_DEF0_1234_5678,
    0x7FFE_FFFF_FFFF_FFFF_FFFF,
    0xFFFE_FFFF_FFFF_FFFF_FFFF,
];

/// The trigonometric instructions, by name.
const TRIGONOMETRIC: [(&str, Instruction); 4] = [
    ("fsin", x86::fsin),
    ("fcos", x86::fcos),
    ("fsincos", x86::fsincos),
    ("fptan", x86::fptan),
];

/// An argument of 2^63 or more in magnitude sets C2 and changes no register
/// and no flag (Intel SDM Volume 2A, `FSIN`, `FCOS`, `FSINCOS`, and
/// `FPTAN`): `FSINCOS` and `FPTAN` push nothing.
#[test]
fn arguments_out_of_range_set_c2_and_change_nothing() {
    for (name, instruction) in TRIGONOMETRIC {
        for (_, rounding_field) in X87_ROUNDINGS {
            for control in controls(rounding_field) {
                for x in OUT_OF_RANGE {
                    let theirs = instruction(MARKER, x, control);
                    let context = format!("{name} {x:#x} control {control:#06x} {theirs:x?}");
                    assert_eq!(theirs.st0, x, "{context}: ST(0)");
                    assert_eq!(theirs.st1, MARKER, "{context}: ST(1)");
                    assert_eq!(theirs.status & (EXCEPTIONS | C2), C2, "{context}: status");
                }
            }
        }
    }
}

/// The registers that a special operand of a trigonometric instruction gives,
/// and the flags: floaty's correctly rounded results, which are exact.
///
/// A NaN result fills both registers of `FSINCOS` and `FPTAN`: `FPTAN` then
/// pushes the NaN, not 1.0. The SDM states only the value of ST(0), at
/// `FPTAN`, Table 3-33, and Volume 1, Table 8-10. The test host does this,
/// and so does Bochs 3.0 (`bochs/cpu/fpu/fpu_trans.cc`, line 126).
fn trigonometric_specials(name: &str, x: F80, env: Env) -> ([u128; 2], Flags) {
    let (sine, sine_flags) = x.sin_with(env);
    let (cosine, cosine_flags) = x.cos_with(env);
    let (tangent, tangent_flags) = x.tan_with(env);
    match name {
        "fsin" => ([sine.to_bits(), MARKER], sine_flags),
        "fcos" => ([cosine.to_bits(), MARKER], cosine_flags),
        "fsincos" if sine.is_nan() => ([sine.to_bits(); 2], sine_flags),
        "fsincos" => (
            [cosine.to_bits(), sine.to_bits()],
            sine_flags | cosine_flags,
        ),
        _ if tangent.is_nan() => ([tangent.to_bits(); 2], tangent_flags),
        _ => ([ONE, tangent.to_bits()], tangent_flags),
    }
}

/// The special operands of every transcendental instruction give exactly
/// floaty's results and flags, in every control setting. A zero gives a zero
/// or 1.0, an infinity is invalid, and a NaN propagates by the x87 rule. The
/// two-argument `FPATAN` follows IEEE 754 `atan2` for zeros and infinities:
/// it computes no `0 / 0` or `inf / inf` (Volume 2A, `FPATAN`, Table 3-30),
/// and its multiples of pi round as the rounding control selects.
#[test]
fn special_operands_give_the_exact_results() {
    let pair_operands: Vec<u128> = SPECIALS.into_iter().chain([ONE, ONE | (1 << 79)]).collect();
    for (rounding, rounding_field) in X87_ROUNDINGS {
        let env = x87_env(rounding);
        for control in controls(rounding_field) {
            for &x in &SPECIALS {
                let value = F80::from_bits(x);
                for (name, instruction) in TRIGONOMETRIC {
                    let theirs = instruction(MARKER, x, control);
                    let (registers, flags) = trigonometric_specials(name, value, env);
                    let context = format!("{name} {x:#x} control {control:#06x} {theirs:x?}");
                    assert_eq!([theirs.st0, theirs.st1], registers, "{context}: registers");
                    let status = x87_arithmetic_status(flags, value.is_nan());
                    assert_eq!(
                        theirs.status & (EXCEPTIONS | C2),
                        status & EXCEPTIONS,
                        "{context}: status"
                    );
                }
                let theirs = x86::f2xm1(MARKER, x, control);
                let (result, flags) = value.exp2_m1_with(env);
                let context = format!("f2xm1 {x:#x} control {control:#06x} {theirs:x?}");
                assert_eq!(
                    [theirs.st0, theirs.st1],
                    [result.to_bits(), MARKER],
                    "{context}: registers"
                );
                let status = x87_arithmetic_status(flags, value.is_nan());
                assert_eq!(
                    theirs.status & EXCEPTIONS,
                    status & EXCEPTIONS,
                    "{context}: status"
                );
            }
            for &y in &pair_operands {
                for &x in &pair_operands {
                    let (y_value, x_value) = (F80::from_bits(y), F80::from_bits(x));
                    if y_value.is_finite()
                        && !y_value.is_zero()
                        && x_value.is_finite()
                        && !x_value.is_zero()
                    {
                        continue;
                    }
                    let theirs = x86::fpatan(y, x, control);
                    let (result, flags) = y_value.atan2_with(x_value, env);
                    let context =
                        format!("fpatan {y:#x} {x:#x} control {control:#06x} {theirs:x?}");
                    assert_eq!(
                        [theirs.st0, theirs.st1],
                        [result.to_bits(), INDEFINITE],
                        "{context}: registers"
                    );
                    let status = x87_arithmetic_status(flags, y_value.is_nan() || x_value.is_nan());
                    assert_eq!(
                        theirs.status & EXCEPTIONS,
                        status & EXCEPTIONS,
                        "{context}: status"
                    );
                }
            }
        }
    }
}

/// Pi / 2, with Pi the 66-bit approximation of pi that the processor holds
/// for the argument reduction: pi rounded to nearest at 66 bits.
fn half_pi() -> F256 {
    F256::from_int((0xC90F_DAA2_2168_C234_u128 << 2) | 0b11).scale_b(-65)
}

/// Returns the sine and the cosine of `x * pi / Pi`, with Pi the 66-bit
/// approximation of pi. The SDM states that `FSIN`, `FCOS`, `FSINCOS`, and
/// `FPTAN` approximate these functions, not the sine and the cosine of `x`
/// (Volume 1, section 8.3.8). The reduction `x - k * Pi / 2` is exact in
/// binary256, which holds the 64 bits of `x` and the 129 bits of `k * Pi /
/// 2`.
fn sine_and_cosine(x: F80) -> (F256, F256) {
    let (wide, _): (F256, _) = x.convert_with(Env::IEEE);
    let half_pi = half_pi();
    let k = (wide / half_pi)
        .round_to_integral_with(Rounding::TiesToEven)
        .0;
    let reduced = wide - k * half_pi;
    let ToInt::Value(multiple) = k.to_int_with::<i64>(Env::IEEE).0 else {
        unreachable!("the multiple of pi/2 of an argument below 2^63 fits an i64");
    };
    let (sine, cosine) = (reduced.sin(), reduced.cos());
    match multiple.rem_euclid(4) {
        0 => (sine, cosine),
        1 => (cosine, -sine),
        2 => (-sine, -cosine),
        _ => (-cosine, sine),
    }
}

/// Returns random normal arguments below 2^63 in magnitude: small ones,
/// ones up to 2^63, and ones next to multiples of Pi / 2.
fn trigonometric_arguments(random: &mut SplitMix64) -> Vec<u128> {
    (0..3_000)
        .map(|index| {
            let negative = random.coin_flip();
            match index % 3 {
                0 => Layout::X87_EXTENDED.encode(
                    negative,
                    16_343 + random.below(40),
                    significand(random),
                ),
                1 => Layout::X87_EXTENDED.encode(
                    negative,
                    16_383 + random.below(63),
                    significand(random),
                ),
                _ => {
                    let k = i64::try_from(random.next_u64() >> (2 + random.below(61)))
                        .expect("below 2^62")
                        .max(1);
                    let near: F80 = (F256::from_int(k) * half_pi()).convert_with(Env::IEEE).0;
                    let step = u128::from(random.below(5));
                    (near.to_bits() + step - 2) | (u128::from(negative) << 79)
                }
            }
        })
        .collect()
}

/// Returns a random 64-bit significand with the integer bit set.
fn significand(random: &mut SplitMix64) -> u128 {
    u128::from(random.next_u64() | (1 << 63))
}

/// Checks one result of the processor against floaty's, within the bound of
/// the SDM.
fn check_bound(theirs: u128, ours: F80, rounding: Rounding, context: &str) {
    assert!(
        distance(theirs, ours.to_bits()) <= tolerance(rounding),
        "{context}: {theirs:#x} is not within the bound of {:#x}",
        ours.to_bits()
    );
}

/// The flags of a result that floaty gives exactly: IE, DE, ZE, OE, UE, and
/// PE, without C1. C1 reports whether the processor rounded its own
/// approximation up, which floaty does not model.
fn exceptions(flags: Flags, nan_operand: bool) -> u16 {
    x87_arithmetic_status(flags, nan_operand) & EXCEPTIONS
}

/// `FSIN`, `FCOS`, `FSINCOS`, and `FPTAN` lie within the bound of the SDM
/// from the correctly rounded functions of `x * pi / Pi`, in every control
/// setting, also next to the multiples of Pi / 2. Precision control does not
/// apply (Volume 1, section 8.1.5.2). The flags match floaty's: every
/// result is inexact, and a tiny one underflows.
#[test]
fn trigonometric_results_lie_within_the_bound() {
    let mut random = SplitMix64::new(0x0087_5100);
    let arguments = trigonometric_arguments(&mut random);
    let models: Vec<(F256, F256)> = arguments
        .iter()
        .map(|&x| sine_and_cosine(F80::from_bits(x)))
        .collect();
    for (rounding, rounding_field) in X87_ROUNDINGS {
        let env = x87_env(rounding);
        let rounded: Vec<[(F80, Flags); 3]> = models
            .iter()
            .map(|&(sine, cosine)| {
                [sine, cosine, sine / cosine].map(|value| value.convert_with(env))
            })
            .collect();
        for control in controls(rounding_field) {
            for (
                &x,
                [
                    (sine, sine_flags),
                    (cosine, cosine_flags),
                    (tangent, tangent_flags),
                ],
            ) in arguments.iter().zip(&rounded)
            {
                let input = F80::from_bits(x).sin_with(env).1 & Flags::DENORMAL_INPUT;
                for (name, instruction) in TRIGONOMETRIC {
                    let theirs = instruction(MARKER, x, control);
                    let context = format!("{name} {x:#x} control {control:#06x} {theirs:x?}");
                    let flags = match name {
                        "fsin" => {
                            check_bound(theirs.st0, *sine, rounding, &context);
                            assert_eq!(theirs.st1, MARKER, "{context}: ST(1)");
                            *sine_flags
                        }
                        "fcos" => {
                            check_bound(theirs.st0, *cosine, rounding, &context);
                            assert_eq!(theirs.st1, MARKER, "{context}: ST(1)");
                            *cosine_flags
                        }
                        "fsincos" => {
                            check_bound(theirs.st0, *cosine, rounding, &context);
                            check_bound(theirs.st1, *sine, rounding, &context);
                            *sine_flags | *cosine_flags
                        }
                        _ => {
                            assert_eq!(theirs.st0, ONE, "{context}: ST(0)");
                            check_bound(theirs.st1, *tangent, rounding, &context);
                            *tangent_flags
                        }
                    };
                    assert_eq!(
                        theirs.status & (EXCEPTIONS | C2),
                        exceptions(flags | input, false),
                        "{context}: status"
                    );
                }
            }
        }
    }
}

/// `FPATAN` and `F2XM1` lie within the bound of the SDM from floaty's
/// correctly rounded `atan2` and `exp2_m1`, in every control setting, and
/// give its flags. `F2XM1` takes arguments from -1 to +1 (Volume 2A,
/// `F2XM1`).
///
/// Conflict: `F2XM1` of -1 and of +1 has the exact results -0.5 and 1. The
/// SDM, Volume 2A, `F2XM1`, signals PE when the value cannot be represented
/// exactly, but Volume 1, section 8.5.6, states that the transcendental
/// instructions by nature give inexact results. The test host signals PE
/// for both, and in a rounding down or toward zero gives 1 - 2^-64 for +1,
/// which lies within the bound. Resolution: the test checks the value
/// within the bound and does not compare PE for these two arguments.
#[test]
fn arctangent_and_power_results_lie_within_the_bound() {
    let mut random = SplitMix64::new(0x0087_A7A2);
    let mut normal = |low: u64, span: u64| {
        let negative = random.coin_flip();
        let field = low + random.below(span);
        Layout::X87_EXTENDED.encode(negative, field, u128::from(random.next_u64() | (1 << 63)))
    };
    let pairs: Vec<(u128, u128)> = (0..6_000)
        .map(|_| (normal(16_343, 80), normal(16_343, 80)))
        .collect();
    let mut powers: Vec<u128> = (0..6_000).map(|_| normal(16_383 - 80, 80)).collect();
    powers.extend([
        ONE,
        ONE | (1 << 79),
        1,
        (1 << 79) | 1,
        0x0000_8000_0000_0000_0003,
    ]);
    for (rounding, rounding_field) in X87_ROUNDINGS {
        let env = x87_env(rounding);
        for control in controls(rounding_field) {
            for &(y, x) in &pairs {
                let theirs = x86::fpatan(y, x, control);
                let (result, flags) = F80::from_bits(y).atan2_with(F80::from_bits(x), env);
                let context = format!("fpatan {y:#x} {x:#x} control {control:#06x} {theirs:x?}");
                check_bound(theirs.st0, result, rounding, &context);
                assert_eq!(
                    theirs.status & EXCEPTIONS,
                    exceptions(flags, false),
                    "{context}: status"
                );
            }
            for &x in &powers {
                let theirs = x86::f2xm1(MARKER, x, control);
                let (result, flags) = F80::from_bits(x).exp2_m1_with(env);
                let context = format!("f2xm1 {x:#x} control {control:#06x} {theirs:x?}");
                check_bound(theirs.st0, result, rounding, &context);
                assert_eq!(theirs.st1, MARKER, "{context}: ST(1)");
                let compared = if x & !(1 << 79) == ONE {
                    EXCEPTIONS & !x86::X87_PE
                } else {
                    EXCEPTIONS
                };
                assert_eq!(
                    theirs.status & compared,
                    exceptions(flags, false) & compared,
                    "{context}: status"
                );
            }
        }
    }
}
