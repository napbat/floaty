//! Compares the NaN of a fused multiply-add under each hardware fused NaN
//! order with its reference.
//!
//! `FusedNanOrder::AddendFirst` follows the `FPMulAdd` pseudocode of the Arm
//! Architecture Reference Manual. No Arm processor or emulator runs in the
//! harness, so the test evaluates the published definition.
//!
//! `FPMulAdd(addend, op1, op2)` passes its operands to `FPProcessNaNs3` in
//! the order addend, op1, op2. `FPProcessNaNs3` returns the first signaling
//! NaN in that order, made quiet, or else the first quiet NaN, and signals
//! invalid for a signaling NaN. With the default-NaN mode, `FPProcessNaN`
//! gives the default NaN instead. A quiet NaN addend with a product of an
//! infinity and a zero gives the default NaN and signals invalid. QEMU's
//! `pickNaNMulAdd` for Arm, in `fpu/softfloat-specialize.c.inc`, gives the
//! same results. floaty's `mul_add(x, y, z)` is `FPMulAdd(z, x, y)`.
//!
//! A case without a NaN result must give the result of the default order,
//! which TestFloat checks, because the order only selects a NaN.
//!
//! `FusedNanOrder::AddendSecond` follows the PowerPC `fmadd` and `fmsub`
//! instructions under QEMU, which the `ibm_ldouble` batch program executes.
//! That test compares every result and every flag.

use floaty::env::{FusedNanOrder, NanPropagation, NanRule};
use floaty::{Class, D64Bid, Env, F32, F64, Flags};
use floaty_verify::ibm_ldouble::{
    self, Flags as ReferenceFlags, Instruction, InstructionCase, Rounding,
};

/// The Arm behaviors: `SignalingFirst` for the default-NaN mode off, and
/// `DefaultNan` for it on, both with `InvalidProduct::Signals`.
fn arm_behaviors() -> [(Env, bool); 2] {
    let rule = |propagation| {
        Env::IEEE.with_nan(NanRule::new(propagation).with_fused_order(FusedNanOrder::AddendFirst))
    };
    [
        (rule(NanPropagation::SignalingFirst), false),
        (rule(NanPropagation::DefaultNan), true),
    ]
}

/// Returns the class of an operand for the pseudocode.
fn kind(class: Class) -> Kind {
    match class {
        Class::QuietNan => Kind::Quiet,
        Class::SignalingNan => Kind::Signaling,
        Class::Infinite => Kind::Infinity,
        Class::Zero => Kind::Zero,
        _ => Kind::Number,
    }
}

/// The operand types that `FPMulAdd` tells apart.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Quiet,
    Signaling,
    Infinity,
    Zero,
    Number,
}

/// Returns the index of the operand whose NaN `FPMulAdd` returns, or `None`
/// for the default NaN, and whether it signals invalid. `kinds` holds the
/// addend, op1, and op2. Returns `Err` when the result is not a NaN.
fn fp_mul_add(kinds: [Kind; 3], default_nan_mode: bool) -> Result<(Option<usize>, bool), ()> {
    let [addend, op1, op2] = kinds;
    let invalid_product = (op1 == Kind::Infinity && op2 == Kind::Zero)
        || (op1 == Kind::Zero && op2 == Kind::Infinity);
    // FPProcessNaNs3: the signaling NaNs in order, then the quiet NaNs.
    let chosen = kinds
        .iter()
        .position(|&kind| kind == Kind::Signaling)
        .or_else(|| kinds.iter().position(|&kind| kind == Kind::Quiet));
    let signaling = kinds.contains(&Kind::Signaling);
    if addend == Kind::Quiet && invalid_product {
        return Ok((None, true));
    }
    match chosen {
        Some(index) => Ok(((!default_nan_mode).then_some(index), signaling)),
        None if invalid_product => Ok((None, true)),
        None => Err(()),
    }
}

/// Runs every triple of the special values of one format.
macro_rules! check {
    ($format:ty, $values:expr, $quiet:expr, $default:expr) => {{
        let values: Vec<$format> = $values
            .iter()
            .map(|&bits| <$format>::from_bits(bits))
            .collect();
        let mut nan_results = 0;
        for (env, default_nan_mode) in arm_behaviors() {
            for &x in &values {
                for &y in &values {
                    for &z in &values {
                        let (result, flags) = x.mul_add_with(y, z, env);
                        let kinds = [kind(z.classify()), kind(x.classify()), kind(y.classify())];
                        let operands = [z, x, y];
                        let context = format!(
                            "{} fma({:x?}, {:x?}, {:x?}) in {env:?}",
                            stringify!($format),
                            x.to_bits(),
                            y.to_bits(),
                            z.to_bits()
                        );
                        match fp_mul_add(kinds, default_nan_mode) {
                            Ok((chosen, invalid)) => {
                                nan_results += 1;
                                let expected = chosen
                                    .map_or($default, |index| $quiet(operands[index].to_bits()));
                                let expected_flags =
                                    if invalid { Flags::INVALID } else { Flags::NONE };
                                assert_eq!(
                                    (result.to_bits(), flags),
                                    (expected, expected_flags),
                                    "{context}"
                                );
                            }
                            Err(()) => {
                                let default = x.mul_add_with(
                                    y,
                                    z,
                                    env.with_nan(
                                        env.nan.with_fused_order(FusedNanOrder::ProductFirst),
                                    ),
                                );
                                assert_eq!(
                                    (result.to_bits(), flags),
                                    (default.0.to_bits(), default.1),
                                    "{context}"
                                );
                            }
                        }
                    }
                }
            }
        }
        nan_results
    }};
}

#[test]
fn the_addend_first_order_follows_the_arm_pseudocode() {
    // +0, -0, 1, -2, +inf, -inf, quiet NaNs, and signaling NaNs of both signs.
    let binary32 = [
        0_u32,
        0x8000_0000,
        0x3F80_0000,
        0xC000_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0001,
        0xFFC0_0002,
        0x7F80_0003,
        0xFF80_0004,
    ];
    let nans = check!(F32, binary32, |bits: u32| bits | 0x0040_0000, 0x7FC0_0000);
    assert_eq!(nans, 1_664, "the binary32 cases with a NaN result");
    let binary64 = [
        0_u64,
        0x8000_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0xC000_0000_0000_0000,
        0x7FF0_0000_0000_0000,
        0xFFF0_0000_0000_0000,
        0x7FF8_0000_0000_0001,
        0xFFF8_0000_0000_0002,
        0x7FF0_0000_0000_0003,
        0xFFF0_0000_0000_0004,
    ];
    let quiet64 = |bits: u64| bits | 0x0008_0000_0000_0000;
    assert_eq!(check!(F64, binary64, quiet64, 0x7FF8_0000_0000_0000), 1_664);
    let decimal64 = [
        0x31C0_0000_0000_0000_u64,
        0xB1C0_0000_0000_0000,
        0x31C0_0000_0000_0001,
        0xB1C0_0000_0000_0002,
        0x7800_0000_0000_0000,
        0xF800_0000_0000_0000,
        0x7C00_0000_0000_0001,
        0xFC00_0000_0000_0002,
        0x7E00_0000_0000_0003,
        0xFE00_0000_0000_0004,
    ];
    // A decimal NaN is quiet when the bit below its five leading combination
    // bits is clear.
    let quiet_decimal = |bits: u64| bits & !0x0200_0000_0000_0000;
    let decimal_nans = check!(D64Bid, decimal64, quiet_decimal, 0x7C00_0000_0000_0000);
    assert_eq!(decimal_nans, 1_664);
}

#[test]
fn the_addend_second_order_follows_powerpc() {
    // The special values of the Arm test, the smallest subnormal, and the
    // largest finite value.
    let values = [
        0_u64,
        0x8000_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0xC000_0000_0000_0000,
        0x7FF0_0000_0000_0000,
        0xFFF0_0000_0000_0000,
        0x7FF8_0000_0000_0001,
        0xFFF8_0000_0000_0002,
        0x7FF0_0000_0000_0003,
        0xFFF0_0000_0000_0004,
        0x0000_0000_0000_0001,
        0x7FEF_FFFF_FFFF_FFFF,
    ];
    let values = &values;
    let cases: Vec<InstructionCase> = Instruction::ALL
        .into_iter()
        .flat_map(|instruction| Rounding::ALL.map(|rounding| (instruction, rounding)))
        .flat_map(|(instruction, rounding)| {
            values.iter().flat_map(move |&a| {
                values.iter().flat_map(move |&c| {
                    values.iter().map(move |&b| InstructionCase {
                        instruction,
                        rounding,
                        a,
                        c,
                        b,
                    })
                })
            })
        })
        .collect();
    let outcomes = ibm_ldouble::run_instructions(&cases);
    let mut nan_results = 0;
    for (case, outcome) in cases.iter().zip(&outcomes) {
        let (a, c, b) = (
            F64::from_bits(case.a),
            F64::from_bits(case.c),
            F64::from_bits(case.b),
        );
        // `fmsub` selects a NaN operand before it negates the addend.
        let addend = match case.instruction {
            Instruction::MultiplySubtract if !b.is_nan() => -b,
            _ => b,
        };
        let (result, flags) = a.mul_add_with(c, addend, ibm_ldouble::behavior(case.rounding));
        assert_eq!(
            (result.to_bits(), ReferenceFlags::from_floaty(flags)),
            (outcome.result, outcome.flags),
            "{case:x?}"
        );
        nan_results += usize::from(result.is_nan());
    }
    assert_eq!(
        (cases.len(), nan_results),
        (13_824, 10_400),
        "the cases, and those with a NaN result"
    );
}
