//! Checks the two double-double references: libgcc's IBM `long double`
//! routines under QEMU, and QD's `dd_real`.
//!
//! The tests show that each reference gives known values, applies each
//! rounding direction, clears the flags before each operation, and returns
//! the flags of each operation. They also show that one QEMU process runs a
//! large batch whose outcomes do not depend on the order of the cases, and
//! that the QD calls restore the floating-point
//! environment of the thread.
//!
//! The expected values come from the references, and the tests pin them. A
//! comment gives the reason for a value where the reason is not plain.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty_verify::ibm_ldouble::{self, Case, Flags, Operation, Outcome, Pair, Rounding};
use floaty_verify::qd;
use floaty_verify::random::SplitMix64;

/// The encoding of 1.
const ONE: u64 = 0x3FF0_0000_0000_0000;
/// The encoding of 2.
const TWO: u64 = 0x4000_0000_0000_0000;
/// The encoding of 3.
const THREE: u64 = 0x4008_0000_0000_0000;
/// The encoding of 4.
const FOUR: u64 = 0x4010_0000_0000_0000;
/// The encoding of 0.5.
const HALF: u64 = 0x3FE0_0000_0000_0000;
/// The encoding of 2^-60.
const TWO_TO_MINUS_60: u64 = 0x3C30_0000_0000_0000;
/// The encoding of the largest finite binary64 value.
const MAX: u64 = 0x7FEF_FFFF_FFFF_FFFF;
/// The encoding of the smallest normal binary64 value plus one unit in the
/// last place. Half of it is not a binary64 value.
const MIN_NORMAL_PLUS_ULP: u64 = 0x0010_0000_0000_0001;
/// The encoding of positive infinity.
const INFINITY: u64 = 0x7FF0_0000_0000_0000;
/// The encoding of +0.
const ZERO: u64 = 0;
/// The encoding of -0.
const NEGATIVE_ZERO: u64 = 0x8000_0000_0000_0000;
/// The encoding of -1.
const NEGATIVE_ONE: u64 = 0xBFF0_0000_0000_0000;
/// The encoding of a signaling NaN.
const SIGNALING_NAN: u64 = 0x7FF4_0000_0000_0000;

/// The double-double value of one binary64 value.
const fn single(hi: u64) -> Pair {
    Pair::new(hi, ZERO)
}

/// 1/3 to nearest: the high half is the binary64 value nearest 1/3, and the
/// low half is the binary64 value nearest the remainder. Both references
/// give this pair.
const THIRD: Pair = Pair::new(0x3FD5_5555_5555_5555, 0x3C75_5555_5555_5555);

/// A quotient whose result differs in each rounding direction, in both
/// references.
const DIVIDEND: Pair = Pair::new(0x3FFC_2FF0_2BFA_FF60, 0x3CA0_C14D_401A_4E64);
/// The divisor of [`DIVIDEND`].
const DIVISOR: Pair = Pair::new(0x3FFD_D848_E7DA_C88C, 0xBCA0_E607_2710_DEB6);

/// Returns a case of the libgcc program.
const fn case(operation: Operation, rounding: Rounding, a: Pair, b: Pair) -> Case {
    Case {
        operation,
        rounding,
        a,
        b,
    }
}

/// Returns an outcome.
const fn outcome(result: Pair, flags: Flags) -> Outcome {
    Outcome { result, flags }
}

/// Runs the cases in one QEMU process and compares each outcome.
fn check_libgcc(expected: &[(Case, Outcome)]) {
    let cases: Vec<Case> = expected.iter().map(|&(case, _)| case).collect();
    let outcomes = ibm_ldouble::run(&cases);
    for (&(case, want), got) in expected.iter().zip(outcomes) {
        assert_eq!(got, want, "{case:?}");
    }
}

#[test]
fn libgcc_gives_known_values() {
    use Operation::{Add, Div};
    use Rounding::TiesToEven;
    check_libgcc(&[
        (
            case(Add, TiesToEven, single(ONE), single(ONE)),
            outcome(single(TWO), Flags::NONE),
        ),
        // 1 + 2^-60 needs two halves. The first binary64 addition rounds, so
        // the routine raises inexact although the pair is exact.
        (
            case(Add, TiesToEven, single(ONE), single(TWO_TO_MINUS_60)),
            outcome(Pair::new(ONE, TWO_TO_MINUS_60), Flags::INEXACT),
        ),
        (
            case(Div, TiesToEven, single(ONE), single(THREE)),
            outcome(THIRD, Flags::INEXACT),
        ),
    ]);
}

#[test]
fn libgcc_applies_each_rounding_direction() {
    use Operation::Div;
    use Rounding::{TiesToEven, TowardNegative, TowardPositive, TowardZero};
    // 1/3 has the same pair to nearest, toward zero, and toward negative.
    // Only the upward direction rounds the high half up.
    let third_up = Pair::new(0x3FD5_5555_5555_5556, 0xBC85_5555_5555_5555);
    let expected = [
        (TiesToEven, THIRD),
        (TowardZero, THIRD),
        (TowardPositive, third_up),
        (TowardNegative, THIRD),
    ]
    .map(|(rounding, result)| {
        (
            case(Div, rounding, single(ONE), single(THREE)),
            outcome(result, Flags::INEXACT),
        )
    });
    check_libgcc(&expected);

    let results = [
        (TiesToEven, 0x3FEE_3902_C02D_B5D4, 0x3C73_3221_DB20_34E8),
        (TowardZero, 0x3FEE_3902_C02D_B5D4, 0x3C73_3221_DB20_34D8),
        (TowardPositive, 0x3FEE_3902_C02D_B5D5, 0xBC9B_3377_8937_F2C8),
        (TowardNegative, 0x3FEE_3902_C02D_B5D4, 0x3C73_3221_DB20_34E0),
    ];
    let expected = results.map(|(rounding, hi, lo)| {
        (
            case(Div, rounding, DIVIDEND, DIVISOR),
            outcome(Pair::new(hi, lo), Flags::INEXACT),
        )
    });
    check_libgcc(&expected);
    assert_distinct(&expected.map(|(_, outcome)| outcome.result));
}

#[test]
fn libgcc_returns_the_flags_of_each_operation() {
    use Operation::{Add, Div, Mul, Sub};
    use Rounding::{TiesToEven, TowardPositive};
    check_libgcc(&[
        // `__gcc_qdiv` returns the first quotient when it is not finite.
        (
            case(Div, TiesToEven, single(ONE), single(ZERO)),
            outcome(single(INFINITY), Flags::DIVIDE_BY_ZERO),
        ),
        (
            case(Mul, TiesToEven, single(MAX), single(TWO)),
            outcome(single(INFINITY), Flags::OVERFLOW | Flags::INEXACT),
        ),
        // The default NaN of PowerPC is positive.
        (
            case(Sub, TiesToEven, single(INFINITY), single(INFINITY)),
            outcome(single(0x7FF8_0000_0000_0000), Flags::INVALID),
        ),
        // The product is tiny and inexact, so it underflows.
        (
            case(Mul, TiesToEven, single(MIN_NORMAL_PLUS_ULP), single(HALF)),
            outcome(
                single(0x0008_0000_0000_0000),
                Flags::UNDERFLOW | Flags::INEXACT,
            ),
        ),
        (
            case(Add, TiesToEven, single(SIGNALING_NAN), single(ONE)),
            outcome(single(0x7FFC_0000_0000_0000), Flags::INVALID),
        ),
        // Each case starts with clear flags and its own direction: an exact
        // case after an inexact one raises nothing, and a case to nearest
        // after an upward one rounds to nearest.
        (
            case(Div, TowardPositive, single(ONE), single(THREE)),
            outcome(
                Pair::new(0x3FD5_5555_5555_5556, 0xBC85_5555_5555_5555),
                Flags::INEXACT,
            ),
        ),
        (
            case(Add, TiesToEven, single(ONE), single(ONE)),
            outcome(single(TWO), Flags::NONE),
        ),
        (
            case(Div, TiesToEven, single(ONE), single(THREE)),
            outcome(THIRD, Flags::INEXACT),
        ),
    ]);
}

/// Returns a random case. Each half has random bits, so the cases include
/// every class of value and pairs that no operation returns.
fn random_case(random: &mut SplitMix64) -> Case {
    let operation = pick(random, &Operation::ALL);
    let rounding = pick(random, &Rounding::ALL);
    let a = Pair::new(random.next_u64(), random.next_u64());
    let b = Pair::new(random.next_u64(), random.next_u64());
    case(operation, rounding, a, b)
}

/// Returns a random element of `items`.
fn pick<T: Copy>(random: &mut SplitMix64, items: &[T]) -> T {
    let count = u64::try_from(items.len()).expect("a slice length fits in 64 bits");
    let index = usize::try_from(random.below(count)).expect("an index fits in usize");
    items[index]
}

#[test]
fn libgcc_runs_a_large_batch_in_one_process() {
    const COUNT: usize = 100_000;
    let mut random = SplitMix64::new(0xD0_0B1E);
    let cases: Vec<Case> = (0..COUNT).map(|_| random_case(&mut random)).collect();
    let outcomes = ibm_ldouble::run(&cases);
    assert_eq!(outcomes.len(), COUNT);

    // The outcome of a case does not depend on the cases before it.
    let reversed: Vec<Case> = cases.iter().rev().copied().collect();
    let again = ibm_ldouble::run(&reversed);
    assert!(
        again.iter().rev().eq(outcomes.iter()),
        "an outcome depends on the order of the cases"
    );
}

#[test]
fn qd_gives_known_values() {
    use Rounding::TiesToEven;
    let cases = [
        (
            qd::add(single(ONE), single(ONE), TiesToEven),
            outcome(single(TWO), Flags::NONE),
        ),
        (
            qd::add(single(ONE), single(TWO_TO_MINUS_60), TiesToEven),
            outcome(Pair::new(ONE, TWO_TO_MINUS_60), Flags::INEXACT),
        ),
        (
            qd::div(single(ONE), single(THREE), TiesToEven),
            outcome(THIRD, Flags::INEXACT),
        ),
        // QD's square root is not correctly rounded. The correctly rounded
        // low half of sqrt(2) is 0xBC9B_DD34_13B2_6456, and QD's is two units
        // in its last place lower. Each step of QD's algorithm, run with
        // Python's binary64 arithmetic and `math.fma`, gives QD's pair.
        (
            qd::sqrt(single(TWO), TiesToEven),
            outcome(
                Pair::new(0x3FF6_A09E_667F_3BCD, 0xBC9B_DD34_13B2_6458),
                Flags::INEXACT,
            ),
        ),
        (
            qd::sqrt(single(FOUR), TiesToEven),
            outcome(single(TWO), Flags::NONE),
        ),
        // QD returns +0 for the square root of a zero, and its own NaN
        // without a flag for the square root of a negative value.
        (
            qd::sqrt(single(NEGATIVE_ZERO), TiesToEven),
            outcome(single(ZERO), Flags::NONE),
        ),
        (
            qd::sqrt(single(NEGATIVE_ONE), TiesToEven),
            outcome(
                Pair::new(0x7FF8_0000_0000_0000, 0x7FF8_0000_0000_0000),
                Flags::NONE,
            ),
        ),
    ];
    for (index, (got, want)) in cases.into_iter().enumerate() {
        assert_eq!(got, want, "case {index}");
    }
}

#[test]
fn qd_applies_each_rounding_direction() {
    let results = [
        (
            Rounding::TiesToEven,
            0x3FEE_3902_C02D_B5D4,
            0x3C73_3221_DB20_34E9,
        ),
        (
            Rounding::TowardZero,
            0x3FEE_3902_C02D_B5D4,
            0x3C73_3221_DB20_34E4,
        ),
        (
            Rounding::TowardPositive,
            0x3FEE_3902_C02D_B5D5,
            0xBC9B_3377_8937_F2C6,
        ),
        (
            Rounding::TowardNegative,
            0x3FEE_3902_C02D_B5D4,
            0x3C73_3221_DB20_34E6,
        ),
    ];
    for (rounding, hi, lo) in results {
        assert_eq!(
            qd::div(DIVIDEND, DIVISOR, rounding),
            outcome(Pair::new(hi, lo), Flags::INEXACT),
            "{rounding:?}"
        );
    }
    assert_distinct(&results.map(|(_, hi, lo)| Pair::new(hi, lo)));
}

#[test]
fn qd_returns_the_flags_of_each_operation() {
    use Rounding::{TiesToEven, TowardPositive};
    // QD has no special cases in its arithmetic. The default NaN of x86 is
    // negative.
    let nan = Pair::new(0xFFF8_0000_0000_0000, 0xFFF8_0000_0000_0000);
    let cases = [
        (
            qd::div(single(ONE), single(ZERO), TiesToEven),
            outcome(nan, Flags::INVALID | Flags::DIVIDE_BY_ZERO),
        ),
        (
            qd::mul(single(MAX), single(TWO), TiesToEven),
            outcome(nan, Flags::INVALID | Flags::OVERFLOW | Flags::INEXACT),
        ),
        (
            qd::sub(single(INFINITY), single(INFINITY), TiesToEven),
            outcome(nan, Flags::INVALID),
        ),
        (
            qd::mul(single(MIN_NORMAL_PLUS_ULP), single(HALF), TiesToEven),
            outcome(
                single(0x0008_0000_0000_0000),
                Flags::UNDERFLOW | Flags::INEXACT,
            ),
        ),
        // Each call starts with clear flags and its own direction.
        (
            qd::div(single(ONE), single(THREE), TowardPositive),
            outcome(
                Pair::new(0x3FD5_5555_5555_5556, 0xBC85_5555_5555_5555),
                Flags::INEXACT,
            ),
        ),
        (
            qd::add(single(ONE), single(ONE), TiesToEven),
            outcome(single(TWO), Flags::NONE),
        ),
        (
            qd::div(single(ONE), single(THREE), TiesToEven),
            outcome(THIRD, Flags::INEXACT),
        ),
    ];
    for (index, (got, want)) in cases.into_iter().enumerate() {
        assert_eq!(got, want, "case {index}");
    }
}

#[test]
fn qd_restores_the_floating_point_environment() {
    use floaty_verify::x86::{
        MXCSR_DAZ, MXCSR_FTZ, MXCSR_MASKED, MXCSR_ROUNDINGS, mxcsr, with_mxcsr,
    };
    // The precision flag, PE, of MXCSR.
    const PRECISION: u32 = 1 << 5;
    // The shim keeps the FTZ and DAZ bits too, which `qd::with_flush` sets.
    for (rounding, field) in MXCSR_ROUNDINGS {
        for flags in [0, PRECISION, MXCSR_FTZ, MXCSR_DAZ | PRECISION] {
            let control = MXCSR_MASKED | field | flags;
            let after = with_mxcsr(control, || {
                for direction in Rounding::ALL {
                    let _ = qd::div(DIVIDEND, DIVISOR, direction);
                    let _ = qd::sqrt(single(TWO), direction);
                }
                mxcsr()
            });
            assert_eq!(after, control, "{rounding:?}, flags {flags:#x}");
        }
    }
}

/// Checks that no two results are equal.
fn assert_distinct(results: &[Pair]) {
    for (index, result) in results.iter().enumerate() {
        assert!(
            !results[index + 1..].contains(result),
            "{result:?} appears twice in {results:?}"
        );
    }
}
