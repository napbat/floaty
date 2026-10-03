//! Compares the operations of `Gcc` on the exact value with the IBM `long
//! double` functions of the libm of glibc 2.43 under QEMU, for canonical
//! pairs: `logbl`, `copysignl`, `ilogbl`, the roundings to integral values,
//! `scalbnl`, the minimum and maximum operations, `totalorderl`,
//! `totalordermagl`, `llrintl`, and `lroundl`.
//!
//! floaty reads the exact value `hi + lo` by the rules of `DoubleDouble`,
//! and glibc reads the halves. For a canonical pair the two agree, but for
//! the conflicts that the comments of the comparison functions record. Each
//! case compares both halves bit for bit and the five IEEE flags, in each of
//! the four rounding directions, under the behavior of PowerPC,
//! `ibm_ldouble::behavior`. In the direction to nearest, the operations of
//! the default mode give the results of the `_with` methods.
//! `double-double-rounding.rs` checks floaty's rules with exact rationals,
//! also where glibc conflicts with them.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::cmp::Ordering;

use floaty::env::Rounding as Direction;
use floaty::{DoubleDouble, F64, Flags, Gcc, ToInt};
use floaty_verify::double_double::{canonical, operand};
use floaty_verify::encodings::Layout;
use floaty_verify::ibm_ldouble::{
    self, Flags as ReferenceFlags, Function, FunctionCase, Outcome, Pair, Rounding,
};
use floaty_verify::random::SplitMix64;

/// The layout of the halves.
const BINARY64: Layout = Layout::BINARY64;

fn value(pair: Pair) -> DoubleDouble<Gcc> {
    DoubleDouble::from_parts(F64::from_bits(pair.hi), F64::from_bits(pair.lo))
}

fn halves(value: DoubleDouble<Gcc>) -> Pair {
    Pair::new(value.hi().to_bits(), value.lo().to_bits())
}

/// Returns floaty's outcome of a result and its flags.
fn outcome((result, flags): (DoubleDouble<Gcc>, Flags)) -> Outcome {
    Outcome {
        result: halves(result),
        flags: ReferenceFlags::from_floaty(flags),
    }
}

/// Returns the outcome of an integer result, as the program gives it: the
/// bits of the integer as the high half.
fn integer(value: i64, flags: Flags) -> Outcome {
    Outcome {
        result: Pair::new(value.cast_unsigned(), 0),
        flags: ReferenceFlags::from_floaty(flags),
    }
}

/// Returns `true` for the encoding of a NaN or an infinity.
fn special(bits: u64) -> bool {
    bits & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000
}

/// Returns `flags` without [`ReferenceFlags::INEXACT`].
fn exact_flags(flags: ReferenceFlags) -> ReferenceFlags {
    ReferenceFlags::from_bits(flags.bits() & !ReferenceFlags::INEXACT.bits())
        .expect("the flags keep the five bits")
}

/// Returns the message of a mismatch, or `None` when the outcomes are equal.
fn differ(case: &FunctionCase, ours: Outcome, theirs: Outcome) -> Option<String> {
    (ours != theirs).then(|| format!("{case:?}: floaty {ours:?}, glibc {theirs:?}"))
}

/// Compares a result pair with glibc's.
///
/// Conflict: the sign of a zero low half. glibc 2.43
/// `sysdeps/ieee754/ldbl-128ibm/s_ceill.c` line 53, and the same line of
/// the floor, trunc, round, roundeven, and rint functions, takes the low
/// half from the binary64 rounding of the low half, so `ceill(-2^80 - 0.5)`
/// gives a `-0` low half. Lines 37-40 return a zero pair as it is, and
/// `s_scalbnl.c` line 97 keeps the sign of a dropped low half. floaty gives
/// a pair a `+0` low half when the rest is zero, as the documentation of
/// `DoubleDouble` states. Both pairs hold one exact value. Resolution:
/// floaty keeps its rule, and two zero low halves compare equal.
///
/// Conflict: the low half of a NaN or an infinity. `s_ceill.c` lines 37-40
/// and 59-61, and `s_rintl.c` lines 46-49 and 126-128, return a NaN or an
/// infinity with its low half, and `math/s_fmax_template.c` line 34 returns
/// a NaN operand as it is. floaty gives a NaN result, and an infinity that
/// rounds to an integral value, a `+0` low half, as `round_to_integral_with`
/// and the minimum and maximum operations state. Resolution: floaty keeps
/// its rule. For a NaN or an infinity with a `+0` low half from floaty, the
/// test compares the high half and the flags.
fn same_pair(case: &FunctionCase, ours: Outcome, theirs: Outcome) -> Option<String> {
    let result = if special(ours.result.hi) && ours.result.lo == 0 {
        Pair::new(theirs.result.hi, 0)
    } else if (ours.result.lo | theirs.result.lo) << 1 == 0 {
        Pair::new(theirs.result.hi, ours.result.lo)
    } else {
        theirs.result
    };
    differ(case, ours, Outcome { result, ..theirs })
}

/// Compares a rounding to an integral value in the direction `direction`.
/// `floorl`, `ceill`, `truncl`, `roundl`, `roundevenl`, and `nearbyintl`
/// are the IEEE 754 `roundToIntegral` operations, which do not signal
/// inexact. floaty's `round_to_integral_with` is `roundToIntegralExact`, so
/// their flags compare without inexact.
///
/// Conflict: the inexact flag of `rintl`. C23 section F.10.6.4 requires
/// `rint` to raise inexact only when the result differs from the operand.
/// glibc 2.43 `s_rintl.c` lines 74 and 81 canonicalize intermediate pairs
/// with binary64 additions, which raise inexact also when the result is
/// exact. floaty follows IEEE 754 `roundToIntegralExact`. Resolution: glibc
/// must raise inexact where floaty does, and can also raise it elsewhere.
fn rounding(case: &FunctionCase, direction: Direction, theirs: Outcome) -> Option<String> {
    let env = ibm_ldouble::behavior(case.rounding).with_rounding(direction);
    let ours = outcome(value(case.operands[0]).round_to_integral_with(env));
    if case.function != Function::Rint {
        let ours = Outcome {
            flags: exact_flags(ours.flags),
            ..ours
        };
        return same_pair(case, ours, theirs);
    }
    let inexact = ReferenceFlags::INEXACT;
    if ours.flags.contains(inexact) && !theirs.flags.contains(inexact) {
        return differ(case, ours, theirs);
    }
    let ours = Outcome {
        flags: exact_flags(ours.flags),
        ..ours
    };
    let theirs = Outcome {
        flags: exact_flags(theirs.flags),
        ..theirs
    };
    same_pair(case, ours, theirs)
}

/// Compares `scalbnl` with `scale_b_with`.
///
/// Conflict: a scaled value outside the normal range. glibc 2.43
/// `s_scalbnl.c` scales each half on its own. Lines 81-82 drop a low half
/// that falls 54 or more exponents below the normal range, without flags.
/// Lines 94-102 round the high half alone for a subnormal result. Lines
/// 59-60 give an overflow as the libgcc product of two huge values, which in
/// a directed rounding is an infinity of the wrong sign, such as `-inf` for
/// a positive value toward negative infinity. floaty rounds the exact value
/// to a pair by the rule of `DoubleDouble`. Resolution: floaty keeps its
/// rule. The test compares every result that floaty gives exact. Otherwise
/// it compares the high half of a result at or above 2^-968 that does not
/// overflow. There a subnormal low half rounds to less than half an ulp of
/// the high half, so both give the scaled high half.
fn scaled(case: &FunctionCase, theirs: Outcome) -> Option<String> {
    let env = ibm_ldouble::behavior(case.rounding);
    let scale = i32::try_from(case.operands[1].hi.cast_signed())
        .expect("the test gives exponents in the range of i32");
    let ours = outcome(value(case.operands[0]).scale_b_with(scale, env));
    if ours.flags == ReferenceFlags::NONE {
        return same_pair(case, ours, theirs);
    }
    let field = (ours.result.hi >> 52) & 0x7FF;
    let scaled_high =
        (55..0x7FF).contains(&field) && !ours.flags.contains(ReferenceFlags::OVERFLOW);
    (scaled_high && ours.result.hi != theirs.result.hi)
        .then(|| format!("{case:?}: floaty {ours:?}, glibc {theirs:?}"))
}

/// Returns `true` for two different pairs of one exact value: the same
/// number, or zeros of different signs when `zeros` is `true`.
fn one_value(a: DoubleDouble<Gcc>, b: DoubleDouble<Gcc>, zeros: bool) -> bool {
    let equal = a.compare_quiet(b) == Some(Ordering::Equal);
    let signs = zeros || a.is_sign_negative() == b.is_sign_negative();
    equal && signs && halves(a) != halves(b)
}

/// Compares a minimum or maximum operation.
///
/// Conflict: two different pairs of one exact value. glibc 2.43
/// `math/s_fmax_template.c` lines 27-28, and the same lines of the other
/// templates, return the first operand when the operands compare equal.
/// `fmaxl` and `fminl` do so also for zeros of different signs, which IEEE
/// 754-2008 `maxNum` and `minNum` allow. floaty orders two pairs of one
/// value by their halves, as `total_cmp_with` with `TotalOrder::Encoding`
/// does, and orders `-0` below `+0`, as the minimum and maximum operations
/// state. Resolution: floaty keeps its rule, and the test skips such pairs
/// of operands.
fn min_max(case: &FunctionCase, theirs: Outcome) -> Option<String> {
    let env = ibm_ldouble::behavior(case.rounding);
    let [a, b, _] = case.operands.map(value);
    let zeros = matches!(case.function, Function::Fmax | Function::Fmin);
    if one_value(a, b, zeros) {
        return None;
    }
    let result = match case.function {
        Function::Fmax => a.max_num_with(b, env),
        Function::Fmin => a.min_num_with(b, env),
        Function::Fmaximum => a.maximum_with(b, env),
        Function::Fminimum => a.minimum_with(b, env),
        Function::FmaximumNumber => a.maximum_number_with(b, env),
        Function::FminimumNumber => a.minimum_number_with(b, env),
        Function::FmaximumMagnitude => a.maximum_magnitude_with(b, env),
        Function::FminimumMagnitude => a.minimum_magnitude_with(b, env),
        Function::FmaximumMagnitudeNumber => a.maximum_magnitude_number_with(b, env),
        Function::FminimumMagnitudeNumber => a.minimum_magnitude_number_with(b, env),
        _ => unreachable!("the caller passes a minimum or maximum operation"),
    };
    same_pair(case, outcome(result), theirs)
}

/// Compares `totalorderl` and `totalordermagl`. IEEE 754 defines
/// `totalOrderMag(x, y)` as `totalOrder(abs(x), abs(y))`.
///
/// glibc 2.43 `s_totalorderl.c` lines 47-55 and `s_totalordermagl.c` lines
/// 47-55 return 1 when the high halves are one NaN or one infinity, or when
/// both low halves are zeros: two canonical pairs of one datum. The default
/// mode of `Gcc` orders by `TotalOrder::Datum`, which makes such pairs equal,
/// so `total_cmp` gives 1 in both orders too.
fn total_order(case: &FunctionCase, theirs: Outcome) -> Option<String> {
    let [a, b, _] = case.operands.map(value);
    let order = if case.function == Function::TotalOrder {
        a.total_cmp(b)
    } else {
        a.abs().total_cmp(b.abs())
    };
    differ(
        case,
        integer(i64::from(order != Ordering::Greater), Flags::NONE),
        theirs,
    )
}

/// Compares `llrintl` and `lroundl` with `to_int_with` into `i64`.
///
/// Conflict: the flags of an integer out of range. glibc 2.43
/// `s_llrintl.c` line 68 and `s_lroundl.c` line 79 convert the fraction to
/// an integer before lines 118-119 and 113-114 find the overflow, so they
/// can raise inexact with invalid. IEEE 754 section 5.8 signals only
/// invalid, and C23 leaves the integer unspecified. Resolution: floaty keeps
/// its rule. For an integer out of range, the test checks only that glibc
/// raises invalid.
fn to_int(case: &FunctionCase, theirs: Outcome) -> Option<String> {
    let env = ibm_ldouble::behavior(case.rounding);
    let env = if case.function == Function::LRound {
        env.with_rounding(Direction::TiesToAway)
    } else {
        env
    };
    match value(case.operands[0]).to_int_with::<i64>(env) {
        (ToInt::Value(result), flags) => differ(case, integer(result, flags), theirs),
        (_, flags) => (!theirs.flags.contains(ReferenceFlags::INVALID)).then(|| {
            format!("{case:?}: floaty signals {flags:?}, glibc {theirs:?} is not invalid")
        }),
    }
}

/// Compares `ilogbl` with `log_b` for a finite nonzero operand. floaty has
/// no `ilogb`, so the integers of a zero, an infinity, and a NaN have no
/// counterpart.
fn exponent(case: &FunctionCase, theirs: Outcome) -> Option<String> {
    let a = value(case.operands[0]);
    if !a.is_finite() || a.is_zero() {
        return None;
    }
    let ToInt::Value(exponent) = a.log_b().to_int::<i64>() else {
        unreachable!("the exponent of a finite pair fits an i64");
    };
    differ(case, integer(exponent, Flags::NONE), theirs)
}

/// Returns the message of a mismatch between floaty and glibc's outcome of
/// a case, or `None`.
fn check(case: &FunctionCase, theirs: Outcome) -> Option<String> {
    let env = ibm_ldouble::behavior(case.rounding);
    let [a, b, _] = case.operands.map(value);
    match case.function {
        Function::LogB => differ(case, outcome(a.log_b_with(env)), theirs),
        Function::CopySign => differ(case, outcome((a.copy_sign(b), Flags::NONE)), theirs),
        Function::ILogB => exponent(case, theirs),
        Function::Floor => rounding(case, Direction::TowardNegative, theirs),
        Function::Ceil => rounding(case, Direction::TowardPositive, theirs),
        Function::Trunc => rounding(case, Direction::TowardZero, theirs),
        Function::Round => rounding(case, Direction::TiesToAway, theirs),
        Function::RoundEven => rounding(case, Direction::TiesToEven, theirs),
        Function::Rint | Function::NearbyInt => rounding(case, case.rounding.into(), theirs),
        Function::ScaleB => scaled(case, theirs),
        Function::TotalOrder | Function::TotalOrderMagnitude => total_order(case, theirs),
        Function::LlRint | Function::LRound => to_int(case, theirs),
        Function::Fmax
        | Function::Fmin
        | Function::Fmaximum
        | Function::Fminimum
        | Function::FmaximumNumber
        | Function::FminimumNumber
        | Function::FmaximumMagnitude
        | Function::FminimumMagnitude
        | Function::FmaximumMagnitudeNumber
        | Function::FminimumMagnitudeNumber => min_max(case, theirs),
        _ => unreachable!("double-double-functions.rs compares the other functions"),
    }
}

/// Returns the message of a mismatch between an operation of the default
/// mode and its `_with` method, or `None`. The default mode of `Gcc` rounds
/// to nearest.
fn check_default(case: &FunctionCase) -> Option<String> {
    let env = ibm_ldouble::behavior(Rounding::TiesToEven);
    let [a, b, _] = case.operands.map(value);
    let (ours, with) = match case.function {
        Function::LogB => (a.log_b(), a.log_b_with(env).0),
        Function::RoundEven => (a.round_to_integral(), a.round_to_integral_with(env).0),
        Function::ScaleB => {
            let scale = i32::try_from(case.operands[1].hi.cast_signed())
                .expect("the test gives exponents in the range of i32");
            (a.scale_b(scale), a.scale_b_with(scale, env).0)
        }
        Function::Fmax => (a.max_num(b), a.max_num_with(b, env).0),
        Function::Fminimum => (a.minimum(b), a.minimum_with(b, env).0),
        _ => return None,
    };
    (halves(ours) != halves(with)).then(|| format!("{case:?} in the default mode: {ours:?}"))
}

/// Compares every case with glibc, and returns the number of cases.
fn compare(cases: &[FunctionCase]) -> usize {
    let outcomes = ibm_ldouble::run_functions(cases);
    let failures: Vec<String> = cases
        .iter()
        .zip(outcomes)
        .flat_map(|(case, theirs)| {
            let default = (case.rounding == Rounding::TiesToEven)
                .then(|| check_default(case))
                .flatten();
            [check(case, theirs), default]
        })
        .flatten()
        .take(40)
        .collect();
    for failure in &failures {
        println!("FAILED {failure}");
    }
    assert!(failures.is_empty(), "floaty matches glibc");
    cases.len()
}

/// Zeros, infinities, NaNs with and without a low half, ones with low halves
/// of each sign and a `-0` low half, the smallest subnormals, the largest
/// pairs, and pairs at the edges of the range of `i64`.
const EDGES: [Pair; 22] = [
    Pair::new(0, 0),
    Pair::new(1 << 63, 0),
    Pair::new(0x7FF0_0000_0000_0000, 0),
    Pair::new(0xFFF0_0000_0000_0000, 0),
    Pair::new(0x7FF0_0000_0000_0000, 0x3FF0_0000_0000_0000),
    Pair::new(0x7FF8_0000_0000_0001, 0),
    Pair::new(0xFFF8_0000_0000_0002, 0),
    Pair::new(0x7FF4_0000_0000_0000, 0),
    Pair::new(0x7FF8_0000_0000_0001, 0x8000_0000_0000_0001),
    Pair::new(0x3FF0_0000_0000_0000, 0),
    Pair::new(0xBFF0_0000_0000_0000, 0),
    Pair::new(0x3FF0_0000_0000_0000, 0x3C80_0000_0000_0000),
    Pair::new(0x3FF0_0000_0000_0000, 0xBC80_0000_0000_0000),
    Pair::new(0x3FF0_0000_0000_0000, 0x8000_0000_0000_0000),
    Pair::new(0x0000_0000_0000_0001, 0),
    Pair::new(0x8000_0000_0000_0001, 0),
    Pair::new(0x7FEF_FFFF_FFFF_FFFF, 0x7C8F_FFFF_FFFF_FFFF),
    Pair::new(0xFFEF_FFFF_FFFF_FFFF, 0xFC8F_FFFF_FFFF_FFFF),
    Pair::new(0x43E0_0000_0000_0000, 0xBFE0_0000_0000_0000),
    Pair::new(0xC3E0_0000_0000_0000, 0x3FE0_0000_0000_0000),
    Pair::new(0xC3E0_0000_0000_0000, 0xBFE0_0000_0000_0000),
    Pair::new(0x43E0_0000_0000_0000, 0xBFF0_0000_0000_0000),
];

/// Returns a pair near an integer: a high half between 2^-2 and 2^110,
/// often integral, and a low half of a half, of one, of zero, or of any
/// magnitude below the high half.
fn near_integer(random: &mut SplitMix64) -> Pair {
    let negative = random.coin_flip();
    let field = 1021 + random.below(112);
    let mut hi = BINARY64.encode_u64(negative, field, random.next_u64());
    if field >= 1023 + 52 && random.coin_flip() {
        hi &= !0xF;
    }
    let lo = match random.below(5) {
        0 => 0x3FE0_0000_0000_0000 | (random.next_u64() & (1 << 63)),
        1 => 0x3FF0_0000_0000_0000 | (random.next_u64() & (1 << 63)),
        2 => 0,
        _ => {
            let low_field = field.saturating_sub(54 + random.below(80)).max(1);
            BINARY64.encode_u64(random.coin_flip(), low_field, random.next_u64())
        }
    };
    Pair::new(hi, lo)
}

/// Returns canonical operands: the edges, and random pairs of `generate`.
fn operands(
    random: &mut SplitMix64,
    count: usize,
    generate: fn(&mut SplitMix64) -> Pair,
) -> Vec<Pair> {
    EDGES
        .into_iter()
        .chain((0..count).map(|_| canonical(random, generate)))
        .collect()
}

/// Returns the cases of a function of one operand for every operand and
/// rounding direction.
fn unary(function: Function, operands: &[Pair]) -> Vec<FunctionCase> {
    operands
        .iter()
        .flat_map(|&a| {
            Rounding::ALL.map(|rounding| FunctionCase {
                function,
                rounding,
                operands: [a, Pair::default(), Pair::default()],
            })
        })
        .collect()
}

/// Returns the cases of a function of two operands: every pair of edges,
/// then each operand with a random operand or with itself, in every
/// rounding direction.
fn binary(function: Function, operands: &[Pair], random: &mut SplitMix64) -> Vec<FunctionCase> {
    let count = u64::try_from(operands.len()).expect("a length fits a u64");
    let mut pairs: Vec<(Pair, Pair)> = EDGES
        .iter()
        .flat_map(|&a| EDGES.iter().map(move |&b| (a, b)))
        .collect();
    pairs.extend(operands.iter().map(|&a| {
        let index = usize::try_from(random.below(count)).expect("an index fits a usize");
        (
            a,
            if random.below(4) == 0 {
                a
            } else {
                operands[index]
            },
        )
    }));
    pairs
        .into_iter()
        .flat_map(|(a, b)| {
            Rounding::ALL.map(|rounding| FunctionCase {
                function,
                rounding,
                operands: [a, b, Pair::default()],
            })
        })
        .collect()
}

#[test]
fn log_b_and_copy_sign_match_glibc_for_canonical_pairs() {
    let mut random = SplitMix64::new(0x6_10B8);
    let operands = operands(&mut random, 20_000, operand);
    let mut count = compare(&unary(Function::LogB, &operands));
    count += compare(&unary(Function::ILogB, &operands));
    count += compare(&binary(Function::CopySign, &operands, &mut random));
    assert_eq!(count, 4 * (3 * 20_022 + 22 * 22));
}

#[test]
fn roundings_to_integral_values_match_glibc_for_canonical_pairs() {
    let mut random = SplitMix64::new(0xE8AC_7001);
    let mut operands = operands(&mut random, 10_000, near_integer);
    operands.extend((0..4_000).map(|_| canonical(&mut random, operand)));
    let mut count = 0;
    for function in [
        Function::Floor,
        Function::Ceil,
        Function::Trunc,
        Function::Round,
        Function::RoundEven,
        Function::Rint,
        Function::NearbyInt,
        Function::LlRint,
        Function::LRound,
    ] {
        count += compare(&unary(function, &operands));
    }
    assert_eq!(count, 9 * 4 * 14_022);
}

#[test]
fn scale_b_matches_glibc_for_canonical_pairs() {
    let mut random = SplitMix64::new(0xE8AC_7002);
    let operands = operands(&mut random, 20_000, operand);
    let cases: Vec<FunctionCase> = operands
        .iter()
        .flat_map(|&a| {
            // Small exponents, exponents that cross the subnormal range or
            // overflow, and exponents beyond every pair.
            let scale: i64 = match random.below(4) {
                0 => i64::try_from(random.below(21)).expect("fits") - 10,
                1 => i64::try_from(random.below(201)).expect("fits") - 100,
                2 => i64::try_from(random.below(4401)).expect("fits") - 2200,
                _ => [i64::from(i32::MIN), -60_000, 60_000, i64::from(i32::MAX)]
                    [usize::try_from(random.below(4)).expect("fits")],
            };
            Rounding::ALL.map(move |rounding| FunctionCase {
                function: Function::ScaleB,
                rounding,
                operands: [a, Pair::new(scale.cast_unsigned(), 0), Pair::default()],
            })
        })
        .collect();
    assert_eq!(compare(&cases), 4 * 20_022);
}

#[test]
fn minimum_maximum_and_total_order_match_glibc_for_canonical_pairs() {
    let mut random = SplitMix64::new(0xE8AC_7003);
    let operands = operands(&mut random, 6_000, operand);
    let functions = [
        Function::Fmax,
        Function::Fmin,
        Function::Fmaximum,
        Function::Fminimum,
        Function::FmaximumNumber,
        Function::FminimumNumber,
        Function::FmaximumMagnitude,
        Function::FminimumMagnitude,
        Function::FmaximumMagnitudeNumber,
        Function::FminimumMagnitudeNumber,
        Function::TotalOrder,
        Function::TotalOrderMagnitude,
    ];
    let count: usize = functions
        .into_iter()
        .map(|function| compare(&binary(function, &operands, &mut random)))
        .sum();
    assert_eq!(count, 12 * 4 * (6_022 + 22 * 22));
}
