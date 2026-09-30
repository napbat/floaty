//! floaty's DPD decimal formats against decTest and decNumber.
//!
//! - [`vectors`]: every case of decTest's `dd` and `dq` files runs through
//!   `D64Dpd` and `D128Dpd`. The result must match, and so must the five
//!   IEEE 754 flags. The `canonical` cases and the `apply` cases with an
//!   explicit encoding, also those of the `ds` files, check `is_canonical`,
//!   the conversion to the same format, and `decode`.
//! - [`random`]: seeded random operands, which [`operands`] biases toward
//!   the edges of each format, run through floaty and through decNumber, in
//!   the eight rounding modes of decNumber. decimal64 and decimal128 use
//!   `decDouble` and `decQuad`. decimal32 uses decNumber's
//!   arbitrary-precision numbers in the decimal32 context, because
//!   `decSingle` has no arithmetic.
//! - [`square_root`] and [`conversions`]: the square root and the
//!   conversions between the widths, which decNumber's fixed-size formats
//!   lack as operations, against the wrappers of `floaty_verify::decnumber`.
//! - [`integers`]: the conversions to and from integers, against decNumber's
//!   rounding to an integral value and its conversion from a string.
//! - [`limit`]: the precision limit of `Env`, against decNumber's
//!   arbitrary-precision numbers at fewer digits.
//! - [`flush`]: flush-to-zero and denormals-are-zero.
//!
//! The rounding modes `ceiling`, `floor`, `down`, `up`, `half_even`,
//! `half_up`, `half_down`, and `05up` are floaty's `TowardPositive`,
//! `TowardNegative`, `TowardZero`, `AwayFromZero`, `TiesToEven`,
//! `TiesToAway`, `TiesTowardZero`, and `ToOdd`. The conditions map to the
//! IEEE 754 flags: `Invalid_operation` and the conditions that decNumber
//! folds into it give `INVALID`, and `Division_by_zero`, `Overflow`,
//! `Underflow`, and `Inexact` give the flag of that name. The fixed-size
//! formats never report `Clamped`, `Rounded`, or `Subnormal`.
//!
//! The random cases, the square roots, the conversions, and the precision
//! limit also compare floaty's `TINY` and `ROUNDED_UP`, by the rules that
//! [`Report`] lists. decNumber's result rounded toward zero gives both. The
//! exact result is below `10^emin` exactly when that result is subnormal,
//! or is a zero that is inexact. A result is above the exact result in
//! magnitude exactly when it is inexact and differs from that result. The
//! vectors ignore both flags, because the decTest conditions do not give
//! `ROUNDED_UP`. Only [`flush`] checks `DENORMAL_INPUT`.
//!
//! The test counts each case that it skips, with the reason:
//!
//! - Some decNumber operations are not operations of floaty: the rounding
//!   `abs`, `minus`, and `plus`, `reduce`, the logical operations, the
//!   integer quotient `divideint`, `maxmag`, `minmag`, `nexttoward`, and the
//!   text conversions.
//! - decNumber gives `Division_impossible` when the integer quotient of
//!   `remainder` or `remaindernear` has more than `p` digits, as
//!   `decBasic.c` of decNumber 3.68 does at lines 591 to 595. floaty follows
//!   IEEE 754 and gives the exact remainder, as the Intel decimal library
//!   does: `1E+384` remaindernear `1` is `0`. The test identifies these cases
//!   by the condition of decNumber, not by floaty's result, and counts them.
//! - floaty's `scale_b` takes an `i32`. decNumber takes a scale operand that
//!   is an integer with exponent 0, up to `2 * (emax + p)`, and signals
//!   invalid for every other operand.
//! - A fused multiply-add with a signaling NaN factor and a different
//!   signaling NaN addend. floaty selects the NaN in SoftFloat's order,
//!   `FusedNanOrder::ProductFirst`, so the quiet NaN of the factors meets
//!   the addend and the addend wins. decNumber gives the first signaling
//!   NaN. IEEE 754-2019 section 6.2.3 does not say which NaN gives the
//!   payload.
//!
//! Every fact that decides a skip or a note comes from decNumber: its class
//! names and its scientific strings, not floaty's classification.
//!
//! decNumber's fused multiply-add gives a quiet NaN addend precedence over
//! the invalid product `0 * inf`, and IEEE 754-2019 section 7.2 (c) lets the
//! implementation choose. The test runs `fma` with the floaty setting of
//! that choice, `InvalidProduct::YieldsToNan`.
//!
//! The random cases take `fma`, `comparetotal`, and `comparetotmag` from
//! decNumber's arbitrary-precision numbers. `decDoubleFMA`, `decQuadFMA`,
//! and the `CompareTotal` operations of the fixed-size formats give wrong
//! results in cases that decTest does not cover, as `Arithmetic::fma_wide`
//! and `Arithmetic::total_order` describe.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

mod conversions;
mod flush;
mod integers;
mod limit;
mod operands;
mod random;
mod square_root;
mod total_order;
mod vectors;

use std::cmp::Ordering;
use std::collections::BTreeMap;

use floaty::env::InvalidProduct;
use floaty::format::{Decimal, Dpd, Standard, Storage, Width};
use floaty::{Class, Env, Flags, Float, Rounding};
use floaty_verify::decnumber::{self, Arithmetic, Binary, Format, Status, Unary};
use floaty_verify::dectest::Operation;

use self::operands::signed;

/// A floaty DPD format of width `W`.
type DpdFloat<const W: usize> = Float<Decimal<Dpd>, W>;

/// The five IEEE 754 flags, which decNumber also reports.
const IEEE_FLAGS: [Flags; 5] = [
    Flags::INVALID,
    Flags::DIVIDE_BY_ZERO,
    Flags::OVERFLOW,
    Flags::UNDERFLOW,
    Flags::INEXACT,
];

/// Returns the IEEE 754 flags of a set of flags.
fn ieee(flags: Flags) -> Flags {
    IEEE_FLAGS
        .into_iter()
        .filter(|&flag| flags.contains(flag))
        .fold(Flags::NONE, |all, flag| all | flag)
}

/// Returns the flags that the random tests compare: the IEEE 754 flags,
/// `TINY`, and `ROUNDED_UP`.
fn compared(flags: Flags) -> Flags {
    [Flags::TINY, Flags::ROUNDED_UP]
        .into_iter()
        .filter(|&flag| flags.contains(flag))
        .fold(ieee(flags), |all, flag| all | flag)
}

/// Returns the IEEE 754 flags of decNumber conditions.
fn flags_of(status: Status) -> Flags {
    [
        (Status::IEEE_INVALID_OPERATION, Flags::INVALID),
        (Status::DIVISION_BY_ZERO, Flags::DIVIDE_BY_ZERO),
        (Status::OVERFLOW, Flags::OVERFLOW),
        (Status::UNDERFLOW, Flags::UNDERFLOW),
        (Status::INEXACT, Flags::INEXACT),
    ]
    .into_iter()
    .filter(|&(condition, _)| status.intersects(condition))
    .fold(Flags::NONE, |all, (_, flag)| all | flag)
}

/// The rule by which floaty reports `TINY` and `ROUNDED_UP` for an
/// operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Report {
    /// A rounded operation: `TINY` when the exact result is not zero and
    /// is below `10^emin` in magnitude, and `ROUNDED_UP` when the result
    /// is above the exact result in magnitude, as the rounding routine
    /// reports them.
    Rounded,
    /// `quantize` and rounding to an integral value: `ROUNDED_UP` when the
    /// magnitude grew. `quantize_with` reports no `TINY`, and an integral
    /// value is never tiny.
    Magnitude,
    /// The remainder, which is exact: `TINY` for a subnormal result.
    Subnormal,
    /// Neither flag: the operations that give an operand or its neighbor,
    /// the comparisons, `log_b`, and the operations that give no number.
    Neither,
}

impl Report {
    /// Returns the rule of an operation.
    fn of(operation: &Operation) -> Self {
        match operation {
            Operation::Fma
            | Operation::Binary(
                Binary::Add | Binary::Subtract | Binary::Multiply | Binary::Divide | Binary::ScaleB,
            ) => Self::Rounded,
            Operation::Binary(Binary::Quantize) | Operation::Unary(Unary::ToIntegralExact) => {
                Self::Magnitude
            }
            Operation::Binary(Binary::Remainder | Binary::RemainderNear) => Self::Subnormal,
            _ => Self::Neither,
        }
    }

    /// Returns whether the rule needs decNumber's result rounded toward
    /// zero.
    fn needs_toward_zero(self) -> bool {
        matches!(self, Self::Rounded | Self::Magnitude)
    }

    /// Returns the `TINY` and `ROUNDED_UP` flags of a result, from
    /// decNumber's result and status in the rounding mode, and its result
    /// and status rounded toward zero. The module documentation gives the
    /// reasoning.
    fn flags<F: Arithmetic>(
        self,
        (result, status): (F::Bits, Status),
        (toward_zero, toward_zero_status): (F::Bits, Status),
    ) -> Flags {
        let inexact = |status: Status| status.intersects(Status::INEXACT);
        let rounded_up = inexact(status) && result != toward_zero;
        let tiny = || {
            let class = F::class(toward_zero);
            class.ends_with("Subnormal") || (class.ends_with("Zero") && inexact(toward_zero_status))
        };
        let (tiny, rounded_up) = match self {
            Self::Rounded => (tiny(), rounded_up),
            Self::Magnitude => (false, rounded_up),
            Self::Subnormal => (F::class(result).ends_with("Subnormal"), false),
            Self::Neither => (false, false),
        };
        let tiny = if tiny { Flags::TINY } else { Flags::NONE };
        let rounded_up = if rounded_up {
            Flags::ROUNDED_UP
        } else {
            Flags::NONE
        };
        tiny | rounded_up
    }
}

/// Returns the floaty direction of a decNumber rounding mode.
fn direction(rounding: decnumber::Rounding) -> Rounding {
    match rounding {
        decnumber::Rounding::Ceiling => Rounding::TowardPositive,
        decnumber::Rounding::Floor => Rounding::TowardNegative,
        decnumber::Rounding::Down => Rounding::TowardZero,
        decnumber::Rounding::Up => Rounding::AwayFromZero,
        decnumber::Rounding::HalfEven => Rounding::TiesToEven,
        decnumber::Rounding::HalfUp => Rounding::TiesToAway,
        decnumber::Rounding::HalfDown => Rounding::TiesTowardZero,
        decnumber::Rounding::ZeroFiveUp => Rounding::ToOdd,
    }
}

/// The rounding modes of decNumber, which floaty shares.
const SHARED_ROUNDINGS: [decnumber::Rounding; 8] = [
    decnumber::Rounding::Ceiling,
    decnumber::Rounding::Floor,
    decnumber::Rounding::Down,
    decnumber::Rounding::Up,
    decnumber::Rounding::HalfEven,
    decnumber::Rounding::HalfUp,
    decnumber::Rounding::HalfDown,
    decnumber::Rounding::ZeroFiveUp,
];

/// Returns why floaty has no operation for a decTest operation, or `None`
/// when it has one. An `apply` case with an explicit encoding and a
/// `canonical` case need no string; [`vectors`] checks them before it asks.
fn unmapped(operation: &Operation) -> Option<&'static str> {
    const LOGICAL: &str = "logical operation: floaty has none";
    match operation {
        Operation::Unary(Unary::Abs | Unary::Minus | Unary::Plus) => {
            Some("abs, minus, plus: decNumber rounds these; floaty has the quiet sign operations")
        }
        Operation::Unary(Unary::Reduce) => Some("reduce: not an IEEE 754 operation"),
        Operation::Unary(Unary::Invert)
        | Operation::Binary(
            Binary::And | Binary::Or | Binary::Xor | Binary::Rotate | Binary::Shift,
        ) => Some(LOGICAL),
        Operation::Binary(Binary::DivideInteger) => {
            Some("divideint: an integer quotient, not IEEE 754")
        }
        Operation::Binary(Binary::MaxMag | Binary::MinMag | Binary::NextToward) => {
            Some("maxmag, minmag, nexttoward: floaty has no such operation")
        }
        Operation::Apply | Operation::ToSci | Operation::ToEng => {
            Some("apply of a string, tosci, toeng: text conversions; floaty has no strings")
        }
        Operation::Other(_) => Some("an operation that decDouble and decQuad lack"),
        _ => None,
    }
}

/// Returns why a case with these operands and oracle conditions is
/// excluded, or `None`. The reasons are in the module documentation.
fn excluded<F: Arithmetic>(
    operation: &Operation,
    operands: &[F::Bits],
    conditions: Status,
) -> Option<&'static str> {
    match operation {
        Operation::Binary(Binary::Remainder | Binary::RemainderNear)
            if conditions.contains(Status::DIVISION_IMPOSSIBLE) =>
        {
            Some(
                "remainder, remaindernear: decNumber's Division_impossible; floaty follows IEEE 754",
            )
        }
        Operation::Binary(Binary::ScaleB) => scale::<F>(operands[1]).err(),
        Operation::Fma if signaling_pair::<F>(operands) => {
            Some("fma: signaling NaN factor and addend; floaty takes SoftFloat's NaN order")
        }
        _ => None,
    }
}

/// Returns whether a fused multiply-add has a signaling NaN factor and a
/// different signaling NaN addend. decNumber gives the first signaling
/// factor, made quiet, and floaty gives the addend. decNumber's strings
/// tell the sign and the payload of the two NaNs apart.
fn signaling_pair<F: Arithmetic>(operands: &[F::Bits]) -> bool {
    let &[x, y, z] = operands else {
        return false;
    };
    let signaling = |bits: F::Bits| F::class(bits) == "sNaN";
    let factor = [x, y].into_iter().find(|&bits| signaling(bits));
    factor.is_some_and(|factor| signaling(z) && F::to_string(factor) != F::to_string(z))
}

/// Returns a note for a case that runs with a floaty setting other than
/// the default, or `None`.
///
/// A fused multiply-add whose product is the invalid `0 * inf` and whose
/// addend is a quiet NaN gives the addend in decNumber and signals nothing.
/// That is `InvalidProduct::YieldsToNan`, with which every `fma` case runs.
/// floaty's default, `Signals`, gives the default NaN and signals invalid.
fn noted<F: Arithmetic>(operation: &Operation, operands: &[F::Bits]) -> Option<&'static str> {
    let (Operation::Fma, &[x, y, z]) = (operation, operands) else {
        return None;
    };
    let invalid = |infinite: F::Bits, zero: F::Bits| {
        F::class(infinite).ends_with("Infinity") && F::class(zero).ends_with("Zero")
    };
    ((invalid(x, y) || invalid(y, x)) && F::class(z) == "NaN")
        .then_some("fma: 0 * inf + quiet NaN, which runs with InvalidProduct::YieldsToNan")
}

/// Returns the scale of a decNumber `scaleb` operand, or why floaty's `i32`
/// scale cannot take it. decNumber's scientific string of an integer with
/// exponent 0 is its digits, with a sign when it is negative.
fn scale<F: Arithmetic>(operand: F::Bits) -> Result<i32, &'static str> {
    if F::class(operand).ends_with("NaN") {
        return Err("scaleb: a NaN scale operand");
    }
    let text = F::to_string(operand);
    let (negative, digits) = match text.strip_prefix('-') {
        Some(digits) => (true, digits),
        None => (false, text.as_str()),
    };
    if digits.is_empty() || !digits.bytes().all(|digit| digit.is_ascii_digit()) {
        return Err("scaleb: a scale operand that is not an integer with exponent 0");
    }
    let limit = 2 * (F::EMAX + signed(F::PRECISION));
    match digits.parse::<i32>() {
        Ok(magnitude) if magnitude <= limit => Ok(if negative { -magnitude } else { magnitude }),
        _ => Err("scaleb: a scale operand beyond decNumber's limit of 2 * (emax + p)"),
    }
}

/// A result that floaty and decNumber both give.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Answer<B> {
    /// An encoding.
    Encoding(B),
    /// The order of a comparison, or `None` for unordered.
    Order(Option<Ordering>),
    /// A class name, or `0` or `1` for `samequantum`.
    Text(&'static str),
}

impl<B: Copy> Answer<B> {
    /// Returns the encoding of an answer that is an encoding.
    fn encoding(self) -> Option<B> {
        match self {
            Self::Encoding(bits) => Some(bits),
            _ => None,
        }
    }
}

/// Returns the answer that a comparison result of decNumber encodes: the
/// order for -1, 0, or 1, and unordered for a NaN. Any other result stays
/// an encoding, which no floaty order matches.
fn comparison<F: Format>(bits: F::Bits) -> Answer<F::Bits> {
    let text = F::to_string(bits);
    match text.as_str() {
        "-1" => Answer::Order(Some(Ordering::Less)),
        "0" => Answer::Order(Some(Ordering::Equal)),
        "1" => Answer::Order(Some(Ordering::Greater)),
        _ if text.contains("NaN") => Answer::Order(None),
        _ => Answer::Encoding(bits),
    }
}

/// Returns whether an operation returns the bits of its operand with at
/// most a new sign, as the IEEE 754 quiet operations do.
fn keeps_encoding(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Unary(Unary::Copy | Unary::CopyAbs | Unary::CopyNegate)
            | Operation::Binary(Binary::CopySign)
    )
}

/// Returns whether an operation gives a comparison result.
fn compares(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Binary(
            Binary::Compare
                | Binary::CompareSignal
                | Binary::CompareTotal
                | Binary::CompareTotalMag
        )
    )
}

/// Returns the class name that decNumber gives for a floaty value.
fn class_name<const W: usize>(value: DpdFloat<W>) -> &'static str
where
    Decimal<Dpd>: Standard<W>,
{
    let signed = |negative: &'static str, positive: &'static str| {
        if value.is_sign_negative() {
            negative
        } else {
            positive
        }
    };
    match value.classify() {
        Class::SignalingNan => "sNaN",
        Class::QuietNan => "NaN",
        Class::Infinite => signed("-Infinity", "+Infinity"),
        Class::Normal => signed("-Normal", "+Normal"),
        Class::Subnormal => signed("-Subnormal", "+Subnormal"),
        Class::Zero => signed("-Zero", "+Zero"),
        _ => "Unsupported",
    }
}

/// Runs an operation that [`unmapped`] accepts on floaty in a rounding
/// direction. Returns the result and every flag.
fn run_floaty<F: Arithmetic, const W: usize>(
    operation: &Operation,
    operands: &[F::Bits],
    rounding: Rounding,
) -> (Answer<F::Bits>, Flags)
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    run_floaty_in::<F, W>(operation, operands, Env::IEEE.with_rounding(rounding))
}

/// Runs an operation that [`unmapped`] accepts on floaty with a behavior.
/// Returns the result and every flag.
fn run_floaty_in<F: Arithmetic, const W: usize>(
    operation: &Operation,
    operands: &[F::Bits],
    env: Env,
) -> (Answer<F::Bits>, Flags)
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let value = |index: usize| DpdFloat::<W>::from_bits(operands[index]);
    let rounded =
        |(result, flags): (DpdFloat<W>, Flags)| (Answer::Encoding(result.to_bits()), flags);
    let quiet = |result: DpdFloat<W>| (Answer::Encoding(result.to_bits()), Flags::NONE);
    let order = |(order, flags)| (Answer::Order(order), flags);
    let (answer, flags) = match operation {
        Operation::Unary(Unary::Copy) => quiet(value(0)),
        Operation::Unary(Unary::CopyAbs) => quiet(value(0).abs()),
        Operation::Unary(Unary::CopyNegate) => quiet(-value(0)),
        Operation::Unary(Unary::LogB) => rounded(value(0).log_b_with(env)),
        Operation::Unary(Unary::NextMinus) => rounded(value(0).next_down_with(env)),
        Operation::Unary(Unary::NextPlus) => rounded(value(0).next_up_with(env)),
        Operation::Unary(Unary::ToIntegralExact) => rounded(value(0).round_to_integral_with(env)),
        Operation::Binary(binary) => {
            let (x, y) = (value(0), value(1));
            match binary {
                Binary::Add => rounded(x.add_with(y, env)),
                Binary::Subtract => rounded(x.sub_with(y, env)),
                Binary::Multiply => rounded(x.mul_with(y, env)),
                Binary::Divide => rounded(x.div_with(y, env)),
                Binary::Compare => order(x.compare_quiet_with(y, env)),
                Binary::CompareSignal => order(x.compare_signaling_with(y, env)),
                Binary::CompareTotal => {
                    (Answer::Order(Some(x.total_cmp_with(y, env))), Flags::NONE)
                }
                Binary::CompareTotalMag => (
                    Answer::Order(Some(x.abs().total_cmp_with(y.abs(), env))),
                    Flags::NONE,
                ),
                Binary::CopySign => quiet(x.copy_sign(y)),
                Binary::Max => rounded(x.max_num_with(y, env)),
                Binary::Min => rounded(x.min_num_with(y, env)),
                Binary::Quantize => rounded(x.quantize_with(y, env)),
                Binary::Remainder => rounded(x.truncated_remainder_with(y, env)),
                Binary::RemainderNear => rounded(x.remainder_with(y, env)),
                Binary::ScaleB => {
                    let scale =
                        scale::<F>(operands[1]).expect("the caller excludes other scale operands");
                    rounded(x.scale_b_with(scale, env))
                }
                _ => unreachable!("the caller skips the operations that floaty lacks"),
            }
        }
        Operation::Fma => {
            let nan = env.nan.with_invalid_product(InvalidProduct::YieldsToNan);
            rounded(value(0).mul_add_with(value(1), value(2), env.with_nan(nan)))
        }
        Operation::Class => (Answer::Text(class_name(value(0))), Flags::NONE),
        Operation::SameQuantum => {
            let same = value(0).same_quantum(value(1));
            (Answer::Text(if same { "1" } else { "0" }), Flags::NONE)
        }
        _ => unreachable!("the caller skips the operations that floaty lacks"),
    };
    (answer, flags)
}

/// Runs the add, subtract, multiply, or divide of [`run_floaty`] with the
/// static mode `Rounded<Ieee, R>` of the direction. Returns `None` for the
/// other operations.
fn run_floaty_static<F: Arithmetic, const W: usize>(
    operation: &Operation,
    operands: &[F::Bits],
    rounding: Rounding,
) -> Option<(Answer<F::Bits>, Flags)>
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let Operation::Binary(binary) = operation else {
        return None;
    };
    if !matches!(
        binary,
        Binary::Add | Binary::Subtract | Binary::Multiply | Binary::Divide
    ) {
        return None;
    }
    let outcome = floaty_verify::with_rounding_mode!(rounding, floaty::mode::Ieee, Mode => {
        let value = |index: usize| DpdFloat::<W>::from_bits(operands[index]).with_mode::<Mode>();
        let (x, y) = (value(0), value(1));
        let (result, flags) = match binary {
            Binary::Add => x.add_with(y, Mode::default()),
            Binary::Subtract => x.sub_with(y, Mode::default()),
            Binary::Multiply => x.mul_with(y, Mode::default()),
            _ => x.div_with(y, Mode::default()),
        };
        (Answer::Encoding(result.to_bits()), flags)
    });
    Some(outcome)
}

/// Runs an operation on decNumber. Returns the result and the conditions.
///
/// `fma`, `comparetotal`, and `comparetotmag` go through decNumber's
/// arbitrary-precision numbers, because the fixed-size formats give wrong
/// results in cases that decTest does not cover. `Arithmetic::fma_wide`
/// and `Arithmetic::total_order` describe them.
fn run_oracle<F: Arithmetic>(
    operation: &Operation,
    operands: &[F::Bits],
    rounding: decnumber::Rounding,
) -> (Answer<F::Bits>, Status) {
    let outcome = match (operation, operands) {
        (Operation::Unary(unary), &[x]) => F::unary(*unary, x, rounding),
        (Operation::Binary(Binary::CompareTotal), &[x, y]) => {
            return (
                Answer::Order(Some(F::total_order(x, y, false))),
                Status::NONE,
            );
        }
        (Operation::Binary(Binary::CompareTotalMag), &[x, y]) => {
            return (
                Answer::Order(Some(F::total_order(x, y, true))),
                Status::NONE,
            );
        }
        (Operation::Binary(binary), &[x, y]) => F::binary(*binary, x, y, rounding),
        (Operation::Fma, &[x, y, z]) => F::fma_wide(x, y, z, rounding),
        (Operation::Class, &[x]) => return (Answer::Text(F::class(x)), Status::NONE),
        (Operation::SameQuantum, &[x, y]) => {
            let same = F::same_quantum(x, y);
            return (Answer::Text(if same { "1" } else { "0" }), Status::NONE);
        }
        _ => unreachable!("the random cases use operations with their operand counts"),
    };
    let answer = if compares(operation) {
        comparison::<F>(outcome.value)
    } else {
        Answer::Encoding(outcome.value)
    };
    (answer, outcome.status)
}

/// Returns the `TINY` and `ROUNDED_UP` flags that floaty must report for an
/// operation, from decNumber's answer and status and, for the rules that
/// need it, decNumber's answer and status rounded toward zero.
fn details<F: Arithmetic>(
    operation: &Operation,
    (answer, status): (Answer<F::Bits>, Status),
    toward_zero: Option<(Answer<F::Bits>, Status)>,
) -> Flags {
    let report = Report::of(operation);
    let Some(result) = answer.encoding() else {
        return Flags::NONE;
    };
    let (toward_zero, toward_zero_status) = if let Some((answer, status)) = toward_zero {
        (answer.encoding().unwrap_or(result), status)
    } else {
        assert!(
            !report.needs_toward_zero(),
            "the rule of {operation:?} needs a result rounded toward zero"
        );
        (result, status)
    };
    report.flags::<F>((result, status), (toward_zero, toward_zero_status))
}

/// Writes an answer with the decNumber string of an encoding.
fn describe<F: Format>(answer: &Answer<F::Bits>) -> String {
    match answer {
        Answer::Encoding(bits) => format!("{} ({bits:#x})", F::to_string(*bits)),
        other => format!("{other:?}"),
    }
}

/// Writes operands as decNumber strings and encodings.
fn describe_operands<F: Format>(operands: &[F::Bits]) -> String {
    let texts: Vec<String> = operands
        .iter()
        .map(|&bits| describe::<F>(&Answer::Encoding(bits)))
        .collect();
    texts.join(", ")
}

/// Returns counts in the order of their reasons.
fn sorted<'a>(counts: &[(&'a str, usize)]) -> Vec<(&'a str, usize)> {
    let mut counts = counts.to_vec();
    counts.sort_unstable();
    counts
}

/// The most failure messages that a run keeps.
const KEPT_FAILURES: usize = 200;

/// Counts of the cases of one run.
#[derive(Default)]
struct Tally {
    passed: usize,
    failed: usize,
    failures: Vec<String>,
    skipped: BTreeMap<String, usize>,
    noted: BTreeMap<&'static str, usize>,
}

impl Tally {
    /// Counts a case that is skipped for a reason.
    fn skip(&mut self, reason: impl Into<String>) {
        *self.skipped.entry(reason.into()).or_default() += 1;
    }

    /// Counts a case that runs with a note from [`noted`].
    fn note(&mut self, note: &'static str) {
        *self.noted.entry(note).or_default() += 1;
    }

    /// Counts a failure, and keeps its message while there is room.
    fn fail(&mut self, message: impl FnOnce() -> String) {
        self.failed += 1;
        if self.failures.len() < KEPT_FAILURES {
            self.failures.push(message());
        }
    }

    /// Asserts that no case failed, and the exact counts of the cases that
    /// passed, that each reason skipped, and that each note marked. The
    /// inputs are pinned or seeded, so every count is exact.
    fn assert_counts(&self, passed: usize, skipped: &[(&str, usize)], noted: &[(&str, usize)]) {
        assert_eq!(self.failed, 0, "no case fails");
        assert_eq!(self.passed, passed, "the count of passed cases");
        let actual: Vec<(&str, usize)> = self
            .skipped
            .iter()
            .map(|(reason, &count)| (reason.as_str(), count))
            .collect();
        assert_eq!(actual, sorted(skipped), "the skip counts");
        let actual: Vec<(&str, usize)> = self
            .noted
            .iter()
            .map(|(&note, &count)| (note, count))
            .collect();
        assert_eq!(actual, sorted(noted), "the note counts");
    }

    /// Prints the counts, and the failures in full.
    fn report(&self, title: &str) {
        println!("{title}: {} passed, {} failed", self.passed, self.failed);
        for (reason, count) in &self.skipped {
            println!("  skipped {count:7}: {reason}");
        }
        for (note, count) in &self.noted {
            println!("  noted   {count:7}: {note}");
        }
        for failure in &self.failures {
            println!("  FAILED {failure}");
        }
    }
}
