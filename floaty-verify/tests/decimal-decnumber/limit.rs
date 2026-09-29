//! The precision limit of `Env` in the decimal formats.
//!
//! `add`, `sub`, `mul`, `div`, `mul_add`, `sqrt`, and `scale_b` of
//! decimal64 and decimal128 run at limits of 1, 2, 3, 7, and `p - 1`
//! digits, in the six shared rounding modes. The test compares the result
//! bits, the five IEEE 754 flags, `TINY`, and `ROUNDED_UP`.
//!
//! No decimal library has a precision limit, so the test applies floaty's
//! rule to decNumber's results. The limit moves only the rounding
//! position. `decnumber::limited` rounds the exact result once to `L`
//! digits with the exponent range of the format and no clamp: the
//! subnormal quantum is `10^(emin - L + 1)`, a result overflows when its
//! adjusted exponent exceeds `emax`, and a result is tiny before rounding.
//! The result then takes its exponent by the rules of the format at its
//! full precision `p`:
//!
//! - An exact result takes the exponent nearest the preferred exponent of
//!   the operation that has room for `p` digits, in the range of the
//!   format. The preferred exponent is `min(qx, qy)` for a sum, `qx + qy`
//!   for a product, `qx - qy` for a quotient, `min(qx + qy, qz)` for a
//!   fused multiply-add, `floor(qx / 2)` for a square root, and `qx + n`
//!   for `scale_b`.
//! - An inexact result has the least possible exponent: its coefficient
//!   has `p` digits, or its exponent is `emin - p + 1`.
//! - A zero from an underflow has the exponent `emin - p + 1`.
//! - An overflow toward zero gives `(10^L - 1) * 10^(p - L)` with the
//!   exponent `emax - p + 1`.
//!
//! `TINY` and `ROUNDED_UP` follow from the result rounded toward zero, as
//! for every rounded operation. A special operand gives a special result,
//! which the limit does not change, so decNumber's operation at the full
//! precision gives it.

use core::num::NonZeroU32;

use floaty::format::{Decimal, Dpd, Standard, Storage, Width};
use floaty::{Env, Flags};
use floaty_verify::decnumber::{self, Arithmetic, Binary, Double, Limited, Quad, Status};
use floaty_verify::dectest::Operation;

use super::operands::{Generator, Number, Shape, digit_count, power_of_ten, signed};
use super::{
    Answer, DpdFloat, Report, SHARED_ROUNDINGS, Tally, compared, describe, describe_operands,
    direction, excluded, flags_of, noted, run_floaty_in, run_oracle, scale, square_root,
};

/// The operations of the test.
const OPERATIONS: [Limited; 7] = [
    Limited::Binary(Binary::Add),
    Limited::Binary(Binary::Subtract),
    Limited::Binary(Binary::Multiply),
    Limited::Binary(Binary::Divide),
    Limited::Binary(Binary::ScaleB),
    Limited::Fma,
    Limited::SquareRoot,
];

/// Returns the limits of the test for format `F`.
fn limits<F: Arithmetic>() -> [u32; 5] {
    [1, 2, 3, 7, F::PRECISION - 1]
}

/// Returns the decTest operation of an operation, or `None` for the square
/// root, which decTest lacks.
fn operation(limited: Limited) -> Option<Operation> {
    match limited {
        Limited::Binary(binary) => Some(Operation::Binary(binary)),
        Limited::Fma => Some(Operation::Fma),
        Limited::SquareRoot => None,
    }
}

/// Returns a number of `limit + 1` digits that ends in 5: a tie at the
/// limit. A quarter of them carry to the next power of 10.
fn tie<F: Arithmetic>(generator: &mut Generator<F>, limit: u32) -> F::Bits {
    let kept = if generator.chance(25) {
        power_of_ten(limit) - 1
    } else {
        generator.full(limit)
    };
    Generator::<F>::encode(Number {
        negative: generator.chance(50),
        coefficient: kept * 10 + 5,
        exponent: generator.exponent(limit + 1),
    })
}

/// Returns the operands of a random case. A quarter of the cases with a
/// first operand that is not a radicand take a tie at the limit there.
fn case<F: Arithmetic>(generator: &mut Generator<F>, limited: Limited, limit: u32) -> Vec<F::Bits> {
    let Some(operation) = operation(limited) else {
        return vec![square_root::operand(generator)];
    };
    let mut operands = generator.operands(&operation);
    if generator.chance(25) {
        operands[0] = tie(generator, limit);
    }
    operands
}

/// Returns the exponent of a finite operand, from decNumber's string.
fn exponent_of<F: Arithmetic>(bits: F::Bits) -> i64 {
    let number = Number::parse(&F::to_string(bits)).expect("the operand is finite");
    i64::from(number.exponent)
}

/// Returns the preferred exponent of an operation on finite operands.
fn preferred<F: Arithmetic>(limited: Limited, operands: &[F::Bits]) -> i64 {
    let exponent = |index: usize| exponent_of::<F>(operands[index]);
    match limited {
        Limited::Binary(Binary::Add | Binary::Subtract) => exponent(0).min(exponent(1)),
        Limited::Binary(Binary::Multiply) => exponent(0) + exponent(1),
        Limited::Binary(Binary::Divide) => exponent(0) - exponent(1),
        Limited::Binary(Binary::ScaleB) => {
            let scale = scale::<F>(operands[1]).expect("the caller excludes other scale operands");
            exponent(0) + i64::from(scale)
        }
        Limited::Fma => (exponent(0) + exponent(1)).min(exponent(2)),
        Limited::SquareRoot => exponent(0).div_euclid(2),
        Limited::Binary(_) => unreachable!("the test runs only the operations that it lists"),
    }
}

/// Returns the member of the cohort of an exact nonzero result whose
/// exponent is nearest `preferred`, with room for `p` digits, in the range
/// of format `F`.
fn nearest<F: Arithmetic>(coefficient: u128, exponent: i32, preferred: i64) -> (u128, i32) {
    let (lowest, highest) = Generator::<F>::exponents();
    // Without its trailing zeros, the coefficient gives the largest
    // exponent of the cohort.
    let (mut coefficient, mut largest) = (coefficient, exponent);
    while coefficient % 10 == 0 {
        coefficient /= 10;
        largest += 1;
    }
    let least = largest - signed(F::PRECISION - digit_count(coefficient));
    let target = i32::try_from(preferred.clamp(i64::from(lowest), i64::from(highest)))
        .expect("an exponent of the format fits an i32");
    let chosen = target.clamp(least.max(lowest), largest);
    let shift = u32::try_from(largest - chosen).expect("the chosen exponent is not larger");
    (coefficient * power_of_ten(shift), chosen)
}

/// Returns the encoding that floaty's rule gives to decNumber's
/// result at a precision limit, with its status and the preferred exponent
/// of the operation.
fn represent<F: Arithmetic>(text: &str, status: Status, preferred: i64, limit: u32) -> F::Bits {
    let Some(Number {
        negative,
        coefficient,
        exponent,
    }) = Number::parse(text)
    else {
        return Generator::<F>::from_text(text);
    };
    let (lowest, highest) = Generator::<F>::exponents();
    let inexact = status.intersects(Status::INEXACT);
    let (coefficient, exponent) = if status.contains(Status::OVERFLOW) {
        let spare = power_of_ten(F::PRECISION - limit);
        ((power_of_ten(limit) - 1) * spare, highest)
    } else if coefficient == 0 && inexact {
        (0, lowest)
    } else if coefficient == 0 {
        let exponent = preferred.clamp(i64::from(lowest), i64::from(highest));
        (
            0,
            i32::try_from(exponent).expect("an exponent of the format fits an i32"),
        )
    } else if inexact {
        let room = u32::try_from(exponent - lowest).expect("a result is not below the quantum");
        let pad = (F::PRECISION - digit_count(coefficient)).min(room);
        (coefficient * power_of_ten(pad), exponent - signed(pad))
    } else {
        nearest::<F>(coefficient, exponent, preferred)
    };
    Generator::<F>::encode(Number {
        negative,
        coefficient,
        exponent,
    })
}

/// Returns the expected result and status of a case at a precision limit.
fn expected<F: Arithmetic>(
    limited: Limited,
    operands: &[F::Bits],
    limit: u32,
    rounding: decnumber::Rounding,
) -> (F::Bits, Status) {
    let special = operands.iter().any(|&bits| {
        let class = F::class(bits);
        class.ends_with("Infinity") || class.ends_with("NaN")
    });
    if !special {
        let outcome = decnumber::limited::<F>(limited, operands, limit, rounding);
        let preferred = preferred::<F>(limited, operands);
        let bits = represent::<F>(&outcome.value, outcome.status, preferred, limit);
        return (bits, outcome.status);
    }
    let Some(operation) = operation(limited) else {
        let root = F::square_root(operands[0], rounding);
        return (root.value, root.status);
    };
    let (answer, status) = run_oracle::<F>(&operation, operands, rounding);
    let bits = answer
        .encoding()
        .expect("a rounded operation gives an encoding");
    (bits, status)
}

/// Runs an operation on floaty with a behavior, or the square root for
/// `None`. Returns the result and every flag.
fn run_floaty<F: Arithmetic, const W: usize>(
    operation: Option<&Operation>,
    operands: &[F::Bits],
    env: Env,
) -> (F::Bits, Flags)
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let Some(operation) = operation else {
        let (root, flags) = DpdFloat::<W>::from_bits(operands[0]).sqrt_with(env);
        return (root.to_bits(), flags);
    };
    let (answer, flags) = run_floaty_in::<F, W>(operation, operands, env);
    let bits = answer
        .encoding()
        .expect("a rounded operation gives an encoding");
    (bits, flags)
}

/// Runs one case on floaty at a precision limit, in every shared rounding
/// mode, and compares it with the rule applied to decNumber.
fn check_case<F: Arithmetic, const W: usize>(
    tally: &mut Tally,
    limited: Limited,
    operands: &[F::Bits],
    limit: u32,
) where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let operation = operation(limited);
    if let Some(reason) = operation
        .as_ref()
        .and_then(|operation| excluded::<F>(operation, operands, Status::NONE))
    {
        for _ in SHARED_ROUNDINGS {
            tally.skip(reason);
        }
        return;
    }
    let toward_zero = expected::<F>(limited, operands, limit, decnumber::Rounding::Down);
    for rounding in SHARED_ROUNDINGS {
        if let Some(note) = operation
            .as_ref()
            .and_then(|operation| noted::<F>(operation, operands))
        {
            tally.note(note);
        }
        let (want, status) = expected::<F>(limited, operands, limit, rounding);
        let want_flags = flags_of(status) | Report::Rounded.flags::<F>((want, status), toward_zero);
        let direction = direction(rounding).expect("a shared mode has a direction");
        let env = Env::IEEE
            .with_rounding(direction)
            .with_precision(NonZeroU32::new(limit));
        let (result, flags) = run_floaty::<F, W>(operation.as_ref(), operands, env);
        let flags = compared(flags);
        if result == want && flags == want_flags {
            tally.passed += 1;
            continue;
        }
        tally.fail(|| {
            format!(
                "{limited:?} limit {limit} {rounding:?} [{}]: floaty gives {} {flags:?}, \
                 expected {} {want_flags:?} ({status})",
                describe_operands::<F>(operands),
                describe::<F>(&Answer::Encoding(result)),
                describe::<F>(&Answer::Encoding(want)),
            )
        });
    }
}

/// Runs `count` random cases of each operation at each limit in format
/// `F`. The operands probe the edges of the format and the subnormal range
/// of the limit.
fn run<F: Arithmetic, const W: usize>(count: usize, seed: u64) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let mut tally = Tally::default();
    for limit in limits::<F>() {
        let shape = Shape {
            precision: limit,
            ..Shape::of::<F>()
        };
        let mut generator = Generator::<F>::new(seed ^ u64::from(limit), shape);
        for limited in OPERATIONS {
            for _ in 0..count {
                let operands = case(&mut generator, limited, limit);
                check_case::<F, W>(&mut tally, limited, &operands, limit);
            }
        }
    }
    tally
}

/// The skip counts of the scale operands that floaty's `i32` cannot take,
/// and of a fused multiply-add with two signaling NaNs.
fn skips(nan: usize, limit: usize, exponent: usize, pair: usize) -> [(&'static str, usize); 4] {
    [
        (
            "fma: signaling NaN factor and addend; floaty takes SoftFloat's NaN order",
            pair,
        ),
        ("scaleb: a NaN scale operand", nan),
        (
            "scaleb: a scale operand beyond decNumber's limit of 2 * (emax + p)",
            limit,
        ),
        (
            "scaleb: a scale operand that is not an integer with exponent 0",
            exponent,
        ),
    ]
}

/// The note of a fused multiply-add of `0 * inf` and a quiet NaN.
const INVALID_PRODUCT: &str =
    "fma: 0 * inf + quiet NaN, which runs with InvalidProduct::YieldsToNan";

#[test]
fn decimal64_precision_limit() {
    let tally = run::<Double, 64>(3_000, 0x6464_0017);
    tally.report("decimal64 precision limit");
    // The generator is seeded, so the counts are exact.
    tally.assert_counts(
        616_668,
        &skips(552, 4554, 8184, 42),
        &[(INVALID_PRODUCT, 288)],
    );
}

#[test]
fn decimal128_precision_limit() {
    let tally = run::<Quad, 128>(3_000, 0x0128_0017);
    tally.report("decimal128 precision limit");
    tally.assert_counts(
        616_494,
        &skips(750, 4320, 8388, 48),
        &[(INVALID_PRODUCT, 270)],
    );
}
