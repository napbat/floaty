//! Square roots against decNumber's arbitrary-precision square root, which
//! `Arithmetic::square_root` rounds to the format: result bits, the five
//! IEEE 754 flags, `TINY`, and `ROUNDED_UP`. A root is never tiny, and it
//! is rounded up when it is inexact and differs from the root rounded
//! toward zero.

use floaty::Env;
use floaty::format::{Decimal, Dpd, Standard, Storage, Width};
use floaty_verify::decnumber::{self, Arithmetic, Double, Quad, Single, Unary};

use super::operands::{Generator, Number, Shape, digit_count, signed};
use super::{
    Answer, DpdFloat, Report, SHARED_ROUNDINGS, Tally, compared, describe, describe_operands,
    direction, flags_of,
};

/// Returns an operand for a square root: any operand, or an exact square
/// or a neighbor of one, with an even or an odd exponent.
pub(super) fn operand<F: Arithmetic>(generator: &mut Generator<F>) -> F::Bits {
    if generator.chance(50) {
        return generator.operand().0;
    }
    // A root of at most p / 2 digits has a square of at most p digits.
    let length = u32::try_from(generator.between(1, signed(F::PRECISION / 2)))
        .expect("a length is positive");
    let root = generator.full(length);
    let coefficient = root * root;
    let square = Generator::<F>::encode(Number {
        negative: false,
        coefficient,
        exponent: generator.exponent(digit_count(coefficient)),
    });
    let rounding = decnumber::Rounding::HalfEven;
    match generator.below(4) {
        0 => F::unary(Unary::NextPlus, square, rounding).value,
        1 => F::unary(Unary::NextMinus, square, rounding).value,
        _ => square,
    }
}

/// Compares `count` random square roots of format `F` in every shared
/// rounding mode.
fn run_square_roots<F: Arithmetic, const W: usize>(count: usize, seed: u64) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let mut generator = Generator::<F>::new(seed, Shape::of::<F>());
    let mut tally = Tally::default();
    for _ in 0..count {
        let x = operand(&mut generator);
        let toward_zero = F::square_root(x, decnumber::Rounding::Down);
        for rounding in SHARED_ROUNDINGS {
            let expected = F::square_root(x, rounding);
            let direction = direction(rounding);
            let (root, flags) =
                DpdFloat::<W>::from_bits(x).sqrt_with(Env::IEEE.with_rounding(direction));
            let (root, flags) = (root.to_bits(), compared(flags));
            let details = Report::Rounded.flags::<F>(
                (expected.value, expected.status),
                (toward_zero.value, toward_zero.status),
            );
            let expected_flags = flags_of(expected.status) | details;
            if root == expected.value && flags == expected_flags {
                tally.passed += 1;
                continue;
            }
            tally.fail(|| {
                format!(
                    "sqrt {rounding:?} [{}]: floaty gives {} {flags:?}, decNumber {} \
                     {expected_flags:?}",
                    describe_operands::<F>(&[x]),
                    describe::<F>(&Answer::Encoding(root)),
                    describe::<F>(&Answer::Encoding(expected.value)),
                )
            });
        }
    }
    tally
}

#[test]
fn random_decimal64_square_root() {
    let tally = run_square_roots::<Double, 64>(20_000, 0x6464_0003);
    tally.report("random decimal64 square root");
    tally.assert_counts(160_000, &[], &[]);
}

#[test]
fn random_decimal128_square_root() {
    let tally = run_square_roots::<Quad, 128>(20_000, 0x0128_0003);
    tally.report("random decimal128 square root");
    tally.assert_counts(160_000, &[], &[]);
}

#[test]
fn random_decimal32_square_root() {
    let tally = run_square_roots::<Single, 32>(20_000, 0x0032_0004);
    tally.report("random decimal32 square root");
    tally.assert_counts(160_000, &[], &[]);
}
