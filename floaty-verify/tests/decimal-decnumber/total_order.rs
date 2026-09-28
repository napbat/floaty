//! The total order of two encodings of one datum, in both rules of
//! `TotalOrder`.
//!
//! decNumber's `canonical` gives the canonical twin of a random encoding, and
//! its arbitrary-precision total order orders two data. With
//! `TotalOrder::Datum` floaty must give that order. With
//! `TotalOrder::Encoding` it must give that order for two data, and the
//! order of the bits below the sign for two encodings of one datum, reversed
//! for a negative sign, as `DESIGN.md` states.

use std::cmp::Ordering;

use floaty::TotalOrder;
use floaty::format::{Decimal, Dpd, Standard, Storage, Width};
use floaty_verify::decnumber::{self, Arithmetic, Double, Quad, Unary};

use super::operands::{Generator, Shape};
use super::{DpdFloat, Tally};

/// The note of a pair of two different encodings of one datum.
const TWINS: &str = "two different encodings of one datum";

/// Returns the expected order of two encodings under a rule, from the order
/// of their data.
fn expected<F: Arithmetic>(x: F::Bits, y: F::Bits, data: Ordering, order: TotalOrder) -> Ordering {
    if data != Ordering::Equal || order == TotalOrder::Datum {
        return data;
    }
    let bits = |value: F::Bits| -> u128 { value.into() };
    let width = 8 * u32::try_from(size_of::<F::Bits>()).expect("a width fits a u32");
    let sign = 1_u128 << (width - 1);
    let magnitude = (bits(x) & !sign).cmp(&(bits(y) & !sign));
    if bits(x) & sign == 0 {
        magnitude
    } else {
        magnitude.reverse()
    }
}

/// Compares `count` random encodings with their canonical twins and with
/// other random operands, in both rules and for both the values and the
/// magnitudes.
fn run<F: Arithmetic, const W: usize>(count: usize, seed: u64) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let mut generator = Generator::<F>::new(seed, Shape::of::<F>());
    let mut tally = Tally::default();
    for _ in 0..count {
        let x = generator.raw();
        let twin = F::unary(Unary::Canonical, x, decnumber::Rounding::HalfEven).value;
        let (other, _) = generator.operand();
        if x != twin {
            tally.note(TWINS);
        }
        for (a, b) in [(x, twin), (twin, x), (x, other), (other, twin)] {
            for magnitude in [false, true] {
                let data = F::total_order(a, b, magnitude);
                let (a_value, b_value) = (DpdFloat::<W>::from_bits(a), DpdFloat::<W>::from_bits(b));
                let (a_value, b_value) = if magnitude {
                    (a_value.abs(), b_value.abs())
                } else {
                    (a_value, b_value)
                };
                for order in [TotalOrder::Datum, TotalOrder::Encoding] {
                    let (a_bits, b_bits) = (a_value.to_bits(), b_value.to_bits());
                    let want = expected::<F>(a_bits, b_bits, data, order);
                    let got = a_value.total_cmp_with(b_value, order);
                    if got == want {
                        tally.passed += 1;
                        continue;
                    }
                    tally.fail(|| {
                        format!(
                            "{order:?} magnitude {magnitude} {} ({a:#x}) against {} ({b:#x}): \
                             floaty {got:?}, expected {want:?}",
                            F::to_string(a),
                            F::to_string(b),
                        )
                    });
                }
            }
        }
    }
    tally
}

#[test]
fn decimal64_total_order_of_twins() {
    let tally = run::<Double, 64>(20_000, 0x6464_7070);
    tally.report("decimal64 total order");
    // The generator is seeded, so the counts are exact.
    tally.assert_counts(320_000, &[], &[(TWINS, 3_352)]);
}

#[test]
fn decimal128_total_order_of_twins() {
    let tally = run::<Quad, 128>(20_000, 0x0128_7070);
    tally.report("decimal128 total order");
    tally.assert_counts(320_000, &[], &[(TWINS, 5_529)]);
}
