//! Compares `next_up`, `next_down`, negation, `abs`, and `copy_sign` of the
//! FP8 formats with `ml_dtypes`, for every operand and every operand pair:
//! the table `data/fp8-operations.bin`.
//!
//! `np.nextafter` steps toward a target. The table uses the infinities as
//! targets, or the largest finite values of a format without an infinity.
//! There `np.nextafter` of the largest value gives the value itself, so the
//! test checks that value by floaty's rule instead: the NaN, or the
//! largest value when the behavior saturates.
//!
//! `ml_dtypes` 0.6.0 differs from floaty in two cases, which the test checks
//! by floaty's rules only:
//!
//! - `np.nextafter` of a NaN gives a canonical NaN without the sign and the
//!   payload of the operand (E4M3 `0xFF` gives `0x7F`). A result with a NaN
//!   operand only has to be a NaN.
//! - `np.copysign` of the FNUZ zero and a negative sign gives `0x80`, the
//!   NaN, although `np.negative` keeps the zero. The FNUZ formats have no
//!   negative zero, and IEEE 754-2019 section 5.5.1 changes only the sign
//!   bit, so floaty keeps the one zero encoding.

use floaty::format::Standard;
use floaty::{B11Fnuz, Binary, Env, Float, Fnuz, NoInf};

const TABLE: &[u8] = include_bytes!("../data/fp8-operations.bin");

/// The bytes of each format: four operations on every operand, and
/// `copysign` on every pair.
const FORMAT_BYTES: usize = 4 * 256 + 65536;

/// The largest finite encoding of one sign in a format without an infinity,
/// and the NaN past it.
#[derive(Clone, Copy)]
struct Limit {
    largest: u8,
    nan: u8,
}

/// Checks `next_up` or `next_down` of one operand against the table, and
/// against floaty's rule at the limit.
fn check_next<S: Standard<8, Bits = u8>>(
    x: Float<S, 8>,
    (result, saturated): (Float<S, 8>, Float<S, 8>),
    expected: u8,
    limit: Option<Limit>,
    context: &str,
) {
    match limit {
        _ if x.is_nan() => assert!(
            result.is_nan() && !result.is_signaling_nan(),
            "{context}: {result:?} is a quiet NaN"
        ),
        Some(Limit { largest, nan }) if x.to_bits() == largest => {
            assert_eq!(result.to_bits(), nan, "{context}: the NaN");
            assert_eq!(saturated.to_bits(), largest, "{context}: saturated");
        }
        _ => assert_eq!(result.to_bits(), expected, "{context}"),
    }
}

/// Checks every operand and pair of one format. `largest` is the encoding of
/// the largest finite value of a format without an infinity.
fn check_format<S: Standard<8, Bits = u8>>(index: usize, largest: Option<u8>) {
    let block = &TABLE[index * FORMAT_BYTES..(index + 1) * FORMAT_BYTES];
    let (up, rest) = block.split_at(256);
    let (down, rest) = rest.split_at(256);
    let (negative, rest) = rest.split_at(256);
    let (abs, copy_sign) = rest.split_at(256);
    // The one NaN of an FNUZ format is 0x80. Its zero has one encoding.
    let fnuz = Float::<S, 8>::from_bits(0x80).is_nan();
    let nan = |sign: u8| if fnuz { 0x80 } else { 0x7F | sign };
    let above = largest.map(|largest| Limit {
        largest,
        nan: nan(0),
    });
    let below = largest.map(|largest| Limit {
        largest: largest | 0x80,
        nan: nan(0x80),
    });
    let saturate = Env::IEEE.with_saturate(true);
    for (a, bits) in (0..=u8::MAX).zip(0_usize..) {
        let x = Float::<S, 8>::from_bits(a);
        let context = format!("F8 format {index} {x:?}");
        let steps = (x.next_up(), x.next_up_with(saturate).0);
        check_next(x, steps, up[bits], above, &format!("next_up {context}"));
        let steps = (x.next_down(), x.next_down_with(saturate).0);
        check_next(x, steps, down[bits], below, &format!("next_down {context}"));
        assert_eq!((-x).to_bits(), negative[bits], "negate {context}");
        assert_eq!(x.abs().to_bits(), abs[bits], "abs {context}");
        for (b, pair) in (0..=u8::MAX).zip(bits * 256..) {
            let y = Float::<S, 8>::from_bits(b);
            let expected = if fnuz && a == 0 { 0 } else { copy_sign[pair] };
            assert_eq!(
                x.copy_sign(y).to_bits(),
                expected,
                "copy_sign {context} {y:?}"
            );
        }
    }
}

#[test]
fn every_fp8_operand_steps_and_copies_signs_like_ml_dtypes() {
    assert_eq!(
        TABLE.len(),
        7 * FORMAT_BYTES,
        "the table has one block for each format"
    );
    check_format::<Binary<4, NoInf>>(0, Some(0x7E));
    check_format::<Binary<5>>(1, None);
    check_format::<Binary<4, Fnuz>>(2, Some(0x7F));
    check_format::<Binary<5, Fnuz>>(3, Some(0x7F));
    check_format::<Binary<4>>(4, None);
    check_format::<Binary<3>>(5, None);
    check_format::<Binary<4, B11Fnuz>>(6, Some(0x7F));
}
