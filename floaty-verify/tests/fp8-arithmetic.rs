//! Compares the arithmetic of the FP8 formats with `ml_dtypes`, rounding to
//! nearest even, for every operand pair: the table `data/fp8-arithmetic.bin`.
//!
//! `ml_dtypes` computes in binary32 on an x86 host, so an invalid operation
//! gives the negative x86 default NaN. The test therefore uses the SSE NaN
//! rule, whose default NaN is negative, and compares every result exactly. The
//! one exception is a NaN operand: `ml_dtypes` does not keep the sign of the
//! NaN it propagates (E4M3 `0x00 + 0xFF` gives `0x7F`), so a result with a NaN
//! operand only has to be a NaN.

use floaty::env::{NanPropagation, NanRule};
use floaty::{Env, F8E4M3Fn, F8E4M3Fnuz, F8E5M2, F8E5M2Fnuz, Flags};

/// The behavior of the table: round to nearest even with the SSE NaN rule.
const SSE: Env =
    Env::IEEE.with_nan(NanRule::new(NanPropagation::FirstOperand).with_default_negative(true));

const TABLE: &[u8] = include_bytes!("../data/fp8-arithmetic.bin");

/// The bytes of each format in the table: four operations on every pair, and
/// the square root of every operand.
const FORMAT_BYTES: usize = 4 * 65536 + 256;

/// Checks one result against the table.
macro_rules! check {
    ($alias:ty, $operation:literal, $operands:expr, $ours:expr, $expected:expr) => {{
        let (ours, flags): ($alias, Flags) = $ours;
        let expected = <$alias>::from_bits($expected);
        let nan_operand = $operands.iter().any(|operand: &$alias| operand.is_nan());
        let context = format!(
            "{} {} {:?}: {ours:?} {flags:?}",
            stringify!($alias),
            $operation,
            $operands
        );
        if expected.is_nan() && nan_operand {
            assert!(ours.is_nan(), "{context}");
        } else {
            assert_eq!(ours.to_bits(), expected.to_bits(), "{context}");
        }
    }};
}

/// Checks every operand pair and every operand of one format.
macro_rules! check_format {
    ($alias:ty, $index:literal) => {{
        let block = &TABLE[$index * FORMAT_BYTES..($index + 1) * FORMAT_BYTES];
        for a in 0..=u8::MAX {
            let x = <$alias>::from_bits(a);
            for b in 0..=u8::MAX {
                let y = <$alias>::from_bits(b);
                let pair = usize::from(a) * 256 + usize::from(b);
                check!($alias, "add", [x, y], x.add_with(y, SSE), block[pair]);
                check!(
                    $alias,
                    "sub",
                    [x, y],
                    x.sub_with(y, SSE),
                    block[65536 + pair]
                );
                check!(
                    $alias,
                    "mul",
                    [x, y],
                    x.mul_with(y, SSE),
                    block[2 * 65536 + pair]
                );
                check!(
                    $alias,
                    "div",
                    [x, y],
                    x.div_with(y, SSE),
                    block[3 * 65536 + pair]
                );
            }
            check!(
                $alias,
                "sqrt",
                [x],
                x.sqrt_with(SSE),
                block[4 * 65536 + usize::from(a)]
            );
        }
    }};
}

#[test]
fn every_fp8_operand_pair_matches_ml_dtypes() {
    assert_eq!(
        TABLE.len(),
        4 * FORMAT_BYTES,
        "the table has one block for each format"
    );
    check_format!(F8E4M3Fn, 0);
    check_format!(F8E5M2, 1);
    check_format!(F8E4M3Fnuz, 2);
    check_format!(F8E5M2Fnuz, 3);
}
