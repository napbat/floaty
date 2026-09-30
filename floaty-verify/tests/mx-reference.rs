//! Compares the MX formats FP4 E2M1, FP6 E2M3, and FP6 E3M2 with the
//! `ml_dtypes` tables in `data/`: every encoding, the conversion of every
//! binary16 encoding, and the arithmetic of every operand pair, rounding to
//! nearest even.
//!
//! `ml_dtypes` and floaty both give the largest finite value for an overflow
//! and for an infinity. They differ where a NaN meets a format without one:
//! floaty gives positive zero and signals invalid, as the `Finite` encoding
//! states. `ml_dtypes` 0.6.0 gives a zero whose sign is the opposite of the
//! sign of the NaN: `0x8` for the NaN of `np.nan` in FP4. The OCP MX
//! specification leaves that conversion to the implementation. So a result
//! that is a NaN in binary32, an invalid operation or a NaN operand, is
//! checked by floaty's rule.

use floaty::{Class, Decoded, Env, F16, Flags};
use floaty_verify::ml_dtypes::{Row, check, rows};

const TABLE: &str = include_str!("../data/mx-reference.txt");
const FROM_F16: &[u8] = include_bytes!("../data/mx-from-f16.bin");
const ARITHMETIC: &[u8] = include_bytes!("../data/mx-arithmetic.bin");

/// Returns the class and the decoded value of the encoding `bits` of the MX
/// format `alias`.
///
/// # Panics
///
/// Panics for a format that is not in the MX list.
fn decode(alias: &str, bits: u8) -> (Class, Decoded<1>) {
    macro_rules! decode_listed {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {
            if alias == stringify!($alias) {
                let value = floaty::$alias::from_bits(bits);
                return (value.classify(), value.decode::<1>());
            }
        };
    }
    floaty_verify::for_each_mx_format!(decode_listed);
    panic!("unknown format {alias}")
}

/// Returns the precision, `emax`, and `emin` of the MX format `alias`.
///
/// # Panics
///
/// Panics for a format that is not in the MX list.
fn parameters(alias: &str) -> (u32, i32, i32) {
    macro_rules! parameters_listed {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {
            if alias == stringify!($alias) {
                return (
                    floaty::$alias::PRECISION,
                    floaty::$alias::EMAX,
                    floaty::$alias::EMIN,
                );
            }
        };
    }
    floaty_verify::for_each_mx_format!(parameters_listed);
    panic!("unknown format {alias}")
}

#[test]
fn every_mx_encoding_matches_ml_dtypes() {
    let (mut formats, mut encodings) = (0, 0);
    for row in rows(TABLE) {
        match row {
            Row::Format {
                alias,
                precision,
                emax,
                emin,
            } => {
                let ours = parameters(alias);
                assert_eq!(ours, (precision, emax, emin), "{alias} parameters");
                formats += 1;
            }
            Row::Encoding {
                alias,
                bits,
                class,
                sign,
                value,
            } => {
                let (ours, decoded) = decode(alias, bits);
                let context = format!("{alias} {bits:#04x}");
                check(&context, ours, decoded, (class, sign, value));
                encodings += 1;
            }
        }
    }
    assert_eq!((formats, encodings), (3, 16 + 64 + 64));
}

/// Checks the conversion of every binary16 encoding to one format of the MX
/// list.
macro_rules! check_from_f16 {
    ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {{
        let block = &FROM_F16[$block * 65536..($block + 1) * 65536];
        for bits in 0..=u16::MAX {
            let half = F16::from_bits(bits);
            let (ours, flags) = half.convert_with::<floaty::$alias>(Env::IEEE);
            let context = format!(
                "{} from {bits:#06x}: {ours:?} {flags:?}",
                stringify!($alias)
            );
            if half.is_nan() {
                assert_eq!((ours.to_bits(), flags), (0, Flags::INVALID), "{context}");
            } else {
                assert_eq!(ours.to_bits(), block[usize::from(bits)], "{context}");
            }
        }
    }};
}

#[test]
fn every_binary16_encoding_converts_as_ml_dtypes_does() {
    floaty_verify::for_each_mx_format!(check_from_f16);
}

/// Checks one result against the table. `invalid` says that the operation
/// gives a NaN in binary32, where floaty gives positive zero.
fn compare(context: &str, (ours, flags): (u8, Flags), expected: u8, invalid: bool) {
    if invalid {
        assert_eq!(
            (ours, flags.contains(Flags::INVALID)),
            (0, true),
            "{context}: {flags:?}"
        );
    } else {
        assert_eq!(ours, expected, "{context}: {flags:?}");
    }
}

/// Checks every operand pair and every operand of one format of `$count`
/// encodings, whose block starts at byte `$start` of the table.
macro_rules! check_arithmetic {
    ($alias:ty, $count:expr, $start:expr) => {{
        let pairs = $count * $count;
        let block = &ARITHMETIC[$start..$start + 4 * pairs + $count];
        let encodings = || (0..$count).map(|bits| u8::try_from(bits).expect("8 bits"));
        for a in encodings() {
            let x = <$alias>::from_bits(a);
            for b in encodings() {
                let y = <$alias>::from_bits(b);
                let pair = usize::from(a) * $count + usize::from(b);
                let context = format!("{} {a:#04x} {b:#04x}", stringify!($alias));
                let result = |(value, flags): ($alias, Flags)| (value.to_bits(), flags);
                let both_zero = x.is_zero() && y.is_zero();
                compare(
                    &format!("{context} +"),
                    result(x.add_with(y, Env::IEEE)),
                    block[pair],
                    false,
                );
                compare(
                    &format!("{context} -"),
                    result(x.sub_with(y, Env::IEEE)),
                    block[pairs + pair],
                    false,
                );
                compare(
                    &format!("{context} *"),
                    result(x.mul_with(y, Env::IEEE)),
                    block[2 * pairs + pair],
                    false,
                );
                compare(
                    &format!("{context} /"),
                    result(x.div_with(y, Env::IEEE)),
                    block[3 * pairs + pair],
                    both_zero,
                );
            }
            let negative = x.is_sign_negative() && !x.is_zero();
            let context = format!("{} sqrt {a:#04x}", stringify!($alias));
            let root = x.sqrt_with(Env::IEEE);
            compare(
                &context,
                (root.0.to_bits(), root.1),
                block[4 * pairs + usize::from(a)],
                negative,
            );
        }
        $start + 4 * pairs + $count
    }};
}

#[test]
fn every_mx_operand_pair_computes_as_ml_dtypes_does() {
    // The blocks have different sizes, so each block starts where the block
    // before it ends. The list gives the formats in the order of the blocks.
    let (mut start, mut blocks) = (0, 0);
    macro_rules! check_listed {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {{
            use floaty::$alias;
            assert_eq!($block, blocks, "the list gives the blocks in order");
            start = check_arithmetic!($alias, 1_usize << $width, start);
            blocks += 1;
        }};
    }
    floaty_verify::for_each_mx_format!(check_listed);
    assert_eq!(
        (start, blocks),
        (ARITHMETIC.len(), 3),
        "the table holds the three formats"
    );
}
