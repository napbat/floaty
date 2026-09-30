//! The remainder.

use floaty::{BF16, F16, F64, F80};
use floaty_verify::encodings::Layout;
use floaty_verify::random::SplitMix64;

use super::operands::{close_pairs, every_16_bit_pair, random_pairs};

#[test]
fn remainders_give_the_default_mode_results() {
    // The x87 path takes operands whose exponents differ by up to 630, and
    // the pairs reach past that bound.
    let mut random = SplitMix64::new(0xE3E3);
    let mut singles = close_pairs(&mut random, Layout::BINARY32, 300);
    singles.extend(random_pairs::<u128>(&mut random, Layout::BINARY32, 4_000));
    for (a, b) in singles {
        let bits = |value: u128| u32::try_from(value).expect("a binary32 encoding");
        let (x, y) = (
            floaty::F32::from_bits(bits(a)),
            floaty::F32::from_bits(bits(b)),
        );
        let expected = x.remainder_with(y, floaty::F32::ENV).0;
        assert_eq!(
            x.remainder(y).to_bits(),
            expected.to_bits(),
            "{a:#x} {b:#x}"
        );
    }
    let mut doubles = close_pairs(&mut random, Layout::BINARY64, 900);
    doubles.extend(random_pairs::<u128>(&mut random, Layout::BINARY64, 4_000));
    for (a, b) in doubles {
        let bits = |value: u128| u64::try_from(value).expect("a binary64 encoding");
        let (x, y) = (F64::from_bits(bits(a)), F64::from_bits(bits(b)));
        let expected = x.remainder_with(y, F64::ENV).0;
        assert_eq!(
            x.remainder(y).to_bits(),
            expected.to_bits(),
            "{a:#x} {b:#x}"
        );
    }
    let mut extended = close_pairs(&mut random, Layout::X87_EXTENDED, 900);
    extended.extend(random_pairs::<u128>(
        &mut random,
        Layout::X87_EXTENDED,
        4_000,
    ));
    for (a, b) in extended {
        let (x, y) = (F80::from_bits(a), F80::from_bits(b));
        let expected = x.remainder_with(y, F80::ENV).0;
        assert_eq!(
            x.remainder(y).to_bits(),
            expected.to_bits(),
            "{a:#x} {b:#x}"
        );
    }
    // binary16 and bfloat16 take the binary32 path on their widened values.
    let mut halves = close_pairs(&mut random, Layout::BINARY16, 40);
    halves.extend(random_pairs::<u128>(&mut random, Layout::BINARY16, 4_000));
    let mut bfloats = close_pairs(&mut random, Layout::BFLOAT16, 300);
    bfloats.extend(random_pairs::<u128>(&mut random, Layout::BFLOAT16, 4_000));
    let bits = |value: u128| u16::try_from(value).expect("a 16-bit encoding");
    for (a, b) in halves {
        let (x, y) = (F16::from_bits(bits(a)), F16::from_bits(bits(b)));
        let expected = x.remainder_with(y, F16::ENV).0;
        assert_eq!(
            x.remainder(y).to_bits(),
            expected.to_bits(),
            "{a:#x} {b:#x}"
        );
    }
    for (a, b) in bfloats {
        let (x, y) = (BF16::from_bits(bits(a)), BF16::from_bits(bits(b)));
        let expected = x.remainder_with(y, BF16::ENV).0;
        assert_eq!(
            x.remainder(y).to_bits(),
            expected.to_bits(),
            "{a:#x} {b:#x}"
        );
    }
}

/// Every remainder of binary16 and of bfloat16 operands, in 16 threads. The
/// remainder of the widened binary32 values must narrow exactly for each
/// pair, ties and exact multiples included. Run time in a release build:
/// about 70 seconds on x86-64.
#[test]
#[ignore = "exhaustive sweep; run with --ignored"]
fn every_16_bit_remainder_gives_the_default_mode_result() {
    every_16_bit_pair(|high, low| {
        let (x, y) = (F16::from_bits(high), F16::from_bits(low));
        let engine = x.remainder_with(y, F16::ENV).0;
        assert_eq!(
            x.remainder(y).to_bits(),
            engine.to_bits(),
            "F16 {high:#06x} {low:#06x}"
        );
        let (x, y) = (BF16::from_bits(high), BF16::from_bits(low));
        let engine = x.remainder_with(y, BF16::ENV).0;
        assert_eq!(
            x.remainder(y).to_bits(),
            engine.to_bits(),
            "BF16 {high:#06x} {low:#06x}"
        );
    });
}

/// Checks the remainder of the format `$alias` on dividends whose quotient
/// by the divisor is an exact half, which rounds to the even integer, and on
/// exact multiples of the divisor, whose remainder is a zero with the sign of
/// the dividend. Each divisor is an odd integer below `2^$bits` times a power
/// of two, so every product is exact.
macro_rules! ties_and_multiples_match {
    ($alias:ty, $bits:literal, $random:expr) => {{
        let random: &mut SplitMix64 = $random;
        for _ in 0..4_000 {
            let odd = i64::try_from(random.next_u64() >> (64 - $bits)).expect("small") | 1;
            let quotient = i64::try_from(random.next_u64() >> 54).expect("small");
            let scale = i32::try_from(random.next_u64() % 60).expect("small") - 30;
            let sign = if random.next_u64() & 1 == 0 { 1 } else { -1 };
            let divisor = <$alias>::from_int(odd).scale_b(scale);
            let tie = <$alias>::from_int(sign * (2 * quotient + 1) * odd).scale_b(scale - 1);
            let multiple = <$alias>::from_int(sign * quotient * odd).scale_b(scale);
            for dividend in [tie, multiple] {
                assert_eq!(
                    dividend.remainder(divisor).to_bits(),
                    dividend.remainder_with(divisor, <$alias>::ENV).0.to_bits(),
                    "{dividend:?} {divisor:?}"
                );
            }
        }
    }};
}

#[test]
fn remainders_of_ties_and_multiples_give_the_default_mode_results() {
    // `FPREM1` rounds a quotient that lies halfway between two integers to
    // the even one, and a remainder of zero keeps the sign of the dividend.
    let mut random = SplitMix64::new(0xE3E4);
    ties_and_multiples_match!(floaty::F32, 12, &mut random);
    ties_and_multiples_match!(F64, 40, &mut random);
    ties_and_multiples_match!(F80, 50, &mut random);
    // A dividend one binade below a divisor whose exponent field is the
    // precision gives a subnormal remainder.
    let (x, y) = (
        floaty::F32::from_bits(0x0BFF_FFFF),
        floaty::F32::from_bits(0x0C00_0000),
    );
    assert_eq!(x.remainder(y).to_bits(), 0x8040_0000);
    let (x, y) = (
        F64::from_bits(0x034F_FFFF_FFFF_FFFF),
        F64::from_bits(0x0350_0000_0000_0000),
    );
    assert_eq!(x.remainder(y).to_bits(), 0x8008_0000_0000_0000);
    let (x, y) = (
        F80::from_bits(0x003F_FFFF_FFFF_FFFF_FFFF),
        F80::from_bits(0x0040_8000_0000_0000_0000),
    );
    assert_eq!(x.remainder(y).to_bits(), 0x8000_4000_0000_0000_0000);
}
