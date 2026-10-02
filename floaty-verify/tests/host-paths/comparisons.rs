//! The comparison, and each minimum and maximum operation.

use floaty::{BF16, F64};
use floaty_verify::encodings::Layout;
use floaty_verify::entry_points::{comparisons, comparisons_with};
use floaty_verify::random::SplitMix64;

use super::operands::{boundary_pairs, random_pairs};

/// Checks that the comparison entry points of [`comparisons`] of the type
/// `$alias` give the results of their `_with` methods in the default mode,
/// for each pair of `$pairs`.
macro_rules! comparisons_match {
    ($alias:ty, $pairs:expr) => {
        for &(a, b) in $pairs {
            let (x, y) = (<$alias>::from_bits(a), <$alias>::from_bits(b));
            assert_eq!(comparisons(x, y), comparisons_with(x, y), "{a:#x} {b:#x}");
        }
    };
}

#[test]
fn comparisons_give_the_default_mode_results() {
    let mut random = SplitMix64::new(0xC0AA);
    let mut halves = boundary_pairs::<u16>(Layout::BINARY16);
    halves.extend(random_pairs::<u16>(&mut random, Layout::BINARY16, 20_000));
    comparisons_match!(floaty::F16, &halves);
    let mut bfloats = boundary_pairs::<u16>(Layout::BFLOAT16);
    bfloats.extend(random_pairs::<u16>(&mut random, Layout::BFLOAT16, 20_000));
    comparisons_match!(BF16, &bfloats);
    let mut singles = boundary_pairs::<u32>(Layout::BINARY32);
    singles.extend(random_pairs::<u32>(&mut random, Layout::BINARY32, 20_000));
    comparisons_match!(floaty::F32, &singles);
    let mut doubles = boundary_pairs::<u64>(Layout::BINARY64);
    doubles.extend(random_pairs::<u64>(&mut random, Layout::BINARY64, 20_000));
    comparisons_match!(F64, &doubles);
    // Equal values of each sign, which random pairs rarely give.
    let equal: Vec<(u32, u32)> = (0..2_000)
        .map(|_| {
            let bits = u32::try_from(random.next_u64() >> 32).expect("the shift keeps 32 bits");
            (bits, bits)
        })
        .collect();
    comparisons_match!(floaty::F32, &equal);
    let mut quads = boundary_pairs::<u128>(Layout::BINARY128);
    quads.extend(random_pairs::<u128>(&mut random, Layout::BINARY128, 20_000));
    let equal_quads: Vec<(u128, u128)> = (0..2_000)
        .map(|_| {
            let bits = random.next_u128();
            (bits, bits)
        })
        .collect();
    comparisons_match!(floaty::F128, &quads);
    comparisons_match!(floaty::F128, &equal_quads);
}
