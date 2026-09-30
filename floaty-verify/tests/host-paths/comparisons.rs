//! The comparison, and each minimum and maximum operation.

use floaty::{BF16, F64};
use floaty_verify::random::SplitMix64;

use super::operands::{boundary_pairs, random_pairs};

/// Checks that the comparison and each minimum and maximum operation of the
/// type `$alias` give the results of their `_with` methods in the default
/// mode, for each pair of `$pairs`.
macro_rules! comparisons_match {
    ($alias:ty, $pairs:expr) => {
        for &(a, b) in $pairs {
            let (x, y) = (<$alias>::from_bits(a), <$alias>::from_bits(b));
            let env = <$alias>::ENV;
            let order = x.compare_quiet_with(y, env).0;
            assert_eq!(x.partial_cmp(&y), order, "{a:#x} {b:#x}: partial_cmp");
            assert_eq!(
                x == y,
                order == Some(core::cmp::Ordering::Equal),
                "{a:#x} {b:#x}: eq"
            );
            let ours = [
                x.minimum(y),
                x.maximum(y),
                x.minimum_number(y),
                x.maximum_number(y),
                x.min_num(y),
                x.max_num(y),
            ]
            .map(|value| value.to_bits());
            let engine = [
                x.minimum_with(y, env).0,
                x.maximum_with(y, env).0,
                x.minimum_number_with(y, env).0,
                x.maximum_number_with(y, env).0,
                x.min_num_with(y, env).0,
                x.max_num_with(y, env).0,
            ]
            .map(|value| value.to_bits());
            assert_eq!(ours, engine, "{a:#x} {b:#x}: minimum and maximum");
        }
    };
}

#[test]
fn comparisons_give_the_default_mode_results() {
    let mut random = SplitMix64::new(0xC0AA);
    let mut halves = boundary_pairs::<u16>(16, 5);
    halves.extend(random_pairs::<u16>(&mut random, 16, 5, 20_000));
    comparisons_match!(floaty::F16, &halves);
    let mut bfloats = boundary_pairs::<u16>(16, 8);
    bfloats.extend(random_pairs::<u16>(&mut random, 16, 8, 20_000));
    comparisons_match!(BF16, &bfloats);
    let mut singles = boundary_pairs::<u32>(32, 8);
    singles.extend(random_pairs::<u32>(&mut random, 32, 8, 20_000));
    comparisons_match!(floaty::F32, &singles);
    let mut doubles = boundary_pairs::<u64>(64, 11);
    doubles.extend(random_pairs::<u64>(&mut random, 64, 11, 20_000));
    comparisons_match!(F64, &doubles);
    // Equal values of each sign, which random pairs rarely give.
    let equal: Vec<(u32, u32)> = (0..2_000)
        .map(|_| {
            let bits = u32::try_from(random.next_u64() >> 32).expect("the shift keeps 32 bits");
            (bits, bits)
        })
        .collect();
    comparisons_match!(floaty::F32, &equal);
}
