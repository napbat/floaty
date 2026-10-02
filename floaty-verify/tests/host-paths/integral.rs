//! Round to integral, in the default mode and in each rounding direction.

use floaty::{BF16, F64, F80};
use floaty_verify::random::SplitMix64;

use super::operands::{conversion_operands, extended_operands, quad_operands};

#[test]
fn rounding_to_integral_gives_the_default_mode_result() {
    let mut random = SplitMix64::new(0x2B2B);
    for bits in 0..=u16::MAX {
        let value = floaty::F16::from_bits(bits);
        assert_eq!(
            value.round_to_integral().to_bits(),
            value.round_to_integral_with(floaty::F16::ENV).0.to_bits(),
            "{value:?}"
        );
        let value = BF16::from_bits(bits);
        assert_eq!(
            value.round_to_integral().to_bits(),
            value.round_to_integral_with(BF16::ENV).0.to_bits(),
            "{value:?}"
        );
    }
    let doubles = conversion_operands(&mut random);
    // Exact halves, which a tie rounds to even.
    let halves: Vec<F64> = (0..10_000)
        .map(|_| {
            let integer = F64::from_int(random.next_u64() >> 12);
            integer
                .add_with(F64::from_bits(0x3FE0_0000_0000_0000), floaty::Env::IEEE)
                .0
        })
        .collect();
    for value in doubles.iter().chain(&halves).copied() {
        assert_eq!(
            value.round_to_integral().to_bits(),
            value.round_to_integral_with(F64::ENV).0.to_bits(),
            "{value:?}"
        );
        let single: floaty::F32 = value.convert();
        assert_eq!(
            single.round_to_integral().to_bits(),
            single.round_to_integral_with(floaty::F32::ENV).0.to_bits(),
            "{single:?}"
        );
    }
    for value in extended_operands(&mut random) {
        assert_eq!(
            value.round_to_integral().to_bits(),
            value.round_to_integral_with(F80::ENV).0.to_bits(),
            "{value:?}"
        );
    }
    for value in quad_operands(&mut random) {
        assert_eq!(
            value.round_to_integral().to_bits(),
            value.round_to_integral_with(floaty::F128::ENV).0.to_bits(),
            "{value:?}"
        );
    }
}

/// Checks that `round_to_integral` of the format `$standard` at width
/// `$width`, in the mode that rounds in the direction `$direction`, gives the
/// result of its `_with` method in that mode, for each encoding of
/// `$encodings`.
macro_rules! directed_rounding_matches {
    ($standard:ty, $width:literal, $direction:ty, $encodings:expr) => {{
        type Value =
            floaty::Float<$standard, $width, floaty::mode::Rounded<floaty::mode::Ieee, $direction>>;
        for bits in $encodings {
            let value = Value::from_bits(bits);
            assert_eq!(
                value.round_to_integral().to_bits(),
                value.round_to_integral_with(Value::ENV).0.to_bits(),
                "{value:?} {}",
                stringify!($direction)
            );
        }
    }};
}

/// Checks `directed_rounding_matches` in each rounding direction.
macro_rules! every_direction_matches {
    ($standard:ty, $width:literal, $encodings:expr) => {{
        use floaty::mode::direction::{
            AwayFromZero, TiesToAway, TiesToEven, TiesTowardZero, ToOdd, TowardNegative,
            TowardPositive, TowardZero,
        };
        directed_rounding_matches!($standard, $width, TiesToEven, $encodings);
        directed_rounding_matches!($standard, $width, TiesToAway, $encodings);
        directed_rounding_matches!($standard, $width, TiesTowardZero, $encodings);
        directed_rounding_matches!($standard, $width, AwayFromZero, $encodings);
        directed_rounding_matches!($standard, $width, TowardPositive, $encodings);
        directed_rounding_matches!($standard, $width, TowardNegative, $encodings);
        directed_rounding_matches!($standard, $width, TowardZero, $encodings);
        directed_rounding_matches!($standard, $width, ToOdd, $encodings);
    }};
}

#[test]
fn rounding_to_integral_in_each_direction_gives_the_mode_result() {
    // The paths take the direction of the mode in the encoding of the
    // instruction. x87 extended rounds only to nearest even on its path.
    let mut random = SplitMix64::new(0x2B2D);
    every_direction_matches!(floaty::Binary<5>, 16, 0..=u16::MAX);
    every_direction_matches!(floaty::Binary<8>, 16, 0..=u16::MAX);
    let doubles = conversion_operands(&mut random);
    // Exact halves, and values a quarter and three quarters above an integer.
    let fractions: Vec<F64> = (0..10_000)
        .map(|index| {
            let integer = F64::from_int(random.next_u64() >> 12);
            let fraction = [
                0x3FE0_0000_0000_0000,
                0x3FD0_0000_0000_0000,
                0x3FE8_0000_0000_0000,
            ];
            let sum = integer
                .add_with(F64::from_bits(fraction[index % 3]), floaty::Env::IEEE)
                .0;
            if index % 2 == 0 { sum } else { -sum }
        })
        .collect();
    let values: Vec<F64> = doubles.iter().chain(&fractions).copied().collect();
    every_direction_matches!(
        floaty::Binary<11>,
        64,
        values.iter().map(|value| value.to_bits())
    );
    let singles = values
        .iter()
        .map(|&value| value.convert::<floaty::F32>().to_bits());
    every_direction_matches!(floaty::Binary<8>, 32, singles.clone());
    let extended: Vec<u128> = extended_operands(&mut random)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    every_direction_matches!(
        floaty::Binary<15, floaty::X87>,
        80,
        extended.iter().copied()
    );
    let quads: Vec<u128> = quad_operands(&mut random)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    every_direction_matches!(floaty::Binary<15>, 128, quads.iter().copied());
}
