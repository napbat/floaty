//! Tests of the wrappers and helpers of the Intel library.

use core::cmp::Ordering;
use std::collections::BTreeSet;

use super::{
    Bid32, Bid64, Bid128, Class, Extremum, Flags, Format, Inexact, Integer, Predicate, Rounding,
    bid64_to_bid128, equal_operand_choice,
};
use crate::random::SplitMix64;

#[test]
fn maps_codes() {
    for (code, rounding) in (0..).zip(Rounding::ALL) {
        assert_eq!(rounding.code(), code);
        assert_eq!(Rounding::from_code(code), Some(rounding));
    }
    assert_eq!(Rounding::from_code(5), None);
    for (code, class) in (0..).zip(Class::ALL) {
        assert_eq!(Class::from_code(code), Some(class));
    }
    assert_eq!(Class::from_code(10), None);
}

#[test]
fn computes_each_width() {
    let one = Bid64::from_string("1", Rounding::TiesToEven);
    assert_eq!(one.value, 0x31c0_0000_0000_0001);
    assert_eq!(one.flags, Flags::NONE);
    let two = Bid64::add(one.value, one.value, Rounding::TiesToEven);
    assert_eq!(two.value, 0x31c0_0000_0000_0002);
    assert_eq!(Bid64::to_string(two.value).value, "+2E+0");
    let third = Bid32::div(
        Bid32::from_int32(1, Rounding::TiesToEven).value,
        Bid32::from_int32(3, Rounding::TiesToEven).value,
        Rounding::TowardZero,
    );
    assert_eq!(Bid32::to_string(third.value).value, "+3333333E-7");
    assert_eq!(third.flags, Flags::INEXACT);
    let wide = bid64_to_bid128(one.value).value;
    assert_eq!(wide, 0x3040_0000_0000_0000_0000_0000_0000_0001);
    assert_eq!(Bid128::from_dpd(Bid128::to_dpd(wide)), wide);
    assert_eq!(Bid128::class(wide), Class::PositiveNormal);
    let less = Bid128::compare(wide, Bid128::negate(wide), Predicate::QuietLess);
    assert!(!less.value);
}

#[test]
fn converts_binary_and_integers() {
    let one = Bid64::from_int64(1, Rounding::TiesToEven).value;
    assert_eq!(
        Bid64::to_binary64(one, Rounding::TiesToEven).value,
        0x3ff0_0000_0000_0000
    );
    assert_eq!(
        Bid64::to_binary32(one, Rounding::TiesToEven).value,
        0x3f80_0000
    );
    assert_eq!(
        Bid64::to_binary80(one, Rounding::TiesToEven).value,
        0x3fff_8000_0000_0000_0000
    );
    assert_eq!(
        Bid64::to_binary128(one, Rounding::TiesToEven).value,
        0x3fff_0000_0000_0000_0000_0000_0000_0000
    );
    assert_eq!(
        Bid64::from_binary80(0x3fff_8000_0000_0000_0000, Rounding::TiesToEven).value,
        one
    );
    let half = Bid64::from_string("-2.5", Rounding::TiesToEven).value;
    let rounded = Bid64::to_integer(half, Integer::Int8, Rounding::TiesToEven, Inexact::Signaled);
    assert_eq!(rounded.value, -2);
    assert_eq!(rounded.flags, Flags::INEXACT);
    let invalid = Bid64::to_integer(half, Integer::UInt8, Rounding::TowardZero, Inexact::Ignored);
    assert_eq!(invalid.flags, Flags::INVALID);
}

#[test]
fn builds_encodings_of_each_layout() {
    let parameters = [Bid32::LAYOUT, Bid64::LAYOUT, Bid128::LAYOUT].map(|layout| {
        (
            layout.precision(),
            layout.emax(),
            layout.bias(),
            layout.largest_field(),
        )
    });
    assert_eq!(
        parameters,
        [
            (7, 96, 101, 191),
            (16, 384, 398, 767),
            (34, 6144, 6176, 12287)
        ]
    );
    let layout = Bid64::LAYOUT;
    assert_eq!(layout.number(false, 398, 1), 0x31c0_0000_0000_0001);
    // The largest decimal64 value takes the form with the implicit bits 100.
    let largest = layout.number(false, 767, 9_999_999_999_999_999);
    assert_eq!(largest, 0x77fb_86f2_6fc0_ffff);
    assert_eq!(layout.fields(largest), Some((767, 9_999_999_999_999_999)));
    assert_eq!(layout.infinity(true), 0xf800_0000_0000_0000);
    assert_eq!(layout.fields(layout.infinity(false)), None);
    assert_eq!(layout.nan(false, true, 5, 1), 0x7e04_0000_0000_0005);
    // A coefficient above 10^16 - 1 gives a non-canonical encoding.
    let wide = layout.number(true, 0, 10_000_000_000_000_000);
    assert!(!Bid64::is_canonical(narrow(wide)));
    assert_eq!(Bid64::class(narrow(wide)), Class::NegativeZero);
}

/// Returns a decimal64 encoding from its bits.
fn narrow(bits: u128) -> u64 {
    u64::try_from(bits).expect("a decimal64 encoding has 64 bits")
}

#[test]
fn random_encodings_fit_and_read_back() {
    let mut rng = SplitMix64::new(1);
    for layout in [Bid32::LAYOUT, Bid64::LAYOUT, Bid128::LAYOUT] {
        let mut classes = [0_usize; 10];
        for _ in 0..10_000 {
            let bits = layout.random(&mut rng);
            assert!(bits >> (layout.width - 1) <= 1, "{bits:#x} fits the width");
            if let Some((field, coefficient)) = layout.fields(bits) {
                let negative = bits >> (layout.width - 1) == 1;
                assert_eq!(layout.number(negative, field, coefficient), bits);
            }
            let class = match layout.width {
                32 => Bid32::class(u32::try_from(bits).expect("32 bits")),
                64 => Bid64::class(narrow(bits)),
                _ => Bid128::class(bits),
            };
            let index = usize::try_from(class.code()).expect("a class code fits a usize");
            classes[index] += 1;
        }
        assert!(classes.iter().all(|&count| count > 0), "{classes:?}");
    }
}

#[test]
fn draws_operands_with_their_properties() {
    let mut rng = SplitMix64::new(2);
    let decimal32 = |bits: u128| u32::try_from(bits).expect("a decimal32 encoding has 32 bits");
    let mut exact_roots = 0;
    let mut bounds = BTreeSet::new();
    for _ in 0..4_000 {
        let (x, y) = Bid32::LAYOUT.equal_pair(&mut rng);
        let equal = Bid32::compare(decimal32(x), decimal32(y), Predicate::QuietEqual);
        assert!(equal.value, "{x:#x} and {y:#x} compare equal");
        let (x, y) = Bid64::LAYOUT.equal_pair(&mut rng);
        let equal = Bid64::compare(narrow(x), narrow(y), Predicate::QuietEqual);
        assert!(equal.value, "{x:#x} and {y:#x} compare equal");
        let root = Bid32::sqrt(
            decimal32(Bid32::LAYOUT.square(&mut rng)),
            Rounding::TiesToEven,
        );
        exact_roots += usize::from(root.flags == Flags::NONE);
        bounds.insert(Bid32::LAYOUT.near_integer(&mut rng, Integer::Int32));
    }
    // Half the radicands are squares, and every square has an exact root.
    assert!(exact_roots > 1_900, "{exact_roots} exact roots");
    // decimal32 gets its nearest values on both sides of 2^31 - 1 and -2^31.
    let bias = Bid32::LAYOUT.bias();
    for (negative, coefficient) in [(false, 2_147_483), (false, 2_147_484), (true, 2_147_484)] {
        let value = Bid32::LAYOUT.number(negative, bias + 3, coefficient);
        assert!(bounds.contains(&value), "{value:#x} is drawn");
    }
}

#[test]
fn builds_canonical_encodings_and_chooses_between_equal_operands() {
    let layout = Bid64::LAYOUT;
    let wide = layout.number(true, 5, 10_000_000_000_000_000);
    assert_eq!(layout.canonical(wide), layout.number(true, 5, 0));
    let infinity = layout.infinity(false);
    assert_eq!(layout.canonical(infinity | 1), infinity);
    let nan = layout.nan(false, true, 1_000_000_000_000_000, 3);
    assert_eq!(layout.canonical(nan), layout.nan(false, true, 0, 0));
    let choose = |x: u128, y: u128, extremum| {
        u128::from(equal_operand_choice::<Bid64>(
            narrow(x),
            narrow(y),
            extremum,
        ))
    };
    let (plus_zero, minus_zero) = (layout.number(false, 398, 0), layout.number(true, 398, 0));
    assert_eq!(choose(plus_zero, minus_zero, Extremum::Minimum), minus_zero);
    assert_eq!(choose(minus_zero, plus_zero, Extremum::Maximum), plus_zero);
    // For a negative zero, totalOrder puts the larger exponent first.
    assert_eq!(choose(wide, minus_zero, Extremum::Minimum), minus_zero);
    assert_eq!(
        choose(wide, minus_zero, Extremum::Maximum),
        layout.number(true, 5, 0)
    );
    // 1.0 and 1.00: totalOrder puts the smaller exponent first for a
    // positive value, and last for a negative value.
    let (one_tenth, one_hundredth) = (
        layout.number(false, 397, 10),
        layout.number(false, 396, 100),
    );
    assert_eq!(
        choose(one_tenth, one_hundredth, Extremum::Minimum),
        one_hundredth
    );
    assert_eq!(
        choose(one_hundredth, one_tenth, Extremum::Maximum),
        one_tenth
    );
    let negate = |bits: u128| bits | 1 << 63;
    assert_eq!(
        choose(negate(one_hundredth), negate(one_tenth), Extremum::Minimum),
        negate(one_tenth)
    );
}

#[test]
fn maps_floaty_flags_integers_and_predicates() {
    let flags = floaty::Flags::INVALID
        | floaty::Flags::INEXACT
        | floaty::Flags::TINY
        | floaty::Flags::DENORMAL_INPUT;
    assert_eq!(
        Flags::from_floaty(flags),
        Flags::INVALID | Flags::INEXACT | Flags::DENORMAL
    );
    assert_eq!(
        Flags::from_floaty(flags).difference(Flags::DENORMAL),
        Flags::INVALID | Flags::INEXACT
    );
    let indefinite = Integer::ALL.map(Integer::indefinite);
    assert_eq!(
        indefinite,
        [
            -0x80,
            -0x8000,
            -0x8000_0000,
            -0x8000_0000_0000_0000,
            0x80,
            0x8000,
            0x8000_0000,
            0x8000_0000_0000_0000
        ]
    );
    assert!(Predicate::QuietGreaterUnordered.holds(None));
    assert!(!Predicate::QuietNotEqual.holds(Some(Ordering::Equal)));
    assert!(Predicate::SignalingNotLess.is_signaling());
    assert!(!Predicate::QuietLess.is_signaling());
    assert_eq!(
        Class::from_floaty(floaty::Class::Subnormal, true),
        Some(Class::NegativeSubnormal)
    );
    assert_eq!(Class::from_floaty(floaty::Class::Unsupported, false), None);
    assert_eq!(
        floaty::Rounding::from(Rounding::TowardNegative),
        floaty::Rounding::TowardNegative
    );
}
