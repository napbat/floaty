//! Compares `atan2` and `atan2Pi` of the binary formats with the MPFR oracle
//! in `floaty_verify::operations`, in every behavior of
//! `operations::BEHAVIORS`.
//!
//! Every pair of encodings of the FP8 and MX formats runs. The other formats
//! run each boundary encoding against zeros, ones, threes, infinities, and a
//! NaN in both orders, random pairs of boundary encodings and of any
//! encodings, pairs of one magnitude and next to one magnitude, and pairs
//! whose quotient `|y| / |x|` lies next to each bound of the shortcuts of the
//! engine: `2^-(p + 3)` and `2^(p + 4)`.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::format::Standard;
use floaty::{Binary, F32, F64, F80, F128, Float, TF32};
use floaty_verify::encodings::{Layout, boundary_encodings, to_limbs};
use floaty_verify::mpfr::{Format, Specials};
use floaty_verify::operations::{BEHAVIORS, check};
use floaty_verify::random::SplitMix64;
use rug::Integer;
use rug::integer::Order;

/// Checks every pair of encodings of a format of the small format lists.
macro_rules! every_small_pair {
    ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {{
        let format = Format::of::<Float<$standard, $width>>($specials);
        let value = |bits: u16| {
            Float::<$standard, $width>::from_bits(
                u8::try_from(bits).expect("the format has at most 8 bits"),
            )
        };
        for y in 0..1_u16 << $width {
            for x in 0..1_u16 << $width {
                for env in &BEHAVIORS {
                    check::check_atan2(value(y), value(x), &format, env);
                }
            }
        }
    }};
}

#[test]
fn every_fp8_pair() {
    floaty_verify::for_each_fp8_format!(every_small_pair);
}

#[test]
fn every_mx_pair() {
    floaty_verify::for_each_mx_format!(every_small_pair);
}

/// Returns an encoding of `width` random bits.
fn random_encoding(width: u32, random: &mut SplitMix64) -> Integer {
    let words: Vec<u64> = (0..width.div_ceil(64)).map(|_| random.next_u64()).collect();
    Integer::from_digits(&words, Order::Lsf).keep_bits(width)
}

/// Returns the pairs of a format of the IEEE layout, `y` first.
fn pairs<S: Standard<W>, const W: usize>(
    layout: Layout,
    make: &dyn Fn(&Integer) -> Float<S, W>,
    (count, seed): (usize, u64),
) -> Vec<(Float<S, W>, Float<S, W>)> {
    let mut random = SplitMix64::new(seed);
    let boundary: Vec<Float<S, W>> = boundary_encodings(layout).iter().map(make).collect();
    let (zero, one, three) = (
        Float::<S, W>::from_int(0),
        Float::<S, W>::from_int(1),
        Float::<S, W>::from_int(3),
    );
    let infinity = one.scale_b(i32::MAX);
    let partners = [
        zero,
        -zero,
        one,
        -one,
        three,
        -three,
        infinity,
        -infinity,
        zero * infinity,
    ];
    let mut pairs = Vec::new();
    for &value in &boundary {
        for &partner in &partners {
            pairs.extend([(value, partner), (partner, value)]);
        }
    }
    let pick = |random: &mut SplitMix64| {
        let index = random.below(u64::try_from(boundary.len()).expect("a count fits"));
        boundary[usize::try_from(index).expect("an index fits")]
    };
    pairs.extend((0..count).map(|_| (pick(&mut random), pick(&mut random))));
    let width = layout.width;
    pairs.extend((0..count).map(|_| {
        let y = make(&random_encoding(width, &mut random));
        (y, make(&random_encoding(width, &mut random)))
    }));
    // `|y| = |x|` in each quadrant, and `|y|` one step above `|x|`.
    for value in [one, three.scale_b(-5), three.scale_b(7)] {
        for (y, x) in [(value, value), (value, value.next_up())] {
            pairs.extend([(y, x), (y, -x), (-y, x), (-y, -x)]);
        }
    }
    // The quotient next to the bounds of the shortcuts, on the grid and off
    // it: `2^k / 1` and `3 2^k / 3` lie on it, and `2^k / 3` off it.
    let precision = i32::try_from(layout.precision()).expect("a precision fits an i32");
    for k in (-(precision + 7)..=-(precision + 1)).chain(precision + 2..=precision + 7) {
        let power = one.scale_b(k);
        for (y, x) in [(power, one), (three.scale_b(k), three), (power, three)] {
            pairs.extend([(y, x), (y, -x), (-y, x), (-y, -x)]);
        }
    }
    pairs
}

/// Checks the pairs of [`pairs`] in every behavior.
fn check_format<S: Standard<W>, const W: usize>(
    layout: Layout,
    make: &dyn Fn(&Integer) -> Float<S, W>,
    samples: (usize, u64),
) {
    let format = Format::of::<Float<S, W>>(Specials::Ieee);
    let pairs = pairs(layout, make, samples);
    for env in &BEHAVIORS {
        for &(y, x) in &pairs {
            check::check_atan2(y, x, &format, env);
        }
    }
}

#[test]
fn binary16_bfloat16_tf32_binary32_binary64_x87_and_binary128() {
    let bits16 = |bits: &Integer| bits.to_u16().expect("a 16-bit encoding fits a u16");
    check_format(
        Layout::BINARY16,
        &|bits: &Integer| Float::<Binary<5>, 16>::from_bits(bits16(bits)),
        (2_000, 0xA216),
    );
    check_format(
        Layout::BFLOAT16,
        &|bits: &Integer| Float::<Binary<8>, 16>::from_bits(bits16(bits)),
        (2_000, 0xAB16),
    );
    check_format(
        Layout::TF32,
        &|bits: &Integer| TF32::from_bits(bits.to_u32().expect("a TF32 encoding has 19 bits")),
        (1_000, 0xA219),
    );
    check_format(
        Layout::BINARY32,
        &|bits: &Integer| F32::from_bits(bits.to_u32().expect("a binary32 encoding fits a u32")),
        (1_000, 0xA232),
    );
    check_format(
        Layout::BINARY64,
        &|bits: &Integer| F64::from_bits(bits.to_u64().expect("a binary64 encoding fits a u64")),
        (1_000, 0xA264),
    );
    check_format(
        Layout::X87_EXTENDED,
        &|bits: &Integer| F80::from_bits(bits.to_u128().expect("an x87 encoding fits a u128")),
        (500, 0xA280),
    );
    check_format(
        Layout::BINARY128,
        &|bits: &Integer| {
            F128::from_bits(bits.to_u128().expect("a binary128 encoding fits a u128"))
        },
        (500, 0xA2128),
    );
}

/// Layouts whose precision fills the storage up to two bits, and a layout of
/// the most exponent bits, 28, with four bits of precision, whose quotients
/// reach `2^(±2^28)`.
#[test]
fn layouts_that_fill_or_cross_limbs() {
    check_format(
        Layout::ieee(64, 2),
        &|bits: &Integer| {
            Float::<Binary<2>, 64>::from_bits(bits.to_u64().expect("the encoding fits a u64"))
        },
        (300, 0xA2264),
    );
    check_format(
        Layout::ieee(72, 15),
        &|bits: &Integer| {
            Float::<Binary<15>, 72>::from_bits(bits.to_u128().expect("the encoding fits a u128"))
        },
        (300, 0xA2072),
    );
    check_format(
        Layout::ieee(32, 28),
        &|bits: &Integer| {
            Float::<Binary<28>, 32>::from_bits(bits.to_u32().expect("the encoding fits a u32"))
        },
        (300, 0xA2832),
    );
}

/// Checks one format of the wide format list.
macro_rules! wide_format {
    ($alias:ident, $width:literal, $exponent_bits:literal, $limbs:literal) => {
        check_format(
            Layout::ieee($width, $exponent_bits),
            &|bits: &Integer| floaty::$alias::from_bits(to_limbs::<$limbs>(bits)),
            (100, $width),
        )
    };
}

#[test]
fn wide_formats() {
    floaty_verify::for_each_wide_format!(wide_format);
}
