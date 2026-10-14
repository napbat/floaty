//! Compares `y log2(x)` and `y log2(1 + x)` of the binary formats,
//! `mul_log2_with` and `mul_log2_p1_with`, with the MPFR oracle in
//! `floaty_verify::operations`, in every behavior of `operations::BEHAVIORS`.
//!
//! Every pair of encodings of the FP8 and MX formats runs. The other formats
//! run each boundary encoding against zeros, ones, twos, halves, threes,
//! infinities, and a NaN in both orders, random pairs of boundary encodings
//! and of any encodings, and pairs that reach each case of the products:
//! powers of 2 and their neighbors, `2^k - 1`, arguments next to 1 and to
//! -1, and factors that overflow or underflow the product.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::format::{Encoding, Standard, Storage, Width};
use floaty::{Binary, F32, F64, F80, F128, Float, TF32};
use floaty_verify::encodings::{Layout, boundary_encodings, to_limbs};
use floaty_verify::mpfr::{Format, Specials};
use floaty_verify::operations::BEHAVIORS;
use floaty_verify::operations::check::{self, LogProduct};
use floaty_verify::random::SplitMix64;
use rug::Integer;
use rug::integer::Order;

/// Checks both products on every pair of encodings of a format of the
/// small format lists.
macro_rules! every_small_pair {
    ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {{
        let format = Format::of::<Float<$standard, $width>>($specials);
        let value = |bits: u16| {
            Float::<$standard, $width>::from_bits(
                u8::try_from(bits).expect("the format has at most 8 bits"),
            )
        };
        let products: [LogProduct<$standard, $width>; 2] = [
            |y, x, env| y.mul_log2_with(x, env),
            |y, x, env| y.mul_log2_p1_with(x, env),
        ];
        for y in 0..1_u16 << $width {
            for x in 0..1_u16 << $width {
                for env in &BEHAVIORS {
                    check::check_log_products(value(y), value(x), &format, env, products);
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
    let integer = Float::<S, W>::from_int;
    let (zero, one, three) = (integer(0), integer(1), integer(3));
    let infinity = one.scale_b(i32::MAX);
    let partners = [
        zero,
        -zero,
        one,
        -one,
        integer(2),
        one.scale_b(-1),
        -one.scale_b(-1),
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
    let any = |random: &mut SplitMix64| make(&random_encoding(width, random));
    pairs.extend((0..count).map(|_| (any(&mut random), any(&mut random))));
    let precision = i32::try_from(layout.precision()).expect("a precision fits an i32");
    let factors = [
        one,
        three,
        -three.scale_b(-7),
        any(&mut random),
        any(&mut random),
    ];
    let largest = boundary
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(zero, |largest, value| {
            if value.abs() > largest {
                value.abs()
            } else {
                largest
            }
        });
    let smallest = one.scale_b(Float::<S, W>::EMIN - precision + 1);
    // Powers of 2 and their neighbors; `2^k - 1`, where `1 + x` is a power
    // of 2; arguments next to 1 and to -1; and `2^k` past the shortcut of
    // `log2(1 + x)` at `k = p + 5`.
    let mut arguments = Vec::new();
    for k in [
        -3,
        -1,
        1,
        2,
        10,
        precision,
        precision + 4,
        precision + 5,
        precision + 6,
        precision + 12,
    ] {
        let power = one.scale_b(k);
        arguments.extend([power, power.next_up(), power.next_down(), power - one]);
    }
    arguments.extend([
        one.next_up(),
        one.next_down(),
        -one.next_up(),
        -one.next_down(),
    ]);
    arguments.extend([
        one.scale_b(-precision - 3),
        -one.scale_b(-precision - 3),
        largest,
    ]);
    for &x in &arguments {
        for &y in &factors {
            pairs.extend([(y, x), (-y, x)]);
        }
        // Products past the range: the largest and the smallest factor.
        pairs.extend([(largest, x), (-largest, x), (smallest, x), (-smallest, x)]);
    }
    pairs
}

/// Checks both products on the pairs of [`pairs`] in every behavior.
fn check_format<const E: u32, Enc: Encoding, const W: usize>(
    layout: Layout,
    make: &dyn Fn(&Integer) -> Float<Binary<E, Enc>, W>,
    samples: (usize, u64),
) where
    Width<W>: Storage,
    Binary<E, Enc>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
{
    let format = Format::of::<Float<Binary<E, Enc>, W>>(Specials::Ieee);
    let products: [LogProduct<Binary<E, Enc>, W>; 2] = [
        |y, x, env| y.mul_log2_with(x, env),
        |y, x, env| y.mul_log2_p1_with(x, env),
    ];
    for env in &BEHAVIORS {
        for &(y, x) in &pairs(layout, make, samples) {
            check::check_log_products(y, x, &format, env, products);
        }
    }
}

#[test]
fn binary16_bfloat16_tf32_binary32_and_binary64() {
    let bits16 = |bits: &Integer| bits.to_u16().expect("a 16-bit encoding fits a u16");
    check_format(
        Layout::BINARY16,
        &|bits: &Integer| Float::<Binary<5>, 16>::from_bits(bits16(bits)),
        (2_000, 0x1216),
    );
    check_format(
        Layout::BFLOAT16,
        &|bits: &Integer| Float::<Binary<8>, 16>::from_bits(bits16(bits)),
        (2_000, 0x12B6),
    );
    check_format(
        Layout::TF32,
        &|bits: &Integer| TF32::from_bits(bits.to_u32().expect("a TF32 encoding has 19 bits")),
        (1_000, 0x1219),
    );
    check_format(
        Layout::BINARY32,
        &|bits: &Integer| F32::from_bits(bits.to_u32().expect("a binary32 encoding fits a u32")),
        (1_000, 0x1232),
    );
    check_format(
        Layout::BINARY64,
        &|bits: &Integer| F64::from_bits(bits.to_u64().expect("a binary64 encoding fits a u64")),
        (1_000, 0x1264),
    );
}

#[test]
fn x87_and_binary128() {
    check_format(
        Layout::X87_EXTENDED,
        &|bits: &Integer| F80::from_bits(bits.to_u128().expect("an x87 encoding fits a u128")),
        (500, 0x1280),
    );
    check_format(
        Layout::BINARY128,
        &|bits: &Integer| {
            F128::from_bits(bits.to_u128().expect("a binary128 encoding fits a u128"))
        },
        (500, 0x12128),
    );
}

/// A layout whose precision fills the storage up to two bits, and a layout
/// of the most exponent bits, 28, whose logarithms reach `2^28`.
#[test]
fn layouts_that_fill_storage_or_reach_far() {
    check_format(
        Layout::ieee(64, 2),
        &|bits: &Integer| {
            Float::<Binary<2>, 64>::from_bits(bits.to_u64().expect("the encoding fits a u64"))
        },
        (300, 0x12264),
    );
    check_format(
        Layout::ieee(32, 28),
        &|bits: &Integer| {
            Float::<Binary<28>, 32>::from_bits(bits.to_u32().expect("the encoding fits a u32"))
        },
        (300, 0x12832),
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
