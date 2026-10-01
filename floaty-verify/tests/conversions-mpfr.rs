//! Compares conversions from and to the formats that TestFloat lacks with the
//! MPFR oracle: bfloat16, TF32, the FP8 and MX formats, and every wide format
//! from binary160 to binary512. The sources include binary32, binary64, x87
//! extended, and binary128, and the unsupported x87 encodings.
//!
//! MPFR rounds the finite values. The special values follow floaty's
//! conversion rules. A NaN result must also have the payload of the rule of
//! `convert_with`: the high-order payload bits of the source NaN, and a zero
//! payload in a format with one NaN encoding.
//!
//! The lists `for_each_small_format!` and `for_each_wide_format!` give the
//! sources and the destinations of most tests, so a new format joins them:
//! every pair of small formats, each small format and the standard formats,
//! each small format and each wide format, and every pair of wide formats.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::marker::PhantomData;

use floaty::env::NanPropagation;
use floaty::format::Standard;
use floaty::{
    BF16, Binary, Decoded, Env, F8E4M3Fn, F8E4M3Fnuz, F8E5M2, F16, F32, F64, F128, F160, F192,
    F224, F256, F288, F320, F352, F384, F416, F448, F480, F512, Float, TF32, X87,
};
use floaty_verify::encodings::{
    IntegerBit, Layout, boundary_encodings, boundary_encodings_u128, to_limbs, to_u128,
};
use floaty_verify::mpfr::decimal::align;
use floaty_verify::mpfr::{self, CONVERSION_BEHAVIORS, Format, Operand, Specials, Value};
use floaty_verify::random::SplitMix64;
use rug::Integer;
use rug::integer::Order;

/// Returns the payload of the NaN that converting `source` gives in a format
/// with a payload of `to_bits` bits. The payload of a NaN source moves from
/// `from_bits` bits to `to_bits` bits and keeps its high-order bits. The
/// payload is zero for the `DefaultNan` rule, for a destination without a
/// payload field (`NoInf` and `Fnuz`), and for an infinite source.
fn nan_payload(
    source: &Decoded<8>,
    from_bits: u32,
    to_bits: u32,
    specials: Specials,
    env: &Env,
) -> [u64; 8] {
    let Decoded::Nan { payload, .. } = source else {
        return [0; 8];
    };
    if specials != Specials::Ieee || env.nan.propagation == NanPropagation::DefaultNan {
        return [0; 8];
    }
    let payload = Integer::from_digits(payload, Order::Lsf);
    to_limbs::<8>(&align(payload, from_bits, to_bits))
}

/// Converts every source value to every destination in every behavior, by
/// [`check_conversion`].
macro_rules! check {
    ($source:ty, $values:expr => $($destination:ty: $specials:expr),+) => {{
        let values: Vec<$source> = $values.iter().map(|&bits| <$source>::from_bits(bits)).collect();
        $(
            check_conversion(
                &values,
                PhantomData::<$destination>,
                $specials,
                (stringify!($source), stringify!($destination)),
            );
        )+
    }};
}

/// Converts every source value to the format `D`, which has the special
/// values `specials`, in every behavior. A NaN result must also have the
/// payload of [`nan_payload`]. `names` names the source and the destination
/// in a failure.
fn check_conversion<S: Standard<SW>, const SW: usize, D: Standard<DW>, const DW: usize>(
    values: &[Float<S, SW>],
    _destination: PhantomData<Float<D, DW>>,
    specials: Specials,
    names: (&str, &str),
) {
    let format = Format::of::<Float<D, DW>>(specials);
    // A binary NaN has two bits less payload than precision.
    let from_bits = Float::<S, SW>::PRECISION - 2;
    let to_bits = Float::<D, DW>::PRECISION - 2;
    for env in CONVERSION_BEHAVIORS {
        for &source in values {
            let operand = Operand::<8>::of(source);
            let (ours, flags): (Float<D, DW>, _) = source.convert_with(env);
            let bits = source.to_bits();
            let context = format!("{} {bits:x?} to {} {env:?}", names.0, names.1);
            assert!(ours.is_canonical(), "{context}: {ours:?} is canonical");
            let result = ours.decode::<8>();
            let expected = mpfr::convert(&operand, &format, &env);
            assert_eq!((Value::from_decoded(result), flags), expected, "{context}");
            if let Decoded::Nan {
                signaling, payload, ..
            } = result
            {
                assert!(!signaling, "{context}: a NaN result is quiet");
                let expected = nan_payload(&operand.decoded, from_bits, to_bits, specials, &env);
                assert_eq!(payload, expected, "{context}: payload");
            }
        }
    }
}

/// Converts every source value, of the format named `name`, to every format
/// of `for_each_small_format!`.
fn to_every_small_format<S: Standard<SW>, const SW: usize>(values: &[Float<S, SW>], name: &str) {
    macro_rules! to_listed {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {
            check_conversion(
                values,
                PhantomData::<Float<$standard, $width>>,
                $specials,
                (name, stringify!($alias)),
            )
        };
    }
    floaty_verify::for_each_small_format!(to_listed);
}

/// Converts every source value, of the format named `name`, to every format
/// of `for_each_wide_format!`.
fn to_every_wide_format<S: Standard<SW>, const SW: usize>(values: &[Float<S, SW>], name: &str) {
    macro_rules! to_listed {
        ($alias:ident, $width:literal, $exponent_bits:literal, $limbs:literal) => {
            check_conversion(
                values,
                PhantomData::<Float<Binary<$exponent_bits>, $width>>,
                Specials::Ieee,
                (name, stringify!($alias)),
            )
        };
    }
    floaty_verify::for_each_wide_format!(to_listed);
}

/// Converts every source value, of the format named `name`, to the standard
/// formats: binary16, bfloat16, TF32, binary32, binary64, and binary128.
fn to_the_standard_formats<S: Standard<SW>, const SW: usize>(values: &[Float<S, SW>], name: &str) {
    let ieee = Specials::Ieee;
    check_conversion(values, PhantomData::<F16>, ieee, (name, "F16"));
    check_conversion(values, PhantomData::<BF16>, ieee, (name, "BF16"));
    check_conversion(values, PhantomData::<TF32>, ieee, (name, "TF32"));
    check_conversion(values, PhantomData::<F32>, ieee, (name, "F32"));
    check_conversion(values, PhantomData::<F64>, ieee, (name, "F64"));
    check_conversion(values, PhantomData::<F128>, ieee, (name, "F128"));
}

/// Returns the value of every encoding of a format of at most 8 bits.
fn every_value<S: Standard<W, Bits = u8>, const W: usize>() -> Vec<Float<S, W>> {
    (0..1_u16 << W)
        .map(|bits| Float::from_bits(u8::try_from(bits).expect("the format has at most 8 bits")))
        .collect()
}

/// Returns the values of `encodings` in the format `S`.
fn values_of<S: Standard<W>, const W: usize>(encodings: &[S::Bits]) -> Vec<Float<S, W>> {
    encodings
        .iter()
        .map(|&bits| Float::from_bits(bits))
        .collect()
}

/// Every encoding of each small format converts to every small format. The
/// infinities and NaNs of the sources meet destinations without them.
#[test]
fn every_small_format_pair() {
    macro_rules! from_listed {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {
            to_every_small_format(&every_value::<$standard, $width>(), stringify!($alias))
        };
    }
    floaty_verify::for_each_small_format!(from_listed);
}

#[test]
fn from_the_small_formats_to_the_standard_formats() {
    macro_rules! from_listed {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {
            to_the_standard_formats(&every_value::<$standard, $width>(), stringify!($alias))
        };
    }
    floaty_verify::for_each_small_format!(from_listed);
}

/// Every binary16 NaN drops payload bits in bfloat16 and in E5M2.
#[test]
fn from_binary16() {
    let halves: Vec<u16> = (0..=u16::MAX).collect();
    check!(F16, halves => BF16: Specials::Ieee);
    to_every_small_format(&values_of::<Binary<5>, 16>(&halves), "F16");
}

#[test]
fn from_bfloat16_and_tf32() {
    let every: Vec<u16> = (0..=u16::MAX).collect();
    check!(BF16, every => F16: Specials::Ieee, TF32: Specials::Ieee);
    to_every_small_format(&values_of::<Binary<8>, 16>(&every), "BF16");
    let mut random = SplitMix64::new(0x7F32);
    let mut samples: Vec<u32> = boundary_encodings(Layout::TF32)
        .iter()
        .map(|encoding| u32::try_from(to_u128(encoding)).expect("19 bits"))
        .collect();
    samples.extend((0..20_000).map(|_| u32::try_from(random.next_u64() >> 45).expect("19 bits")));
    check!(TF32, samples => BF16: Specials::Ieee, F16: Specials::Ieee);
    to_every_small_format(&values_of::<Binary<8>, 19>(&samples), "TF32");
}

/// Returns the boundary encodings and `count` random encodings of a format
/// with the layout `$layout` in `$limbs` limbs.
macro_rules! samples {
    ($layout:expr, $limbs:literal, $count:literal, $random:expr) => {{
        let layout: Layout = $layout;
        let mut samples: Vec<[u64; $limbs]> = boundary_encodings(layout)
            .iter()
            .map(to_limbs::<$limbs>)
            .collect();
        let mask: Integer = (Integer::from(1) << layout.width) - 1u32;
        let nan_field: Integer = Integer::from(layout.largest_field()) << layout.fraction_bits();
        samples.extend((0..$count).map(|index| {
            let digits: [u64; $limbs] = core::array::from_fn(|_| $random.next_u64());
            let mut value = Integer::from_digits(&digits, Order::Lsf) & &mask;
            // One encoding in 16 gets the exponent field of the NaNs, so that
            // NaNs with random payloads occur.
            if index % 16 == 0 {
                value |= &nan_field;
            }
            to_limbs::<$limbs>(&value)
        }));
        samples
    }};
}

/// Converts binary128 and every wide format to a wider and a narrower wide
/// format, and to a smaller format.
#[test]
fn between_the_wide_formats() {
    let mut random = SplitMix64::new(0x0160);
    let quads: Vec<u128> = boundary_encodings(Layout::BINARY128)
        .iter()
        .map(to_u128)
        .chain((0..1_000).map(|_| random.next_u128()))
        .collect();
    check!(F128, quads => F160: Specials::Ieee, F256: Specials::Ieee, F480: Specials::Ieee);
    let f160 = samples!(Layout::BINARY160, 3, 1_000, random);
    check!(F160, f160 => F128: Specials::Ieee, F192: Specials::Ieee, BF16: Specials::Ieee);
    let f192 = samples!(Layout::BINARY192, 3, 1_000, random);
    check!(F192, f192 => F160: Specials::Ieee, F224: Specials::Ieee, F16: Specials::Ieee);
    let f224 = samples!(Layout::BINARY224, 4, 1_000, random);
    check!(F224, f224 => F192: Specials::Ieee, F288: Specials::Ieee, F32: Specials::Ieee);
    let f256 = samples!(Layout::BINARY256, 4, 1_000, random);
    check!(F256, f256 => F128: Specials::Ieee, F288: Specials::Ieee);
    let f288 = samples!(Layout::BINARY288, 5, 1_000, random);
    check!(F288, f288 => F256: Specials::Ieee, F320: Specials::Ieee, F8E5M2: Specials::Ieee);
    let f320 = samples!(Layout::BINARY320, 5, 1_000, random);
    check!(F320, f320 => F288: Specials::Ieee, F352: Specials::Ieee, F128: Specials::Ieee);
    let f352 = samples!(Layout::BINARY352, 6, 1_000, random);
    check!(F352, f352 => F320: Specials::Ieee, F384: Specials::Ieee, F64: Specials::Ieee);
    let f384 = samples!(Layout::BINARY384, 6, 1_000, random);
    check!(F384, f384 => F352: Specials::Ieee, F416: Specials::Ieee, F8E4M3Fn: Specials::NoInf);
    let f416 = samples!(Layout::BINARY416, 7, 1_000, random);
    check!(F416, f416 => F384: Specials::Ieee, F448: Specials::Ieee, F64: Specials::Ieee);
    let f448 = samples!(Layout::BINARY448, 7, 1_000, random);
    check!(F448, f448 => F416: Specials::Ieee, F480: Specials::Ieee, F128: Specials::Ieee);
    let f480 = samples!(Layout::BINARY480, 8, 1_000, random);
    check!(F480, f480 => F448: Specials::Ieee, F512: Specials::Ieee, F160: Specials::Ieee);
    let f512 = samples!(Layout::BINARY512, 8, 1_000, random);
    check!(F512, f512 => F480: Specials::Ieee, F128: Specials::Ieee);
}

/// Every encoding of each small format converts to every wide format, and
/// the boundary and random encodings of each wide format convert to every
/// small format.
#[test]
fn between_the_small_and_the_wide_formats() {
    macro_rules! from_small {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {
            to_every_wide_format(&every_value::<$standard, $width>(), stringify!($alias))
        };
    }
    floaty_verify::for_each_small_format!(from_small);
    let mut random = SplitMix64::new(0x5A11);
    macro_rules! from_wide {
        ($alias:ident, $width:literal, $exponent_bits:literal, $limbs:literal) => {{
            let encodings = samples!(Layout::ieee($width, $exponent_bits), $limbs, 200, random);
            let values = values_of::<Binary<$exponent_bits>, $width>(&encodings);
            to_every_small_format(&values, stringify!($alias));
        }};
    }
    floaty_verify::for_each_wide_format!(from_wide);
}

/// The boundary and random encodings of each wide format convert to every
/// wide format.
#[test]
fn every_wide_format_pair() {
    let mut random = SplitMix64::new(0x0161);
    macro_rules! from_wide {
        ($alias:ident, $width:literal, $exponent_bits:literal, $limbs:literal) => {{
            let encodings = samples!(Layout::ieee($width, $exponent_bits), $limbs, 100, random);
            let values = values_of::<Binary<$exponent_bits>, $width>(&encodings);
            to_every_wide_format(&values, stringify!($alias));
        }};
    }
    floaty_verify::for_each_wide_format!(from_wide);
}

#[test]
fn from_binary256_and_binary512() {
    let mut random = SplitMix64::new(0x0512);
    let mut wide: Vec<[u64; 8]> = boundary_encodings(Layout::BINARY512)
        .iter()
        .map(to_limbs::<8>)
        .collect();
    wide.extend((0..3_000).map(|_| core::array::from_fn(|_| random.next_u64())));
    check!(F512, wide => F256: Specials::Ieee, BF16: Specials::Ieee, F8E4M3Fnuz: Specials::Fnuz, F64: Specials::Ieee);
    let mut narrow: Vec<[u64; 4]> = boundary_encodings(Layout::BINARY256)
        .iter()
        .map(to_limbs::<4>)
        .collect();
    narrow.extend((0..3_000).map(|_| core::array::from_fn(|_| random.next_u64())));
    check!(F256, narrow => F512: Specials::Ieee, F16: Specials::Ieee, F8E4M3Fn: Specials::NoInf);
}

/// Returns the boundary encodings and `count` random encodings of a format
/// of at most 128 bits. One random encoding in 16 gets the exponent field of
/// the NaNs. Three x87 encodings in four get the integer bit of a canonical
/// encoding, and the fourth keeps a random integer bit, which gives unnormals
/// and pseudo-denormals.
fn encodings_up_to_128(layout: Layout, count: usize, random: &mut SplitMix64) -> Vec<u128> {
    let mask = u128::MAX >> (128 - layout.width);
    let fraction_bits = layout.fraction_bits();
    let nan_field = u128::from(layout.largest_field()) << fraction_bits;
    let mut encodings = boundary_encodings_u128(layout);
    encodings.extend((0..count).map(|index| {
        let mut bits = random.next_u128() & mask;
        if index % 16 == 0 {
            bits |= nan_field;
        }
        if layout.integer_bit == IntegerBit::Explicit && index % 4 != 0 {
            let integer_bit = 1 << (fraction_bits - 1);
            let field_is_zero = bits & (u128::from(layout.largest_field()) << fraction_bits) == 0;
            bits = if field_is_zero {
                bits & !integer_bit
            } else {
                bits | integer_bit
            };
        }
        bits
    }));
    encodings
}

/// Converts every source value, of the format named `name`, to bfloat16,
/// TF32, and every format of `for_each_small_format!`.
fn to_the_formats_that_testfloat_lacks<S: Standard<SW>, const SW: usize>(
    values: &[Float<S, SW>],
    name: &str,
) {
    check_conversion(values, PhantomData::<BF16>, Specials::Ieee, (name, "BF16"));
    check_conversion(values, PhantomData::<TF32>, Specials::Ieee, (name, "TF32"));
    to_every_small_format(values, name);
}

/// binary32, binary64, x87 extended, and binary128 convert to bfloat16,
/// TF32, and every small format. TestFloat has none of these destinations.
#[test]
fn from_binary32_binary64_x87_extended_and_binary128() {
    let mut random = SplitMix64::new(0x5741);
    let narrow = |bits: u128| u64::try_from(bits).expect("the format has at most 64 bits");
    let singles: Vec<u32> = encodings_up_to_128(Layout::BINARY32, 4_000, &mut random)
        .into_iter()
        .map(|bits| u32::try_from(bits).expect("the format has 32 bits"))
        .collect();
    to_the_formats_that_testfloat_lacks(&values_of::<Binary<8>, 32>(&singles), "F32");
    let doubles: Vec<u64> = encodings_up_to_128(Layout::BINARY64, 4_000, &mut random)
        .into_iter()
        .map(narrow)
        .collect();
    to_the_formats_that_testfloat_lacks(&values_of::<Binary<11>, 64>(&doubles), "F64");
    let extended = encodings_up_to_128(Layout::X87_EXTENDED, 4_000, &mut random);
    to_the_formats_that_testfloat_lacks(&values_of::<Binary<15, X87>, 80>(&extended), "F80");
    let quads = encodings_up_to_128(Layout::BINARY128, 4_000, &mut random);
    to_the_formats_that_testfloat_lacks(&values_of::<Binary<15>, 128>(&quads), "F128");
}
