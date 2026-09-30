//! Compares conversions from and to the formats that TestFloat lacks with the
//! MPFR oracle: bfloat16, TF32, the FP8 and MX formats, and every wide format
//! from binary160 to binary512.
//!
//! MPFR rounds the finite values. The special values follow floaty's
//! conversion rules. A NaN result must also have the payload of the rule of
//! `convert_with`: the high-order payload bits of the source NaN, and a zero
//! payload in a format with one NaN encoding.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::num::NonZeroU32;

use floaty::env::{NanPropagation, NanRule, Tininess};
use floaty::{
    BF16, Class, Decoded, Env, F4E2M1Fn, F6E2M3Fn, F6E3M2Fn, F8E4M3Fn, F8E4M3Fnuz, F8E5M2,
    F8E5M2Fnuz, F16, F32, F64, F128, F160, F192, F224, F256, F288, F320, F352, F384, F416, F448,
    F480, F512, Rounding, TF32,
};
use floaty_verify::encodings::{IntegerBit, boundary_encodings, to_limbs, to_u128};
use floaty_verify::mpfr::{self, Format, Specials, Value};
use floaty_verify::random::SplitMix64;
use rug::Integer;
use rug::integer::Order;

/// The behaviors of each conversion.
fn behaviors() -> [Env; 10] {
    [
        Env::IEEE,
        Env::IEEE
            .with_rounding(Rounding::TowardZero)
            .with_tininess(Tininess::BeforeRounding),
        Env::IEEE
            .with_rounding(Rounding::TowardPositive)
            .with_flush_to_zero(true),
        Env::IEEE.with_rounding(Rounding::ToOdd).with_saturate(true),
        Env::IEEE
            .with_rounding(Rounding::TiesToAway)
            .with_denormals_are_zero(true),
        Env::IEEE
            .with_rounding(Rounding::TowardNegative)
            .with_precision(NonZeroU32::new(3)),
        // A saturated infinity and a saturated overflow both use the limit.
        Env::IEEE
            .with_saturate(true)
            .with_precision(NonZeroU32::new(2)),
        Env::IEEE
            .with_rounding(Rounding::TiesTowardZero)
            .with_flush_to_zero(true),
        Env::IEEE.with_rounding(Rounding::AwayFromZero),
        Env::IEEE.with_nan(NanRule::new(NanPropagation::DefaultNan).with_default_negative(true)),
    ]
}

/// The oracle parameters of a destination.
macro_rules! oracle_format {
    ($alias:ty, $specials:expr) => {
        Format {
            precision: <$alias>::PRECISION,
            emin: <$alias>::EMIN,
            emax: <$alias>::EMAX,
            specials: $specials,
        }
    };
}

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
    let aligned = if to_bits >= from_bits {
        payload << (to_bits - from_bits)
    } else {
        payload >> (from_bits - to_bits)
    };
    to_limbs::<8>(&aligned)
}

/// Converts every source value to every destination in every behavior.
macro_rules! check {
    ($source:ty, $values:expr => $($destination:ty: $specials:expr),+) => {{
        for env in behaviors() {
            for &bits in &$values {
                let source = <$source>::from_bits(bits);
                let decoded = source.decode::<8>();
                let subnormal = source.classify() == Class::Subnormal;
                $(
                    let format = oracle_format!($destination, $specials);
                    let (ours, flags): ($destination, _) = source.convert_with(env);
                    assert!(ours.is_canonical(), "{ours:?} is canonical");
                    let result = ours.decode::<8>();
                    let ours = (Value::from_decoded(result), flags);
                    let expected = mpfr::convert(&decoded, subnormal, &format, &env);
                    let context = format!(
                        "{} {bits:x?} to {} {env:?}",
                        stringify!($source),
                        stringify!($destination)
                    );
                    assert_eq!(ours, expected, "{context}");
                    if let Decoded::Nan { signaling, payload, .. } = result {
                        assert!(!signaling, "{context}: a NaN result is quiet");
                        // A binary NaN has two bits less payload than precision.
                        let from_bits = <$source>::PRECISION - 2;
                        let to_bits = <$destination>::PRECISION - 2;
                        let expected = nan_payload(&decoded, from_bits, to_bits, $specials, &env);
                        assert_eq!(payload, expected, "{context}: payload");
                    }
                )+
            }
        }
    }};
}

#[test]
fn from_the_fp8_formats() {
    let every: Vec<u8> = (0..=u8::MAX).collect();
    check!(F8E4M3Fn, every => F8E5M2Fnuz: Specials::Fnuz, F8E5M2: Specials::Ieee, BF16: Specials::Ieee, F16: Specials::Ieee);
    check!(F8E5M2, every => F8E4M3Fn: Specials::NoInf, F8E4M3Fnuz: Specials::Fnuz, TF32: Specials::Ieee);
    check!(F8E4M3Fnuz, every => F8E4M3Fn: Specials::NoInf, F8E5M2: Specials::Ieee, F64: Specials::Ieee);
    check!(F8E5M2Fnuz, every => F8E4M3Fnuz: Specials::Fnuz, F8E4M3Fn: Specials::NoInf, BF16: Specials::Ieee);
}

#[test]
fn from_and_to_the_mx_formats() {
    let four: Vec<u8> = (0..16).collect();
    let six: Vec<u8> = (0..64).collect();
    check!(F4E2M1Fn, four => F6E2M3Fn: Specials::Finite, F8E4M3Fn: Specials::NoInf, F16: Specials::Ieee);
    check!(F6E2M3Fn, six => F4E2M1Fn: Specials::Finite, F6E3M2Fn: Specials::Finite, BF16: Specials::Ieee);
    check!(F6E3M2Fn, six => F6E2M3Fn: Specials::Finite, F8E5M2: Specials::Ieee, F4E2M1Fn: Specials::Finite);
    // The infinities and NaNs of the sources meet destinations without them.
    let every: Vec<u8> = (0..=u8::MAX).collect();
    check!(F8E5M2, every => F4E2M1Fn: Specials::Finite, F6E3M2Fn: Specials::Finite);
    check!(F8E4M3Fn, every => F6E2M3Fn: Specials::Finite);
    let halves: Vec<u16> = (0..=u16::MAX).collect();
    check!(F16, halves => F6E3M2Fn: Specials::Finite, F4E2M1Fn: Specials::Finite);
}

/// Every binary16 NaN drops payload bits in bfloat16 and in E5M2.
#[test]
fn from_binary16() {
    let halves: Vec<u16> = (0..=u16::MAX).collect();
    check!(F16, halves => BF16: Specials::Ieee, F8E5M2: Specials::Ieee, F8E4M3Fnuz: Specials::Fnuz);
}

#[test]
fn from_bfloat16_and_tf32() {
    let every: Vec<u16> = (0..=u16::MAX).collect();
    check!(BF16, every => F8E4M3Fn: Specials::NoInf, F8E5M2Fnuz: Specials::Fnuz, F16: Specials::Ieee, TF32: Specials::Ieee);
    let mut random = SplitMix64::new(0x7F32);
    let mut samples: Vec<u32> = boundary_encodings(19, 8, IntegerBit::Implicit)
        .iter()
        .map(|encoding| u32::try_from(to_u128(encoding)).expect("19 bits"))
        .collect();
    samples.extend((0..20_000).map(|_| u32::try_from(random.next_u64() >> 45).expect("19 bits")));
    check!(TF32, samples => BF16: Specials::Ieee, F8E5M2: Specials::Ieee, F8E4M3Fn: Specials::NoInf, F16: Specials::Ieee);
}

/// Returns the boundary encodings and `count` random encodings of a format of
/// `$width` bits in `$limbs` limbs.
macro_rules! samples {
    ($width:literal, $exponent_bits:literal, $limbs:literal, $count:literal, $random:expr) => {{
        let mut samples: Vec<[u64; $limbs]> =
            boundary_encodings($width, $exponent_bits, IntegerBit::Implicit)
                .iter()
                .map(to_limbs::<$limbs>)
                .collect();
        let mask: Integer = (Integer::from(1) << $width) - 1u32;
        let nan_field: Integer =
            ((Integer::from(1) << $exponent_bits) - 1u32) << ($width - 1 - $exponent_bits);
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
    let quads: Vec<u128> = boundary_encodings(128, 15, IntegerBit::Implicit)
        .iter()
        .map(to_u128)
        .chain((0..1_000).map(|_| random.next_u128()))
        .collect();
    check!(F128, quads => F160: Specials::Ieee, F256: Specials::Ieee, F480: Specials::Ieee);
    let f160 = samples!(160, 16, 3, 1_000, random);
    check!(F160, f160 => F128: Specials::Ieee, F192: Specials::Ieee, BF16: Specials::Ieee);
    let f192 = samples!(192, 17, 3, 1_000, random);
    check!(F192, f192 => F160: Specials::Ieee, F224: Specials::Ieee, F16: Specials::Ieee);
    let f224 = samples!(224, 18, 4, 1_000, random);
    check!(F224, f224 => F192: Specials::Ieee, F288: Specials::Ieee, F32: Specials::Ieee);
    let f256 = samples!(256, 19, 4, 1_000, random);
    check!(F256, f256 => F128: Specials::Ieee, F288: Specials::Ieee);
    let f288 = samples!(288, 20, 5, 1_000, random);
    check!(F288, f288 => F256: Specials::Ieee, F320: Specials::Ieee, F8E5M2: Specials::Ieee);
    let f320 = samples!(320, 20, 5, 1_000, random);
    check!(F320, f320 => F288: Specials::Ieee, F352: Specials::Ieee, F128: Specials::Ieee);
    let f352 = samples!(352, 21, 6, 1_000, random);
    check!(F352, f352 => F320: Specials::Ieee, F384: Specials::Ieee, F64: Specials::Ieee);
    let f384 = samples!(384, 21, 6, 1_000, random);
    check!(F384, f384 => F352: Specials::Ieee, F416: Specials::Ieee, F8E4M3Fn: Specials::NoInf);
    let f416 = samples!(416, 22, 7, 1_000, random);
    check!(F416, f416 => F384: Specials::Ieee, F448: Specials::Ieee, F64: Specials::Ieee);
    let f448 = samples!(448, 22, 7, 1_000, random);
    check!(F448, f448 => F416: Specials::Ieee, F480: Specials::Ieee, F128: Specials::Ieee);
    let f480 = samples!(480, 23, 8, 1_000, random);
    check!(F480, f480 => F448: Specials::Ieee, F512: Specials::Ieee, F160: Specials::Ieee);
    let f512 = samples!(512, 23, 8, 1_000, random);
    check!(F512, f512 => F480: Specials::Ieee, F128: Specials::Ieee);
}

#[test]
fn from_binary256_and_binary512() {
    let mut random = SplitMix64::new(0x0512);
    let mut wide: Vec<[u64; 8]> = boundary_encodings(512, 23, IntegerBit::Implicit)
        .iter()
        .map(to_limbs::<8>)
        .collect();
    wide.extend((0..3_000).map(|_| core::array::from_fn(|_| random.next_u64())));
    check!(F512, wide => F256: Specials::Ieee, BF16: Specials::Ieee, F8E4M3Fnuz: Specials::Fnuz, F64: Specials::Ieee);
    let mut narrow: Vec<[u64; 4]> = boundary_encodings(256, 19, IntegerBit::Implicit)
        .iter()
        .map(to_limbs::<4>)
        .collect();
    narrow.extend((0..3_000).map(|_| core::array::from_fn(|_| random.next_u64())));
    check!(F256, narrow => F512: Specials::Ieee, F16: Specials::Ieee, F8E4M3Fn: Specials::NoInf);
}
