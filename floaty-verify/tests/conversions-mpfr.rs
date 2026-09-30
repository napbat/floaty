//! Compares conversions from the formats that TestFloat lacks, bfloat16,
//! TF32, the FP8 formats, and binary256 and binary512, with the MPFR oracle.
//!
//! MPFR rounds the finite values. The special values follow floaty's
//! conversion rules. TestFloat checks NaN payloads for the IEEE formats,
//! so this test checks only that a NaN is a NaN with the right sign.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::num::NonZeroU32;

use floaty::env::Tininess;
use floaty::{
    BF16, Class, Env, F4E2M1Fn, F6E2M3Fn, F6E3M2Fn, F8E4M3Fn, F8E4M3Fnuz, F8E5M2, F8E5M2Fnuz, F16,
    F64, F256, F512, Rounding, TF32,
};
use floaty_verify::encodings::{IntegerBit, boundary_encodings, to_limbs, to_u128};
use floaty_verify::mpfr::{self, Format, Specials, Value};
use floaty_verify::random::SplitMix64;

/// The behaviors of each conversion.
fn behaviors() -> [Env; 9] {
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
                    let ours = (Value::from_decoded(ours.decode::<8>()), flags);
                    let expected = mpfr::convert(&decoded, subnormal, &format, &env);
                    let context = format!(
                        "{} {bits:x?} to {} {env:?}",
                        stringify!($source),
                        stringify!($destination)
                    );
                    assert_eq!(ours, expected, "{context}");
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
