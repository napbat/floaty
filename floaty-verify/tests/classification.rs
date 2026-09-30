//! Compares classification and decoded values with `rustc_apfloat`, the Rust
//! port of LLVM APFloat, and with the host `f32` and `f64` types.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::{BF16, Binary, Class, F8E4M3Fn, F8E5M2, F16, F32, F64, F80, F128, Float, NoInf, TF32};
use floaty_verify::apfloat::{Tf32, check_value, class};
use floaty_verify::encodings::{Layout, boundary_encodings, to_u128};
use floaty_verify::random::SplitMix64;
use floaty_verify::shape::{self, Payload, trailing_payload};
use rug::Integer;
use rustc_apfloat::Float as _;
use rustc_apfloat::ieee::{
    BFloat, Double, Float8E4M3FN, Float8E5M2, Half, IeeeFloat, NonfiniteBehavior, Quad, Semantics,
    Single, X87DoubleExtended,
};

const RANDOM_SAMPLES: usize = 100_000;

/// A 72-bit layout whose exponent field, bits 56 to 70, crosses a limb boundary.
struct Wide72Semantics;

impl Semantics for Wide72Semantics {
    const BITS: usize = 72;
    const EXP_BITS: usize = 15;
}

/// A 69-bit layout whose exponent field, bits 60 to 67, crosses a limb boundary.
struct Wide69Semantics;

impl Semantics for Wide69Semantics {
    const BITS: usize = 69;
    const EXP_BITS: usize = 8;
}

/// The 69-bit layout with the `NoInf` encoding.
struct Wide69NoInfSemantics;

impl Semantics for Wide69NoInfSemantics {
    const BITS: usize = 69;
    const EXP_BITS: usize = 8;
    const NONFINITE_BEHAVIOR: NonfiniteBehavior = NonfiniteBehavior::NanOnly;
}

/// An 80-bit layout with an implicit integer bit, whose exponent field, bits
/// 59 to 78, crosses a limb boundary.
struct Wide80Semantics;

impl Semantics for Wide80Semantics {
    const BITS: usize = 80;
    const EXP_BITS: usize = 20;
}

type Wide72 = Float<Binary<15>, 72>;
type Wide69 = Float<Binary<8>, 69>;
type Wide69NoInf = Float<Binary<8, NoInf>, 69>;
type Wide80 = Float<Binary<20>, 80>;

/// Compares the class, the sign, and the value of one encoding with
/// `rustc_apfloat`, and checks the form of the decoded value. The last
/// argument says whether the NaNs of the format carry a payload.
macro_rules! check {
    ($ours:ty, $theirs:ty, $bits:expr) => {
        check!($ours, $theirs, $bits, true)
    };
    ($ours:ty, $theirs:ty, $bits:expr, $has_payload:expr) => {{
        let bits: u128 = $bits;
        let ours = <$ours>::from_bits(bits.try_into().expect("the encoding fits the storage"));
        let theirs = <$theirs>::from_bits(bits);
        let context = format!("{} {bits:#x}", stringify!($ours));
        assert_eq!(ours.classify(), class(theirs), "{context}");
        assert_eq!(ours.is_sign_negative(), theirs.is_negative(), "{context}");
        assert!(ours.is_canonical(), "{context}");
        let decoded = ours.decode::<2>();
        check_value(decoded, theirs, &context);
        let payload = if $has_payload {
            trailing_payload(&Integer::from(bits), <$ours>::PRECISION)
        } else {
            Payload::None
        };
        shape::check(
            &decoded,
            ours.classify(),
            <$ours>::PRECISION,
            <$ours>::EMIN,
            &payload,
            &context,
        );
    }};
}

#[test]
fn every_encoding_of_the_small_formats() {
    for bits in 0..1_u128 << 8 {
        check!(F8E4M3Fn, Float8E4M3FN, bits, false);
        check!(F8E5M2, Float8E5M2, bits);
    }
    for bits in 0..1_u128 << 16 {
        check!(F16, Half, bits);
        check!(BF16, BFloat, bits);
    }
    for bits in 0..1_u128 << 19 {
        check!(TF32, Tf32, bits);
    }
}

/// Returns the boundary encodings and random encodings of a format.
fn samples(layout: Layout, seed: u64) -> impl Iterator<Item = u128> {
    let mask = if layout.width == 128 {
        u128::MAX
    } else {
        (1 << layout.width) - 1
    };
    let mut random = SplitMix64::new(seed);
    let boundaries: Vec<u128> = boundary_encodings(layout).iter().map(to_u128).collect();
    boundaries
        .into_iter()
        .chain((0..RANDOM_SAMPLES).map(move |_| random.next_u128() & mask))
}

#[test]
fn boundary_and_random_encodings_of_the_wide_formats() {
    for bits in samples(Layout::BINARY32, 1) {
        check!(F32, Single, bits);
    }
    for bits in samples(Layout::BINARY64, 2) {
        check!(F64, Double, bits);
    }
    for bits in samples(Layout::BINARY128, 3) {
        check!(F128, Quad, bits);
    }
}

#[test]
fn layouts_whose_exponent_field_crosses_a_limb_boundary() {
    for bits in samples(Layout::ieee(72, 15), 4) {
        check!(Wide72, IeeeFloat<Wide72Semantics>, bits);
    }
    for bits in samples(Layout::ieee(69, 8), 5) {
        check!(Wide69, IeeeFloat<Wide69Semantics>, bits);
        check!(Wide69NoInf, IeeeFloat<Wide69NoInfSemantics>, bits, false);
    }
    for bits in samples(Layout::ieee(80, 20), 6) {
        check!(Wide80, IeeeFloat<Wide80Semantics>, bits);
    }
}

/// Returns `true` for a canonical x87 encoding, by the rule of the Intel SDM
/// Volume 1 (253665-093US), Table 8-3: the integer bit is set exactly when the
/// exponent field is not zero.
fn x87_canonical(bits: u128) -> bool {
    let field = (bits >> 64) & 0x7FFF;
    let integer = (bits >> 63) & 1 == 1;
    integer == (field != 0)
}

/// `rustc_apfloat` follows LLVM, not the processor, for the non-canonical x87
/// encodings. `rustc_apfloat` reads an unnormal, a pseudo-NaN, and a
/// pseudo-infinity as NaNs, and a pseudo-denormal as a normal value. The
/// Intel SDM, Volume 1, revision 253665-093US, section 8.2.2 and Table 8-3 on
/// page 8-14, makes the first three invalid operands. The processor reads a
/// pseudo-denormal with the biased exponent 1. So the value comparison covers the canonical encodings
/// only. `x87-hardware.rs` checks the class of every encoding against the
/// processor.
#[test]
fn x87_encodings() {
    let mask = (1_u128 << 80) - 1;
    let mut random = SplitMix64::new(0x0087_0087);
    let boundaries: Vec<u128> = boundary_encodings(Layout::X87_EXTENDED)
        .iter()
        .map(to_u128)
        .collect();
    let mut compared = 0_usize;
    let samples = (0..RANDOM_SAMPLES).map(|_| random.next_u128() & mask);
    for bits in boundaries.into_iter().chain(samples) {
        let ours = F80::from_bits(bits);
        assert_eq!(
            ours.is_canonical(),
            x87_canonical(bits),
            "F80 {bits:#x} canonical"
        );
        if !x87_canonical(bits) {
            continue;
        }
        let theirs = X87DoubleExtended::from_bits(bits);
        let context = format!("F80 {bits:#x}");
        assert_eq!(ours.classify(), class(theirs), "{context}");
        let decoded = ours.decode::<2>();
        check_value(decoded, theirs, &context);
        let payload = trailing_payload(&Integer::from(bits), F80::PRECISION);
        shape::check(
            &decoded,
            ours.classify(),
            F80::PRECISION,
            F80::EMIN,
            &payload,
            &context,
        );
        compared += 1;
    }
    assert!(
        compared > RANDOM_SAMPLES / 4,
        "the samples include canonical encodings"
    );
}

fn host_class_f32(bits: u32) -> Class {
    match f32::from_bits(bits).classify() {
        core::num::FpCategory::Zero => Class::Zero,
        core::num::FpCategory::Subnormal => Class::Subnormal,
        core::num::FpCategory::Normal => Class::Normal,
        core::num::FpCategory::Infinite => Class::Infinite,
        core::num::FpCategory::Nan if bits & 0x0040_0000 == 0 => Class::SignalingNan,
        core::num::FpCategory::Nan => Class::QuietNan,
    }
}

fn host_class_f64(bits: u64) -> Class {
    match f64::from_bits(bits).classify() {
        core::num::FpCategory::Zero => Class::Zero,
        core::num::FpCategory::Subnormal => Class::Subnormal,
        core::num::FpCategory::Normal => Class::Normal,
        core::num::FpCategory::Infinite => Class::Infinite,
        core::num::FpCategory::Nan if bits & 0x0008_0000_0000_0000 == 0 => Class::SignalingNan,
        core::num::FpCategory::Nan => Class::QuietNan,
    }
}

#[test]
fn host_binary32_and_binary64_samples() {
    let mut random = SplitMix64::new(0x0005_EED5);
    for _ in 0..RANDOM_SAMPLES {
        let bits = random.next_u64();
        let high = u32::try_from(bits >> 32).expect("32 bits fit a u32");
        assert_eq!(
            F32::from_bits(high).classify(),
            host_class_f32(high),
            "{high:#x}"
        );
        assert_eq!(
            F64::from_bits(bits).classify(),
            host_class_f64(bits),
            "{bits:#x}"
        );
    }
}

/// Every binary32 encoding against the host. Run time: about 20 seconds in a
/// release build, and several minutes in a debug build.
#[test]
#[ignore = "exhaustive sweep; run with --ignored"]
fn every_binary32_encoding_against_the_host() {
    for bits in 0..=u32::MAX {
        let ours = F32::from_bits(bits);
        assert_eq!(ours.classify(), host_class_f32(bits), "{bits:#x}");
        assert_eq!(
            ours.is_sign_negative(),
            f32::from_bits(bits).is_sign_negative(),
            "{bits:#x}"
        );
    }
}
