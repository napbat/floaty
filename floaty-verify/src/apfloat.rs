//! Adapters for `rustc_apfloat`, the Rust port of LLVM APFloat.

use floaty::{Class, Decoded};
use rustc_apfloat::ieee::{IeeeFloat, Semantics};
use rustc_apfloat::{Category, Float, Round};

/// TF32 semantics: 19 bits, 8 of them exponent bits.
pub struct Tf32Semantics;

impl Semantics for Tf32Semantics {
    const BITS: usize = 19;
    const EXP_BITS: usize = 8;
}

/// TF32 in `rustc_apfloat`.
pub type Tf32 = IeeeFloat<Tf32Semantics>;

/// Returns the class of a `rustc_apfloat` value.
pub fn class<T: Float>(value: T) -> Class {
    match value.category() {
        Category::Zero => Class::Zero,
        Category::Normal if value.is_denormal() => Class::Subnormal,
        Category::Normal => Class::Normal,
        Category::Infinity => Class::Infinite,
        Category::NaN if value.is_signaling() => Class::SignalingNan,
        Category::NaN => Class::QuietNan,
    }
}

/// Checks that a decoded value equals a `rustc_apfloat` value.
///
/// A finite value must scale exactly to its significand. The check needs a
/// format whose range holds `2^PRECISION`. A NaN must agree on its sign and
/// on whether it signals.
///
/// # Panics
///
/// Panics with `context` when the values differ.
pub fn check_value<T: Float>(ours: Decoded<2>, theirs: T, context: &str) {
    match ours {
        Decoded::Zero { negative, .. } => {
            assert!(theirs.is_zero(), "{context}: zero");
            assert_eq!(negative, theirs.is_negative(), "{context}: sign");
        }
        Decoded::Finite {
            negative,
            exponent,
            significand,
        } => {
            assert!(theirs.is_finite_non_zero(), "{context}: finite");
            assert_eq!(negative, theirs.is_negative(), "{context}: sign");
            let mut exact = false;
            let scaled = theirs.abs().scalbn(-exponent);
            let integer = scaled.to_u128_r(128, Round::TowardZero, &mut exact).value;
            let expected = u128::from(significand[0]) | (u128::from(significand[1]) << 64);
            assert!(
                exact,
                "{context}: the value is an integer at the decoded exponent"
            );
            assert_eq!(integer, expected, "{context}: significand");
        }
        Decoded::Infinity { negative } => {
            assert!(theirs.is_infinite(), "{context}: infinity");
            assert_eq!(negative, theirs.is_negative(), "{context}: sign");
        }
        Decoded::Nan {
            negative,
            signaling,
            ..
        } => {
            assert!(theirs.is_nan(), "{context}: NaN");
            assert_eq!(negative, theirs.is_negative(), "{context}: sign");
            assert_eq!(signaling, theirs.is_signaling(), "{context}: signaling");
        }
        Decoded::Unsupported => panic!("{context}: rustc_apfloat has no unsupported encodings"),
    }
}
