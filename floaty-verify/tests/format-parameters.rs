//! Compares the format parameters with IEEE 754 and with `rustc_apfloat`.

use floaty::{
    BF16, F8E4M3Fn, F8E5M2, F16, F32, F64, F80, F128, F160, F192, F224, F256, F288, F320, F352,
    F384, F416, F448, F480, F512, TF32,
};
use floaty_verify::apfloat::Tf32Semantics;
use rustc_apfloat::ieee::{
    BFloatS, DoubleS, Float8E4M3FNS, Float8E5M2S, HalfS, QuadS, Semantics, SingleS,
    X87DoubleExtendedS,
};

/// The IEEE 754 interchange parameters of binary`k`: precision, `emax`, and
/// `emin`. IEEE 754-2019 table 3.5 fixes the exponent width at `k` of 16, 32,
/// and 64, and gives `round(4 * log2(k)) - 13` from `k` = 128.
fn interchange(k: u32) -> (u32, i32, i32) {
    let exponent_bits = match k {
        16 => 5,
        32 => 8,
        64 => 11,
        _ => {
            assert!(
                k >= 128 && k.is_multiple_of(32),
                "binary{k} is not an interchange format"
            );
            // round(4 * log2(k)) is the n with 2^(2n - 1) <= k^8 < 2^(2n + 1).
            let eighth_power = u128::from(k).pow(8);
            eighth_power.ilog2().div_ceil(2) - 13
        }
    };
    let emax = (1 << (exponent_bits - 1)) - 1;
    (k - exponent_bits, emax, 1 - emax)
}

macro_rules! ieee {
    ($($alias:ident = $k:literal),*) => {
        $(assert_eq!(
            ($alias::PRECISION, $alias::EMAX, $alias::EMIN),
            interchange($k),
            stringify!($alias)
        );)*
    };
}

#[test]
fn ieee_aliases_have_the_interchange_parameters() {
    ieee!(
        F16 = 16,
        F32 = 32,
        F64 = 64,
        F128 = 128,
        F160 = 160,
        F192 = 192,
        F224 = 224,
        F256 = 256,
        F288 = 288,
        F320 = 320,
        F352 = 352,
        F384 = 384,
        F416 = 416,
        F448 = 448,
        F480 = 480,
        F512 = 512
    );
}

fn apfloat<S: Semantics>() -> (u32, i32, i32) {
    let precision = u32::try_from(S::PRECISION).expect("a precision fits a u32");
    (precision, S::MAX_EXP, S::MIN_EXP)
}

#[test]
fn aliases_match_rustc_apfloat() {
    assert_eq!((F16::PRECISION, F16::EMAX, F16::EMIN), apfloat::<HalfS>());
    assert_eq!((F32::PRECISION, F32::EMAX, F32::EMIN), apfloat::<SingleS>());
    assert_eq!((F64::PRECISION, F64::EMAX, F64::EMIN), apfloat::<DoubleS>());
    assert_eq!(
        (F128::PRECISION, F128::EMAX, F128::EMIN),
        apfloat::<QuadS>()
    );
    assert_eq!(
        (BF16::PRECISION, BF16::EMAX, BF16::EMIN),
        apfloat::<BFloatS>()
    );
    assert_eq!(
        (TF32::PRECISION, TF32::EMAX, TF32::EMIN),
        apfloat::<Tf32Semantics>()
    );
    assert_eq!(
        (F8E5M2::PRECISION, F8E5M2::EMAX, F8E5M2::EMIN),
        apfloat::<Float8E5M2S>()
    );
    assert_eq!(
        (F8E4M3Fn::PRECISION, F8E4M3Fn::EMAX, F8E4M3Fn::EMIN),
        apfloat::<Float8E4M3FNS>()
    );
    assert_eq!(
        (F80::PRECISION, F80::EMAX, F80::EMIN),
        apfloat::<X87DoubleExtendedS>()
    );
}
