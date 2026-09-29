//! Compares the binary formats wider than 128 bits with the definition in
//! IEEE 754-2019 section 3.4, evaluated exactly with MPFR. No established
//! library decodes these widths, so the definition is the reference.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::{
    Binary, Class, Decoded, F160, F192, F224, F256, F288, F320, F352, F384, F416, F448, F480, F512,
    Float,
};
use floaty_verify::encodings::{IntegerBit, boundary_encodings, to_limbs};
use floaty_verify::random::SplitMix64;
use floaty_verify::shape::{self, trailing_payload};
use rug::integer::Order;
use rug::{Float as BigFloat, Integer};

/// A 200-bit layout whose exponent field, bits 179 to 198, crosses a limb
/// boundary.
type Wide200 = Float<Binary<20>, 200>;

const RANDOM_SAMPLES: usize = 5_000;

/// What IEEE 754-2019 section 3.4 says an encoding is.
#[derive(Debug, PartialEq)]
enum Expected {
    Zero {
        negative: bool,
    },
    Finite {
        negative: bool,
        value: BigFloat,
        subnormal: bool,
    },
    Infinity {
        negative: bool,
    },
    Nan {
        negative: bool,
        quiet: bool,
        payload: Integer,
    },
}

/// Evaluates an encoding of a format `width` bits wide with `exponent_bits`
/// exponent bits by the IEEE 754-2019 section 3.4 definition.
fn ieee_definition(encoding: &Integer, width: u32, exponent_bits: u32) -> Expected {
    let precision = width - exponent_bits;
    let trailing = precision - 1;
    let bias = (1_i32 << (exponent_bits - 1)) - 1;
    let negative = encoding.get_bit(width - 1);
    let field_mask = (Integer::from(1) << exponent_bits) - 1u32;
    let field = (encoding.clone() >> trailing) & field_mask.clone();
    let fraction = encoding.clone() & ((Integer::from(1) << trailing) - 1u32);
    if field == field_mask {
        return if fraction.is_zero() {
            Expected::Infinity { negative }
        } else {
            Expected::Nan {
                negative,
                quiet: fraction.get_bit(trailing - 1),
                payload: fraction.clone() & ((Integer::from(1) << (trailing - 1)) - 1u32),
            }
        };
    }
    let field = field.to_i32().expect("the exponent field fits an i32");
    if field == 0 && fraction.is_zero() {
        return Expected::Zero { negative };
    }
    // v = 2^(e - bias) * (d + 2^(1-p) * T), with d = 1 for a normal number, and
    // v = 2^emin * (0 + 2^(1-p) * T) for a subnormal number.
    let (exponent, leading) = if field == 0 {
        (1 - bias, 0_u32)
    } else {
        (field - bias, 1_u32)
    };
    let working = precision + 2;
    let scale = i32::try_from(trailing).expect("the precision fits an i32");
    let significand = BigFloat::with_val(working, &fraction) >> scale;
    let mut value = (significand + leading) << exponent;
    if negative {
        value = -value;
    }
    Expected::Finite {
        negative,
        value,
        subnormal: field == 0,
    }
}

/// Converts a decoded value to the same shape as the definition.
fn from_decoded<const N: usize>(decoded: Decoded<N>, precision: u32, emin: i32) -> Expected {
    match decoded {
        Decoded::Zero { negative, .. } => Expected::Zero { negative },
        Decoded::Finite {
            negative,
            exponent,
            significand,
        } => {
            let integer = Integer::from_digits(&significand, Order::Lsf);
            let mut value = BigFloat::with_val(precision + 2, &integer) << exponent;
            let subnormal = value.get_exp().is_some_and(|top| top - 1 < emin);
            if negative {
                value = -value;
            }
            Expected::Finite {
                negative,
                value,
                subnormal,
            }
        }
        Decoded::Infinity { negative } => Expected::Infinity { negative },
        Decoded::Nan {
            negative,
            signaling,
            payload,
        } => Expected::Nan {
            negative,
            quiet: !signaling,
            payload: Integer::from_digits(&payload, Order::Lsf),
        },
        Decoded::Unsupported => panic!("an IEEE format has no unsupported encodings"),
    }
}

fn expected_class(expected: &Expected) -> Class {
    match expected {
        Expected::Zero { .. } => Class::Zero,
        Expected::Finite {
            subnormal: true, ..
        } => Class::Subnormal,
        Expected::Finite {
            subnormal: false, ..
        } => Class::Normal,
        Expected::Infinity { .. } => Class::Infinite,
        Expected::Nan { quiet: true, .. } => Class::QuietNan,
        Expected::Nan { quiet: false, .. } => Class::SignalingNan,
    }
}

/// Checks the boundary encodings and random encodings of one format.
macro_rules! check_format {
    ($alias:ty, $width:literal, $exponent_bits:literal, $limbs:literal, $seed:literal) => {{
        let mut random = SplitMix64::new($seed);
        let mask: Integer = (Integer::from(1) << $width) - 1u32;
        let mut encodings = boundary_encodings($width, $exponent_bits, IntegerBit::Implicit);
        for _ in 0..RANDOM_SAMPLES {
            let limbs: Vec<u64> = (0..$limbs).map(|_| random.next_u64()).collect();
            let encoding: Integer = Integer::from_digits(&limbs, Order::Lsf) & &mask;
            encodings.push(encoding);
        }
        for encoding in &encodings {
            let context = format!("{} {encoding:#x}", stringify!($alias));
            let ours = <$alias>::from_bits(to_limbs::<$limbs>(encoding));
            let expected = ieee_definition(encoding, $width, $exponent_bits);
            assert_eq!(ours.classify(), expected_class(&expected), "{context}");
            assert!(ours.is_canonical(), "{context}");
            let decoded = ours.decode::<$limbs>();
            let payload = trailing_payload(encoding, <$alias>::PRECISION);
            let (precision, emin) = (<$alias>::PRECISION, <$alias>::EMIN);
            shape::check(
                &decoded,
                ours.classify(),
                precision,
                emin,
                &payload,
                &context,
            );
            assert_eq!(
                from_decoded(decoded, precision, emin),
                expected,
                "{context}"
            );
        }
    }};
}

#[test]
fn binary160_to_binary512_follow_the_ieee_definition() {
    check_format!(F160, 160, 16, 3, 160);
    check_format!(F192, 192, 17, 3, 192);
    check_format!(F224, 224, 18, 4, 224);
    check_format!(F256, 256, 19, 4, 256);
    check_format!(F288, 288, 20, 5, 288);
    check_format!(F320, 320, 20, 5, 320);
    check_format!(F352, 352, 21, 6, 352);
    check_format!(F384, 384, 21, 6, 384);
    check_format!(F416, 416, 22, 7, 416);
    check_format!(F448, 448, 22, 7, 448);
    check_format!(F480, 480, 23, 8, 480);
    check_format!(F512, 512, 23, 8, 512);
}

#[test]
fn a_wide_layout_whose_exponent_field_crosses_a_limb_boundary() {
    check_format!(Wide200, 200, 20, 4, 200);
}
