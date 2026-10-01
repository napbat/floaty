//! Compares the NaN payload operations of floaty with the payload functions
//! of the host glibc libm, through `floaty_verify::libm`: binary32,
//! binary64, x87 extended precision, and binary128.
//!
//! The operands are the boundary encodings, random encodings of which one in
//! 16 has the exponent field of the NaNs, and the integers and NaN payloads
//! at the edges of the payload range. The test needs glibc 2.32 or later,
//! which gives -1 for a value that is not a NaN, as C23 requires.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::{F32, F64, F80, F128};
use floaty_verify::encodings::{IntegerBit, Layout, sample_encodings_u128};
use floaty_verify::libm::{self, Format, Payload};
use floaty_verify::random::SplitMix64;

/// Returns the number of payload bits: the fraction bits below the quiet
/// bit.
fn payload_bits(layout: Layout) -> u32 {
    match layout.integer_bit {
        IntegerBit::Implicit => layout.precision() - 2,
        IntegerBit::Explicit => 62,
    }
}

/// Returns the encodings of the integers at the edges of the payload range,
/// with both signs, of values half an integer from them, and of NaNs with the
/// edge payloads.
fn payload_edges(layout: Layout) -> Vec<u128> {
    let width = payload_bits(layout);
    let bias = u64::try_from(layout.ieee_bias()).expect("an exponent bias is positive");
    // The positive value `integer * 2^-shift`. Each value is at least 1/2,
    // so it is normal.
    let value = |integer: u128, shift: u32| {
        let length = 128 - integer.leading_zeros();
        let field = bias + u64::from(length) - 1 - u64::from(shift);
        layout.encode(false, field, integer << (layout.precision() - length))
    };
    let top = 1_u128 << width;
    let mut encodings = vec![0, 1 << (layout.width - 1)];
    for integer in [1, 2, 3, top >> 1, top - 1, top, top + 2] {
        let positive = value(integer, 0);
        encodings.extend([positive, positive | 1 << (layout.width - 1)]);
    }
    for odd in [1, 3, 5, 2 * top - 1] {
        encodings.push(value(odd, 1));
    }
    let quiet = 1_u128 << width;
    let integer_bit = match layout.integer_bit {
        IntegerBit::Implicit => 0,
        IntegerBit::Explicit => 1 << 63,
    };
    let field = u64::from(layout.largest_field());
    for payload in [0, 1, top >> 1, top - 1] {
        for negative in [false, true] {
            encodings.push(layout.encode(negative, field, integer_bit | quiet | payload));
            if payload != 0 {
                encodings.push(layout.encode(negative, field, integer_bit | payload));
            }
        }
    }
    encodings
}

/// Returns `true` for an x87 encoding without its integer bit and with a
/// nonzero exponent field: an unnormal, a pseudo-NaN, or a pseudo-infinity.
///
/// Conflict: glibc reads these encodings by their fields, and floaty does
/// not support them. glibc 2.43 `getpayloadl` tests only the exponent field
/// and the fraction below the integer bit, so a pseudo-NaN gives its payload
/// (`sysdeps/ieee754/ldbl-96/s_getpayloadl.c`, lines 30 to 32).
/// `setpayloadl` tests only the exponent field and the bits below the
/// integer, so an unnormal with an integral value gives a NaN
/// (`s_setpayloadl_main.c`, lines 38 to 52). floaty documents that an
/// unsupported encoding is no NaN and no number: `payload` gives -1, and
/// `from_payload` gives +0. Resolution: the test leaves these encodings out.
/// The rule of floaty for them is checked in `operations-wide-mpfr.rs`.
fn unsupported(layout: Layout, bits: u128) -> bool {
    let fraction_bits = layout.fraction_bits();
    layout.integer_bit == IntegerBit::Explicit
        && bits >> fraction_bits & u128::from(layout.largest_field()) != 0
        && bits >> (fraction_bits - 1) & 1 == 0
}

/// Compares the payload operations of floaty, which `run` applies, with the
/// libm functions of `format`.
fn compare(format: Format, layout: Layout, run: impl Fn(u128, Payload) -> u128, seed: u64) {
    let mut random = SplitMix64::new(seed);
    let mut encodings = sample_encodings_u128(layout, 100_000, &mut random);
    encodings.extend(payload_edges(layout));
    for bits in encodings {
        if unsupported(layout, bits) {
            continue;
        }
        for function in Payload::ALL {
            assert_eq!(
                run(bits, function),
                libm::payload(format, function, bits),
                "{format:?} {function:?} {bits:#x}"
            );
        }
    }
}

/// Applies a payload function of floaty to a value of the type `$float`.
macro_rules! apply {
    ($float:ty, $bits:ty) => {
        |bits: u128, function: Payload| {
            let value = <$float>::from_bits(<$bits>::try_from(bits).expect("the encoding fits"));
            let result = match function {
                Payload::Get => value.payload(),
                Payload::Set => <$float>::from_payload(value),
                Payload::SetSignaling => <$float>::from_payload_signaling(value),
            };
            u128::from(result.to_bits())
        }
    };
}

#[test]
fn binary32_payloads_match_glibc() {
    compare(
        Format::Float,
        Layout::BINARY32,
        apply!(F32, u32),
        0x0032_0907,
    );
}

#[test]
fn binary64_payloads_match_glibc() {
    compare(
        Format::Double,
        Layout::BINARY64,
        apply!(F64, u64),
        0x0064_0907,
    );
}

#[test]
fn x87_payloads_match_glibc() {
    compare(
        Format::LongDouble,
        Layout::X87_EXTENDED,
        apply!(F80, u128),
        0x0080_0907,
    );
}

#[test]
fn binary128_payloads_match_glibc() {
    compare(
        Format::Float128,
        Layout::BINARY128,
        apply!(F128, u128),
        0x0128_0907,
    );
}
