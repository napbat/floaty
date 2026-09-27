//! Test encodings at the field boundaries of a binary format.

use rug::Integer;

/// Whether a format stores its integer bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerBit {
    /// The integer bit is implicit, as in the IEEE 754 formats.
    Implicit,
    /// The integer bit is the top fraction bit, as in the x87 format.
    Explicit,
}

/// Returns encodings that exercise every field boundary of a binary format
/// `width` bits wide with `exponent_bits` exponent bits.
///
/// Each encoding combines a sign, an exponent field from both ends and the
/// middle of its range, and a fraction from a set of edge patterns.
#[must_use]
pub fn boundary_encodings(width: u32, exponent_bits: u32, integer_bit: IntegerBit) -> Vec<Integer> {
    let fraction_bits = width - 1 - exponent_bits;
    let field_max = (Integer::from(1) << exponent_bits) - 1u32;
    let fraction_all = (Integer::from(1) << fraction_bits) - 1u32;
    let top = Integer::from(1) << (fraction_bits - 1);
    let next = Integer::from(1) << (fraction_bits - 2);
    let mut fractions = vec![
        Integer::ZERO,
        Integer::from(1),
        top.clone(),
        top.clone() | 1u32,
        next.clone(),
        next.clone() | 1u32,
        fraction_all.clone(),
    ];
    if integer_bit == IntegerBit::Explicit {
        let quiet = next.clone();
        let low_quiet = Integer::from(1) << (fraction_bits - 3);
        fractions.extend([
            quiet.clone() | 1u32,
            top.clone() | &quiet,
            top.clone() | &quiet | 1u32,
            top | low_quiet,
            fraction_all >> 1u32,
        ]);
    }
    let fields = [
        Integer::ZERO,
        Integer::from(1),
        Integer::from(2),
        field_max.clone() >> 1u32,
        field_max.clone() - 1u32,
        field_max,
    ];
    let mut encodings = Vec::new();
    for sign in [Integer::ZERO, Integer::from(1) << (width - 1)] {
        for field in &fields {
            for fraction in &fractions {
                encodings.push(sign.clone() | (field.clone() << fraction_bits) | fraction);
            }
        }
    }
    encodings
}

/// Converts an encoding to a `u128`.
///
/// # Panics
///
/// Panics when the encoding does not fit in 128 bits.
#[must_use]
pub fn to_u128(encoding: &Integer) -> u128 {
    encoding.to_u128().expect("the encoding fits in 128 bits")
}

/// Converts an encoding to `N` little-endian limbs.
///
/// # Panics
///
/// Panics when the encoding does not fit in `N` limbs.
#[must_use]
pub fn to_limbs<const N: usize>(encoding: &Integer) -> [u64; N] {
    let digits = encoding.to_digits::<u64>(rug::integer::Order::Lsf);
    assert!(digits.len() <= N, "the encoding fits in {N} limbs");
    let mut limbs = [0; N];
    limbs[..digits.len()].copy_from_slice(&digits);
    limbs
}
