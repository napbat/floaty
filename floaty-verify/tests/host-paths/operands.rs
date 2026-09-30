//! The operands of the checks: boundary and random encodings, values near
//! the halfway points of the narrower formats and near the bounds of the
//! integer types, and every pair of 16-bit encodings.

use floaty::{F64, F80};
use floaty_verify::encodings::{IntegerBit, boundary_encodings_u128};
use floaty_verify::random::SplitMix64;

/// Returns random encodings of a storage type, and the encodings at the
/// boundaries of a layout, in pairs.
pub(super) fn random_pairs<T: TryFrom<u128>>(
    random: &mut SplitMix64,
    width: u32,
    exponent_bits: u32,
    count: usize,
) -> Vec<(T, T)>
where
    T::Error: core::fmt::Debug,
{
    let integer_bit = if width == 80 {
        IntegerBit::Explicit
    } else {
        IntegerBit::Implicit
    };
    let mut encodings = boundary_encodings_u128(width, exponent_bits, integer_bit);
    encodings.extend((0..count).map(|_| random.next_u128() >> (128 - width)));
    let convert = |bits: u128| T::try_from(bits).expect("the encoding fits the storage");
    encodings
        .iter()
        .zip(encodings.iter().rev())
        .map(|(&a, &b)| (convert(a), convert(b)))
        .collect()
}

/// Returns triples of the operands of `pairs`: each pair with the first
/// operand of a later pair as the addend.
pub(super) fn triples<T: Copy>(pairs: &[(T, T)]) -> Vec<(T, T, T)> {
    let addends = pairs.iter().cycle().skip(pairs.len() / 3 + 1);
    pairs
        .iter()
        .zip(addends)
        .map(|(&(a, b), &(c, _))| (a, b, c))
        .collect()
}

/// Calls `check` on every pair of 16-bit encodings, in 16 threads. Each
/// thread takes the pairs of 4,096 first operands.
pub(super) fn every_16_bit_pair(check: impl Fn(u16, u16) + Sync) {
    let check = &check;
    std::thread::scope(|scope| {
        for part in 0..16_u16 {
            scope.spawn(move || {
                for high in part * 0x1000..(part + 1) * 0x1000 {
                    for low in 0..=u16::MAX {
                        check(high, low);
                    }
                }
            });
        }
    });
}

/// Returns double-double operands: halves from the boundary encodings and
/// random bits of binary64, which give NaN halves, infinities, subnormal
/// halves, and malformed pairs, and well-formed pairs with a small low half.
pub(super) fn double_double_pairs(random: &mut SplitMix64) -> Vec<(F64, F64)> {
    let boundaries = boundary_encodings_u128(64, 11, IntegerBit::Implicit);
    let half = |bits: u128| F64::from_bits(u64::try_from(bits).expect("a binary64 encoding"));
    let mut halves: Vec<(F64, F64)> = boundaries
        .iter()
        .zip(boundaries.iter().rev())
        .map(|(&hi, &lo)| (half(hi), half(lo)))
        .collect();
    halves.extend((0..20_000).map(|_| {
        let hi = F64::from_bits(random.next_u64());
        // A low half near 2^-60 of the high half, or random bits.
        let lo = if random.next_u64() % 4 == 0 {
            F64::from_bits(random.next_u64())
        } else {
            hi.scale_b(-60)
                .mul_with(
                    F64::from_bits(random.next_u64() >> 12 | 0x3FF0_0000_0000_0000),
                    floaty::Env::IEEE,
                )
                .0
        };
        (hi, lo)
    }));
    halves
}

/// Returns binary64 operands for the conversions: boundary and random
/// encodings, values near the halfway points of binary32 and binary16, and
/// values near the bounds of the integer types.
pub(super) fn conversion_operands(random: &mut SplitMix64) -> Vec<F64> {
    let mut bits: Vec<u64> = boundary_encodings_u128(64, 11, IntegerBit::Implicit)
        .into_iter()
        .map(|bits| u64::try_from(bits).expect("a binary64 encoding"))
        .collect();
    bits.extend((0..20_000).map(|_| random.next_u64()));
    // A binary64 value holds 29 bits below binary32 and 42 below binary16:
    // the halfway point, and one unit on each side of it.
    for shift in [28, 41] {
        for _ in 0..5_000 {
            let base = (random.next_u64() >> 1) & !((1 << (shift + 1)) - 1);
            let half = base | (1 << shift);
            bits.extend([half - 1, half, half + 1]);
        }
    }
    let mut values: Vec<F64> = bits.into_iter().map(F64::from_bits).collect();
    // Integers and halves near the bounds of 32- and 64-bit integers.
    for exponent in [31, 32, 53, 63, 64] {
        let bound = F64::from_bits(0x3FF0_0000_0000_0000).scale_b(exponent);
        let half = F64::from_bits(0x3FE0_0000_0000_0000);
        for value in [
            bound,
            -bound,
            bound.next_up(),
            bound.next_down(),
            -bound.next_down(),
        ] {
            values.push(value);
            values.push(value.add_with(half, floaty::Env::IEEE).0);
            values.push(value.sub_with(half, floaty::Env::IEEE).0);
        }
    }
    values
}

/// Returns x87 extended operands: boundary and random encodings, with
/// unsupported encodings and pseudo-denormals among them, values near the
/// halfway points of binary64 and binary32, and integers and halves near the
/// bounds of 32- and 64-bit integers.
pub(super) fn extended_operands(random: &mut SplitMix64) -> Vec<F80> {
    let mut bits = boundary_encodings_u128(80, 15, IntegerBit::Explicit);
    bits.extend((0..20_000).map(|_| random.next_u128() >> 48));
    // An x87 significand holds 11 bits below binary64 and 40 below binary32:
    // the halfway point, and one unit on each side of it, at exponents near
    // 1.0 and near the bounds of 64-bit integers.
    for shift in [10, 39] {
        for _ in 0..5_000 {
            let exponent = u128::from(0x3FFF - 70 + random.next_u64() % 140);
            let high = (random.next_u64() | 1 << 63) & !((1 << (shift + 1)) - 1);
            let half = u128::from(high | 1 << shift);
            bits.extend([half - 1, half, half + 1].map(|significand| exponent << 64 | significand));
        }
    }
    let mut values: Vec<F80> = bits.into_iter().map(F80::from_bits).collect();
    let one = F80::from_bits(0x3FFF_8000_0000_0000_0000);
    let half = F80::from_bits(0x3FFE_8000_0000_0000_0000);
    for exponent in [31, 32, 63, 64] {
        let bound = one.scale_b(exponent);
        for value in [
            bound,
            -bound,
            bound.next_up(),
            bound.next_down(),
            -bound.next_down(),
        ] {
            values.push(value);
            values.push(value.add_with(half, floaty::Env::IEEE).0);
            values.push(value.sub_with(half, floaty::Env::IEEE).0);
        }
    }
    values
}

/// Returns every pair of the boundary encodings of a layout, both orders, so
/// that each pair of special values, such as `-0` and `+0`, occurs.
pub(super) fn boundary_pairs<T: TryFrom<u128>>(width: u32, exponent_bits: u32) -> Vec<(T, T)>
where
    T::Error: core::fmt::Debug,
{
    let encodings = boundary_encodings_u128(width, exponent_bits, IntegerBit::Implicit);
    let convert = |bits: u128| T::try_from(bits).expect("the encoding fits the storage");
    encodings
        .iter()
        .flat_map(|&a| encodings.iter().map(move |&b| (convert(a), convert(b))))
        .collect()
}

/// Returns pairs of normal encodings of a binary format with `exponent_bits`
/// exponent bits and `fraction_bits` bits below them, whose exponent fields
/// differ by up to `spread`, with random signs. A set integer bit at
/// `fraction_bits - 1` makes x87 extended encodings canonical when
/// `integer_bit` is set.
pub(super) fn close_pairs(
    random: &mut SplitMix64,
    exponent_bits: u32,
    fraction_bits: u32,
    integer_bit: bool,
    spread: u64,
) -> Vec<(u128, u128)> {
    let largest = (1_u128 << exponent_bits) - 2;
    let encoding = |random: &mut SplitMix64, exponent: u128| {
        let sign = u128::from(random.next_u64() >> 63) << (exponent_bits + fraction_bits);
        let mut fraction = random.next_u128() >> (128 - fraction_bits);
        if integer_bit {
            fraction |= 1 << (fraction_bits - 1);
        }
        sign | (exponent << fraction_bits) | fraction
    };
    (0..4_000)
        .map(|_| {
            let exponent = 1 + u128::from(random.next_u64()) % largest;
            let gap = u128::from(random.next_u64() % (spread + 1));
            let divisor = exponent.saturating_sub(gap).max(1);
            (encoding(random, exponent), encoding(random, divisor))
        })
        .collect()
}
