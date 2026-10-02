//! The test vectors of the slice kernels of `Lanes`: pairs of vectors of
//! encodings of one binary format, in lengths around the lane counts.
//!
//! Each kind of pair stresses one part of the kernels:
//!
//! - values near 1 with random signs, whose sums round at every step;
//! - the same values with boundary encodings among them: zeros of both
//!   signs, subnormals, the smallest and largest normals, infinities, and
//!   quiet and signaling NaNs;
//! - large values of one sign, whose products and sums overflow;
//! - zeros of both signs alone, for the sign of a zero sum;
//! - subnormals alone, whose products underflow;
//! - a vector whose second half negates its first half, paired with itself,
//!   whose sum cancels and whose differences are zero.

use crate::encodings::{Layout, boundary_encodings_u128};
use crate::random::SplitMix64;

/// The lengths of the vectors: empty, below, at, and above the lane counts
/// 1 to 64, and a long vector whose length no lane count divides.
pub const LENGTHS: [usize; 20] = [
    0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 1027,
];

/// The kinds of vector pairs, as the module documentation lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mix {
    /// Values near 1 with random signs.
    Near,
    /// Values near 1 with boundary encodings among them.
    Edges,
    /// Large values of one sign.
    Overflow,
    /// Zeros of both signs.
    Zeros,
    /// Subnormals.
    Subnormals,
    /// A vector whose second half negates its first half, twice.
    Cancel,
}

impl Mix {
    /// Every kind.
    pub const ALL: [Self; 6] = [
        Self::Near,
        Self::Edges,
        Self::Overflow,
        Self::Zeros,
        Self::Subnormals,
        Self::Cancel,
    ];
}

/// Returns a random encoding of `layout` with an exponent field from
/// `fields`, a random sign unless `negative` fixes it, and a random fraction.
fn encoding(
    random: &mut SplitMix64,
    layout: Layout,
    fields: (u64, u64),
    negative: Option<bool>,
) -> u128 {
    let (low, high) = fields;
    let field = low + random.below(high - low + 1);
    let negative = negative.unwrap_or_else(|| random.coin_flip());
    layout.encode(negative, field, random.next_u128())
}

/// Returns a vector of `length` encodings of `layout` of the kind `mix`. For
/// [`Mix::Cancel`] the vector holds values near 1, and [`pair`] negates it.
///
/// # Panics
///
/// Panics when the layout is wider than 128 bits.
#[must_use]
pub fn vector(layout: Layout, mix: Mix, length: usize, random: &mut SplitMix64) -> Vec<u128> {
    let bias = u64::from(layout.largest_field() >> 1);
    let largest = u64::from(layout.largest_field());
    let near = (bias - 6, bias + 6);
    let boundary = boundary_encodings_u128(layout);
    let count = u64::try_from(boundary.len()).expect("the count fits a u64");
    (0..length)
        .map(|_| match mix {
            Mix::Edges if random.below(4) == 0 => {
                let index = usize::try_from(random.below(count))
                    .expect("an index of the boundary encodings fits a usize");
                boundary[index]
            }
            Mix::Near | Mix::Cancel | Mix::Edges => encoding(random, layout, near, None),
            Mix::Overflow => encoding(random, layout, (largest - 2, largest - 1), Some(false)),
            Mix::Zeros => layout.encode(random.coin_flip(), 0, 0),
            Mix::Subnormals => layout.encode(random.coin_flip(), 0, random.next_u128() | 1),
        })
        .collect()
}

/// Returns a pair of vectors of `length` encodings of `layout` of the kind
/// `mix`. For [`Mix::Cancel`], the second half of the first vector negates
/// its first half, so that its sum cancels, and the second vector is the
/// first, so that each difference is zero.
#[must_use]
pub fn pair(
    layout: Layout,
    mix: Mix,
    length: usize,
    random: &mut SplitMix64,
) -> (Vec<u128>, Vec<u128>) {
    let mut first = vector(layout, mix, length, random);
    if mix != Mix::Cancel {
        let second = vector(layout, mix, length, random);
        return (first, second);
    }
    let sign = 1_u128 << (layout.width - 1);
    let half = length / 2;
    for index in half..2 * half {
        first[index] = first[index - half] ^ sign;
    }
    (first.clone(), first)
}

/// Returns `length` random 8-bit codes, with the extreme codes among them.
#[must_use]
pub fn codes(length: usize, random: &mut SplitMix64) -> Vec<u8> {
    (0..length)
        .map(|index| match index % 7 {
            0 => 0,
            1 => u8::MAX,
            _ => random.next_u64().to_le_bytes()[0],
        })
        .collect()
}
