//! The integer square root.
//!
//! A value of at most 16 bits takes the square root of `core`. A value of at
//! most 128 bits takes no division. A table and two Newton
//! steps give `1 / sqrt(x)` to about 35 bits. The product with `x` gives a
//! root below the true root, and one step with the remainder brings it within
//! one of the true root. A wider value takes Newton's steps on the root, one
//! long division each, from the root of its top 128 bits. Every path ends with
//! a correction by squaring, so the root is exact for every estimate, and a
//! poor estimate only costs time.

use core::cmp::Ordering;

use super::{Limbs, divide, from_u128, high_u64, low_u64, multiply_fit, to_u128};

/// The first approximation of `1 / sqrt(A)` for `A` from 1/4 to 1, in units of
/// 2^-14, for each value `top` of the top 9 bits of `A * 2^64`, from 128 to
/// 511. The entry is `floor(sqrt(2^38 / (2 * top + 1)))`, the value at the
/// middle of the interval of `top`, so it is within 2^-9 of `1 / sqrt(A)`.
const TABLE: [u16; 384] = {
    let mut table = [0; 384];
    let (mut index, mut top) = (0, 128_u64);
    while top < 512 {
        // The root is from 16,392 to 32,705. `u16::try_from` is not callable
        // in a constant, so the two low bytes give the value.
        let [low, high, ..] = ((1 << 38) / (2 * top + 1)).isqrt().to_le_bytes();
        table[index] = u16::from_le_bytes([low, high]);
        index += 1;
        top += 1;
    }
    table
};

/// Returns one Newton step toward `1 / sqrt(A)`, `y * (3 - A * y^2) / 2`,
/// for `a = A * 2^64` and `y` in units of 2^-62. The step never rises above
/// `1 / sqrt(A)`.
#[inline]
fn reciprocal_step(a: u64, y: u64) -> u64 {
    let root = high_u64(u128::from(a) * u128::from(y));
    let square = low_u64((u128::from(root) * u128::from(y)) >> 62);
    let factor = (3 << 62) - square;
    low_u64((u128::from(y) * u128::from(factor)) >> 63)
}

/// Returns `floor(sqrt(value))` for a nonzero `value`, without a division.
fn square_root_u128(value: u128) -> u64 {
    debug_assert!(value != 0, "the value is not zero");
    // An even shift puts the value in [2^126, 2^128), and the root of the
    // shifted value is the root shifted by half as much.
    let shift = value.leading_zeros() & !1;
    let normalized = value << shift;
    let a = high_u64(normalized);
    let index = usize::try_from((a >> 55) - 128).expect("the top 9 bits are from 128 to 511");
    let y = u64::from(TABLE[index]) << 48;
    // Once the steps converge, the truncations can leave `y` up to 3 units
    // above `1 / sqrt(A)`, and each unit moves the root by 4. Four units less
    // keep `y` below `1 / sqrt(A)`, so the estimate is not above the root.
    let y = reciprocal_step(a, reciprocal_step(a, y)) - 4;
    // `y` is below `1 / sqrt(A)` by at most about 2^-34, so `A * y * 2^64` is
    // below the root by at most about 2^30. Its remainder divided by twice the
    // estimate, `rest * y / 2^127`, is the rest of the root to within one.
    let estimate = low_u64((u128::from(a) * u128::from(y)) >> 62);
    let square = |root: u64| u128::from(root) * u128::from(root);
    // The estimate is not above the root, so the remainder is not negative.
    // The saturation keeps the result exact even if it were.
    let rest = normalized.saturating_sub(square(estimate));
    debug_assert!(rest >> 96 == 0, "the estimate is within 2^31 of the root");
    let step = low_u64((u128::from(low_u64(rest >> 32)) * u128::from(y)) >> 95);
    let mut root = estimate.saturating_add(step);
    // The loops make the root exact for every estimate. The assertions check
    // that the estimate is within one of the root, so a test finds a slow
    // correction.
    let start = root;
    while square(root) > normalized {
        root -= 1;
        debug_assert!(start - root <= 1, "the estimate is within one of the root");
    }
    // `normalized - root^2 > 2 * root` means `(root + 1)^2 <= normalized`.
    while normalized - square(root) > 2 * u128::from(root) {
        root += 1;
        debug_assert!(root - start <= 1, "the estimate is within one of the root");
    }
    root >> (shift / 2)
}

/// Returns the integer square root of `value`, rounded down, and `true` when
/// the root is not exact.
pub fn square_root<L: Limbs>(value: L) -> (L, bool) {
    let length = value.bit_length();
    if length <= 16 {
        // The square root of `core` takes a short path for a value of at most
        // 16 bits, which is faster there than the table and Newton steps.
        let narrow = value.limb(0);
        let root = narrow.isqrt();
        return (L::ZERO.with_limb(0, root), root * root != narrow);
    }
    if length <= 128 {
        let wide = to_u128(&value);
        let root = square_root_u128(wide);
        let square = u128::from(root) * u128::from(root);
        return (from_u128(u128::from(root)), square != wide);
    }
    // An even shift keeps the root of the shifted value at half the shift.
    // One more than the root of the top 126 or 127 bits is above the root by
    // less than 2^-62 of it.
    let shift = (length - 127) & !1;
    let top = to_u128(&value.shr(shift));
    let mut root = from_u128::<L>(u128::from(square_root_u128(top)) + 1).shl(shift / 2);
    // Each step from above at least doubles the correct bits and stays at or
    // above the root. Two bits more than the root has leave the root at most
    // one above the true root.
    let needed = length.div_ceil(2) + 2;
    let mut correct = 62;
    while correct < needed {
        let (quotient, _) = divide(value, root);
        root = root.add(quotient).shr(1);
        correct *= 2;
    }
    // The true root fits half the width. A root one above it can reach 2 to
    // the half width, whose square does not fit, so it falls to the largest
    // value of half the width.
    let half = L::ones(L::BITS / 2);
    if root.compare(&half) == Ordering::Greater {
        root = half;
    }
    let one = L::ZERO.with_bit(0);
    let start = root;
    let mut square = multiply_fit(root, root);
    while square.compare(&value) == Ordering::Greater {
        root = root.sub(one);
        square = multiply_fit(root, root);
        debug_assert!(
            start.sub(root).compare(&one).is_le(),
            "the root is at most one above the true root"
        );
    }
    (root, square != value)
}

#[cfg(test)]
mod tests {
    use super::square_root;
    use crate::limbs::{self, Limbs};

    /// A seeded xorshift generator.
    fn generator(mut state: u64) -> impl FnMut() -> u64 {
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    /// Checks the root of a value of at most 128 bits against the square
    /// root of `core`.
    fn check_narrow(value: u128) {
        let (root, inexact) = square_root(limbs::split_u128(value));
        let expected = value.isqrt();
        assert_eq!(limbs::to_u128(&root), expected, "sqrt({value})");
        assert_eq!(inexact, expected * expected != value, "sqrt({value})");
    }

    #[test]
    fn narrow_roots_match_the_core_square_root() {
        (0..=5000).for_each(check_narrow);
        let mut next = generator(0x5DEE_CE66_D1CE_4E5B);
        for bits in 1..=128 {
            for _ in 0..2000 {
                let value = ((u128::from(next()) << 64) | u128::from(next())) >> (128 - bits);
                check_narrow(value);
            }
        }
        for _ in 0..20_000 {
            let root = u128::from(next());
            let square = root * root;
            [square - 1, square, square + 1]
                .into_iter()
                .for_each(check_narrow);
        }
        for shift in 0..128 {
            check_narrow(1 << shift);
            check_narrow((1 << shift) - 1);
        }
        // Each table entry covers the values whose top 9 bits are `top`:
        // test the ends and the middle of each interval.
        for top in 128_u128..512 {
            for low in [0, 1, 1 << 54, (1 << 55) - 1] {
                let high = ((top << 55) | low) << 64;
                check_narrow(high);
                check_narrow(high | u128::from(u64::MAX));
            }
        }
        check_narrow(u128::MAX);
        check_narrow(u128::from(u64::MAX) * u128::from(u64::MAX));
    }

    /// Checks that `root^2 <= value < (root + 1)^2`, and the inexact flag.
    /// The second condition is `value - root^2 <= 2 * root`.
    fn check_wide<L: Limbs>(value: L) {
        let (root, inexact) = square_root(value);
        let square = limbs::multiply_fit(root, root);
        assert!(
            square.compare(&value).is_le(),
            "{value:x?}: the root is too large"
        );
        let rest = value.sub(square);
        assert!(
            rest.compare(&root.shl(1)).is_le(),
            "{value:x?}: the root is too small"
        );
        assert_eq!(inexact, !rest.is_zero(), "{value:x?}");
    }

    #[test]
    fn wide_roots_meet_the_definition() {
        let mut next = generator(0x2545_F491_4F6C_DD1D);
        for _ in 0..2000 {
            let random: [u64; 4] = core::array::from_fn(|_| next());
            let small_root: [u64; 4] = [next(), next(), 0, 0];
            let square = limbs::multiply_fit(small_root, small_root);
            check_wide(random);
            check_wide(random.shr(u32::try_from(next() % 120).expect("a shift fits")));
            check_wide(square);
            check_wide(square.add([1, 0, 0, 0]));
            check_wide(square.sub([1, 0, 0, 0]));
        }
        for _ in 0..500 {
            let random: [u64; 8] = core::array::from_fn(|_| next());
            let root: [u64; 8] = core::array::from_fn(|index| if index < 4 { next() } else { 0 });
            let square = limbs::multiply_fit(root, root);
            check_wide(random);
            check_wide(square);
            check_wide(square.sub([1, 0, 0, 0, 0, 0, 0, 0]));
        }
        for _ in 0..100 {
            let random: [u64; 16] = core::array::from_fn(|_| next());
            check_wide(random);
        }
        check_wide([u64::MAX; 4]);
        check_wide([u64::MAX; 16]);
    }
}
