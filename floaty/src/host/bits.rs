//! Tests on the bits of host encodings. The paths test a NaN with integer
//! instructions: LLVM can move a float comparison above the check of the
//! environment.

use core::cmp::Ordering;

/// Returns `true` for the bits of a binary32 NaN.
#[inline]
pub(super) const fn nan_32(bits: u32) -> bool {
    bits & 0x7FFF_FFFF > 0x7F80_0000
}

/// Returns `true` for the bits of a binary16 NaN.
#[inline]
pub(super) const fn nan_16(bits: u16) -> bool {
    bits & 0x7FFF > 0x7C00
}

/// Returns `true` for the bits of a bfloat16 NaN.
#[inline]
pub(super) const fn nan_bfloat(bits: u16) -> bool {
    bits & 0x7FFF > 0x7F80
}

/// Returns `true` for the bits of a binary64 NaN.
#[inline]
pub(super) const fn nan_64(bits: u64) -> bool {
    bits & 0x7FFF_FFFF_FFFF_FFFF > 0x7FF0_0000_0000_0000
}

/// Returns the order of two encodings of a binary interchange format, as the
/// quiet predicates give it, or `None` when an encoding is a NaN. Each
/// encoding sits in the low `width` bits, and `infinity` is the encoding of
/// +∞. The order of two values that are not NaNs is the order of their
/// sign-magnitude encodings, with -0 equal to +0. Integer instructions
/// compute it, so no floating-point environment changes it.
#[inline]
pub(super) fn order(left: u64, right: u64, width: u32, infinity: u64) -> Option<Ordering> {
    // The shift puts the sign bit of the format in the sign bit of `i64`.
    let shift = 64 - width;
    let (left, right, infinity) = (left << shift, right << shift, infinity << shift);
    // A shift by one more drops the sign bit and keeps the magnitude.
    if (left << 1).max(right << 1) > infinity << 1 {
        return None;
    }
    if (left | right) << 1 == 0 {
        return Some(Ordering::Equal);
    }
    // A negative encoding with its magnitude bits inverted orders as a
    // two's complement integer. -0 then orders below +0, which the test of
    // two zeros above excludes.
    let key = |bits: u64| {
        let signed = bits.cast_signed();
        signed ^ ((signed >> 63) & i64::MAX)
    };
    Some(key(left).cmp(&key(right)))
}

/// Returns `true` for two binary32 encodings where the minimum and maximum
/// instructions of the host differ from the operations: a NaN operand, or two
/// zeros. `MINSS` then gives the second operand, and every operation orders
/// `-0` below `+0`. The test has no branch, so LLVM tests the lanes of a
/// packed path at once.
#[inline]
pub(super) fn min_max_differs(left: u32, right: u32) -> bool {
    let nan = (left & 0x7FFF_FFFF).max(right & 0x7FFF_FFFF) > 0x7F80_0000;
    // The shift drops the sign bits, so the result is zero for two zeros.
    let zeros = (left | right) << 1 == 0;
    nan | zeros
}

/// Returns `true` for two binary16 encodings where the minimum and maximum
/// instructions of the host differ from the operations, as
/// `min_max_differs` does for binary32.
#[inline]
pub(super) fn min_max_differs_16(left: u16, right: u16) -> bool {
    let nan = (left & 0x7FFF).max(right & 0x7FFF) > 0x7C00;
    let zeros = (left | right) << 1 == 0;
    nan | zeros
}

/// Returns `true` for two binary128 encodings, low limb first, where the
/// selection of an operand by the comparison differs from the minimum and
/// maximum operations: a NaN operand, or two zeros, which compare equal.
#[inline]
pub(super) fn min_max_differs_128(left: [u64; 2], right: [u64; 2]) -> bool {
    let nan = |bits: [u64; 2]| {
        let high = bits[1] & 0x7FFF_FFFF_FFFF_FFFF;
        high > 0x7FFF_0000_0000_0000 || (high == 0x7FFF_0000_0000_0000 && bits[0] != 0)
    };
    let zero = |bits: [u64; 2]| (bits[1] << 1) | bits[0] == 0;
    nan(left) | nan(right) | (zero(left) & zero(right))
}

/// Returns `true` for two binary64 encodings where the minimum and maximum
/// instructions of the host differ from the operations, as
/// `min_max_differs` does for binary32.
#[inline]
pub(super) fn min_max_differs_64(left: u64, right: u64) -> bool {
    let magnitude = 0x7FFF_FFFF_FFFF_FFFF;
    let nan = (left & magnitude).max(right & magnitude) > 0x7FF0_0000_0000_0000;
    let zeros = (left | right) << 1 == 0;
    nan | zeros
}
