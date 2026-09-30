//! Tests on the bits of host encodings. The paths test a NaN with integer
//! instructions: LLVM can move a float comparison above the check of the
//! environment.

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
