//! The floating-point control register of AArch64, FPCR, for the tests that
//! run under `qemu-aarch64`.
//!
//! The fields are those of FPCR in the Arm Architecture Registers, DDI 0601:
//! FIZ is bit 0, AH is bit 1, the trap enables IOE, DZE, OFE, UFE, and IXE are
//! bits 8 to 12, IDE is bit 15, FZ16 is bit 19, `RMode` is bits 22 and 23, FZ
//! is bit 24, and AHP is bit 26.

use core::arch::asm;

/// The values of FPCR that change a result of the host unit, or trap. Each
/// has one field set.
pub const FPCR_SETTINGS: [u64; 15] = [
    1 << 22,
    2 << 22,
    3 << 22,
    1 << 24,
    1 << 19,
    1,
    1 << 1,
    1 << 8,
    1 << 9,
    1 << 10,
    1 << 11,
    1 << 12,
    1 << 15,
    (1 << 24) | (1 << 22),
    1 << 26,
];

/// Runs `body` with FPCR set to `control`, and restores FPCR afterward.
///
/// Rust assumes the default floating-point environment, so `body` must not
/// depend on its own host floating-point arithmetic. The tests use it to show
/// that floaty reads FPCR before its host paths.
pub fn with_fpcr<T>(control: u64, body: impl FnOnce() -> T) -> T {
    let saved: u64;
    // SAFETY: MRS reads FPCR, and MSR writes `control` to it. The code changes
    // no memory and no other state.
    unsafe {
        asm!(
            "mrs {saved}, fpcr",
            "msr fpcr, {control}",
            saved = out(reg) saved,
            control = in(reg) control,
            options(nomem, nostack, preserves_flags),
        );
    }
    let result = body();
    // SAFETY: MSR restores the saved FPCR value.
    unsafe {
        asm!(
            "msr fpcr, {saved}",
            saved = in(reg) saved,
            options(nomem, nostack, preserves_flags),
        );
    }
    result
}
