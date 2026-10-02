//! The floating-point-control register of s390x, FPC, for the tests that run
//! under `qemu-s390x`.
//!
//! The fields are those of the FPC register in the z/Architecture Principles
//! of Operation, SA22-7832-13, chapter 9, with bit 0 the most significant: the
//! IEEE masks of invalid operation, division by zero, overflow, underflow,
//! and inexact are bits 0 to 4, and the binary-floating-point rounding mode is
//! bits 29 to 31: 1 toward zero, 2 toward positive infinity, and 3 toward
//! negative infinity.

use core::arch::asm;

/// The values of the FPC register that change a result of the host unit, or
/// trap. Each has one field set, but the last, which sets two.
pub const FPC_SETTINGS: [u32; 9] = [
    1,
    2,
    3,
    1 << 31,
    1 << 30,
    1 << 29,
    1 << 28,
    1 << 27,
    (1 << 29) | 2,
];

/// Runs `body` with the FPC register set to `control`, and restores the FPC
/// register afterward.
///
/// Rust assumes the default floating-point environment, so `body` must not
/// depend on its own host floating-point arithmetic. The tests use it to show
/// that floaty reads the FPC register before its host paths.
pub fn with_fpc<T>(control: u32, body: impl FnOnce() -> T) -> T {
    let saved: u32;
    // SAFETY: EFPC reads the FPC register, and SFPC writes `control` to it.
    // The code changes no memory and no other state.
    unsafe {
        asm!(
            "efpc {saved}",
            "sfpc {control}",
            saved = out(reg) saved,
            control = in(reg) control,
            options(nomem, nostack, preserves_flags),
        );
    }
    let result = body();
    // SAFETY: SFPC restores the saved FPC value.
    unsafe {
        asm!(
            "sfpc {saved}",
            saved = in(reg) saved,
            options(nomem, nostack, preserves_flags),
        );
    }
    result
}

/// Returns a runner that runs a body with the FPC register set to `control`,
/// by [`with_fpc`], for the `assert_*_under` functions of
/// [`crate::entry_points`].
pub fn under_fpc<T>(control: u32) -> impl FnOnce(&dyn Fn() -> T) -> T {
    move |body| with_fpc(control, body)
}
