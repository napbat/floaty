//! Verification harness for `floaty`.
//!
//! The harness compares `floaty` against established reference
//! implementations. The README at the repository root lists which reference
//! checks which format. The tests live in `tests/`, and this library holds
//! what the tests share.
//!
//! The harness is not a default workspace member, because its references
//! need a C toolchain. Run it with `cargo test -p floaty-verify`.
//!
//! The C references, MPFR, and the x86 hardware build only for x86-64. On
//! another target, the harness has the modules that need none of them, and
//! the tests that use only those.

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
pub mod apfloat;
#[cfg(target_arch = "x86_64")]
pub mod arithmetic;
#[cfg(target_arch = "x86_64")]
pub mod decnumber;
#[cfg(target_arch = "x86_64")]
pub mod dectest;
#[cfg(target_arch = "x86_64")]
pub mod double_double;
pub mod encodings;
pub mod entry_points;
pub mod formats;
#[cfg(target_arch = "x86_64")]
pub mod ibm_ldouble;
#[cfg(target_arch = "x86_64")]
pub mod intel_decimal;
#[cfg(target_arch = "x86_64")]
pub mod libm;
#[cfg(target_arch = "x86_64")]
pub mod mesa;
pub mod ml_dtypes;
pub mod modes;
#[cfg(target_arch = "x86_64")]
pub mod mpfr;
#[cfg(target_arch = "x86_64")]
pub mod operations;
#[cfg(target_arch = "x86_64")]
pub mod qd;
pub mod random;
#[cfg(target_arch = "x86_64")]
pub mod readtest;
#[cfg(target_arch = "x86_64")]
pub mod shape;
#[cfg(target_arch = "x86_64")]
pub mod testfloat;
#[cfg(target_arch = "x86_64")]
pub mod x86;

/// The floaty crate, for the macros of [`modes`] and of the format lists, so
/// that a caller need not name floaty.
#[doc(hidden)]
pub use floaty as __floaty;
