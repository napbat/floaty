//! Runs the double-double arithmetic of QD, `dd_real`, as an oracle.
//!
//! `build.rs` builds QD 2.3.24 from its pinned archive, with the IEEE-style
//! addition, the accurate division, and the C `fma` for the error of a
//! product, and with `-O2 -ffp-contract=off`. The shim `shim/qd_shim.cpp`
//! compiles the inline operators of QD with the same options.
//!
//! Each function sets the rounding direction of the calling thread, clears
//! the flags, computes with `dd_real`, and reads the flags with
//! `fetestexcept`. It then restores the rounding direction and the flags that
//! the thread had before the call. A thread that rounds to nearest before a
//! call thus rounds to nearest after it.
//!
//! QD computes on the host: x86-64 SSE for each binary64 step, and the C
//! library `fma`, which [`c_fma`] also calls. glibc's `fetestexcept` does not
//! report the denormal flag (DE of MXCSR), so an [`Outcome`] holds only the
//! five IEEE flags. Compare the five IEEE flags only.
//!
//! QD's `sqrt` of a negative value returns QD's NaN and writes an error
//! message to standard error. The shim discards the message.
//!
//! The value, rounding, and flag types are the types of
//! [`crate::ibm_ldouble`], so one test can compare both references.

use core::ffi::{c_int, c_uint};

pub use crate::ibm_ldouble::{Flags, Outcome, Pair, Rounding};

/// Returns `fma(x, y, z)` of the C library that QD calls, on binary64
/// encodings, rounded to nearest. glibc selects its `fma` at run time, so the
/// result depends on the host processor.
#[must_use]
pub fn c_fma(x: u64, y: u64, z: u64) -> u64 {
    // SAFETY: `fma` takes three values and returns one. It reads no memory.
    let result = unsafe { fma(f64::from_bits(x), f64::from_bits(y), f64::from_bits(z)) };
    result.to_bits()
}

/// The C shape of a binary operation of the shim: the halves of `a`, the
/// halves of `b`, the rounding direction, and the halves of the result. It
/// returns the raised flags.
type BinaryFunction = unsafe extern "C" fn(*const u64, *const u64, c_int, *mut u64) -> c_uint;

unsafe extern "C" {
    fn floaty_qd_add(a: *const u64, b: *const u64, rounding: c_int, result: *mut u64) -> c_uint;
    fn floaty_qd_sub(a: *const u64, b: *const u64, rounding: c_int, result: *mut u64) -> c_uint;
    fn floaty_qd_mul(a: *const u64, b: *const u64, rounding: c_int, result: *mut u64) -> c_uint;
    fn floaty_qd_div(a: *const u64, b: *const u64, rounding: c_int, result: *mut u64) -> c_uint;
    fn floaty_qd_sqrt(a: *const u64, rounding: c_int, result: *mut u64) -> c_uint;
    /// The C library `fma`, which QD calls for the error of a product.
    fn fma(x: f64, y: f64, z: f64) -> f64;
}

/// Returns `a + b` of `dd_real` in `rounding`.
#[must_use]
pub fn add(a: Pair, b: Pair, rounding: Rounding) -> Outcome {
    binary(floaty_qd_add, a, b, rounding)
}

/// Returns `a - b` of `dd_real` in `rounding`.
#[must_use]
pub fn sub(a: Pair, b: Pair, rounding: Rounding) -> Outcome {
    binary(floaty_qd_sub, a, b, rounding)
}

/// Returns `a * b` of `dd_real` in `rounding`.
#[must_use]
pub fn mul(a: Pair, b: Pair, rounding: Rounding) -> Outcome {
    binary(floaty_qd_mul, a, b, rounding)
}

/// Returns `a / b` of `dd_real` in `rounding`.
#[must_use]
pub fn div(a: Pair, b: Pair, rounding: Rounding) -> Outcome {
    binary(floaty_qd_div, a, b, rounding)
}

/// Returns `sqrt(a)` of `dd_real` in `rounding`.
#[must_use]
pub fn sqrt(a: Pair, rounding: Rounding) -> Outcome {
    let a = [a.hi, a.lo];
    let mut result = [0; 2];
    // SAFETY: the function reads the two halves of `a` and writes the two
    // halves of `result`. Both arrays live until it returns. It restores
    // the floating-point environment of this thread before it returns.
    let bits = unsafe { floaty_qd_sqrt(a.as_ptr(), code(rounding), result.as_mut_ptr()) };
    outcome(result, bits)
}

/// Calls a binary operation of the shim.
fn binary(function: BinaryFunction, a: Pair, b: Pair, rounding: Rounding) -> Outcome {
    let a = [a.hi, a.lo];
    let b = [b.hi, b.lo];
    let mut result = [0; 2];
    // SAFETY: the function reads the two halves of `a` and of `b`, and
    // writes the two halves of `result`. The three arrays live until it
    // returns. It restores the floating-point environment of this thread
    // before it returns.
    let bits = unsafe { function(a.as_ptr(), b.as_ptr(), code(rounding), result.as_mut_ptr()) };
    outcome(result, bits)
}

/// Returns the shim code of a rounding direction.
const fn code(rounding: Rounding) -> c_int {
    match rounding {
        Rounding::TiesToEven => 0,
        Rounding::TowardZero => 1,
        Rounding::TowardPositive => 2,
        Rounding::TowardNegative => 3,
    }
}

/// Returns the outcome of the result halves and the flag bits of the shim.
fn outcome(result: [u64; 2], bits: c_uint) -> Outcome {
    let flags = u8::try_from(bits)
        .ok()
        .and_then(Flags::from_bits)
        .expect("the shim knows every rounding direction and returns only the five flag bits");
    Outcome {
        result: Pair::new(result[0], result[1]),
        flags,
    }
}
