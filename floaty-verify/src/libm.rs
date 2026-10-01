//! Runs the NaN payload functions of the host glibc libm as an oracle.
//!
//! The build script compiles the shim `shim/libm_payload.c`, which calls
//! `getpayload`, `setpayload`, and `setpayloadsig` of C23 annex F.10.13 for
//! `float`, `double`, the x87 `long double`, and `_Float128`. IEEE 754-2019
//! section 9.7 names them getPayload, setPayload, and setPayloadSignaling.
//! glibc added them in release 2.25. Since release 2.32 (glibc bug 26073),
//! `getpayload` gives -1 for a value that is not a NaN, as C23 requires.
//! Each function here takes and returns an encoding.

use core::ffi::c_int;

unsafe extern "C" {
    fn floaty_libm_float(operation: c_int, input: *const u8, output: *mut u8);
    fn floaty_libm_double(operation: c_int, input: *const u8, output: *mut u8);
    fn floaty_libm_long_double(operation: c_int, input: *const u8, output: *mut u8);
    fn floaty_libm_float128(operation: c_int, input: *const u8, output: *mut u8);
}

/// A NaN payload function of libm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Payload {
    /// `getpayload`.
    Get,
    /// `setpayload`.
    Set,
    /// `setpayloadsig`.
    SetSignaling,
}

impl Payload {
    /// Every payload function.
    pub const ALL: [Self; 3] = [Self::Get, Self::Set, Self::SetSignaling];

    /// Returns the operation code of the shim.
    const fn code(self) -> c_int {
        match self {
            Self::Get => 0,
            Self::Set => 1,
            Self::SetSignaling => 2,
        }
    }
}

/// A format of the libm functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    /// `float`, binary32.
    Float,
    /// `double`, binary64.
    Double,
    /// The x87 `long double`.
    LongDouble,
    /// `_Float128`, binary128.
    Float128,
}

/// Runs `function` of `format` on an encoding, held in the low bits of
/// `bits`, and returns the encoding of the result.
#[must_use]
pub fn payload(format: Format, function: Payload, bits: u128) -> u128 {
    let shim = match format {
        Format::Float => floaty_libm_float,
        Format::Double => floaty_libm_double,
        Format::LongDouble => floaty_libm_long_double,
        Format::Float128 => floaty_libm_float128,
    };
    // x86-64 is little-endian, so the bytes are in the order of the host.
    let input = bits.to_le_bytes();
    let mut output = [0_u8; 16];
    // SAFETY: the shim reads at most 16 bytes of `input` and writes 16 bytes
    // of `output`, and both arrays hold 16 bytes.
    unsafe { shim(function.code(), input.as_ptr(), output.as_mut_ptr()) };
    u128::from_le_bytes(output)
}
