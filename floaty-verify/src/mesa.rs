//! Runs Mesa's conversions of the unsigned floats of R11G11B10 as an oracle.
//!
//! `build.rs` fetches `format_r11g11b10f.h` and `rounding.h` of Mesa 25.2.0,
//! and the shim `shim/mesa_r11g11b10.c` exposes the inline functions of the
//! header. The conversions follow GL_EXT_packed_float. Each function takes
//! and returns encodings.

unsafe extern "C" {
    fn floaty_mesa_f32_to_uf11(bits: u32) -> u32;
    fn floaty_mesa_f32_to_uf10(bits: u32) -> u32;
    fn floaty_mesa_uf11_to_f32(channel: u16) -> u32;
    fn floaty_mesa_uf10_to_f32(channel: u16) -> u32;
}

/// Returns the 11-bit channel of a binary32 encoding.
#[must_use]
pub fn to_eleven(bits: u32) -> u16 {
    // SAFETY: the shim function takes an encoding by value and reads no
    // memory.
    let channel = unsafe { floaty_mesa_f32_to_uf11(bits) };
    u16::try_from(channel).expect("an 11-bit channel fits 16 bits")
}

/// Returns the 10-bit channel of a binary32 encoding.
#[must_use]
pub fn to_ten(bits: u32) -> u16 {
    // SAFETY: the shim function takes an encoding by value and reads no
    // memory.
    let channel = unsafe { floaty_mesa_f32_to_uf10(bits) };
    u16::try_from(channel).expect("a 10-bit channel fits 16 bits")
}

/// Returns the binary32 encoding of an 11-bit channel.
#[must_use]
pub fn from_eleven(channel: u16) -> u32 {
    // SAFETY: the shim function takes a channel by value and reads no memory.
    unsafe { floaty_mesa_uf11_to_f32(channel) }
}

/// Returns the binary32 encoding of a 10-bit channel.
#[must_use]
pub fn from_ten(channel: u16) -> u32 {
    // SAFETY: the shim function takes a channel by value and reads no memory.
    unsafe { floaty_mesa_uf10_to_f32(channel) }
}
