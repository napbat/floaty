//! Checks the R11G11B10 recipe of the README against Mesa 25.2.0, which
//! converts the unsigned floats of that format as `GL_EXT_packed_float` says.
//! Mesa rounds to nearest even and keeps subnormal values. It gives the
//! largest finite value for a larger finite value, positive infinity for
//! positive infinity, zero for a negative value or negative infinity, and one
//! NaN for every NaN.
//!
//! The recipe converts binary32 to `Float<Binary<5>, 12>` for an 11-bit
//! channel, or to `Float<Binary<5>, 11>` for a 10-bit channel, with
//! saturation. Saturation also clamps an infinity, so the recipe maps
//! positive infinity to the infinity of the channel first. It maps a negative
//! result to zero and a NaN to the NaN of the channel, and drops the sign
//! bit. The conversion to binary32 reads the
//! channel with a zero sign bit, and is exact.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::format::{Standard, Storage, Width};
use floaty::{Binary, Env, F32, Float};
use floaty_verify::mesa;

/// The NaN that Mesa gives an 11-bit channel.
const ELEVEN_NAN: u16 = 0x7C1;

/// The NaN that Mesa gives a 10-bit channel.
const TEN_NAN: u16 = 0x3E1;

/// The positive infinity of an 11-bit channel.
const ELEVEN_INFINITY: u16 = 0x7C0;

/// The positive infinity of a 10-bit channel.
const TEN_INFINITY: u16 = 0x3E0;

/// Returns the channel of a binary32 value, by the recipe: positive infinity
/// for positive infinity, the signed format of width `W` with saturation,
/// zero for a negative result, and `nan` for a NaN. `infinity` is the
/// positive infinity of the channel.
fn channel<const W: usize>(value: F32, nan: u16, infinity: u16) -> u16
where
    Width<W>: Storage<Bits = u16>,
    Binary<5>: Standard<W, Bits = u16>,
{
    if value.is_infinite() && !value.is_sign_negative() {
        return infinity;
    }
    let (signed, _) = value.convert_with::<Float<Binary<5>, W>>(Env::IEEE.with_saturate(true));
    if signed.is_nan() {
        nan
    } else if signed.is_sign_negative() {
        0
    } else {
        signed.to_bits()
    }
}

/// Returns the binary32 value of a channel, by the recipe.
fn value<const W: usize>(channel: u16) -> F32
where
    Width<W>: Storage<Bits = u16>,
    Binary<5>: Standard<W, Bits = u16>,
{
    Float::<Binary<5>, W>::from_bits(channel).convert()
}

/// Returns binary32 encodings that reach every rounding case of both
/// channels: each pattern of the top 16 bits, with low bits that make the
/// dropped part zero, a tie, or just below or above one.
fn encodings() -> impl Iterator<Item = u32> {
    (0..=u32::from(u16::MAX)).flat_map(|high| {
        [0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF].map(|low| high << 16 | low)
    })
}

#[test]
fn binary32_converts_to_each_channel_as_mesa_does() {
    let mut count = 0_usize;
    for bits in encodings() {
        let value = F32::from_bits(bits);
        assert_eq!(
            channel::<12>(value, ELEVEN_NAN, ELEVEN_INFINITY),
            mesa::to_eleven(bits),
            "{bits:#010x} to 11 bits"
        );
        assert_eq!(
            channel::<11>(value, TEN_NAN, TEN_INFINITY),
            mesa::to_ten(bits),
            "{bits:#010x} to 10 bits"
        );
        count += 1;
    }
    assert_eq!(count, 6 << 16);
}

#[test]
fn each_channel_converts_to_binary32_as_mesa_does() {
    for code in 0..0x800_u16 {
        let (ours, theirs) = (value::<12>(code), mesa::from_eleven(code));
        if ours.is_nan() {
            // Mesa keeps the channel bits as the payload of a signaling NaN.
            // The recipe converts a NaN as floaty converts every NaN.
            assert!(F32::from_bits(theirs).is_nan(), "{code:#05x}");
        } else {
            assert_eq!(ours.to_bits(), theirs, "{code:#05x}");
        }
    }
    for code in 0..0x400_u16 {
        let (ours, theirs) = (value::<11>(code), mesa::from_ten(code));
        if ours.is_nan() {
            assert!(F32::from_bits(theirs).is_nan(), "{code:#05x}");
        } else {
            assert_eq!(ours.to_bits(), theirs, "{code:#05x}");
        }
    }
}
