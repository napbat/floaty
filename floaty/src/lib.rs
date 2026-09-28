//! Bit-exact, platform-independent software floating point.
//!
//! `floaty` emulates binary, decimal, and double-double floating-point formats
//! in software. Every operation gives the same bits on every host.
//!
//! A value has the type [`Float<S, W>`](Float): a standard `S` at a width of
//! `W` bits. Type aliases name the common formats, for example [`F32`],
//! [`BF16`], [`F8E4M3`], and [`F80`].
//!
//! ```
//! use floaty::{Class, F8E4M3};
//!
//! // OCP FP8 E4M3 has no infinity: 0x7F is its NaN.
//! assert_eq!(F8E4M3::from_bits(0x7F).classify(), Class::QuietNan);
//! assert_eq!(F8E4M3::from_bits(0x7E).classify(), Class::Normal);
//! ```
//!
//! [`Float::round`] rounds an exact value to a format, and
//! [`Float::convert`] converts between formats. Each type has a default mode,
//! and a call can override its [`Env`] and get the [`Flags`] back.
//!
//! The crate follows the build order in `DESIGN.md` at the repository root.
//! Arithmetic comes in later steps.

#![no_std]

pub mod env;
pub mod format;

mod binary;
mod exact;
mod float;
mod limbs;

pub use env::{Env, Flags, Rounding, mode};
pub use exact::Exact;
pub use float::{
    BF16, Class, Decoded, F8E4M3, F8E4M3Fnuz, F8E5M2, F8E5M2Fnuz, F16, F32, F64, F80, F128, F160,
    F192, F224, F256, F288, F320, F352, F384, F416, F448, F480, F512, Float, FloatType, TF32,
};
pub use format::{Binary, Fnuz, Ieee, NoInf, X87};

mod sealed {
    /// Prevents implementations of a trait outside the crate.
    pub trait Sealed {}
}
