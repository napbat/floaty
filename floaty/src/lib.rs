//! Bit-exact, platform-independent software floating point.
//!
//! `floaty` emulates binary, decimal, and double-double floating-point formats
//! in software. Every operation gives the same bits on every host. The
//! binary32 and binary64 operators use the host unit on x86-64 only where it
//! gives those bits.
//!
//! A value has the type [`Float<S, W>`](Float): a standard `S` at a width of
//! `W` bits. Type aliases name the common formats, for example [`F32`],
//! [`BF16`], [`F8E4M3`], [`F80`], and [`D64Bid`]. A double-double value has
//! the type [`DoubleDouble<Alg>`](DoubleDouble), with the arithmetic of one
//! reference implementation.
//!
//! ```
//! use floaty::{Class, F8E4M3};
//!
//! // OCP FP8 E4M3 has no infinity: 0x7F is its NaN.
//! assert_eq!(F8E4M3::from_bits(0x7F).classify(), Class::QuietNan);
//! assert_eq!(F8E4M3::from_bits(0x7E).classify(), Class::Normal);
//! ```
//!
//! The operators `+`, `-`, `*`, and `/` round with the default mode of the
//! type. Each operation also has a `_with` form that returns the [`Flags`].
//! It takes a [`Rounding`], an [`Env`] chosen at run time, or a mode fixed at
//! compile time, such as [`mode::X86Sse`]:
//!
//! ```
//! use floaty::{F32, Flags, Rounding};
//!
//! let (one, tiny) = (F32::from_bits(0x3F80_0000), F32::from_bits(0x3380_0000));
//! assert_eq!((one + tiny).to_bits(), 0x3F80_0000);
//! let (sum, flags) = one.add_with(tiny, Rounding::TowardPositive);
//! assert_eq!(sum.to_bits(), 0x3F80_0001);
//! assert!(flags.contains(Flags::INEXACT));
//! ```
//!
//! [`Float::round`] rounds an exact value to a format, and
//! [`Float::convert`] converts between formats.
//!
//! The crate follows the build order in `DESIGN.md` at the repository root.

#![no_std]

pub mod env;
pub mod format;

mod binary;
mod decimal;
mod double_double;
mod exact;
mod float;
mod host;
mod integer;
mod lanes;
mod limbs;
mod nan;
mod radix;
mod unpacked;

pub use double_double::{Algorithm, DoubleDouble, Gcc, Qd};
pub use env::{Env, Flags, Rounding, TotalOrder, mode};
pub use exact::Exact;
pub use float::{
    BF16, Class, D32Bid, D32Dpd, D64Bid, D64Dpd, D128Bid, D128Dpd, Decoded, F8E4M3, F8E4M3Fnuz,
    F8E5M2, F8E5M2Fnuz, F16, F32, F64, F80, F128, F160, F192, F224, F256, F288, F320, F352, F384,
    F416, F448, F480, F512, Float, FloatType, TF32,
};
pub use format::{Bid, Binary, Decimal, Dpd, Fnuz, Ieee, NoInf, X87};
pub use integer::{Int, Integer, ToInt, UInt};
pub use lanes::Lanes;

mod sealed {
    /// Prevents implementations of a trait outside the crate.
    pub trait Sealed {}
}
