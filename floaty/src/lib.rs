//! Bit-exact, platform-independent software floating point.
//!
//! `floaty` emulates binary, decimal, and double-double floating-point formats
//! in software. Every operation gives the same bits on every host.
//!
//! The crate is in its design phase and has no public API yet. The private
//! modules below hold the layers that `DESIGN.md` at the repository root
//! describes. Each layer arrives in the build order that `DESIGN.md` states.

#![no_std]

mod binary;
mod env;
mod exact;
mod format;
mod limbs;
