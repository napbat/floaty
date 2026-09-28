//! Verification harness for `floaty`.
//!
//! The harness compares `floaty` against established reference
//! implementations. `DESIGN.md` at the repository root lists which reference
//! checks which format. The tests live in `tests/`, and this library holds
//! what the tests share.
//!
//! The harness is not a default workspace member, because its references
//! need a C toolchain. Run it with `cargo test -p floaty-verify`.

pub mod apfloat;
pub mod encodings;
pub mod mpfr;
pub mod random;
pub mod shape;
