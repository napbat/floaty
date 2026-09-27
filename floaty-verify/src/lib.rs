//! Verification harness for `floaty`.
//!
//! The harness compares `floaty` against established reference
//! implementations: Berkeley TestFloat, MPFR, the host processor, the decTest
//! vectors, Intel's decimal library tests, QD, and libgcc. `DESIGN.md` at the
//! repository root lists which reference checks which format.
//!
//! The harness is not a default workspace member, because the reference
//! libraries need a C toolchain. Run it with `cargo test -p floaty-verify`.
