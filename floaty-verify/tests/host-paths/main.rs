//! Checks that every entry point of a host path gives the result of its
//! `_with` method in the default mode, on every target.
//!
//! The other tests compare the `_with` methods, which always run the engine,
//! with the oracles. The engine gives the same bits on every host. So on
//! AArch64 and s390x, where the C references do not build, this comparison
//! checks each host path against those oracles. In a build without a host
//! path, the entry points take the engine.
//!
//! - [`arithmetic`]: the operators, `sqrt`, and `mul_add`.
//! - [`double_double`]: the operators and `sqrt` of the double-double
//!   algorithms.
//! - [`conversions`]: `convert`, `to_int`, and `from_int`.
//! - [`integral`]: `round_to_integral`, in the default mode and in each
//!   rounding direction.
//! - [`comparisons`]: the comparison, the minimum, and the maximum.
//! - [`remainder`]: the remainder.
//! - [`kernels`]: the slice kernels of `Lanes`, in every kind of vector
//!   and lane count, and the slice conversion.
//! - [`elementwise`]: the elementwise views, stores, integer conversions,
//!   and reductions of `Lanes`, and the slice conversion of host values.
//! - [`blocks`]: `map` and `evaluate` of chains of every step, in binary16,
//!   bfloat16, binary32, and binary64, and their check of MXCSR on x86-64.
//! - [`host_path`]: `host_path` of each format, mode, and control of the
//!   environment of the host unit.
//! - [`levels`]: the runs of these tests in each instruction set that the
//!   slice paths of `Lanes` and blocks can select on the processor.
//!
//! [`operands`] gives the operands of every check.

mod arithmetic;
mod blocks;
mod comparisons;
mod conversions;
mod double_double;
mod elementwise;
mod host_path;
mod integral;
mod kernels;
mod levels;
mod operands;
mod remainder;
