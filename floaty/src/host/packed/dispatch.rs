//! The run-time selection of the instruction set of the slice kernels, the
//! elementwise slice operations, and `convert_chunks`.
//!
//! Each entry point runs in `Build` unless the processor has a larger
//! instruction set than the build enables. It then runs generic over that
//! set. `Isa::run` puts each block of the generic code in a function
//! compiled with the features of the set, so a block that LLVM does not
//! inline keeps them too. The loop of a block calls no closure of its caller
//! for each chunk: a closure that LLVM does not inline has no features of
//! the set, and the call passes the lanes through memory. So the loop of
//! `convert_chunks` stores each chunk itself. The check of the processor runs
//! once, in a build with the feature `std`, and an atomic keeps its answer.
//! So a call pays one load of the atomic and one branch. A build without
//! `std`, and a build that enables the features of the largest set, takes
//! `Build` with no check. Every instruction set gives the bits of the engine,
//! as `Build` does.
//!
//! x86-64 has two larger sets, as the module `x86` states: x86-64-v3, with
//! AVX2, FMA, and F16C, and x86-64-v4, which adds AVX-512. 32-bit x86 takes
//! the same two sets, whose features it has too. AArch64 has its NEON forms
//! in every build, so it has no larger set. Its SVE has no fixed vector
//! length and no register class in the inline assembly of Rust, so no form
//! uses it. s390x runs the scalar instruction of each lane, as its packed
//! paths do, so it has no larger set either.

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
mod x86;

use super::super::environment::packed;
use super::super::paths::ready_in_direction;
use super::super::{Host, Kind, Load, Step, Term};
use super::{Build, available, convert_chunks_on, elementwise, kernel};
use crate::env::{Env, Mode, Rounding};
use crate::float::{Float, FloatType};
use crate::format::Standard;
use crate::format::internal::MinMax;

/// Returns `true` when the slice kernels with a fused step have a host path
/// on this processor: in the build, or in the instruction set that the
/// processor selects.
#[inline]
pub fn fused_kernels() -> bool {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    if x86::selected().is_some() {
        return true;
    }
    const { available(Host::Single, Kind::FusedMultiplyAdd) }
}

/// Returns the rounding direction of the mode when the mode and the
/// environment allow the slice kernels and the elementwise slice operations
/// of binary32, as `ready_in_direction` states, and this processor rounds in
/// the direction. Every instruction set rounds to nearest even. Only the
/// forms with a rounding control round in the other directions, in the
/// build or in the instruction set that the processor selects. Elsewhere a
/// directed mode takes the engine for the whole call, not for each chunk.
#[inline]
pub fn direction(env: &Env) -> Option<Rounding> {
    let rounding = ready_in_direction(Host::Single, env, Host::Single.precision())?;
    if rounding == Rounding::TiesToEven {
        return Some(rounding);
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    if let Some(selected) = x86::selected() {
        return selected.rounding_control().then_some(rounding);
    }
    packed::ROUNDING_CONTROL.then_some(rounding)
}

/// Returns the name of the instruction set that this processor runs:
/// `build`, `v3`, or `v4`.
#[cfg(feature = "override-host-level")]
#[must_use]
pub fn level() -> &'static str {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(selected) = x86::selected() {
        return selected.name();
    }
    "build"
}

/// Returns the names of the instruction sets that this processor can run,
/// as `level` names them, from the smallest.
#[cfg(feature = "override-host-level")]
pub fn levels() -> impl Iterator<Item = &'static str> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let larger = x86::levels();
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let larger: &[&'static str] = &[];
    core::iter::once("build").chain(larger.iter().copied())
}

/// Runs `accumulate_on` in the instruction set of this processor.
#[inline]
pub fn accumulate<const N: usize>(
    x: impl Load,
    y: impl Load,
    term: Term,
    step: Step,
    rounding: Rounding,
) -> Option<u32> {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    if let Some(selected) = x86::selected() {
        return selected.accumulate::<N>(x, y, term, step, rounding);
    }
    kernel::accumulate_on::<Build, N>(x, y, term, step, rounding)
}

/// Runs `rows_on` in the instruction set of this processor.
#[inline]
pub fn rows<const N: usize>(
    rows: impl Load,
    query: impl Load,
    row_count: usize,
    term: Term,
    rounding: Rounding,
    each: impl FnMut(usize, Option<u32>),
) {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    if let Some(selected) = x86::selected() {
        selected.rows::<N>(rows, query, row_count, term, rounding, each);
        return;
    }
    kernel::rows_on::<Build, N>(rows, query, row_count, term, rounding, each);
}

/// Runs `store_on` in the instruction set of this processor.
#[inline]
pub fn store<const N: usize>(
    values: impl Load,
    rounding: Rounding,
    each: impl FnMut(usize, usize, Option<&[u32]>),
) {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    if let Some(selected) = x86::selected() {
        selected.store::<N>(values, rounding, each);
        return;
    }
    elementwise::store_on::<Build, N>(values, rounding, each);
}

/// Runs `to_int_on` in the instruction set of this processor.
#[inline]
pub fn to_int<const N: usize>(
    values: impl Load,
    rounding: Rounding,
    each: impl FnMut(usize, usize, Option<&[i32]>),
) {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    if let Some(selected) = x86::selected() {
        selected.to_int::<N>(values, rounding, each);
        return;
    }
    elementwise::to_int_on::<Build, N>(values, rounding, each);
}

/// Runs `reduce_on` in the instruction set of this processor.
#[inline]
pub fn reduce<const N: usize>(
    values: impl Load,
    operation: MinMax,
    rounding: Rounding,
) -> Option<u32> {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    if let Some(selected) = x86::selected() {
        return selected.reduce::<N>(values, operation, rounding);
    }
    elementwise::reduce_on::<Build, N>(values, operation, rounding)
}

/// Runs `convert_chunks_on` in the instruction set of this processor.
#[inline]
pub fn convert_chunks<S: Standard<W>, const W: usize, M: Mode, T: FloatType, const N: usize>(
    values: &[Float<S, W, M>],
    out: &mut [T],
    fallback: impl FnMut(&[Float<S, W, M>], &mut [T]),
) {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    if let Some(selected) = x86::selected() {
        selected.convert_chunks::<S, W, M, T, N>(values, out, fallback);
        return;
    }
    convert_chunks_on::<Build, S, W, M, T, N>(values, out, fallback);
}
