//! The run-time selection of the instruction set of the slice kernels, the
//! elementwise slice operations, `convert_chunks`, and blocks, which run a
//! `Task` in the set.
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
use super::super::{Host, Isa, Kind};
use super::{Build, available};
use crate::env::{Env, Rounding};

/// Code generic over the instruction set, which `run_task` runs in the set
/// of this processor. A closure cannot be generic, so a type holds the
/// arguments of the code.
pub trait Task {
    /// The result of the code.
    type Output;

    /// Runs the code in the instruction set `I`. The code calls `I::run` to
    /// take the features of the set.
    fn run<I: Isa>(self) -> Self::Output;
}

/// Runs `task` in the instruction set of this processor.
#[inline]
pub fn run_task<T: Task>(task: T) -> T::Output {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    if let Some(selected) = x86::selected() {
        return selected.run_task(task);
    }
    task.run::<Build>()
}

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
