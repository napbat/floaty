//! The host paths: operations on the floating-point unit of the host that give
//! the bits of the engine. `DESIGN.md` lists each path and its rules.
//!
//! The build selects the paths at compile time, from the target architecture
//! and its features. One module in `host/` reads the floating-point
//! environment of each architecture. A build for another architecture, or with
//! `--cfg floaty_engine_only`, has no host path, and each entry point returns
//! `None`.
//!
//! A path serves only entry points that return no flags. The mode must round
//! to nearest even without FTZ, DAZ, or a precision limit below the format
//! precision, and the environment of the host must match it. A NaN result goes
//! back to the engine, which selects the NaN by the rule of the mode.

use crate::format::internal::Host;

#[cfg(not(floaty_engine_only))]
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64"
))]
mod paths;
#[cfg(not(floaty_engine_only))]
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64"
))]
pub use self::paths::{Ready, binary, mul_add, ready, sqrt};

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(floaty_engine_only)
))]
mod x86_64;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(floaty_engine_only)
))]
use self::x86_64 as environment;

#[cfg(all(target_arch = "aarch64", not(floaty_engine_only)))]
mod aarch64;
#[cfg(all(target_arch = "aarch64", not(floaty_engine_only)))]
use self::aarch64 as environment;

/// An arithmetic operation of a host path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
}

/// The kind of a host path, for [`available`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The operators `+`, `-`, `*`, and `/`.
    Arithmetic,
    /// The square root.
    SquareRoot,
    /// The fused multiply-add.
    FusedMultiplyAdd,
}

/// Returns `true` when this build has a host path of `kind` for a format of
/// the host kind `host`. The answer is a constant, so an entry point without
/// a host path compiles to the engine.
#[must_use]
pub const fn available(host: Host, kind: Kind) -> bool {
    let unit = !cfg!(floaty_engine_only)
        && cfg!(any(
            all(target_arch = "x86_64", target_feature = "sse2"),
            target_arch = "aarch64"
        ));
    let fused = cfg!(any(
        all(target_arch = "x86_64", target_feature = "fma"),
        target_arch = "aarch64"
    ));
    let half = cfg!(any(
        all(target_arch = "x86_64", target_feature = "f16c"),
        target_arch = "aarch64"
    ));
    match (host, kind) {
        // Two roundings of a binary16 fused multiply-add can differ from one.
        (Host::None, _) | (Host::Half, Kind::FusedMultiplyAdd) => false,
        (Host::Single | Host::Double, Kind::Arithmetic | Kind::SquareRoot) => unit,
        (Host::Single | Host::Double, Kind::FusedMultiplyAdd) => unit && fused,
        (Host::Half, Kind::Arithmetic | Kind::SquareRoot) => unit && half,
    }
}

#[cfg(any(
    floaty_engine_only,
    not(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        target_arch = "aarch64"
    ))
))]
pub use self::none::{Ready, binary, mul_add, ready, sqrt};

/// The entry points of a build without a host path: each returns `None`.
#[cfg(any(
    floaty_engine_only,
    not(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        target_arch = "aarch64"
    ))
))]
mod none {
    use super::Operation;
    use crate::env::Env;
    use crate::format::Standard;

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn binary<S: Standard<W>, const W: usize>(
        _left: S::Bits,
        _right: S::Bits,
        _operation: Operation,
        _env: &Env,
    ) -> Option<S::Bits> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn sqrt<S: Standard<W>, const W: usize>(_value: S::Bits, _env: &Env) -> Option<S::Bits> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn mul_add<S: Standard<W>, const W: usize>(
        _left: S::Bits,
        _right: S::Bits,
        _addend: S::Bits,
        _env: &Env,
    ) -> Option<S::Bits> {
        None
    }

    /// Proof that the host paths of binary64 apply: none can exist in this
    /// build.
    #[derive(Clone, Copy, Debug)]
    pub enum Ready {}

    impl Ready {
        /// Never runs: no `Ready` exists in this build.
        pub fn binary(self, _left: u64, _right: u64, _operation: Operation) -> Option<u64> {
            match self {}
        }

        /// Never runs: no `Ready` exists in this build.
        pub fn sqrt(self, _value: u64) -> Option<u64> {
            match self {}
        }

        /// Never runs: no `Ready` exists in this build.
        pub fn mul_add(self, _left: u64, _right: u64, _addend: u64) -> Option<u64> {
            match self {}
        }
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn ready(_env: &Env) -> Option<Ready> {
        None
    }
}
