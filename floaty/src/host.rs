//! The host paths: operations on the floating-point unit of the host that give
//! the bits of the engine. The README lists each path.
//!
//! The build selects the paths at compile time, from the target architecture
//! and its features. One module in `host/` reads the floating-point
//! environment of each architecture. A build for another architecture, or with
//! `--cfg floaty_engine_only`, has no host path, and each entry point returns
//! `None`.
//!
//! A path serves only entry points that return no flags. The mode must round
//! to nearest even without FTZ, DAZ, or a precision limit below the format
//! precision, and the environment of the host must match it. The rounding to
//! an integral value also takes a directed rounding, which its instructions
//! take from their encoding. A NaN result goes back to the engine, which
//! selects the NaN by the rule of the mode.

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
pub use self::paths::{
    Ready, binary, compare, convert, from_int, min_max, mul_add, ready, remainder,
    round_to_integral, sqrt, to_int,
};

#[cfg(not(floaty_engine_only))]
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64"
))]
mod narrow;

#[cfg(not(floaty_engine_only))]
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64"
))]
pub mod packed;

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
    /// The rounding to an integral value.
    RoundToIntegral,
    /// The conversion to an integer, to nearest even.
    ToInt,
    /// The conversion from an integer.
    FromInt,
    /// The comparison, and the minimum and maximum operations.
    Comparison,
    /// The IEEE 754 remainder.
    Remainder,
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
    let rounding = cfg!(any(
        all(target_arch = "x86_64", target_feature = "sse4.1"),
        target_arch = "aarch64"
    ));
    // Every x86-64 processor has the x87 unit.
    let x87 = cfg!(target_arch = "x86_64");
    // `FEAT_FP16` computes binary16 in its own precision.
    let fp16 = cfg!(all(target_arch = "aarch64", target_feature = "fp16"));
    match (host, kind) {
        // The x87 unit has no fused multiply-add. Two roundings of a bfloat16
        // fused multiply-add, or of an integer, through binary32 can differ
        // from one. Only the x87 unit has a remainder instruction, and it
        // loads binary32, binary64, and x87 extended values. binary16 and
        // bfloat16 values widen to binary32 exactly for it.
        (Host::None, _)
        | (Host::Extended, Kind::FusedMultiplyAdd | Kind::Comparison)
        | (Host::BFloat, Kind::FusedMultiplyAdd | Kind::FromInt) => false,
        (Host::Half, Kind::FusedMultiplyAdd) => unit && fp16,
        // A bfloat16 value widens to binary32 by a shift, and a binary32
        // result rounds to bfloat16 in integer instructions, so the bfloat16
        // paths need no bfloat16 instruction.
        (
            Host::Single | Host::Double,
            Kind::Arithmetic | Kind::SquareRoot | Kind::ToInt | Kind::FromInt | Kind::Comparison,
        )
        | (Host::BFloat, Kind::Arithmetic | Kind::SquareRoot | Kind::ToInt | Kind::Comparison) => {
            unit
        }
        (Host::Single | Host::Double, Kind::FusedMultiplyAdd) => unit && fused,
        (Host::Single | Host::Double | Host::BFloat, Kind::RoundToIntegral) => unit && rounding,
        (Host::Half, Kind::RoundToIntegral) => unit && half && rounding,
        // binary16 widens to binary32 and rounds back in F16C or `FCVT`, or
        // else in integer instructions. The engine compares binary16 values
        // faster than the integer widening does, so a comparison takes the
        // path only with a widening instruction.
        (Host::Half, Kind::Arithmetic | Kind::SquareRoot | Kind::ToInt | Kind::FromInt) => unit,
        (Host::Half, Kind::Comparison) => unit && half,
        (
            Host::Extended,
            Kind::Arithmetic
            | Kind::SquareRoot
            | Kind::RoundToIntegral
            | Kind::ToInt
            | Kind::FromInt
            | Kind::Remainder,
        )
        | (Host::Single | Host::Double | Host::Half | Host::BFloat, Kind::Remainder) => unit && x87,
    }
}

/// Returns `true` when this build has a host path that converts a format of
/// the host kind `from` to one of the host kind `to`. The answer is a
/// constant, so a conversion without a host path compiles to the engine.
#[must_use]
pub const fn convertible(from: Host, to: Host) -> bool {
    let unit = !cfg!(floaty_engine_only)
        && cfg!(any(
            all(target_arch = "x86_64", target_feature = "sse2"),
            target_arch = "aarch64"
        ));
    let x87 = cfg!(target_arch = "x86_64");
    match (from, to) {
        // A shift widens bfloat16 to binary32 exactly, and integer
        // instructions round binary32 to bfloat16. binary16 widens and rounds
        // in F16C or `FCVT`, or else in integer instructions.
        (Host::Single, Host::Double | Host::BFloat | Host::Half)
        | (Host::Double, Host::Single)
        | (Host::BFloat | Host::Half, Host::Single | Host::Double) => unit,
        // Only AArch64 rounds binary64 to binary16 once.
        (Host::Double, Host::Half) => unit && cfg!(target_arch = "aarch64"),
        (Host::Single | Host::Double, Host::Extended)
        | (Host::Extended, Host::Single | Host::Double) => unit && x87,
        _ => false,
    }
}

#[cfg(any(
    floaty_engine_only,
    not(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        target_arch = "aarch64"
    ))
))]
pub use self::none::{
    Ready, binary, compare, convert, from_int, min_max, mul_add, packed, ready, remainder,
    round_to_integral, sqrt, to_int,
};

/// The entry points of a build without a host path: each returns `None`.
#[cfg(any(
    floaty_engine_only,
    not(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        target_arch = "aarch64"
    ))
))]
mod none {
    use core::cmp::Ordering;

    use super::Operation;
    use crate::env::Env;
    use crate::format::Standard;
    use crate::format::internal::{Host, MinMax};

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn compare<S: Standard<W>, const W: usize>(
        _left: S::Bits,
        _right: S::Bits,
        _env: &Env,
    ) -> Option<Ordering> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn remainder<S: Standard<W>, const W: usize>(
        _dividend: S::Bits,
        _divisor: S::Bits,
        _env: &Env,
    ) -> Option<S::Bits> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn min_max<S: Standard<W>, const W: usize>(
        _left: S::Bits,
        _right: S::Bits,
        _operation: MinMax,
        _env: &Env,
    ) -> Option<S::Bits> {
        None
    }

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

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn round_to_integral<S: Standard<W>, const W: usize>(
        _value: S::Bits,
        _env: &Env,
    ) -> Option<S::Bits> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn to_int<S: Standard<W>, const W: usize>(_value: S::Bits, _env: &Env) -> Option<i64> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn from_int<S: Standard<W>, const W: usize>(_value: i64, _env: &Env) -> Option<S::Bits> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn convert(_from: Host, _to: Host, _bits: [u64; 2], _env: &Env) -> Option<[u64; 2]> {
        None
    }

    /// The packed entry points of a build without a host path: each returns
    /// `None`.
    pub mod packed {
        use core::cmp::Ordering;

        use crate::env::{Env, Mode};
        use crate::float::Float;
        use crate::format::Standard;
        use crate::format::internal::{Host, MinMax};
        use crate::host::{Kind, Operation};

        /// Returns `false`: this build has no host path.
        #[must_use]
        pub const fn lanes_of(_host: Host, _kind: Kind) -> bool {
            false
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn compare<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
            _left: &[Float<S, W, M>; N],
            _right: &[Float<S, W, M>; N],
            _env: &Env,
        ) -> Option<[Option<Ordering>; N]> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn min_max<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
            _left: &[Float<S, W, M>; N],
            _right: &[Float<S, W, M>; N],
            _operation: MinMax,
            _env: &Env,
        ) -> Option<[Float<S, W, M>; N]> {
            None
        }

        /// Returns `false`: this build has no host path.
        #[must_use]
        pub const fn converts(_from: Host, _to: Host) -> bool {
            false
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn binary<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
            _left: &[Float<S, W, M>; N],
            _right: &[Float<S, W, M>; N],
            _operation: Operation,
            _env: &Env,
        ) -> Option<[Float<S, W, M>; N]> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn sqrt<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
            _value: &[Float<S, W, M>; N],
            _env: &Env,
        ) -> Option<[Float<S, W, M>; N]> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn round_to_integral<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
            _value: &[Float<S, W, M>; N],
            _env: &Env,
        ) -> Option<[Float<S, W, M>; N]> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn mul_add<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
            _left: &[Float<S, W, M>; N],
            _right: &[Float<S, W, M>; N],
            _addend: &[Float<S, W, M>; N],
            _env: &Env,
        ) -> Option<[Float<S, W, M>; N]> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn convert<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
            _value: &[Float<S, W, M>; N],
            _to: Host,
            _env: &Env,
        ) -> Option<[u64; N]> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn to_int_i32<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
            _value: &[Float<S, W, M>; N],
            _env: &Env,
        ) -> Option<[i32; N]> {
            None
        }
    }
}
