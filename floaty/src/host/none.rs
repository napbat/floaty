//! The entry points of a build without a host path: each returns `None`.

use core::cmp::Ordering;

use super::{Host, Operation};
use crate::env::Env;
use crate::format::Standard;
use crate::format::internal::MinMax;

/// The block entry points of a build without a host path: each returns
/// `None`.
pub mod block {
    use crate::block::Chain;

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn map<C: Chain<IN, P>, F, const IN: usize, const P: usize>(
        _chain: &C,
        _x: [&[F]; IN],
        _p: [F; P],
        _out: &mut [F],
    ) -> Option<bool> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn evaluate<C: Chain<IN, P>, F, const IN: usize, const P: usize>(
        _chain: &C,
        _x: [F; IN],
        _p: [F; P],
    ) -> Option<F> {
        None
    }
}

/// `false`: this build has no host unit.
pub const UNIT: bool = false;
/// `false`: this build has no fused multiply-add instruction.
pub const FUSED: bool = false;
/// `false`: this build has no binary16 conversion instruction.
pub const HALF: bool = false;
/// `false`: this build has no instruction that rounds to an integral value.
pub const ROUNDING: bool = false;
/// `false`: this build has no x87 unit.
pub const X87: bool = false;
/// `false`: this build has no x87 unit.
pub const X87_FULL_PRECISION: bool = false;
/// `false`: this build has no binary16 fused multiply-add.
pub const HALF_FUSED: bool = false;
/// `false`: this build has no rounding of binary64 to binary16.
pub const DOUBLE_TO_HALF: bool = false;
/// `false`: this build has no rounding to odd.
pub const ROUND_TO_ODD: bool = false;
/// `false`: this build has no binary128 unit.
pub const QUAD: bool = false;

/// How a scalar host path reads the environment of its unit: this build
/// has no host unit, so every path reads nothing.
#[derive(Clone, Copy, Debug)]
pub enum Unit {
    /// The host path reads the environment of its unit.
    Read,
}

impl Unit {
    /// Returns [`Unit::Read`]: this build has no host unit to read.
    #[must_use]
    #[inline]
    pub fn read(_host: Host) -> Self {
        Self::Read
    }
}

/// Returns `to`: this build has no host unit.
#[must_use]
#[inline]
pub const fn conversion_unit(_from: Host, to: Host) -> Host {
    to
}

/// Returns `None`: this build has no host path.
#[inline]
pub fn compare<S: Standard<W>, const W: usize>(
    _left: S::Bits,
    _right: S::Bits,
    _env: &Env,
    _unit: Unit,
) -> Option<Ordering> {
    None
}

/// Returns `None`: this build has no host path.
#[inline]
pub fn remainder<S: Standard<W>, const W: usize>(
    _dividend: S::Bits,
    _divisor: S::Bits,
    _env: &Env,
    _unit: Unit,
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
    _unit: Unit,
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
    _unit: Unit,
) -> Option<S::Bits> {
    None
}

/// Returns `None`: this build has no host path.
#[inline]
pub fn sqrt<S: Standard<W>, const W: usize>(
    _value: S::Bits,
    _env: &Env,
    _unit: Unit,
) -> Option<S::Bits> {
    None
}

/// Returns `None`: this build has no host path.
#[inline]
pub fn mul_add<S: Standard<W>, const W: usize>(
    _left: S::Bits,
    _right: S::Bits,
    _addend: S::Bits,
    _env: &Env,
    _unit: Unit,
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
    _unit: Unit,
) -> Option<S::Bits> {
    None
}

/// Returns `None`: this build has no host path.
#[inline]
pub fn to_int<S: Standard<W>, const W: usize>(
    _value: S::Bits,
    _env: &Env,
    _unit: Unit,
) -> Option<i64> {
    None
}

/// Returns `None`: this build has no host path.
#[inline]
pub fn from_int<S: Standard<W>, const W: usize>(
    _value: i64,
    _env: &Env,
    _unit: Unit,
) -> Option<S::Bits> {
    None
}

/// Returns `None`: this build has no host path.
#[inline]
pub fn convert(
    _from: Host,
    _to: Host,
    _bits: [u64; 2],
    _env: &Env,
    _unit: Unit,
) -> Option<[u64; 2]> {
    None
}

/// The packed entry points of a build without a host path: each returns
/// `None`.
pub mod packed {
    use core::cmp::Ordering;

    use crate::env::{Env, Mode};
    use crate::float::{Float, FloatType};
    use crate::format::Standard;
    use crate::format::internal::MinMax;
    use crate::host::{Direction, Host, Isa, Kind, Load, Operation, Step, Term};

    /// Returns `false`: this build has no host path.
    #[must_use]
    pub const fn available(_host: Host, _kind: Kind) -> bool {
        false
    }

    /// Returns `false`: this build has no host path.
    #[inline]
    pub fn fused_kernels() -> bool {
        false
    }

    /// Returns `"build"`: this build has no host path, and so no
    /// instruction set beyond the build.
    #[cfg(feature = "override-host-level")]
    #[must_use]
    pub fn level() -> &'static str {
        "build"
    }

    /// Returns `"build"` alone, as `level` does.
    #[cfg(feature = "override-host-level")]
    pub fn levels() -> impl Iterator<Item = &'static str> {
        core::iter::once("build")
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
    pub const fn convertible(_from: Host, _to: Host) -> bool {
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
    pub fn convert_chunks<S: Standard<W>, const W: usize, M: Mode, T: FloatType, const N: usize>(
        _values: &[Float<S, W, M>],
        _out: &mut [T],
        _env: &Env,
        _fallback: impl FnMut(&[Float<S, W, M>], &mut [T]),
    ) -> Option<()> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn to_int_i32<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
        _value: &[Float<S, W, M>; N],
        _env: &Env,
    ) -> Option<[Option<i32>; N]> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn accumulate<const N: usize>(
        _x: impl Load,
        _y: impl Load,
        _term: Term,
        _step: Step,
        _env: &Env,
    ) -> Option<u32> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn accumulate_rows<const N: usize>(
        _rows: impl Load,
        _query: impl Load,
        _row_count: usize,
        _term: Term,
        _env: &Env,
        _each: impl FnMut(usize, Option<u32>),
    ) -> Option<()> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn widen_halves<I: Isa, const N: usize>(_halves: &[u16; N]) -> Option<[u32; N]> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn widen_codes<I: Isa, const N: usize>(_codes: &[u8; N]) -> Option<[u32; N]> {
        None
    }

    /// Returns `None`: this build has no host path.
    #[inline]
    pub fn widen_scaled_codes<I: Isa, const N: usize, D: Direction>(
        _codes: &[i8; N],
        _scale: u32,
        _direction: D,
    ) -> Option<[u32; N]> {
        None
    }

    /// The elementwise entry points of a build without a host path:
    /// each returns `None`.
    pub mod elementwise {
        use crate::env::{Env, Rounding};
        use crate::format::internal::MinMax;
        use crate::host::{Direction, Isa, Load, Operation};

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn binary<I: Isa, const C: usize, D: Direction>(
            _x: [u32; C],
            _y: [u32; C],
            _operation: Operation,
            _direction: D,
        ) -> Option<[u32; C]> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn min_max<I: Isa, const C: usize>(
            _x: [u32; C],
            _y: [u32; C],
            _operation: MinMax,
        ) -> Option<[u32; C]> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn round_to_integral<I: Isa, const C: usize>(
            _x: [u32; C],
            _rounding: Rounding,
        ) -> Option<[u32; C]> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn store<const N: usize>(
            _values: impl Load,
            _env: &Env,
            _each: impl FnMut(usize, usize, Option<&[u32]>),
        ) -> Option<()> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn to_int<const N: usize>(
            _values: impl Load,
            _env: &Env,
            _each: impl FnMut(usize, usize, Option<&[i32]>),
        ) -> Option<()> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn reduce<const N: usize>(
            _values: impl Load,
            _operation: MinMax,
            _env: &Env,
        ) -> Option<u32> {
            None
        }
    }
}
