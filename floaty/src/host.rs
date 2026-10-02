//! The host paths: operations on the floating-point unit of the host that give
//! the bits of the engine. The README lists each path.
//!
//! The build selects the paths at compile time, from the target architecture
//! and its features. One module in `host/` reads the floating-point
//! environment of each architecture. A build for another architecture, or with
//! `--cfg floaty_engine_only`, has no host path, and each entry point returns
//! `None`.
//!
//! A path serves only entry points that return no flags, in a mode that
//! `compatible` in `paths` accepts. That function names every field of
//! `Env`, and the README states the rule. The environment of the host must
//! match the mode. The rounding to an integral value also takes a directed
//! rounding, which its instructions take from their encoding. A NaN result
//! goes back to the engine, which selects the NaN by the rule of the mode.

use crate::format::EncodingKind;

#[cfg(not(floaty_engine_only))]
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ),
    target_arch = "aarch64",
    target_arch = "s390x"
))]
mod bits;
#[cfg(not(floaty_engine_only))]
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ),
    target_arch = "aarch64",
    target_arch = "s390x"
))]
mod paths;
#[cfg(not(floaty_engine_only))]
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ),
    target_arch = "aarch64",
    target_arch = "s390x"
))]
pub use self::paths::{
    Ready, binary, compare, convert, from_int, min_max, mul_add, ready, remainder,
    round_to_integral, sqrt, to_int,
};

#[cfg(not(floaty_engine_only))]
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ),
    target_arch = "aarch64",
    target_arch = "s390x"
))]
mod narrow;

#[cfg(not(floaty_engine_only))]
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ),
    target_arch = "aarch64",
    target_arch = "s390x"
))]
pub mod packed;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2",
    not(floaty_engine_only)
))]
mod x86;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2",
    not(floaty_engine_only)
))]
use self::x86 as environment;

#[cfg(all(target_arch = "aarch64", not(floaty_engine_only)))]
mod aarch64;
#[cfg(all(target_arch = "aarch64", not(floaty_engine_only)))]
use self::aarch64 as environment;

#[cfg(all(target_arch = "s390x", not(floaty_engine_only)))]
mod s390x;
#[cfg(all(target_arch = "s390x", not(floaty_engine_only)))]
use self::s390x as environment;

// A build without a host unit takes the capabilities of `none`, which are all
// `false`.
#[cfg(any(
    floaty_engine_only,
    not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        ),
        target_arch = "aarch64",
        target_arch = "s390x"
    ))
))]
use self::none as environment;

/// The host format with the same encoding as a floaty format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Host {
    /// No host format.
    None,
    /// The host `f32`: IEEE 754 binary32.
    Single,
    /// The host `f64`: IEEE 754 binary64.
    Double,
    /// IEEE 754 binary16, which computes in the host `f32` and converts in
    /// host instructions. binary32 holds 2p + 2 bits of binary16, so the
    /// two roundings of add, subtract, multiply, divide, and square root
    /// give the correctly rounded result.
    ///
    /// Without `FEAT_FP16`, the fused multiply-add computes in binary64 and
    /// rounds the binary64 sum to binary16. The product of two binary16
    /// values is exact in binary64: it has at most 22 bits, and its lowest
    /// bit weighs at least 2^-48. The addend has at most 11 bits, and its
    /// lowest bit weighs at least 2^-24. So the binary64 sum is inexact only
    /// when its bits span more than 53 places, in two cases. In the first,
    /// the product is below 2^-30 times the addend. The sum then lies within
    /// 2^-29 times the addend of it, and half a unit of binary16 away from the
    /// addend is at least 2^-12 times the addend, so both roundings give the
    /// addend. In the second, the product is at least 2^28, so both
    /// roundings overflow. Every other binary64 sum is exact, and rounds
    /// once.
    Half,
    /// x87 extended precision, which computes on the x87 unit of x86-64
    /// at the 64-bit precision.
    Extended,
    /// IEEE 754 binary128, which computes on the binary floating-point unit
    /// of s390x in pairs of floating-point registers.
    Quad,
    /// bfloat16, which computes in the host `f32` and rounds in a host
    /// instruction. binary32 holds 2p + 2 bits of bfloat16, so the two
    /// roundings of add, subtract, multiply, divide, and square root give
    /// the correctly rounded result.
    BFloat,
}

impl Host {
    /// Returns the precision of the host format, or 0 for `None`, which has
    /// no host path.
    #[must_use]
    pub const fn precision(self) -> u32 {
        match self {
            Self::None => 0,
            Self::BFloat => 8,
            Self::Half => 11,
            Self::Single => 24,
            Self::Double => 53,
            Self::Extended => 64,
            Self::Quad => 113,
        }
    }

    /// Returns the host format of the binary format with `exponent_bits`
    /// exponent bits and `width` bits in the encoding rules `kind`.
    #[must_use]
    pub const fn of_binary(exponent_bits: u32, width: u32, kind: EncodingKind) -> Self {
        match (exponent_bits, width, kind) {
            (5, 16, EncodingKind::Ieee) => Self::Half,
            (8, 16, EncodingKind::Ieee) => Self::BFloat,
            (8, 32, EncodingKind::Ieee) => Self::Single,
            (11, 64, EncodingKind::Ieee) => Self::Double,
            (15, 80, EncodingKind::X87) => Self::Extended,
            (15, 128, EncodingKind::Ieee) => Self::Quad,
            _ => Self::None,
        }
    }
}

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

/// The term that a slice kernel of `Lanes` adds into a lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Term {
    /// The value `x` of the first vector alone, which adds into the lane in
    /// one rounded addition.
    Value,
    /// The product `x * y`.
    Product,
    /// The square of the difference: `d * d` with `d = x - y`.
    SquareDifference,
}

/// How a slice kernel of `Lanes` adds a term into a lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// The product rounds, and then the sum rounds.
    Separate,
    /// A fused multiply-add rounds once.
    Fused,
}

/// A vector that a slice kernel of `Lanes` reads on the host unit, as the
/// binary32 encodings of its values.
pub trait Load: Copy {
    /// Returns the number of values.
    fn count(self) -> usize;

    /// Returns the binary32 encodings of the `N` values from `start`, or
    /// `None` when the build has no instruction for the conversion. The
    /// vector holds at least `start + N` values.
    fn load<const N: usize>(self, start: usize) -> Option<[u32; N]>;

    /// Returns the binary32 encodings of the values from `start`, with +0
    /// in the lanes past the last value, as [`load`](Self::load) does. The
    /// vector holds more than `start` and fewer than `start + N` values.
    fn load_rest<const N: usize>(self, start: usize) -> Option<[u32; N]>;

    /// Returns the `count` values from `start` as a vector of the same kind.
    /// The vector holds at least `start + count` values.
    #[must_use]
    fn part(self, start: usize, count: usize) -> Self;
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
    let unit = environment::UNIT;
    let fused = environment::FUSED;
    let half = environment::HALF;
    let rounding = environment::ROUNDING;
    let x87 = environment::X87;
    let fp16 = environment::HALF_FUSED;
    let quad = environment::QUAD;
    match (host, kind) {
        // The x87 unit has no fused multiply-add. Two roundings of a bfloat16
        // fused multiply-add, or of an integer, through binary32 can differ
        // from one. Only the x87 unit has a remainder instruction, and it
        // loads binary32, binary64, and x87 extended values. binary16 and
        // bfloat16 values widen to binary32 exactly for it. The binary128
        // unit of s390x has neither a fused multiply-add nor a remainder.
        (Host::None, _)
        | (Host::Extended, Kind::FusedMultiplyAdd | Kind::Comparison)
        | (Host::BFloat, Kind::FusedMultiplyAdd)
        | (Host::Quad, Kind::FusedMultiplyAdd | Kind::Remainder) => false,
        (
            Host::Quad,
            Kind::Arithmetic
            | Kind::SquareRoot
            | Kind::RoundToIntegral
            | Kind::ToInt
            | Kind::FromInt
            | Kind::Comparison,
        ) => unit && quad,
        // An integer below 2^53 in magnitude converts to binary64 exactly,
        // and round to odd to binary32 then keeps the bfloat16 rounding.
        (Host::BFloat, Kind::FromInt) => unit && environment::ROUND_TO_ODD,
        // binary64 holds the exact product of two binary16 values, as
        // `Host::Half` states.
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
    let unit = environment::UNIT;
    let x87 = environment::X87;
    match (from, to) {
        // A shift widens bfloat16 to binary32 exactly, and integer
        // instructions round binary32 to bfloat16. binary16 widens and rounds
        // in F16C or `FCVT`, or else in integer instructions.
        (Host::Single, Host::Double | Host::BFloat | Host::Half)
        | (Host::Double, Host::Single)
        | (Host::BFloat | Host::Half, Host::Single | Host::Double) => unit,
        (Host::Double, Host::Half) => unit && environment::DOUBLE_TO_HALF,
        // Round to odd to binary32 keeps the rounding to bfloat16.
        (Host::Double, Host::BFloat) => unit && environment::ROUND_TO_ODD,
        (Host::Single | Host::Double, Host::Extended)
        | (Host::Extended, Host::Single | Host::Double) => unit && x87,
        (Host::Single | Host::Double, Host::Quad) | (Host::Quad, Host::Single | Host::Double) => {
            unit && environment::QUAD
        }
        _ => false,
    }
}

#[cfg(any(
    floaty_engine_only,
    not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        ),
        target_arch = "aarch64",
        target_arch = "s390x"
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
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        ),
        target_arch = "aarch64",
        target_arch = "s390x"
    ))
))]
mod none {
    use core::cmp::Ordering;

    use super::{Host, Operation};
    use crate::env::Env;
    use crate::format::Standard;
    use crate::format::internal::MinMax;

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
    /// `false`: this build has no binary16 fused multiply-add.
    pub const HALF_FUSED: bool = false;
    /// `false`: this build has no rounding of binary64 to binary16.
    pub const DOUBLE_TO_HALF: bool = false;
    /// `false`: this build has no rounding to odd.
    pub const ROUND_TO_ODD: bool = false;
    /// `false`: this build has no binary128 unit.
    pub const QUAD: bool = false;

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
        use crate::format::internal::MinMax;
        use crate::host::{Host, Kind, Load, Operation, Step, Term};

        /// Returns `false`: this build has no host path.
        #[must_use]
        pub const fn available(_host: Host, _kind: Kind) -> bool {
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
        pub fn convert_chunks<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
            _values: &[Float<S, W, M>],
            _to: Host,
            _env: &Env,
            _each: impl FnMut(usize, Option<[u64; N]>),
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
        pub fn widen_halves<const N: usize>(_halves: &[u16; N]) -> Option<[u32; N]> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn widen_codes<const N: usize>(_codes: &[u8; N]) -> Option<[u32; N]> {
            None
        }

        /// Returns `None`: this build has no host path.
        #[inline]
        pub fn widen_scaled_codes<const N: usize>(
            _codes: &[i8; N],
            _scale: u32,
        ) -> Option<[u32; N]> {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Host, Kind, Operation, available, convertible};
    use crate::env::Env;
    use crate::float::Float;
    use crate::format::internal::{LimbConversion, MinMax};
    use crate::format::{Binary, Standard, X87};
    use crate::limbs::Limbs;

    /// The host kinds, as conversion destinations.
    const HOSTS: [Host; 6] = [
        Host::Half,
        Host::BFloat,
        Host::Single,
        Host::Double,
        Host::Extended,
        Host::Quad,
    ];

    /// Asserts that each host path of the format `S` that `available`,
    /// `convertible`, `packed::available`, or `packed::convertible` claims
    /// gives a result. The operands 3, 2, and 1 are exact in every format,
    /// and no path declines them by value.
    fn claimed_paths_give_results<S: Standard<W>, const W: usize>() {
        let env = Env::IEEE;
        let host = S::HOST;
        let value = |integer: i64| Float::<S, W>::from_int(integer);
        let (three, two, one) = (value(3), value(2), value(1));
        let (left_bits, right_bits, addend_bits) = (three.to_bits(), two.to_bits(), one.to_bits());
        let scalar = [
            (
                Kind::Arithmetic,
                super::binary::<S, W>(left_bits, right_bits, Operation::Add, &env).is_some(),
            ),
            (
                Kind::SquareRoot,
                super::sqrt::<S, W>(left_bits, &env).is_some(),
            ),
            (
                Kind::FusedMultiplyAdd,
                super::mul_add::<S, W>(left_bits, right_bits, addend_bits, &env).is_some(),
            ),
            (
                Kind::RoundToIntegral,
                super::round_to_integral::<S, W>(left_bits, &env).is_some(),
            ),
            (
                Kind::ToInt,
                super::to_int::<S, W>(left_bits, &env).is_some(),
            ),
            (Kind::FromInt, super::from_int::<S, W>(3, &env).is_some()),
            (
                Kind::Comparison,
                super::compare::<S, W>(left_bits, right_bits, &env).is_some()
                    && super::min_max::<S, W>(left_bits, right_bits, MinMax::Minimum, &env)
                        .is_some(),
            ),
            (
                Kind::Remainder,
                super::remainder::<S, W>(left_bits, right_bits, &env).is_some(),
            ),
        ];
        for (kind, taken) in scalar {
            assert!(
                !available(host, kind) || taken,
                "{host:?} {kind:?}: the claimed path gives no result"
            );
        }
        let (left, right, addend) = ([three; 8], [two; 8], [one; 8]);
        let packed = [
            (
                Kind::Arithmetic,
                super::packed::binary(&left, &right, Operation::Add, &env).is_some(),
            ),
            (Kind::SquareRoot, super::packed::sqrt(&left, &env).is_some()),
            (
                Kind::FusedMultiplyAdd,
                super::packed::mul_add(&left, &right, &addend, &env).is_some(),
            ),
            (
                Kind::RoundToIntegral,
                super::packed::round_to_integral(&left, &env).is_some(),
            ),
            (
                Kind::ToInt,
                super::packed::to_int_i32(&left, &env).is_some(),
            ),
            (
                Kind::Comparison,
                super::packed::compare(&left, &right, &env).is_some()
                    && super::packed::min_max(&left, &right, MinMax::Minimum, &env).is_some(),
            ),
        ];
        for (kind, taken) in packed {
            assert!(
                !super::packed::available(host, kind) || taken,
                "{host:?} {kind:?}: the claimed packed path gives no result"
            );
        }
        let limbs = left_bits.to_limbs();
        let encoding = [limbs.limb(0), limbs.limb(1)];
        for to in HOSTS {
            assert!(
                !convertible(host, to) || super::convert(host, to, encoding, &env).is_some(),
                "{host:?} to {to:?}: the claimed conversion gives no result"
            );
            assert!(
                !super::packed::convertible(host, to)
                    || super::packed::convert(&left, to, &env).is_some(),
                "{host:?} to {to:?}: the claimed packed conversion gives no result"
            );
        }
    }

    #[test]
    fn every_claimed_host_path_gives_a_result() {
        claimed_paths_give_results::<Binary<5>, 16>();
        claimed_paths_give_results::<Binary<8>, 16>();
        claimed_paths_give_results::<Binary<8>, 32>();
        claimed_paths_give_results::<Binary<11>, 64>();
        claimed_paths_give_results::<Binary<15, X87>, 80>();
        claimed_paths_give_results::<Binary<15>, 128>();
    }
}
