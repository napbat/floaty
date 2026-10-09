//! The host paths: operations on the floating-point unit of the host that give
//! the bits of the engine. The README lists each path.
//!
//! The build selects the paths at compile time, from the target architecture
//! and its features. One module in `host/` reads the floating-point
//! environment of each architecture. A build for another architecture, or with
//! `--cfg floaty_engine_only`, has no host path, and each entry point returns
//! `None`. The slice kernels and the elementwise slice operations of `Lanes`,
//! and blocks, also take a larger instruction set that the processor has, in
//! a build with the feature `std`, as `packed::dispatch` states. `block`
//! runs the steps of a chain as Rust operations on `f32` and `f64`, as it
//! states.
//!
//! A path serves only entry points that return no flags, in a mode that
//! `compatible` in `paths` accepts. That function names every field of
//! `Env`, and the README states the rule. The environment of the host must
//! match the mode. The rounding to an integral value also takes a directed
//! rounding, which its instructions take from their encoding. So do the
//! slice kernels and the elementwise slice operations in the instruction
//! sets with a rounding control, as `ready_in_direction` in `paths` states.
//! A NaN result goes back to the engine, which selects the NaN by the rule
//! of the mode.

use crate::env::{Mode, Rounding};
use crate::float::Float;
use crate::format::internal::MinMax;
use crate::format::{Binary, EncodingKind};
use crate::sealed::Sealed;

mod load;
pub use self::load::{Direction, Load};

/// Returns host binary32 values as binary32 values of floaty, which have
/// their layout. Each value keeps its bits.
#[must_use]
pub fn floats_of_singles<M: Mode>(values: &[f32]) -> &[Float<Binary<8>, 32, M>] {
    // SAFETY: `Float` is `repr(transparent)` over its encoding, a `u32`,
    // which has the size and the alignment of an `f32`. Every bit pattern is
    // a binary32 encoding, and the slice keeps the length and the lifetime.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

/// Returns host binary32 values as binary32 values of floaty to write, as
/// `floats_of_singles` returns them to read.
#[must_use]
pub fn floats_of_singles_mut<M: Mode>(values: &mut [f32]) -> &mut [Float<Binary<8>, 32, M>] {
    // SAFETY: as in `floats_of_singles`. The slice keeps the unique borrow.
    unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len()) }
}

/// Returns host binary64 values as binary64 values of floaty, which have
/// their layout. Each value keeps its bits.
#[must_use]
pub fn floats_of_doubles<M: Mode>(values: &[f64]) -> &[Float<Binary<11>, 64, M>] {
    // SAFETY: `Float` is `repr(transparent)` over its encoding, a `u64`,
    // which has the size and the alignment of an `f64`. Every bit pattern is
    // a binary64 encoding, and the slice keeps the length and the lifetime.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

/// Returns host binary64 values as binary64 values of floaty to write, as
/// `floats_of_doubles` returns them to read.
#[must_use]
pub fn floats_of_doubles_mut<M: Mode>(values: &mut [f64]) -> &mut [Float<Binary<11>, 64, M>] {
    // SAFETY: as in `floats_of_doubles`. The slice keeps the unique borrow.
    unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len()) }
}

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
pub mod block;
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
    Ready, Unit, binary, compare, conversion_unit, convert, from_int, min_max, mul_add, ready,
    remainder, round_to_integral, sqrt, to_int,
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

/// An instruction set that the slice kernels and the elementwise slice
/// operations of `Lanes` compute in, as a type: the packed forms whose
/// instructions depend on optional features. Each form returns `None` where
/// the instruction set does not have it.
///
/// The type `Build` of `packed` has the forms of the features that the build
/// enables. `packed::dispatch` selects a larger instruction set that the
/// processor has at run time. Each form of each instruction set gives the
/// bits of the instruction of `Build`: the sets differ in the width of the
/// registers and in the encodings, not in the result of a lane.
pub trait Isa: Sealed {
    /// `true` when the instruction set has 256-bit registers, which hold
    /// eight binary32 lanes.
    const WIDE: bool;
    /// `true` when the instruction set has 512-bit registers, which hold
    /// sixteen binary32 lanes.
    const EXTRA_WIDE: bool;
    /// `true` when the instruction set widens binary16 lanes to binary32 and
    /// rounds binary32 lanes to binary16.
    const HALF: bool;
    /// `true` when the instruction set converts binary32 lanes to 32-bit
    /// integers, with the integer indefinite `i32::MIN` in a lane that the
    /// scalar conversion must decide.
    const INTEGERS: bool;
    /// `true` when the instruction set has a binary32 and binary64 fused
    /// multiply-add.
    const FUSED: bool;

    /// Returns `f()`, computed with the features of the instruction set.
    fn run<R>(f: impl FnOnce() -> R) -> R;

    /// Returns `operation` of four pairs of binary32 lanes.
    fn binary_f32x4(left: [f32; 4], right: [f32; 4], operation: Operation) -> Option<[f32; 4]>;
    /// Returns `operation` of eight pairs of binary32 lanes.
    fn binary_f32x8(left: [f32; 8], right: [f32; 8], operation: Operation) -> Option<[f32; 8]>;
    /// Returns `operation` of sixteen pairs of binary32 lanes.
    fn binary_f32x16(left: [f32; 16], right: [f32; 16], operation: Operation) -> Option<[f32; 16]>;
    /// Returns `left * right + addend` of four triples of binary32 lanes,
    /// each rounded once.
    fn mul_add_f32x4(left: [f32; 4], right: [f32; 4], addend: [f32; 4]) -> Option<[f32; 4]>;
    /// Returns `left * right + addend` of eight triples of binary32 lanes,
    /// each rounded once.
    fn mul_add_f32x8(left: [f32; 8], right: [f32; 8], addend: [f32; 8]) -> Option<[f32; 8]>;
    /// Returns `left * right + addend` of sixteen triples of binary32 lanes,
    /// each rounded once.
    fn mul_add_f32x16(left: [f32; 16], right: [f32; 16], addend: [f32; 16]) -> Option<[f32; 16]>;
    /// Returns `operation` of sixteen pairs of binary32 lanes, rounded in the
    /// direction `rounding` by the rounding control of the encoding of each
    /// instruction, or `None` for a direction without a control. The
    /// rounding control of the environment then does not apply.
    fn binary_rounded_f32x16(
        left: [f32; 16],
        right: [f32; 16],
        operation: Operation,
        rounding: Rounding,
    ) -> Option<[f32; 16]>;
    /// Returns `left * right + addend` of sixteen triples of binary32 lanes,
    /// each rounded once in the direction `rounding`, as
    /// `binary_rounded_f32x16` rounds.
    fn mul_add_rounded_f32x16(
        left: [f32; 16],
        right: [f32; 16],
        addend: [f32; 16],
        rounding: Rounding,
    ) -> Option<[f32; 16]>;
    /// Returns the lane of each of four pairs that the minimum or maximum
    /// instruction selects: the smaller or the larger value, and the right
    /// lane when a lane is a NaN or both are zeros.
    fn min_max_f32x4(left: [f32; 4], right: [f32; 4], operation: MinMax) -> Option<[f32; 4]>;
    /// Returns the lane of each of eight pairs that the minimum or maximum
    /// instruction selects, as `min_max_f32x4` does.
    fn min_max_f32x8(left: [f32; 8], right: [f32; 8], operation: MinMax) -> Option<[f32; 8]>;
    /// Returns the lane of each of sixteen pairs that the minimum or maximum
    /// instruction selects, as `min_max_f32x4` does.
    fn min_max_f32x16(left: [f32; 16], right: [f32; 16], operation: MinMax) -> Option<[f32; 16]>;
    /// Returns four binary32 lanes rounded to integral values in the
    /// direction `rounding`, or `None` for a direction that the instruction
    /// does not have.
    fn round_f32x4(value: [f32; 4], rounding: Rounding) -> Option<[f32; 4]>;
    /// Returns eight binary32 lanes rounded as `round_f32x4` rounds them.
    fn round_f32x8(value: [f32; 8], rounding: Rounding) -> Option<[f32; 8]>;
    /// Returns sixteen binary32 lanes rounded as `round_f32x4` rounds them.
    fn round_f32x16(value: [f32; 16], rounding: Rounding) -> Option<[f32; 16]>;
    /// Returns four binary32 lanes rounded to 32-bit integers to nearest
    /// even, with the integer indefinite for a NaN and a value out of range.
    fn to_int_f32x4(value: [f32; 4]) -> Option<[i32; 4]>;
    /// Returns eight binary32 lanes rounded as `to_int_f32x4` rounds them.
    fn to_int_f32x8(value: [f32; 8]) -> Option<[i32; 8]>;
    /// Returns sixteen binary32 lanes rounded as `to_int_f32x4` rounds them.
    fn to_int_f32x16(value: [f32; 16]) -> Option<[i32; 16]>;
    /// Returns sixteen binary32 lanes rounded to 32-bit integers in the
    /// direction `rounding`, as `binary_rounded_f32x16` rounds, with the
    /// integer indefinite as `to_int_f32x4` gives it.
    fn to_int_rounded_f32x16(value: [f32; 16], rounding: Rounding) -> Option<[i32; 16]>;
    /// Returns four 32-bit integers rounded to binary32.
    fn from_int_x4(value: [i32; 4]) -> Option<[f32; 4]>;
    /// Returns eight 32-bit integers rounded to binary32.
    fn from_int_x8(value: [i32; 8]) -> Option<[f32; 8]>;
    /// Returns sixteen 32-bit integers rounded to binary32.
    fn from_int_x16(value: [i32; 16]) -> Option<[f32; 16]>;
    /// Returns four binary16 lanes widened exactly to binary32.
    fn widen_halves_x4(value: [u16; 4]) -> Option<[f32; 4]>;
    /// Returns eight binary16 lanes widened exactly to binary32.
    fn widen_halves_x8(value: [u16; 8]) -> Option<[f32; 8]>;
    /// Returns sixteen binary16 lanes widened exactly to binary32.
    fn widen_halves_x16(value: [u16; 16]) -> Option<[f32; 16]>;
    /// Returns four binary32 lanes rounded to binary16 to nearest even.
    fn narrow_halves_x4(value: [f32; 4]) -> Option<[u16; 4]>;
    /// Returns eight binary32 lanes rounded to binary16 to nearest even.
    fn narrow_halves_x8(value: [f32; 8]) -> Option<[u16; 8]>;
    /// Returns sixteen binary32 lanes rounded to binary16 to nearest even.
    fn narrow_halves_x16(value: [f32; 16]) -> Option<[u16; 16]>;
    /// Returns two binary32 lanes widened exactly to binary64.
    fn widen_x2(value: [f32; 2]) -> Option<[f64; 2]>;
    /// Returns four binary32 lanes widened exactly to binary64.
    fn widen_x4(value: [f32; 4]) -> Option<[f64; 4]>;
    /// Returns two binary64 lanes rounded to binary32.
    fn narrow_x2(value: [f64; 2]) -> Option<[f32; 2]>;
    /// Returns four binary64 lanes rounded to binary32.
    fn narrow_x4(value: [f64; 4]) -> Option<[f32; 4]>;
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
    let x87_full = environment::X87_FULL_PRECISION;
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
        // The x87 arithmetic and square root round to the precision control,
        // which a thread starts at 64 bits on every system but Windows. The
        // other x87 paths give the same bits at every precision.
        (Host::Extended, Kind::Arithmetic | Kind::SquareRoot) => unit && x87 && x87_full,
        (Host::Extended, Kind::RoundToIntegral | Kind::ToInt | Kind::FromInt | Kind::Remainder)
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
    Ready, Unit, binary, block, compare, conversion_unit, convert, from_int, min_max, mul_add,
    packed, ready, remainder, round_to_integral, sqrt, to_int,
};

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
mod none;

#[cfg(test)]
mod tests;
