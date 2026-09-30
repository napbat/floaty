//! The entry points that take a host path where the build has one, and their
//! `_with` counterparts in the default mode, which always run the engine.
//!
//! A host path must give the bits of the engine. A test runs a function of
//! this module and its `_with` companion on the same operands, and compares
//! the two results, which have named fields. A test that checks a control
//! register runs the function inside the closure that sets the register, on
//! operands that pass through [`core::hint::black_box`], so that the
//! compiler cannot evaluate the entry points before the register is set.

use core::cmp::Ordering;

use floaty::env::Mode;
use floaty::format::Standard;
use floaty::{F16, F32, F64, Float, Lanes, ToInt};

/// The results of the arithmetic entry points on three operands `x`, `y`,
/// and `z`, as encodings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arithmetic<B> {
    /// `x + y`.
    pub add: B,
    /// `x - y`.
    pub sub: B,
    /// `x * y`.
    pub mul: B,
    /// `x / y`.
    pub div: B,
    /// The square root of `x`.
    pub sqrt: B,
    /// `x * y + z`, rounded once.
    pub mul_add: B,
}

/// Returns the results of the operators, `sqrt`, and `mul_add`.
#[must_use]
pub fn arithmetic<S: Standard<W>, const W: usize, M: Mode>(
    x: Float<S, W, M>,
    y: Float<S, W, M>,
    z: Float<S, W, M>,
) -> Arithmetic<S::Bits> {
    Arithmetic {
        add: (x + y).to_bits(),
        sub: (x - y).to_bits(),
        mul: (x * y).to_bits(),
        div: (x / y).to_bits(),
        sqrt: x.sqrt().to_bits(),
        mul_add: x.mul_add(y, z).to_bits(),
    }
}

/// Returns the results of the `_with` counterparts of [`arithmetic`] in the
/// default mode.
#[must_use]
pub fn arithmetic_with<S: Standard<W>, const W: usize, M: Mode>(
    x: Float<S, W, M>,
    y: Float<S, W, M>,
    z: Float<S, W, M>,
) -> Arithmetic<S::Bits> {
    let env = Float::<S, W, M>::ENV;
    Arithmetic {
        add: x.add_with(y, env).0.to_bits(),
        sub: x.sub_with(y, env).0.to_bits(),
        mul: x.mul_with(y, env).0.to_bits(),
        div: x.div_with(y, env).0.to_bits(),
        sqrt: x.sqrt_with(env).0.to_bits(),
        mul_add: x.mul_add_with(y, z, env).0.to_bits(),
    }
}

/// The results of the comparison entry points on two operands `x` and `y`:
/// the order, the equality, and the minimum and maximum operations, as
/// encodings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Comparisons<B> {
    /// The order of `x` to `y`, or `None` when they are unordered.
    pub order: Option<Ordering>,
    /// `x == y`.
    pub equal: bool,
    /// `minimum`.
    pub minimum: B,
    /// `maximum`.
    pub maximum: B,
    /// `minimumNumber`.
    pub minimum_number: B,
    /// `maximumNumber`.
    pub maximum_number: B,
    /// `minNum`.
    pub min_num: B,
    /// `maxNum`.
    pub max_num: B,
}

/// Returns the results of `partial_cmp`, `==`, and the minimum and maximum
/// operations.
#[must_use]
pub fn comparisons<S: Standard<W>, const W: usize, M: Mode>(
    x: Float<S, W, M>,
    y: Float<S, W, M>,
) -> Comparisons<S::Bits> {
    Comparisons {
        order: x.partial_cmp(&y),
        equal: x == y,
        minimum: x.minimum(y).to_bits(),
        maximum: x.maximum(y).to_bits(),
        minimum_number: x.minimum_number(y).to_bits(),
        maximum_number: x.maximum_number(y).to_bits(),
        min_num: x.min_num(y).to_bits(),
        max_num: x.max_num(y).to_bits(),
    }
}

/// Returns the results of the `_with` counterparts of [`comparisons`] in the
/// default mode. The order is the quiet comparison, and two operands are
/// equal when they compare equal.
#[must_use]
pub fn comparisons_with<S: Standard<W>, const W: usize, M: Mode>(
    x: Float<S, W, M>,
    y: Float<S, W, M>,
) -> Comparisons<S::Bits> {
    let env = Float::<S, W, M>::ENV;
    let order = x.compare_quiet_with(y, env).0;
    Comparisons {
        order,
        equal: order == Some(Ordering::Equal),
        minimum: x.minimum_with(y, env).0.to_bits(),
        maximum: x.maximum_with(y, env).0.to_bits(),
        minimum_number: x.minimum_number_with(y, env).0.to_bits(),
        maximum_number: x.maximum_number_with(y, env).0.to_bits(),
        min_num: x.min_num_with(y, env).0.to_bits(),
        max_num: x.max_num_with(y, env).0.to_bits(),
    }
}

/// The results of the conversion entry points on an operand `x` and an
/// integer `n`: rounding to an integral value, conversions to and from
/// integers, and conversions to the host formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conversions<B> {
    /// `x` rounded to an integral value.
    pub round_to_integral: B,
    /// `x` converted to an `i64`.
    pub to_i64: ToInt<i64>,
    /// `x` converted to a `u32`.
    pub to_u32: ToInt<u32>,
    /// `n` converted to the format of `x`.
    pub from_i64: B,
    /// `x` converted to binary16.
    pub to_binary16: u16,
    /// `x` converted to binary32.
    pub to_binary32: u32,
    /// `x` converted to binary64.
    pub to_binary64: u64,
}

/// Returns the results of `round_to_integral`, `to_int`, `from_int`, and
/// `convert`.
#[must_use]
pub fn conversions<S: Standard<W>, const W: usize, M: Mode>(
    x: Float<S, W, M>,
    n: i64,
) -> Conversions<S::Bits> {
    Conversions {
        round_to_integral: x.round_to_integral().to_bits(),
        to_i64: x.to_int::<i64>(),
        to_u32: x.to_int::<u32>(),
        from_i64: Float::<S, W, M>::from_int(n).to_bits(),
        to_binary16: x.convert::<F16>().to_bits(),
        to_binary32: x.convert::<F32>().to_bits(),
        to_binary64: x.convert::<F64>().to_bits(),
    }
}

/// Returns the results of the `_with` counterparts of [`conversions`] in the
/// default mode of each result. A conversion to a float runs in the default
/// mode of the destination, as `convert` does.
#[must_use]
pub fn conversions_with<S: Standard<W>, const W: usize, M: Mode>(
    x: Float<S, W, M>,
    n: i64,
) -> Conversions<S::Bits> {
    let env = Float::<S, W, M>::ENV;
    Conversions {
        round_to_integral: x.round_to_integral_with(env).0.to_bits(),
        to_i64: x.to_int_with::<i64>(env).0,
        to_u32: x.to_int_with::<u32>(env).0,
        from_i64: Float::<S, W, M>::from_int_with(n, env).0.to_bits(),
        to_binary16: x.convert_with::<F16>(F16::ENV).0.to_bits(),
        to_binary32: x.convert_with::<F32>(F32::ENV).0.to_bits(),
        to_binary64: x.convert_with::<F64>(F64::ENV).0.to_bits(),
    }
}

/// The results of the arithmetic entry points of `Lanes` on three lane sets
/// `x`, `y`, and `z`, as the encodings of the lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneArithmetic<B, const N: usize> {
    /// `x + y`.
    pub add: [B; N],
    /// `x - y`.
    pub sub: [B; N],
    /// `x * y`.
    pub mul: [B; N],
    /// `x / y`.
    pub div: [B; N],
    /// The square root of `x`.
    pub sqrt: [B; N],
    /// `x * y + z`, rounded once in each lane.
    pub mul_add: [B; N],
    /// `x` rounded to an integral value.
    pub round_to_integral: [B; N],
}

/// Returns the results of the operators, `sqrt`, `mul_add`, and
/// `round_to_integral` of `Lanes`.
#[must_use]
pub fn lane_arithmetic<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    x: Lanes<Float<S, W, M>, N>,
    y: Lanes<Float<S, W, M>, N>,
    z: Lanes<Float<S, W, M>, N>,
) -> LaneArithmetic<S::Bits, N> {
    LaneArithmetic {
        add: (x + y).to_bits(),
        sub: (x - y).to_bits(),
        mul: (x * y).to_bits(),
        div: (x / y).to_bits(),
        sqrt: x.sqrt().to_bits(),
        mul_add: x.mul_add(y, z).to_bits(),
        round_to_integral: x.round_to_integral().to_bits(),
    }
}

/// Returns the results of the `_with` counterparts of [`lane_arithmetic`] in
/// the default mode.
#[must_use]
pub fn lane_arithmetic_with<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    x: Lanes<Float<S, W, M>, N>,
    y: Lanes<Float<S, W, M>, N>,
    z: Lanes<Float<S, W, M>, N>,
) -> LaneArithmetic<S::Bits, N> {
    let env = Float::<S, W, M>::ENV;
    LaneArithmetic {
        add: x.add_with(y, env).0.to_bits(),
        sub: x.sub_with(y, env).0.to_bits(),
        mul: x.mul_with(y, env).0.to_bits(),
        div: x.div_with(y, env).0.to_bits(),
        sqrt: x.sqrt_with(env).0.to_bits(),
        mul_add: x.mul_add_with(y, z, env).0.to_bits(),
        round_to_integral: x.round_to_integral_with(env).0.to_bits(),
    }
}
