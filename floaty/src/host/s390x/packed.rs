//! The chunks of lanes of `Lanes` on s390x: four binary32 or two binary64
//! lanes, each through the scalar instruction of its lane.
//!
//! The base z/Architecture has no vector registers, so each function runs the
//! scalar instruction of each lane. The packed paths then check the FPC
//! register once for all lanes, as the x87 extended lanes check the control
//! word once. Each function for a 256-bit or a 512-bit chunk returns `None`,
//! and so does each function for binary16 lanes.

use core::cmp::Ordering;

use super::super::Operation;
use super::super::packed::Masks;
use crate::env::Rounding;
use crate::format::internal::MinMax;

/// `false`: s390x has no vector registers in the base z/Architecture.
pub const WIDE: bool = false;

/// `false`: s390x has no vector registers in the base z/Architecture.
pub const EXTRA_WIDE: bool = false;

/// `false`: the packed paths of s390x run the scalar instruction of each
/// lane, which takes no rounding direction from its encoding.
pub const ROUNDING_CONTROL: bool = false;

/// `false`: s390x has no binary16 conversion instruction, so binary16 lanes
/// take the scalar path of each lane.
pub const HALF: bool = false;

/// `false`: no packed path converts lanes to integers on s390x.
pub const INTEGERS: bool = false;

/// `false`: s390x has no binary16 arithmetic.
pub const NATIVE_HALF: bool = false;

/// Returns `operation` of four pairs of binary32 lanes.
#[inline]
pub fn binary_f32x4(left: [f32; 4], right: [f32; 4], operation: Operation) -> [f32; 4] {
    core::array::from_fn(|lane| super::binary_f32(left[lane], right[lane], operation))
}

/// Returns `operation` of two pairs of binary64 lanes.
#[inline]
pub fn binary_f64x2(left: [f64; 2], right: [f64; 2], operation: Operation) -> [f64; 2] {
    core::array::from_fn(|lane| super::binary_f64(left[lane], right[lane], operation))
}

/// Returns the square roots of four binary32 lanes.
#[inline]
pub fn sqrt_f32x4(value: [f32; 4]) -> [f32; 4] {
    value.map(super::sqrt_f32)
}

/// Returns the square roots of two binary64 lanes.
#[inline]
pub fn sqrt_f64x2(value: [f64; 2]) -> [f64; 2] {
    value.map(super::sqrt_f64)
}

/// Returns lanes 0 and 1 of `first` and then of `second`, and lanes 2 and 3
/// of `first` and then of `second`. A move of lanes is no floating-point
/// operation.
#[inline]
pub fn halves_f32x4(first: [f32; 4], second: [f32; 4]) -> ([f32; 4], [f32; 4]) {
    let ([a0, a1, a2, a3], [b0, b1, b2, b3]) = (first, second);
    ([a0, a1, b0, b1], [a2, a3, b2, b3])
}

/// Returns lanes 0 and 2 of `first` and then of `second`, and lanes 1 and 3
/// of `first` and then of `second`, as `halves_f32x4` states.
#[inline]
pub fn evens_odds_f32x4(first: [f32; 4], second: [f32; 4]) -> ([f32; 4], [f32; 4]) {
    let ([a0, a1, a2, a3], [b0, b1, b2, b3]) = (first, second);
    ([a0, a2, b0, b2], [a1, a3, b1, b3])
}

/// Returns four binary32 lanes rounded to integral values in the direction
/// `rounding`, or `None` for a direction that no instruction encodes.
#[inline]
pub fn round_f32x4(value: [f32; 4], rounding: Rounding) -> Option<[f32; 4]> {
    let [a, b, c, d] = value;
    Some([
        super::round_f32(a, rounding)?,
        super::round_f32(b, rounding)?,
        super::round_f32(c, rounding)?,
        super::round_f32(d, rounding)?,
    ])
}

/// Returns two binary64 lanes rounded to integral values in the direction
/// `rounding`, or `None` for a direction that no instruction encodes.
#[inline]
pub fn round_f64x2(value: [f64; 2], rounding: Rounding) -> Option<[f64; 2]> {
    let [a, b] = value;
    Some([
        super::round_f64(a, rounding)?,
        super::round_f64(b, rounding)?,
    ])
}

/// Returns `left * right + addend` of four triples of binary32 lanes, each
/// rounded once.
#[inline]
pub fn mul_add_f32x4(left: [f32; 4], right: [f32; 4], addend: [f32; 4]) -> Option<[f32; 4]> {
    let lane = |index: usize| super::mul_add_f32(left[index], right[index], addend[index]);
    Some([lane(0)?, lane(1)?, lane(2)?, lane(3)?])
}

/// Returns `left * right + addend` of two triples of binary64 lanes, each
/// rounded once.
#[inline]
pub fn mul_add_f64x2(left: [f64; 2], right: [f64; 2], addend: [f64; 2]) -> Option<[f64; 2]> {
    let lane = |index: usize| super::mul_add_f64(left[index], right[index], addend[index]);
    Some([lane(0)?, lane(1)?])
}

/// Returns two binary32 lanes widened exactly to binary64.
#[inline]
pub fn widen_x2(value: [f32; 2]) -> [f64; 2] {
    value.map(super::widen_single)
}

/// Returns two binary64 lanes rounded to binary32.
#[inline]
pub fn narrow_x2(value: [f64; 2]) -> [f32; 2] {
    value.map(super::narrow_double)
}

/// Returns the masks of the orders of each pair of lanes: all ones in a lane
/// where the test holds.
fn masks<T: Copy + Default, const N: usize>(orders: [Option<Ordering>; N], ones: T) -> Masks<T, N> {
    let lanes = |test: fn(Option<Ordering>) -> bool| {
        orders.map(|order| if test(order) { ones } else { T::default() })
    };
    Masks {
        less: lanes(|order| order == Some(Ordering::Less)),
        greater: lanes(|order| order == Some(Ordering::Greater)),
        unordered: lanes(|order| order.is_none()),
    }
}

/// Returns the masks of a comparison of each pair of binary32 lanes.
#[inline]
pub fn compare_f32x4(left: [f32; 4], right: [f32; 4]) -> Masks<u32, 4> {
    let orders = core::array::from_fn(|lane| super::compare_f32(left[lane], right[lane]));
    masks(orders, u32::MAX)
}

/// Returns the masks of a comparison of each pair of binary64 lanes.
#[inline]
pub fn compare_f64x2(left: [f64; 2], right: [f64; 2]) -> Masks<u64, 2> {
    let orders = core::array::from_fn(|lane| super::compare_f64(left[lane], right[lane]));
    masks(orders, u64::MAX)
}

/// Returns the smaller or the larger of each pair of binary32 lanes, as
/// `operation` selects.
#[inline]
pub fn min_max_f32x4(left: [f32; 4], right: [f32; 4], operation: MinMax) -> [f32; 4] {
    let select = if operation.is_minimum() {
        super::min_f32
    } else {
        super::max_f32
    };
    core::array::from_fn(|lane| select(left[lane], right[lane]))
}

/// Returns the smaller or the larger of each pair of binary64 lanes, as
/// `operation` selects.
#[inline]
pub fn min_max_f64x2(left: [f64; 2], right: [f64; 2], operation: MinMax) -> [f64; 2] {
    let select = if operation.is_minimum() {
        super::min_f64
    } else {
        super::max_f64
    };
    core::array::from_fn(|lane| select(left[lane], right[lane]))
}

/// Returns `None`: s390x has no binary16 conversion instruction.
#[inline]
pub fn widen_halves_x4(_value: [u16; 4]) -> Option<[f32; 4]> {
    None
}

/// Returns `None`: s390x has no binary16 conversion instruction.
#[inline]
pub fn narrow_halves_x4(_value: [f32; 4]) -> Option<[u16; 4]> {
    None
}

/// Returns the integer indefinite in every lane, which sends each lane to
/// the scalar conversion. `INTEGERS` keeps the packed path from calling this
/// function.
#[inline]
pub fn to_int_f32x4(_value: [f32; 4]) -> [i32; 4] {
    [i32::MIN; 4]
}

/// Returns the integer indefinite in every lane, as `to_int_f32x4` does.
#[inline]
pub fn to_int_f64x2(_value: [f64; 2]) -> [i32; 2] {
    [i32::MIN; 2]
}

/// Returns four 32-bit integers converted to binary32 in the rounding mode
/// of the FPC register, each by `CEGBR`. The conversion is exact for an
/// integer below `2^24` in magnitude.
#[inline]
pub fn from_int_x4(value: [i32; 4]) -> [f32; 4] {
    value.map(|lane| {
        super::from_int_f32(i64::from(lane)).expect("CEGBR converts every 64-bit integer")
    })
}

/// Defines a function that returns `None`, for a chunk that s390x does not
/// compute: a 256-bit or a 512-bit chunk, or eight binary16 lanes.
macro_rules! no_chunk {
    ($doc:literal, $name:ident, ($($type:ty),+) -> $result:ty) => {
        #[doc = $doc]
        #[inline]
        pub fn $name($(_: $type),+) -> Option<$result> {
            None
        }
    };
}

/// Defines the functions of the 256-bit and the 512-bit chunks, which return
/// `None`.
macro_rules! no_wide {
    ($($name:ident, ($($type:ty),+) -> $result:ty);+ $(;)?) => {
        $(no_chunk!("Returns `None`: the base z/Architecture has no vector registers.", $name, ($($type),+) -> $result);)+
    };
}

/// Defines the functions of eight binary16 lanes, which return `None`.
macro_rules! no_native {
    ($($name:ident, ($($type:ty),+) -> $result:ty);+ $(;)?) => {
        $(no_chunk!("Returns `None`: s390x has no binary16 arithmetic.", $name, ($($type),+) -> $result);)+
    };
}

no_wide!(
    binary_f32x8, ([f32; 8], [f32; 8], Operation) -> [f32; 8];
    binary_f64x4, ([f64; 4], [f64; 4], Operation) -> [f64; 4];
    sqrt_f32x8, ([f32; 8]) -> [f32; 8];
    sqrt_f64x4, ([f64; 4]) -> [f64; 4];
    round_f32x8, ([f32; 8], Rounding) -> [f32; 8];
    round_f64x4, ([f64; 4], Rounding) -> [f64; 4];
    mul_add_f32x8, ([f32; 8], [f32; 8], [f32; 8]) -> [f32; 8];
    mul_add_f64x4, ([f64; 4], [f64; 4], [f64; 4]) -> [f64; 4];
    widen_x4, ([f32; 4]) -> [f64; 4];
    narrow_x4, ([f64; 4]) -> [f32; 4];
    widen_halves_x8, ([u16; 8]) -> [f32; 8];
    narrow_halves_x8, ([f32; 8]) -> [u16; 8];
    compare_f32x8, ([f32; 8], [f32; 8]) -> Masks<u32, 8>;
    compare_f64x4, ([f64; 4], [f64; 4]) -> Masks<u64, 4>;
    min_max_f32x8, ([f32; 8], [f32; 8], MinMax) -> [f32; 8];
    min_max_f64x4, ([f64; 4], [f64; 4], MinMax) -> [f64; 4];
    to_int_f32x8, ([f32; 8]) -> [i32; 8];
    to_int_f64x4, ([f64; 4]) -> [i32; 4];
    from_int_x8, ([i32; 8]) -> [f32; 8];
    binary_f32x16, ([f32; 16], [f32; 16], Operation) -> [f32; 16];
    mul_add_f32x16, ([f32; 16], [f32; 16], [f32; 16]) -> [f32; 16];
    min_max_f32x16, ([f32; 16], [f32; 16], MinMax) -> [f32; 16];
    round_f32x16, ([f32; 16], Rounding) -> [f32; 16];
    to_int_f32x16, ([f32; 16]) -> [i32; 16];
    from_int_x16, ([i32; 16]) -> [f32; 16];
    widen_halves_x16, ([u16; 16]) -> [f32; 16];
    narrow_halves_x16, ([f32; 16]) -> [u16; 16];
    binary_rounded_f32x16, ([f32; 16], [f32; 16], Operation, Rounding) -> [f32; 16];
    mul_add_rounded_f32x16, ([f32; 16], [f32; 16], [f32; 16], Rounding) -> [f32; 16];
    to_int_rounded_f32x16, ([f32; 16], Rounding) -> [i32; 16];
);

no_native!(
    binary_f16x8, ([u16; 8], [u16; 8], Operation) -> [u16; 8];
    sqrt_f16x8, ([u16; 8]) -> [u16; 8];
    round_f16x8, ([u16; 8], Rounding) -> [u16; 8];
    mul_add_f16x8, ([u16; 8], [u16; 8], [u16; 8]) -> [u16; 8];
    compare_f16x8, ([u16; 8], [u16; 8]) -> Masks<u16, 8>;
    min_max_f16x8, ([u16; 8], [u16; 8], MinMax) -> [u16; 8];
);
