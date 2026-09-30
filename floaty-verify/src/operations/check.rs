//! Checks that compare the operations beyond arithmetic of floaty with the
//! oracle.
//!
//! Each check panics with the operation, the operands, and the behavior when
//! floaty differs from the oracle. A float result must also be canonical.

use floaty::format::Standard;
use floaty::{Env, Flags, Float, TotalOrder};
use rug::Integer;

use crate::encodings::Layout;

use super::compare::{self, MinMax};
use super::integral::{self, IntegerValue, to_int_value};
use super::{
    Direction, Outcome, Sample, next, outcome, remainder, scale_b, truncated_remainder, with_sign,
};
use crate::mpfr::{Format, Specials};

/// A floaty value, and its sample for the oracle.
pub struct Case<S: Standard<W>, const W: usize> {
    /// The floaty value.
    pub value: Float<S, W>,
    /// The encoding and decoded operand.
    pub sample: Sample<8>,
}

impl<S: Standard<W>, const W: usize> Case<S, W> {
    /// Makes a case from an encoding, its field layout, and the floaty value
    /// of that encoding.
    #[must_use]
    pub fn new(bits: Integer, layout: Layout, value: Float<S, W>) -> Self {
        Self {
            value,
            sample: Sample::new(bits, layout, value),
        }
    }

    /// Returns `true` when the sign bit of the encoding is set.
    fn sign_bit(&self) -> bool {
        self.sample.bits.get_bit(self.sample.layout.width - 1)
    }
}

/// Checks a float result against the expected outcome and flags.
fn check_float<S: Standard<W>, const W: usize>(
    (result, flags): (Float<S, W>, Flags),
    expected: &(Outcome, Flags),
    context: &dyn Fn() -> String,
) {
    assert!(
        result.is_canonical(),
        "{}: {result:?} is canonical",
        context()
    );
    assert_eq!(&(outcome(result), flags), expected, "{}", context());
}

/// Checks the operations of a pair that read no behavior: the total order,
/// the sign operations, and `PartialEq` and `PartialOrd`, which use the
/// default mode. `make` gives the value of an encoding.
///
/// # Panics
///
/// Panics when floaty differs from the oracle.
pub fn check_order_and_signs<S: Standard<W>, const W: usize>(
    x: &Case<S, W>,
    y: &Case<S, W>,
    specials: Specials,
    make: &dyn Fn(&Integer) -> Float<S, W>,
) {
    let context = || format!("{:?} {:?}", x.value, y.value);
    assert_eq!(
        x.value.total_cmp(y.value),
        compare::total_order(&x.sample, &y.sample, TotalOrder::Datum),
        "total_cmp {}",
        context()
    );
    assert_eq!(
        x.value.total_cmp_with(y.value, TotalOrder::Encoding),
        compare::total_order(&x.sample, &y.sample, TotalOrder::Encoding),
        "total_cmp by encoding {}",
        context()
    );
    let (order, _) = compare::compare_quiet(&x.sample.operand, &y.sample.operand, &Env::IEEE);
    assert_eq!(
        x.value.partial_cmp(&y.value),
        order,
        "partial_cmp {}",
        context()
    );
    assert_eq!(
        x.value == y.value,
        order == Some(core::cmp::Ordering::Equal),
        "eq {}",
        context()
    );
    let sign = |result: Float<S, W>, negative: bool, name: &str| {
        let expected = make(&with_sign(&x.sample, specials, negative));
        assert_eq!(result.to_bits(), expected.to_bits(), "{name} {}", context());
    };
    sign(x.value.abs(), false, "abs");
    sign(-x.value, !x.sign_bit(), "negate");
    sign(x.value.copy_sign(y.value), y.sign_bit(), "copy_sign");
}

/// Checks the quiet and the signaling comparison of a pair.
///
/// # Panics
///
/// Panics when floaty differs from the oracle.
pub fn check_comparisons<S: Standard<W>, const W: usize>(
    x: &Case<S, W>,
    y: &Case<S, W>,
    env: &Env,
) {
    let (first, second) = (&x.sample.operand, &y.sample.operand);
    assert_eq!(
        x.value.compare_quiet_with(y.value, *env),
        compare::compare_quiet(first, second, env),
        "compare_quiet {:?} {:?} {env:?}",
        x.value,
        y.value
    );
    assert_eq!(
        x.value.compare_signaling_with(y.value, *env),
        compare::compare_signaling(first, second, env),
        "compare_signaling {:?} {:?} {env:?}",
        x.value,
        y.value
    );
}

/// Checks the six minimum and maximum operations on a pair.
///
/// # Panics
///
/// Panics when floaty differs from the oracle.
pub fn check_min_max<S: Standard<W>, const W: usize>(
    x: &Case<S, W>,
    y: &Case<S, W>,
    format: &Format,
    env: &Env,
) {
    for operation in MinMax::ALL {
        let (a, b) = (x.value, y.value);
        let ours = match operation {
            MinMax::Minimum => a.minimum_with(b, *env),
            MinMax::Maximum => a.maximum_with(b, *env),
            MinMax::MinimumNumber => a.minimum_number_with(b, *env),
            MinMax::MaximumNumber => a.maximum_number_with(b, *env),
            MinMax::MinNum => a.min_num_with(b, *env),
            MinMax::MaxNum => a.max_num_with(b, *env),
        };
        let expected =
            compare::min_max(operation, &x.sample.operand, &y.sample.operand, format, env);
        check_float(ours, &expected, &|| {
            format!("{operation:?} {a:?} {b:?} {env:?}")
        });
    }
}

/// Checks the IEEE 754 remainder and the truncated remainder of a pair.
///
/// # Panics
///
/// Panics when floaty differs from the oracle.
pub fn check_remainder<S: Standard<W>, const W: usize>(
    x: &Case<S, W>,
    y: &Case<S, W>,
    format: &Format,
    env: &Env,
) {
    check_float(
        x.value.remainder_with(y.value, *env),
        &remainder(&x.sample.operand, &y.sample.operand, format, env),
        &|| format!("remainder {:?} {:?} {env:?}", x.value, y.value),
    );
    check_float(
        x.value.truncated_remainder_with(y.value, *env),
        &truncated_remainder(&x.sample.operand, &y.sample.operand, format, env),
        &|| format!("truncated_remainder {:?} {:?} {env:?}", x.value, y.value),
    );
}

/// Checks `round_to_integral_with`, and `next_up_with` and `next_down_with`,
/// on one value.
///
/// # Panics
///
/// Panics when floaty differs from the oracle.
pub fn check_integral_and_next<S: Standard<W>, const W: usize>(
    x: &Case<S, W>,
    format: &Format,
    env: &Env,
) {
    let operand = &x.sample.operand;
    check_float(
        x.value.round_to_integral_with(*env),
        &integral::round_to_integral(operand, format, env),
        &|| format!("round_to_integral {:?} {env:?}", x.value),
    );
    check_float(
        x.value.next_up_with(*env),
        &next(operand, Direction::Up, format, env),
        &|| format!("next_up {:?} {env:?}", x.value),
    );
    check_float(
        x.value.next_down_with(*env),
        &next(operand, Direction::Down, format, env),
        &|| format!("next_down {:?} {env:?}", x.value),
    );
}

/// Checks `scale_b_with` on one value, for each scale.
///
/// # Panics
///
/// Panics when floaty differs from the oracle.
pub fn check_scale_b<S: Standard<W>, const W: usize>(
    x: &Case<S, W>,
    scales: &[i32],
    format: &Format,
    env: &Env,
) {
    for &scale in scales {
        check_float(
            x.value.scale_b_with(scale, *env),
            &scale_b(&x.sample.operand, scale, format, env),
            &|| format!("scale_b {:?} {scale} {env:?}", x.value),
        );
    }
}

/// Checks `to_int_with` into the integer type `I` on one value.
///
/// # Panics
///
/// Panics when floaty differs from the oracle.
pub fn check_to_int<I: IntegerValue, S: Standard<W>, const W: usize>(x: &Case<S, W>, env: &Env) {
    let (ours, flags) = x.value.to_int_with::<I>(*env);
    assert_eq!(
        (to_int_value(ours), flags),
        integral::to_int::<I, 8>(&x.sample.operand, env),
        "to_int into {} {:?} {env:?}",
        core::any::type_name::<I>(),
        x.value
    );
}

/// Checks `from_int_with` from one integer.
///
/// # Panics
///
/// Panics when floaty differs from the oracle.
pub fn check_from_int<I: IntegerValue, S: Standard<W>, const W: usize>(
    value: I,
    format: &Format,
    env: &Env,
) {
    let integer = value.to_integer();
    check_float(
        Float::<S, W>::from_int_with(value, *env),
        &integral::from_int(&integer, format, env),
        &|| format!("from_int {integer} {env:?}"),
    );
}

/// Checks the operations with the default mode, which call the `_with`
/// operations with the behavior of the type.
///
/// # Panics
///
/// Panics when floaty differs from the oracle.
pub fn check_default_mode<S: Standard<W>, const W: usize>(
    x: &Case<S, W>,
    y: &Case<S, W>,
    format: &Format,
) {
    let env = Float::<S, W>::ENV;
    let pairs = [
        (x.value.minimum(y.value), x.value.minimum_with(y.value, env)),
        (x.value.maximum(y.value), x.value.maximum_with(y.value, env)),
        (
            x.value.minimum_number(y.value),
            x.value.minimum_number_with(y.value, env),
        ),
        (
            x.value.maximum_number(y.value),
            x.value.maximum_number_with(y.value, env),
        ),
        (x.value.min_num(y.value), x.value.min_num_with(y.value, env)),
        (x.value.max_num(y.value), x.value.max_num_with(y.value, env)),
        (
            x.value.remainder(y.value),
            x.value.remainder_with(y.value, env),
        ),
        (
            x.value.round_to_integral(),
            x.value.round_to_integral_with(env),
        ),
        (x.value.scale_b(3), x.value.scale_b_with(3, env)),
        (x.value.next_up(), x.value.next_up_with(env)),
        (x.value.next_down(), x.value.next_down_with(env)),
    ];
    for (index, (plain, (with, _))) in pairs.into_iter().enumerate() {
        assert_eq!(
            plain.to_bits(),
            with.to_bits(),
            "operation {index} {:?} {:?}",
            x.value,
            y.value
        );
    }
    let (expected, _) = integral::to_int::<i8, 8>(&x.sample.operand, &env);
    assert_eq!(
        to_int_value(x.value.to_int::<i8>()),
        expected,
        "to_int {:?}",
        x.value
    );
    let (expected, _) = remainder(&x.sample.operand, &y.sample.operand, format, &env);
    assert_eq!(
        outcome(x.value.remainder(y.value)),
        expected,
        "remainder {:?} {:?}",
        x.value,
        y.value
    );
}
