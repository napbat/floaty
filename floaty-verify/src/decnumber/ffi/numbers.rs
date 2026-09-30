//! decNumber's operations on arbitrary-precision numbers, which
//! [`Single`](super::Single), [`limited`](super::limited), and the wide
//! oracles use.

use core::ffi::{CStr, c_char};

use super::super::{Binary, Unary};
use super::{Context, Number as Data};

declare_unary!("decNumber":
    abs = "Abs", invert = "Invert", log_b = "LogB", minus = "Minus",
    next_minus = "NextMinus", next_plus = "NextPlus", plus = "Plus",
    reduce = "Reduce", to_integral_exact = "ToIntegralExact",
);
declare_binary!("decNumber":
    add = "Add", and = "And", compare = "Compare", compare_signal = "CompareSignal",
    compare_total = "CompareTotal", compare_total_mag = "CompareTotalMag",
    divide = "Divide", divide_integer = "DivideInteger", max = "Max",
    max_mag = "MaxMag", min = "Min", min_mag = "MinMag", multiply = "Multiply",
    next_toward = "NextToward", or = "Or", quantize = "Quantize",
    remainder = "Remainder", remainder_near = "RemainderNear", rotate = "Rotate",
    scale_b = "ScaleB", shift = "Shift", subtract = "Subtract", xor = "Xor",
);

unsafe extern "C" {
    #[link_name = "decNumberCopy"]
    fn copy(result: *mut Data, x: *const Data) -> *mut Data;
    #[link_name = "decNumberCopyAbs"]
    fn copy_abs(result: *mut Data, x: *const Data) -> *mut Data;
    #[link_name = "decNumberCopyNegate"]
    fn copy_negate(result: *mut Data, x: *const Data) -> *mut Data;
    #[link_name = "decNumberCopySign"]
    fn copy_sign(result: *mut Data, x: *const Data, y: *const Data) -> *mut Data;
    #[link_name = "decNumberSameQuantum"]
    fn same_quantum(result: *mut Data, x: *const Data, y: *const Data) -> *mut Data;
    #[link_name = "decNumberClass"]
    fn class(x: *const Data, context: *mut Context) -> i32;
    #[link_name = "decNumberClassToString"]
    fn class_string(class: i32) -> *const c_char;
}

/// A C operation with one operand and a context.
type UnaryFunction = unsafe extern "C" fn(*mut Data, *const Data, *mut Context) -> *mut Data;
/// A C operation with one operand and no context.
type PlainFunction = unsafe extern "C" fn(*mut Data, *const Data) -> *mut Data;
/// A C operation with two operands and a context.
type BinaryFunction =
    unsafe extern "C" fn(*mut Data, *const Data, *const Data, *mut Context) -> *mut Data;

/// Runs an operation with one operand. The copies read no context.
/// `Canonical` is a copy, because a number has no encoding: the caller
/// encodes the result canonically.
pub(super) fn unary(operation: Unary, x: &Data, context: &mut Context) -> Data {
    let mut result = Data::zero();
    let function: UnaryFunction = match operation {
        Unary::Abs => abs,
        Unary::Invert => invert,
        Unary::LogB => log_b,
        Unary::Minus => minus,
        Unary::NextMinus => next_minus,
        Unary::NextPlus => next_plus,
        Unary::Plus => plus,
        Unary::Reduce => reduce,
        Unary::ToIntegralExact => to_integral_exact,
        Unary::Canonical | Unary::Copy | Unary::CopyAbs | Unary::CopyNegate => {
            let plain: PlainFunction = match operation {
                Unary::CopyAbs => copy_abs,
                Unary::CopyNegate => copy_negate,
                _ => copy,
            };
            // SAFETY: the pointers name valid, distinct values, and the
            // function only reads `x` and writes `result`.
            unsafe { plain(&raw mut result, x) };
            return result;
        }
    };
    // SAFETY: `result` has room for the digits of the context, and the
    // function only reads `x` and writes the others.
    unsafe { function(&raw mut result, x, context) };
    result
}

/// Runs an operation with two operands. `CopySign` reads no context.
pub(super) fn binary(operation: Binary, x: &Data, y: &Data, context: &mut Context) -> Data {
    let mut result = Data::zero();
    let function: BinaryFunction = match operation {
        Binary::Add => add,
        Binary::And => and,
        Binary::Compare => compare,
        Binary::CompareSignal => compare_signal,
        Binary::CompareTotal => compare_total,
        Binary::CompareTotalMag => compare_total_mag,
        Binary::Divide => divide,
        Binary::DivideInteger => divide_integer,
        Binary::Max => max,
        Binary::MaxMag => max_mag,
        Binary::Min => min,
        Binary::MinMag => min_mag,
        Binary::Multiply => multiply,
        Binary::NextToward => next_toward,
        Binary::Or => or,
        Binary::Quantize => quantize,
        Binary::Remainder => remainder,
        Binary::RemainderNear => remainder_near,
        Binary::Rotate => rotate,
        Binary::ScaleB => scale_b,
        Binary::Shift => shift,
        Binary::Subtract => subtract,
        Binary::Xor => xor,
        Binary::CopySign => {
            // SAFETY: the pointers name valid, distinct values, and the
            // function only reads `x` and `y` and writes `result`.
            unsafe { copy_sign(&raw mut result, x, y) };
            return result;
        }
    };
    // SAFETY: `result` has room for the digits of the context, and the
    // function only reads `x` and `y` and writes the others.
    unsafe { function(&raw mut result, x, y, context) };
    result
}

/// Returns whether two numbers have the same exponent, or are both
/// infinities, or are both NaNs.
pub(super) fn quantum_matches(x: &Data, y: &Data) -> bool {
    let mut result = Data::zero();
    // SAFETY: the function reads `x` and `y` and writes a result of one
    // digit.
    unsafe { same_quantum(&raw mut result, x, y) };
    !result.is_zero()
}

/// Returns the class name of a number in a context, which gives the
/// smallest normal exponent.
pub(super) fn class_name(x: &Data, context: &mut Context) -> &'static str {
    // SAFETY: `decNumberClass` only reads its operands, and
    // `decNumberClassToString` returns one of the string literals
    // `DEC_ClassString_*`, which live for the whole program.
    let name = unsafe { CStr::from_ptr(class_string(class(x, context))) };
    name.to_str().expect("a class name is ASCII")
}
