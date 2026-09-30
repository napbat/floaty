//! The C interface of decNumber, and the format implementations on it.
//!
//! decNumber stores an encoding as an array of bytes in the byte order of
//! the host, which `DECLITEND` fixes to little-endian. The build runs on
//! x86-64 hosts only.

use core::ffi::{CStr, c_char};
use std::ffi::CString;

use super::{
    Arithmetic, Binary, Double, Format, Limited, Outcome, Quad, Rounding, Single, Status, Unary,
    Widening,
};

/// decNumber's `decContext`, for a build without `DECSUBSET`.
#[repr(C)]
struct Context {
    digits: i32,
    emax: i32,
    emin: i32,
    round: u32,
    traps: u32,
    status: u32,
    clamp: u8,
}

unsafe extern "C" {
    /// Sets up a context for a kind of format, with every trap off for the
    /// fixed-size kinds.
    fn decContextDefault(context: *mut Context, kind: i32) -> *mut Context;
}

impl Context {
    /// Returns the context of a fixed-size format, `DEC_INIT_DECSINGLE`,
    /// `DEC_INIT_DECDOUBLE`, or `DEC_INIT_DECQUAD`, with a rounding mode.
    fn new(kind: i32, rounding: Rounding) -> Self {
        let mut context = Self {
            digits: 0,
            emax: 0,
            emin: 0,
            round: 0,
            traps: 0,
            status: 0,
            clamp: 0,
        };
        // SAFETY: `context` is a `decContext`, and `kind` is a kind that
        // decNumber defines, so the call only writes the fields.
        unsafe { decContextDefault(&raw mut context, kind) };
        context.round = rounding.code();
        context
    }

    /// Returns the conditions that the operations raised.
    fn status(&self) -> Status {
        Status(self.status)
    }

    /// Returns a context of the arbitrary-precision numbers, `DEC_INIT_BASE`,
    /// with `digits` digits, a rounding mode, the exponent range of
    /// decNumber's mathematical functions, no clamp, and every trap off.
    ///
    /// `decNumberFMA` rejects a wider exponent range with
    /// `Invalid_context`. The range still holds every exact result of the
    /// fixed-size formats.
    fn numbers(digits: u32, rounding: Rounding) -> Self {
        let mut context = Self::new(BASE_KIND, rounding);
        context.digits = i32::try_from(digits).expect("a working precision fits an i32");
        context.emax = MATH_EXPONENT;
        context.emin = -MATH_EXPONENT;
        context.traps = 0;
        context
    }

    /// Returns the context of a format with `digits` digits and the exponent
    /// range of format `F`, without the clamp, as [`limited`] describes.
    fn limited<F: Format + ?Sized>(digits: u32, rounding: Rounding) -> Self {
        let mut context = Self::numbers(digits, rounding);
        context.emax = F::EMAX;
        context.emin = F::EMIN;
        context
    }
}

/// `DEC_INIT_BASE`, the kind of `decContextDefault` for arbitrary-precision
/// numbers.
const BASE_KIND: i32 = 0;

/// `DEC_MAX_MATH`, the largest exponent of the operands and results of
/// `decNumberFMA`.
const MATH_EXPONENT: i32 = 999_999;

/// The coefficient units of a [`Number`]. A unit holds three digits, so a
/// number has room for 96 digits, more than the `2 * 34 + 4` digits of the
/// widest operation here.
const NUMBER_UNITS: usize = 32;

/// The capacity of the string buffer of a [`Number`]: its digits and 14
/// bytes of sign, point, exponent, and NUL, as `decNumberToString` needs.
const NUMBER_STRING_CAPACITY: usize = 3 * NUMBER_UNITS + 14;

/// decNumber's `decNumber`, built with three digits in each unit of its
/// coefficient, and room for [`NUMBER_UNITS`] units.
#[repr(C)]
struct Number {
    digits: i32,
    exponent: i32,
    bits: u8,
    units: [u16; NUMBER_UNITS],
}

/// The bits of a `decNumber` that mark an infinity or a NaN.
const NUMBER_SPECIAL: u8 = 0x70;

unsafe extern "C" {
    fn decNumberFromString(
        result: *mut Number,
        text: *const c_char,
        context: *mut Context,
    ) -> *mut Number;
    fn decNumberToString(x: *const Number, text: *mut c_char) -> *mut c_char;
    fn decNumberSquareRoot(
        result: *mut Number,
        x: *const Number,
        context: *mut Context,
    ) -> *mut Number;
    fn decNumberFMA(
        result: *mut Number,
        x: *const Number,
        y: *const Number,
        z: *const Number,
        context: *mut Context,
    ) -> *mut Number;
}

impl Number {
    /// Returns a zero, for an operation to write.
    const fn zero() -> Self {
        Self {
            digits: 1,
            exponent: 0,
            bits: 0,
            units: [0; NUMBER_UNITS],
        }
    }

    /// Reads an encoding of format `F` exactly. The context must have at
    /// least the digits of `F`.
    fn read<F: Format + ?Sized>(x: F::Bits, context: &mut Context) -> Self {
        Self::parse(&F::to_string(x), context)
    }

    /// Reads an encoding of format `F` exactly, in a context of its own.
    fn exact<F: Format + ?Sized>(x: F::Bits) -> Self {
        Self::read::<F>(x, &mut Context::numbers(F::PRECISION, Rounding::HalfEven))
    }

    /// Converts a numeric string, rounding to the context.
    fn parse(text: &str, context: &mut Context) -> Self {
        let text = CString::new(text).expect("decNumber writes no NUL character");
        let mut number = Self::zero();
        // SAFETY: `number` has room for the digits of the context, and
        // `text` is a NUL-terminated string.
        unsafe { decNumberFromString(&raw mut number, text.as_ptr(), context) };
        number
    }

    /// Returns the scientific string of the number.
    fn text(&self) -> String {
        let mut buffer: [c_char; NUMBER_STRING_CAPACITY] = [0; NUMBER_STRING_CAPACITY];
        // SAFETY: the buffer has room for the string of a number of
        // `NUMBER_UNITS` units, and decNumber ends it with a NUL.
        let text = unsafe {
            decNumberToString(self, buffer.as_mut_ptr());
            CStr::from_ptr(buffer.as_ptr())
        };
        text.to_str()
            .expect("decNumber writes ASCII strings")
            .to_owned()
    }

    /// Returns whether the number is an infinity or a NaN.
    fn is_special(&self) -> bool {
        self.bits & NUMBER_SPECIAL != 0
    }

    /// Returns whether the number is a zero.
    fn is_zero(&self) -> bool {
        self.digits == 1 && self.units[0] == 0 && !self.is_special()
    }
}

/// Rounds the result of an operation on numbers to format `F`, and adds
/// the conditions of the operation.
fn round_to<F: Format + ?Sized>(
    result: &Number,
    context: &Context,
    rounding: Rounding,
) -> Outcome<F::Bits> {
    let rounded = F::from_string(&result.text(), rounding);
    Outcome {
        value: rounded.value,
        status: (context.status() | rounded.status).without(Status::UNREPORTED),
    }
}

/// Runs an operation on operands of format `F` at `digits` digits, with the
/// exponent range of [`Context::numbers`]. Returns the result and the
/// context, which holds the conditions.
///
/// # Panics
///
/// Panics when the operand count does not match the operation.
fn wide<F: Format + ?Sized>(
    operation: Limited,
    operands: &[F::Bits],
    digits: u32,
    rounding: Rounding,
) -> (Number, Context) {
    let mut context = Context::numbers(digits, rounding);
    let values: Vec<Number> = operands
        .iter()
        .map(|&bits| Number::read::<F>(bits, &mut context))
        .collect();
    let mut result = Number::zero();
    match (operation, values.as_slice()) {
        (Limited::Binary(binary), [x, y]) => result = numbers::binary(binary, x, y, &mut context),
        // SAFETY: `result` has room for the digits of the context, and the
        // pointers name distinct, valid values.
        (Limited::Fma, [x, y, z]) => unsafe {
            decNumberFMA(&raw mut result, x, y, z, &raw mut context);
        },
        // SAFETY: as for the fused multiply-add.
        (Limited::SquareRoot, [x]) => unsafe {
            decNumberSquareRoot(&raw mut result, x, &raw mut context);
        },
        _ => panic!("{operation:?} does not take {} operands", operands.len()),
    }
    (result, context)
}

/// Runs `run` in 05up, which keeps a later rounding to fewer digits
/// correct. A zero result is exact, and its sign comes from a run in
/// `rounding`.
fn prepared(run: impl Fn(Rounding) -> (Number, Context), rounding: Rounding) -> (Number, Context) {
    let (result, context) = run(Rounding::ZeroFiveUp);
    if result.is_zero() {
        run(rounding)
    } else {
        (result, context)
    }
}

/// Returns the square root of `x` in format `F`, as
/// [`Arithmetic::square_root`] describes.
pub(super) fn square_root<F: Format + ?Sized>(x: F::Bits, rounding: Rounding) -> Outcome<F::Bits> {
    let digits = 2 * F::PRECISION + 4;
    let (root, context) = wide::<F>(Limited::SquareRoot, &[x], digits, Rounding::HalfEven);
    round_to::<F>(&root, &context, rounding)
}

/// Returns `x * y + z` in format `F`, as [`Arithmetic::fma_wide`]
/// describes. The operands are finite.
pub(super) fn fma_wide<F: Format + ?Sized>(
    x: F::Bits,
    y: F::Bits,
    z: F::Bits,
    rounding: Rounding,
) -> Outcome<F::Bits> {
    let digits = 2 * F::PRECISION + 3;
    let run = |mode| wide::<F>(Limited::Fma, &[x, y, z], digits, mode);
    let (result, context) = prepared(run, rounding);
    round_to::<F>(&result, &context, rounding)
}

/// Returns the result of an operation on finite operands of format `F`,
/// rounded once to a precision limit, as [`super::limited`] describes.
pub(super) fn limited<F: Format + ?Sized>(
    operation: Limited,
    operands: &[F::Bits],
    digits: u32,
    rounding: Rounding,
) -> Outcome<String> {
    assert!(
        operands
            .iter()
            .all(|&bits| !Number::exact::<F>(bits).is_special()),
        "the precision limit applies only to finite operands"
    );
    let wide_digits = 2 * F::PRECISION + 4;
    let run = |mode| wide::<F>(operation, operands, wide_digits, mode);
    let (result, context) = match operation {
        // decNumber rounds a square root to nearest even in every mode.
        Limited::SquareRoot => run(Rounding::HalfEven),
        _ => prepared(run, rounding),
    };
    let mut limit = Context::limited::<F>(digits, rounding);
    let rounded = Number::parse(&result.text(), &mut limit);
    Outcome {
        value: rounded.text(),
        status: context.status() | limit.status(),
    }
}

/// Orders two encodings of format `F` by the total order of decNumber's
/// arbitrary-precision numbers, of the values or of their magnitudes.
pub(super) fn total_order<F: Format + ?Sized>(
    x: F::Bits,
    y: F::Bits,
    magnitude: bool,
) -> core::cmp::Ordering {
    let (x, y) = (Number::exact::<F>(x), Number::exact::<F>(y));
    let operation = if magnitude {
        Binary::CompareTotalMag
    } else {
        Binary::CompareTotal
    };
    let mut context = Context::numbers(F::PRECISION, Rounding::HalfEven);
    let result = numbers::binary(operation, &x, &y, &mut context);
    match result.text().as_str() {
        "-1" => core::cmp::Ordering::Less,
        "0" => core::cmp::Ordering::Equal,
        "1" => core::cmp::Ordering::Greater,
        other => unreachable!("a total order gives -1, 0, or 1, not {other}"),
    }
}

/// The capacity of a string buffer. The longest decQuad string has 43 bytes
/// with its terminating NUL.
const STRING_CAPACITY: usize = 64;

/// Reads the string that decNumber wrote into a buffer.
fn read_string(buffer: &[c_char; STRING_CAPACITY]) -> String {
    // SAFETY: decNumber terminates the string with a NUL inside the buffer,
    // because the buffer is longer than the longest string of each format.
    let text = unsafe { CStr::from_ptr(buffer.as_ptr()) };
    text.to_str()
        .expect("decNumber writes ASCII strings")
        .to_owned()
}

/// Declares C operations with one operand and a context. The names start
/// with `$c`.
macro_rules! declare_unary {
    ($c:literal: $($name:ident = $suffix:literal),* $(,)?) => {
        unsafe extern "C" {
            $(
                #[link_name = concat!($c, $suffix)]
                fn $name(result: *mut Data, x: *const Data, context: *mut Context) -> *mut Data;
            )*
        }
    };
}

/// Declares C operations with two operands and a context. The names start
/// with `$c`.
macro_rules! declare_binary {
    ($c:literal: $($name:ident = $suffix:literal),* $(,)?) => {
        unsafe extern "C" {
            $(
                #[link_name = concat!($c, $suffix)]
                fn $name(
                    result: *mut Data,
                    x: *const Data,
                    y: *const Data,
                    context: *mut Context,
                ) -> *mut Data;
            )*
        }
    };
}

/// Declares the C type and conversions of a fixed-size format in `$module`,
/// and implements [`Format`] on them. The C names start with `$c`.
macro_rules! dpd_format {
    (
        $format:ident, $module:ident, $c:literal, $bits:ty, $bytes:literal, $align:literal,
        $kind:literal, $precision:literal, $emax:literal, $emin:literal
    ) => {
        mod $module {
            use core::ffi::c_char;

            use super::Context;

            /// The C union of the format: the encoding in host byte order.
            #[repr(C, align($align))]
            #[derive(Clone, Copy)]
            pub(super) struct Data {
                bytes: [u8; $bytes],
            }

            impl Data {
                pub(super) fn new(bits: $bits) -> Self {
                    Self {
                        bytes: bits.to_le_bytes(),
                    }
                }

                pub(super) fn bits(self) -> $bits {
                    <$bits>::from_le_bytes(self.bytes)
                }
            }

            /// The kind of the format for `decContextDefault`.
            pub(super) const KIND: i32 = $kind;

            unsafe extern "C" {
                #[link_name = concat!($c, "FromString")]
                pub(super) fn from_string(
                    result: *mut Data,
                    text: *const c_char,
                    context: *mut Context,
                ) -> *mut Data;
                #[link_name = concat!($c, "ToString")]
                pub(super) fn to_string(x: *const Data, text: *mut c_char) -> *mut c_char;
                #[link_name = concat!($c, "ToEngString")]
                pub(super) fn to_eng_string(x: *const Data, text: *mut c_char) -> *mut c_char;
            }
        }

        impl Format for $format {
            type Bits = $bits;

            const NAME: &'static str = $c;
            const PRECISION: u32 = $precision;
            const EMAX: i32 = $emax;
            const EMIN: i32 = $emin;

            fn from_string(text: &str, rounding: Rounding) -> Outcome<$bits> {
                let text = CString::new(text).expect("a numeric string has no NUL character");
                let mut context = Context::new($module::KIND, rounding);
                let mut result = $module::Data::new(0);
                // SAFETY: `result` and `context` are valid for writes, and
                // `text` is a NUL-terminated string.
                unsafe { $module::from_string(&raw mut result, text.as_ptr(), &raw mut context) };
                Outcome {
                    value: result.bits(),
                    status: context.status(),
                }
            }

            fn to_string(x: $bits) -> String {
                let x = $module::Data::new(x);
                let mut buffer = [0; STRING_CAPACITY];
                // SAFETY: the buffer is longer than the longest string of the
                // format, and every bit pattern is an encoding.
                unsafe { $module::to_string(&raw const x, buffer.as_mut_ptr()) };
                read_string(&buffer)
            }

            fn to_eng_string(x: $bits) -> String {
                let x = $module::Data::new(x);
                let mut buffer = [0; STRING_CAPACITY];
                // SAFETY: as for `to_string`.
                unsafe { $module::to_eng_string(&raw const x, buffer.as_mut_ptr()) };
                read_string(&buffer)
            }
        }
    };
}

/// Declares the C operations of a format that [`dpd_format`] declared in
/// `$module`, in `$operations`, and implements [`Arithmetic`] on them.
macro_rules! dpd_arithmetic {
    ($format:ident, $module:ident, $operations:ident, $c:literal) => {
        mod $operations {
            use core::ffi::c_char;

            use super::super::{Outcome, Status};
            use super::$module::{Data, KIND};
            use super::{Context, Rounding};

            declare_unary!($c:
                abs = "Abs", invert = "Invert", log_b = "LogB", minus = "Minus",
                next_minus = "NextMinus", next_plus = "NextPlus", plus = "Plus",
                reduce = "Reduce", to_integral_exact = "ToIntegralExact",
            );
            declare_binary!($c:
                add = "Add", and = "And", compare = "Compare", compare_signal = "CompareSignal",
                divide = "Divide", divide_integer = "DivideInteger", max = "Max",
                max_mag = "MaxMag", min = "Min", min_mag = "MinMag", multiply = "Multiply",
                next_toward = "NextToward", or = "Or", quantize = "Quantize",
                remainder = "Remainder", remainder_near = "RemainderNear", rotate = "Rotate",
                scale_b = "ScaleB", shift = "Shift", subtract = "Subtract", xor = "Xor",
            );

            unsafe extern "C" {
                #[link_name = concat!($c, "Canonical")]
                fn canonical(result: *mut Data, x: *const Data) -> *mut Data;
                #[link_name = concat!($c, "Copy")]
                fn copy(result: *mut Data, x: *const Data) -> *mut Data;
                #[link_name = concat!($c, "CopyAbs")]
                fn copy_abs(result: *mut Data, x: *const Data) -> *mut Data;
                #[link_name = concat!($c, "CopyNegate")]
                fn copy_negate(result: *mut Data, x: *const Data) -> *mut Data;
                #[link_name = concat!($c, "CompareTotal")]
                fn compare_total(result: *mut Data, x: *const Data, y: *const Data) -> *mut Data;
                #[link_name = concat!($c, "CompareTotalMag")]
                fn compare_total_mag(
                    result: *mut Data,
                    x: *const Data,
                    y: *const Data,
                ) -> *mut Data;
                #[link_name = concat!($c, "CopySign")]
                fn copy_sign(result: *mut Data, x: *const Data, y: *const Data) -> *mut Data;
                #[link_name = concat!($c, "FMA")]
                fn fma(
                    result: *mut Data,
                    x: *const Data,
                    y: *const Data,
                    z: *const Data,
                    context: *mut Context,
                ) -> *mut Data;
                #[link_name = concat!($c, "SameQuantum")]
                fn same_quantum(x: *const Data, y: *const Data) -> u32;
                #[link_name = concat!($c, "ClassString")]
                fn class_string(x: *const Data) -> *const c_char;
            }

            /// A C operation with one operand and a context.
            type UnaryFunction = unsafe extern "C" fn(*mut Data, *const Data, *mut Context) -> *mut Data;
            /// A C operation with one operand and no context.
            type UnaryPlainFunction = unsafe extern "C" fn(*mut Data, *const Data) -> *mut Data;
            /// A C operation with two operands and a context.
            type BinaryFunction =
                unsafe extern "C" fn(*mut Data, *const Data, *const Data, *mut Context) -> *mut Data;
            /// A C operation with two operands and no context.
            type BinaryPlainFunction = unsafe extern "C" fn(*mut Data, *const Data, *const Data) -> *mut Data;

            pub(super) fn unary(operation: super::Unary, x: Data, rounding: Rounding) -> Outcome<Data> {
                use super::Unary as Operation;
                let function: UnaryFunction = match operation {
                    Operation::Abs => abs,
                    Operation::Invert => invert,
                    Operation::LogB => log_b,
                    Operation::Minus => minus,
                    Operation::NextMinus => next_minus,
                    Operation::NextPlus => next_plus,
                    Operation::Plus => plus,
                    Operation::Reduce => reduce,
                    Operation::ToIntegralExact => to_integral_exact,
                    Operation::Canonical => return unary_plain(canonical, x),
                    Operation::Copy => return unary_plain(copy, x),
                    Operation::CopyAbs => return unary_plain(copy_abs, x),
                    Operation::CopyNegate => return unary_plain(copy_negate, x),
                };
                let mut result = Data::new(0);
                let mut context = Context::new(KIND, rounding);
                // SAFETY: the pointers name valid, distinct values, and the
                // function only reads `x` and writes the others.
                unsafe { function(&raw mut result, &raw const x, &raw mut context) };
                Outcome {
                    value: result,
                    status: context.status(),
                }
            }

            fn unary_plain(function: UnaryPlainFunction, x: Data) -> Outcome<Data> {
                let mut result = Data::new(0);
                // SAFETY: the pointers name valid, distinct values, and the
                // function only reads `x` and writes `result`.
                unsafe { function(&raw mut result, &raw const x) };
                Outcome {
                    value: result,
                    status: Status::NONE,
                }
            }

            pub(super) fn binary(
                operation: super::Binary,
                x: Data,
                y: Data,
                rounding: Rounding,
            ) -> Outcome<Data> {
                use super::Binary as Operation;
                let function: BinaryFunction = match operation {
                    Operation::Add => add,
                    Operation::And => and,
                    Operation::Compare => compare,
                    Operation::CompareSignal => compare_signal,
                    Operation::Divide => divide,
                    Operation::DivideInteger => divide_integer,
                    Operation::Max => max,
                    Operation::MaxMag => max_mag,
                    Operation::Min => min,
                    Operation::MinMag => min_mag,
                    Operation::Multiply => multiply,
                    Operation::NextToward => next_toward,
                    Operation::Or => or,
                    Operation::Quantize => quantize,
                    Operation::Remainder => remainder,
                    Operation::RemainderNear => remainder_near,
                    Operation::Rotate => rotate,
                    Operation::ScaleB => scale_b,
                    Operation::Shift => shift,
                    Operation::Subtract => subtract,
                    Operation::Xor => xor,
                    Operation::CompareTotal => return binary_plain(compare_total, x, y),
                    Operation::CompareTotalMag => return binary_plain(compare_total_mag, x, y),
                    Operation::CopySign => return binary_plain(copy_sign, x, y),
                };
                let mut result = Data::new(0);
                let mut context = Context::new(KIND, rounding);
                // SAFETY: the pointers name valid, distinct values, and the
                // function only reads `x` and `y` and writes the others.
                unsafe { function(&raw mut result, &raw const x, &raw const y, &raw mut context) };
                Outcome {
                    value: result,
                    status: context.status(),
                }
            }

            fn binary_plain(function: BinaryPlainFunction, x: Data, y: Data) -> Outcome<Data> {
                let mut result = Data::new(0);
                // SAFETY: the pointers name valid, distinct values, and the
                // function only reads `x` and `y` and writes `result`.
                unsafe { function(&raw mut result, &raw const x, &raw const y) };
                Outcome {
                    value: result,
                    status: Status::NONE,
                }
            }

            pub(super) fn fused(x: Data, y: Data, z: Data, rounding: Rounding) -> Outcome<Data> {
                let mut result = Data::new(0);
                let mut context = Context::new(KIND, rounding);
                // SAFETY: the pointers name valid, distinct values, and the
                // function only reads the operands and writes the others.
                unsafe {
                    fma(
                        &raw mut result,
                        &raw const x,
                        &raw const y,
                        &raw const z,
                        &raw mut context,
                    )
                };
                Outcome {
                    value: result,
                    status: context.status(),
                }
            }

            pub(super) fn quantum_matches(x: Data, y: Data) -> bool {
                // SAFETY: the function only reads its two operands.
                unsafe { same_quantum(&raw const x, &raw const y) != 0 }
            }

            pub(super) fn class_name(x: Data) -> &'static str {
                // SAFETY: the function only reads `x`. It returns one of the
                // string literals `DEC_ClassString_*`, which live for the
                // whole program.
                let name = unsafe { super::CStr::from_ptr(class_string(&raw const x)) };
                name.to_str().expect("a class name is ASCII")
            }
        }

        impl Arithmetic for $format {
            fn unary(operation: Unary, x: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits> {
                let outcome = $operations::unary(operation, $module::Data::new(x), rounding);
                Outcome {
                    value: outcome.value.bits(),
                    status: outcome.status,
                }
            }

            fn binary(
                operation: Binary,
                x: Self::Bits,
                y: Self::Bits,
                rounding: Rounding,
            ) -> Outcome<Self::Bits> {
                let (x, y) = ($module::Data::new(x), $module::Data::new(y));
                let outcome = $operations::binary(operation, x, y, rounding);
                Outcome {
                    value: outcome.value.bits(),
                    status: outcome.status,
                }
            }

            fn fma(
                x: Self::Bits,
                y: Self::Bits,
                z: Self::Bits,
                rounding: Rounding,
            ) -> Outcome<Self::Bits> {
                let outcome = $operations::fused(
                    $module::Data::new(x),
                    $module::Data::new(y),
                    $module::Data::new(z),
                    rounding,
                );
                Outcome {
                    value: outcome.value.bits(),
                    status: outcome.status,
                }
            }

            fn same_quantum(x: Self::Bits, y: Self::Bits) -> bool {
                $operations::quantum_matches($module::Data::new(x), $module::Data::new(y))
            }

            fn class(x: Self::Bits) -> &'static str {
                $operations::class_name($module::Data::new(x))
            }
        }
    };
}

/// Declares the conversions between a format that [`dpd_format`] declared in
/// `$module` and the next wider format, declared in `$wide`, in
/// `$conversions`, and implements [`Widening`] on them.
macro_rules! dpd_widening {
    ($format:ident, $module:ident, $c:literal, $wider:ident, $wide:ident, $conversions:ident) => {
        mod $conversions {
            use super::Context;
            use super::$module::Data;
            use super::$wide::Data as WideData;

            unsafe extern "C" {
                #[link_name = concat!($c, "ToWider")]
                pub(super) fn to_wider(x: *const Data, result: *mut WideData) -> *mut WideData;
                #[link_name = concat!($c, "FromWider")]
                pub(super) fn from_wider(
                    result: *mut Data,
                    x: *const WideData,
                    context: *mut Context,
                ) -> *mut Data;
            }
        }

        impl Widening for $format {
            type Wider = $wider;

            fn to_wider(x: Self::Bits) -> <$wider as Format>::Bits {
                let x = $module::Data::new(x);
                let mut result = $wide::Data::new(0);
                // SAFETY: the pointers name valid, distinct values, and the
                // function only reads `x` and writes `result`.
                unsafe { $conversions::to_wider(&raw const x, &raw mut result) };
                result.bits()
            }

            fn from_wider(x: <$wider as Format>::Bits, rounding: Rounding) -> Outcome<Self::Bits> {
                let x = $wide::Data::new(x);
                let mut result = $module::Data::new(0);
                let mut context = Context::new($module::KIND, rounding);
                // SAFETY: the pointers name valid, distinct values, and the
                // function only reads `x` and writes the others.
                unsafe {
                    $conversions::from_wider(&raw mut result, &raw const x, &raw mut context)
                };
                Outcome {
                    value: result.bits(),
                    status: context.status(),
                }
            }
        }
    };
}

dpd_format!(Single, single, "decSingle", u32, 4, 4, 32, 7, 96, -95);
dpd_format!(Double, double, "decDouble", u64, 8, 8, 64, 16, 384, -383);
dpd_format!(Quad, quad, "decQuad", u128, 16, 8, 128, 34, 6144, -6143);
dpd_arithmetic!(Double, double, double_operations, "decDouble");
dpd_arithmetic!(Quad, quad, quad_operations, "decQuad");
dpd_widening!(
    Single,
    single,
    "decSingle",
    Double,
    double,
    single_conversions
);
dpd_widening!(Double, double, "decDouble", Quad, quad, double_conversions);

mod numbers;

/// decSingle has no arithmetic. The operations of [`Single`] run on the
/// arbitrary-precision numbers in the decimal32 context of
/// `decContextDefault`: 7 digits, `emax` 96, `emin` -95, and the clamp. The
/// result converts to the format exactly.
impl Arithmetic for Single {
    fn unary(operation: Unary, x: u32, rounding: Rounding) -> Outcome<u32> {
        let x = Number::exact::<Self>(x);
        let mut context = Context::new(single::KIND, rounding);
        let result = numbers::unary(operation, &x, &mut context);
        round_to::<Self>(&result, &context, rounding)
    }

    fn binary(operation: Binary, x: u32, y: u32, rounding: Rounding) -> Outcome<u32> {
        let (x, y) = (Number::exact::<Self>(x), Number::exact::<Self>(y));
        let mut context = Context::new(single::KIND, rounding);
        let result = numbers::binary(operation, &x, &y, &mut context);
        round_to::<Self>(&result, &context, rounding)
    }

    fn fma(x: u32, y: u32, z: u32, rounding: Rounding) -> Outcome<u32> {
        let special = [x, y, z]
            .into_iter()
            .any(|bits| Number::exact::<Self>(bits).is_special());
        if !special {
            return fma_wide::<Self>(x, y, z, rounding);
        }
        // The widening is exact, and a special result narrows exactly.
        let wide = Double::fma(
            Self::to_wider(x),
            Self::to_wider(y),
            Self::to_wider(z),
            rounding,
        );
        let narrow = Self::from_wider(wide.value, rounding);
        Outcome {
            value: narrow.value,
            status: wide.status | narrow.status,
        }
    }

    fn same_quantum(x: u32, y: u32) -> bool {
        numbers::quantum_matches(&Number::exact::<Self>(x), &Number::exact::<Self>(y))
    }

    fn class(x: u32) -> &'static str {
        let mut context = Context::new(single::KIND, Rounding::HalfEven);
        numbers::class_name(&Number::exact::<Self>(x), &mut context)
    }
}
