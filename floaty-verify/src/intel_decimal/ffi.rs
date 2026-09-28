//! The C interface of the Intel library, and the format implementations on
//! it.
//!
//! The build sets `CALL_BY_REF=0`, so every value passes by value. A
//! function that rounds takes the rounding direction after its values, and a
//! function that reports flags then takes a pointer to them. The signatures follow
//! `bid_functions.h` under these settings, and `bid_conf.h` gives every name
//! the prefix `__`. The shim `shim/intel_binary80.c` passes the C
//! `long double` of the binary80 conversions through a 16-byte buffer.

use core::ffi::{CStr, c_char};
use std::ffi::CString;

use super::{
    Bid32, Bid64, Bid128, Class, Flags, Format, Inexact, Integer, Layout, Outcome, Predicate,
    Rounding,
};

/// The C `BID_UINT128`: two 64-bit words, the least significant first,
/// aligned to 16 bytes as the library declares it. It also holds binary128
/// values.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Word128 {
    words: [u64; 2],
}

/// A value and the C type that passes it to the library.
trait Abi: Copy {
    /// The C type.
    type C: Copy;

    /// Returns the C value.
    fn to_c(self) -> Self::C;

    /// Returns the value of a C value.
    fn from_c(c: Self::C) -> Self;
}

impl Abi for u32 {
    type C = u32;

    fn to_c(self) -> u32 {
        self
    }

    fn from_c(c: u32) -> Self {
        c
    }
}

impl Abi for u64 {
    type C = u64;

    fn to_c(self) -> u64 {
        self
    }

    fn from_c(c: u64) -> Self {
        c
    }
}

impl Abi for u128 {
    type C = Word128;

    fn to_c(self) -> Word128 {
        let low = u64::try_from(self & u128::from(u64::MAX)).expect("the mask leaves 64 bits");
        let high = u64::try_from(self >> 64).expect("the shift leaves 64 bits");
        Word128 { words: [low, high] }
    }

    fn from_c(c: Word128) -> Self {
        (u128::from(c.words[1]) << 64) | u128::from(c.words[0])
    }
}

// The helpers below call a library function that takes only values, and
// possibly a rounding direction and a pointer to the flags. Such a function
// reads its arguments and writes only the flags. Each caller passes a
// library function of the shape of the helper.

/// Calls a function with one value, a rounding direction, and the flags.
fn call_1r<A, R>(
    function: unsafe extern "C" fn(A, u32, *mut u32) -> R,
    a: A,
    rounding: Rounding,
) -> Outcome<R> {
    let mut flags = 0;
    // SAFETY: the function reads its arguments and writes only the flags,
    // which live until it returns.
    let value = unsafe { function(a, rounding.code(), &raw mut flags) };
    Outcome {
        value,
        flags: Flags::from_bits(flags),
    }
}

/// Calls a function with two values, a rounding direction, and the flags.
fn call_2r<A, B, R>(
    function: unsafe extern "C" fn(A, B, u32, *mut u32) -> R,
    a: A,
    b: B,
    rounding: Rounding,
) -> Outcome<R> {
    let mut flags = 0;
    // SAFETY: as for `call_1r`.
    let value = unsafe { function(a, b, rounding.code(), &raw mut flags) };
    Outcome {
        value,
        flags: Flags::from_bits(flags),
    }
}

/// Calls a function with three values, a rounding direction, and the flags.
fn call_3r<A, B, C, R>(
    function: unsafe extern "C" fn(A, B, C, u32, *mut u32) -> R,
    a: A,
    b: B,
    c: C,
    rounding: Rounding,
) -> Outcome<R> {
    let mut flags = 0;
    // SAFETY: as for `call_1r`.
    let value = unsafe { function(a, b, c, rounding.code(), &raw mut flags) };
    Outcome {
        value,
        flags: Flags::from_bits(flags),
    }
}

/// Calls a function with one value and the flags.
fn call_1f<A, R>(function: unsafe extern "C" fn(A, *mut u32) -> R, a: A) -> Outcome<R> {
    let mut flags = 0;
    // SAFETY: as for `call_1r`.
    let value = unsafe { function(a, &raw mut flags) };
    Outcome {
        value,
        flags: Flags::from_bits(flags),
    }
}

/// Calls a function with two values and the flags.
fn call_2f<A, B, R>(function: unsafe extern "C" fn(A, B, *mut u32) -> R, a: A, b: B) -> Outcome<R> {
    let mut flags = 0;
    // SAFETY: as for `call_1r`.
    let value = unsafe { function(a, b, &raw mut flags) };
    Outcome {
        value,
        flags: Flags::from_bits(flags),
    }
}

/// Calls a function with one value and no flags.
fn call_1<A, R>(function: unsafe extern "C" fn(A) -> R, a: A) -> R {
    // SAFETY: the function only reads its argument.
    unsafe { function(a) }
}

/// Calls a function with two values and no flags.
fn call_2<A, B, R>(function: unsafe extern "C" fn(A, B) -> R, a: A, b: B) -> R {
    // SAFETY: the function only reads its arguments.
    unsafe { function(a, b) }
}

/// The capacity of a string buffer. The longest string has a sign, 34
/// digits, `E`, a sign, 4 exponent digits, and a NUL.
const STRING_CAPACITY: usize = 64;

/// A 16-byte buffer of the binary80 shim. The low 10 bytes hold the x87
/// encoding, the least significant byte first.
type Binary80 = [u8; 16];

/// Declares a module with the conversions to one integer type in ten
/// functions: five rounding directions, without and with inexact.
macro_rules! to_integer {
    ($w:literal, $c:ty, $module:ident, $name:literal, $integer:ty) => {
        pub(super) mod $module {
            use crate::intel_decimal::{Inexact, Outcome, Rounding};

            unsafe extern "C" {
                #[link_name = concat!("__bid", $w, "_to_", $name, "_rnint")]
                fn rnint(x: $c, flags: *mut u32) -> $integer;
                #[link_name = concat!("__bid", $w, "_to_", $name, "_rninta")]
                fn rninta(x: $c, flags: *mut u32) -> $integer;
                #[link_name = concat!("__bid", $w, "_to_", $name, "_int")]
                fn int(x: $c, flags: *mut u32) -> $integer;
                #[link_name = concat!("__bid", $w, "_to_", $name, "_floor")]
                fn floor(x: $c, flags: *mut u32) -> $integer;
                #[link_name = concat!("__bid", $w, "_to_", $name, "_ceil")]
                fn ceil(x: $c, flags: *mut u32) -> $integer;
                #[link_name = concat!("__bid", $w, "_to_", $name, "_xrnint")]
                fn xrnint(x: $c, flags: *mut u32) -> $integer;
                #[link_name = concat!("__bid", $w, "_to_", $name, "_xrninta")]
                fn xrninta(x: $c, flags: *mut u32) -> $integer;
                #[link_name = concat!("__bid", $w, "_to_", $name, "_xint")]
                fn xint(x: $c, flags: *mut u32) -> $integer;
                #[link_name = concat!("__bid", $w, "_to_", $name, "_xfloor")]
                fn xfloor(x: $c, flags: *mut u32) -> $integer;
                #[link_name = concat!("__bid", $w, "_to_", $name, "_xceil")]
                fn xceil(x: $c, flags: *mut u32) -> $integer;
            }

            pub(in crate::intel_decimal::ffi) fn convert(
                x: $c,
                rounding: Rounding,
                inexact: Inexact,
            ) -> Outcome<i128> {
                let function: unsafe extern "C" fn($c, *mut u32) -> $integer =
                    match (inexact, rounding) {
                        (Inexact::Ignored, Rounding::NearestEven) => rnint,
                        (Inexact::Ignored, Rounding::NearestAway) => rninta,
                        (Inexact::Ignored, Rounding::TowardZero) => int,
                        (Inexact::Ignored, Rounding::TowardNegative) => floor,
                        (Inexact::Ignored, Rounding::TowardPositive) => ceil,
                        (Inexact::Signaled, Rounding::NearestEven) => xrnint,
                        (Inexact::Signaled, Rounding::NearestAway) => xrninta,
                        (Inexact::Signaled, Rounding::TowardZero) => xint,
                        (Inexact::Signaled, Rounding::TowardNegative) => xfloor,
                        (Inexact::Signaled, Rounding::TowardPositive) => xceil,
                    };
                crate::intel_decimal::ffi::call_1f(function, x).map(i128::from)
            }
        }
    };
}

/// Declares a conversion from an integer type. A `rounded` conversion takes
/// a rounding direction and reports flags; an `exact` one takes neither,
/// because every value of the integer type fits the format.
macro_rules! from_integer {
    (rounded, $w:literal, $c:ty, $name:ident, $integer:ty) => {
        pub(super) fn $name(
            value: $integer,
            rounding: crate::intel_decimal::Rounding,
        ) -> crate::intel_decimal::Outcome<$c> {
            unsafe extern "C" {
                #[link_name = concat!("__bid", $w, "_", stringify!($name))]
                fn function(value: $integer, rounding: u32, flags: *mut u32) -> $c;
            }
            crate::intel_decimal::ffi::call_1r(function, value, rounding)
        }
    };
    (exact, $w:literal, $c:ty, $name:ident, $integer:ty) => {
        pub(super) fn $name(
            value: $integer,
            _: crate::intel_decimal::Rounding,
        ) -> crate::intel_decimal::Outcome<$c> {
            unsafe extern "C" {
                #[link_name = concat!("__bid", $w, "_", stringify!($name))]
                fn function(value: $integer) -> $c;
            }
            crate::intel_decimal::Outcome {
                value: crate::intel_decimal::ffi::call_1(function, value),
                flags: crate::intel_decimal::Flags::NONE,
            }
        }
    };
}

/// Declares the comparison predicates, which take two values and the flags.
macro_rules! predicates {
    ($w:literal, $c:ty, $($name:ident),* $(,)?) => {
        unsafe extern "C" {
            $(
                #[link_name = concat!("__bid", $w, "_", stringify!($name))]
                pub(super) fn $name(x: $c, y: $c, flags: *mut u32) -> i32;
            )*
        }
    };
}

/// Declares the C functions of one width in `$module`. The shapes name the
/// four conversions from integers, in the order `int32`, `uint32`, `int64`,
/// `uint64`.
macro_rules! bid_declarations {
    (
        $module:ident, $w:literal, $c:ty,
        [$int32:ident, $uint32:ident, $int64:ident, $uint64:ident]
    ) => {
        mod $module {
            use crate::intel_decimal::ffi::{Binary80, Word128};

            unsafe extern "C" {
                #[link_name = concat!("__bid", $w, "_add")]
                pub(super) fn add(x: $c, y: $c, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_sub")]
                pub(super) fn sub(x: $c, y: $c, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_mul")]
                pub(super) fn mul(x: $c, y: $c, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_div")]
                pub(super) fn div(x: $c, y: $c, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_fma")]
                pub(super) fn fma(x: $c, y: $c, z: $c, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_sqrt")]
                pub(super) fn sqrt(x: $c, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_quantize")]
                pub(super) fn quantize(x: $c, y: $c, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_scalbn")]
                pub(super) fn scalbn(x: $c, n: i32, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_round_integral_exact")]
                pub(super) fn round_integral_exact(x: $c, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_nearbyint")]
                pub(super) fn nearbyint(x: $c, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_round_integral_nearest_even")]
                pub(super) fn round_integral_nearest_even(x: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_round_integral_nearest_away")]
                pub(super) fn round_integral_nearest_away(x: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_round_integral_zero")]
                pub(super) fn round_integral_zero(x: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_round_integral_negative")]
                pub(super) fn round_integral_negative(x: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_round_integral_positive")]
                pub(super) fn round_integral_positive(x: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_rem")]
                pub(super) fn rem(x: $c, y: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_logb")]
                pub(super) fn logb(x: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_ilogb")]
                pub(super) fn ilogb(x: $c, flags: *mut u32) -> i32;
                #[link_name = concat!("__bid", $w, "_quantum")]
                pub(super) fn quantum(x: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_quantexp")]
                pub(super) fn quantexp(x: $c, flags: *mut u32) -> i32;
                #[link_name = concat!("__bid", $w, "_nextup")]
                pub(super) fn nextup(x: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_nextdown")]
                pub(super) fn nextdown(x: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_minnum")]
                pub(super) fn minnum(x: $c, y: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_maxnum")]
                pub(super) fn maxnum(x: $c, y: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_minnum_mag")]
                pub(super) fn minnum_mag(x: $c, y: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_maxnum_mag")]
                pub(super) fn maxnum_mag(x: $c, y: $c, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_totalOrder")]
                pub(super) fn total_order(x: $c, y: $c) -> i32;
                #[link_name = concat!("__bid", $w, "_totalOrderMag")]
                pub(super) fn total_order_mag(x: $c, y: $c) -> i32;
                #[link_name = concat!("__bid", $w, "_sameQuantum")]
                pub(super) fn same_quantum(x: $c, y: $c) -> i32;
                #[link_name = concat!("__bid", $w, "_abs")]
                pub(super) fn abs(x: $c) -> $c;
                #[link_name = concat!("__bid", $w, "_negate")]
                pub(super) fn negate(x: $c) -> $c;
                #[link_name = concat!("__bid", $w, "_copySign")]
                pub(super) fn copy_sign(x: $c, y: $c) -> $c;
                #[link_name = concat!("__bid", $w, "_class")]
                pub(super) fn class(x: $c) -> i32;
                #[link_name = concat!("__bid", $w, "_isCanonical")]
                pub(super) fn is_canonical(x: $c) -> i32;
                #[link_name = concat!("__bid_to_dpd", $w)]
                pub(super) fn to_dpd(x: $c) -> $c;
                #[link_name = concat!("__bid_dpd_to_bid", $w)]
                pub(super) fn from_dpd(x: $c) -> $c;
                #[link_name = concat!("__binary32_to_bid", $w)]
                pub(super) fn from_binary32(x: f32, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("__binary64_to_bid", $w)]
                pub(super) fn from_binary64(x: f64, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("floaty_binary80_to_bid", $w)]
                pub(super) fn from_binary80(
                    x: *const Binary80,
                    rounding: u32,
                    flags: *mut u32,
                ) -> $c;
                #[link_name = concat!("__binary128_to_bid", $w)]
                pub(super) fn from_binary128(x: Word128, rounding: u32, flags: *mut u32) -> $c;
                #[link_name = concat!("__bid", $w, "_to_binary32")]
                pub(super) fn to_binary32(x: $c, rounding: u32, flags: *mut u32) -> f32;
                #[link_name = concat!("__bid", $w, "_to_binary64")]
                pub(super) fn to_binary64(x: $c, rounding: u32, flags: *mut u32) -> f64;
                #[link_name = concat!("floaty_bid", $w, "_to_binary80")]
                pub(super) fn to_binary80(
                    result: *mut Binary80,
                    x: $c,
                    rounding: u32,
                    flags: *mut u32,
                );
                #[link_name = concat!("__bid", $w, "_to_binary128")]
                pub(super) fn to_binary128(x: $c, rounding: u32, flags: *mut u32) -> Word128;
                #[link_name = concat!("__bid", $w, "_from_string")]
                pub(super) fn from_string(
                    text: *mut core::ffi::c_char,
                    rounding: u32,
                    flags: *mut u32,
                ) -> $c;
                #[link_name = concat!("__bid", $w, "_to_string")]
                pub(super) fn to_string(text: *mut core::ffi::c_char, x: $c, flags: *mut u32);
            }

            predicates!(
                $w,
                $c,
                quiet_equal,
                quiet_greater,
                quiet_greater_equal,
                quiet_greater_unordered,
                quiet_less,
                quiet_less_equal,
                quiet_less_unordered,
                quiet_not_equal,
                quiet_not_greater,
                quiet_not_less,
                quiet_ordered,
                quiet_unordered,
                signaling_greater,
                signaling_greater_equal,
                signaling_greater_unordered,
                signaling_less,
                signaling_less_equal,
                signaling_less_unordered,
                signaling_not_greater,
                signaling_not_less,
            );

            to_integer!($w, $c, int8, "int8", i8);
            to_integer!($w, $c, int16, "int16", i16);
            to_integer!($w, $c, int32, "int32", i32);
            to_integer!($w, $c, int64, "int64", i64);
            to_integer!($w, $c, uint8, "uint8", u8);
            to_integer!($w, $c, uint16, "uint16", u16);
            to_integer!($w, $c, uint32, "uint32", u32);
            to_integer!($w, $c, uint64, "uint64", u64);

            from_integer!($int32, $w, $c, from_int32, i32);
            from_integer!($uint32, $w, $c, from_uint32, u32);
            from_integer!($int64, $w, $c, from_int64, i64);
            from_integer!($uint64, $w, $c, from_uint64, u64);
        }
    };
}

/// Implements [`Format`] on the C functions that [`bid_declarations`]
/// declared in `$module`.
macro_rules! bid_format {
    ($format:ident, $module:ident, $w:literal, $bits:ty, $c:ty) => {
        impl Format for $format {
            type Bits = $bits;

            const NAME: &'static str = concat!("bid", $w);

            const LAYOUT: Layout = Layout {
                width: <$bits>::BITS,
            };

            fn add(x: $bits, y: $bits, rounding: Rounding) -> Outcome<$bits> {
                call_2r($module::add, x.to_c(), y.to_c(), rounding).map(<$bits>::from_c)
            }

            fn sub(x: $bits, y: $bits, rounding: Rounding) -> Outcome<$bits> {
                call_2r($module::sub, x.to_c(), y.to_c(), rounding).map(<$bits>::from_c)
            }

            fn mul(x: $bits, y: $bits, rounding: Rounding) -> Outcome<$bits> {
                call_2r($module::mul, x.to_c(), y.to_c(), rounding).map(<$bits>::from_c)
            }

            fn div(x: $bits, y: $bits, rounding: Rounding) -> Outcome<$bits> {
                call_2r($module::div, x.to_c(), y.to_c(), rounding).map(<$bits>::from_c)
            }

            fn fma(x: $bits, y: $bits, z: $bits, rounding: Rounding) -> Outcome<$bits> {
                call_3r($module::fma, x.to_c(), y.to_c(), z.to_c(), rounding).map(<$bits>::from_c)
            }

            fn sqrt(x: $bits, rounding: Rounding) -> Outcome<$bits> {
                call_1r($module::sqrt, x.to_c(), rounding).map(<$bits>::from_c)
            }

            fn quantize(x: $bits, y: $bits, rounding: Rounding) -> Outcome<$bits> {
                call_2r($module::quantize, x.to_c(), y.to_c(), rounding).map(<$bits>::from_c)
            }

            fn scalbn(x: $bits, n: i32, rounding: Rounding) -> Outcome<$bits> {
                call_2r($module::scalbn, x.to_c(), n, rounding).map(<$bits>::from_c)
            }

            fn round_integral_exact(x: $bits, rounding: Rounding) -> Outcome<$bits> {
                call_1r($module::round_integral_exact, x.to_c(), rounding).map(<$bits>::from_c)
            }

            fn nearbyint(x: $bits, rounding: Rounding) -> Outcome<$bits> {
                call_1r($module::nearbyint, x.to_c(), rounding).map(<$bits>::from_c)
            }

            fn round_integral(x: $bits, rounding: Rounding) -> Outcome<$bits> {
                let function: unsafe extern "C" fn($c, *mut u32) -> $c = match rounding {
                    Rounding::NearestEven => $module::round_integral_nearest_even,
                    Rounding::NearestAway => $module::round_integral_nearest_away,
                    Rounding::TowardZero => $module::round_integral_zero,
                    Rounding::TowardNegative => $module::round_integral_negative,
                    Rounding::TowardPositive => $module::round_integral_positive,
                };
                call_1f(function, x.to_c()).map(<$bits>::from_c)
            }

            fn rem(x: $bits, y: $bits) -> Outcome<$bits> {
                call_2f($module::rem, x.to_c(), y.to_c()).map(<$bits>::from_c)
            }

            fn logb(x: $bits) -> Outcome<$bits> {
                call_1f($module::logb, x.to_c()).map(<$bits>::from_c)
            }

            fn ilogb(x: $bits) -> Outcome<i32> {
                call_1f($module::ilogb, x.to_c())
            }

            fn quantum(x: $bits) -> Outcome<$bits> {
                call_1f($module::quantum, x.to_c()).map(<$bits>::from_c)
            }

            fn quantexp(x: $bits) -> Outcome<i32> {
                call_1f($module::quantexp, x.to_c())
            }

            fn nextup(x: $bits) -> Outcome<$bits> {
                call_1f($module::nextup, x.to_c()).map(<$bits>::from_c)
            }

            fn nextdown(x: $bits) -> Outcome<$bits> {
                call_1f($module::nextdown, x.to_c()).map(<$bits>::from_c)
            }

            fn minnum(x: $bits, y: $bits) -> Outcome<$bits> {
                call_2f($module::minnum, x.to_c(), y.to_c()).map(<$bits>::from_c)
            }

            fn maxnum(x: $bits, y: $bits) -> Outcome<$bits> {
                call_2f($module::maxnum, x.to_c(), y.to_c()).map(<$bits>::from_c)
            }

            fn minnum_mag(x: $bits, y: $bits) -> Outcome<$bits> {
                call_2f($module::minnum_mag, x.to_c(), y.to_c()).map(<$bits>::from_c)
            }

            fn maxnum_mag(x: $bits, y: $bits) -> Outcome<$bits> {
                call_2f($module::maxnum_mag, x.to_c(), y.to_c()).map(<$bits>::from_c)
            }

            fn compare(x: $bits, y: $bits, predicate: Predicate) -> Outcome<bool> {
                let function: unsafe extern "C" fn($c, $c, *mut u32) -> i32 = match predicate {
                    Predicate::QuietEqual => $module::quiet_equal,
                    Predicate::QuietGreater => $module::quiet_greater,
                    Predicate::QuietGreaterEqual => $module::quiet_greater_equal,
                    Predicate::QuietGreaterUnordered => $module::quiet_greater_unordered,
                    Predicate::QuietLess => $module::quiet_less,
                    Predicate::QuietLessEqual => $module::quiet_less_equal,
                    Predicate::QuietLessUnordered => $module::quiet_less_unordered,
                    Predicate::QuietNotEqual => $module::quiet_not_equal,
                    Predicate::QuietNotGreater => $module::quiet_not_greater,
                    Predicate::QuietNotLess => $module::quiet_not_less,
                    Predicate::QuietOrdered => $module::quiet_ordered,
                    Predicate::QuietUnordered => $module::quiet_unordered,
                    Predicate::SignalingGreater => $module::signaling_greater,
                    Predicate::SignalingGreaterEqual => $module::signaling_greater_equal,
                    Predicate::SignalingGreaterUnordered => $module::signaling_greater_unordered,
                    Predicate::SignalingLess => $module::signaling_less,
                    Predicate::SignalingLessEqual => $module::signaling_less_equal,
                    Predicate::SignalingLessUnordered => $module::signaling_less_unordered,
                    Predicate::SignalingNotGreater => $module::signaling_not_greater,
                    Predicate::SignalingNotLess => $module::signaling_not_less,
                };
                call_2f(function, x.to_c(), y.to_c()).map(|value| value != 0)
            }

            fn total_order(x: $bits, y: $bits) -> bool {
                call_2($module::total_order, x.to_c(), y.to_c()) != 0
            }

            fn total_order_mag(x: $bits, y: $bits) -> bool {
                call_2($module::total_order_mag, x.to_c(), y.to_c()) != 0
            }

            fn same_quantum(x: $bits, y: $bits) -> bool {
                call_2($module::same_quantum, x.to_c(), y.to_c()) != 0
            }

            fn abs(x: $bits) -> $bits {
                <$bits>::from_c(call_1($module::abs, x.to_c()))
            }

            fn negate(x: $bits) -> $bits {
                <$bits>::from_c(call_1($module::negate, x.to_c()))
            }

            fn copy_sign(x: $bits, y: $bits) -> $bits {
                <$bits>::from_c(call_2($module::copy_sign, x.to_c(), y.to_c()))
            }

            fn class(x: $bits) -> Class {
                Class::from_code(call_1($module::class, x.to_c()))
                    .expect("the library returns a value of class_types")
            }

            fn is_canonical(x: $bits) -> bool {
                call_1($module::is_canonical, x.to_c()) != 0
            }

            fn to_integer(
                x: $bits,
                integer: Integer,
                rounding: Rounding,
                inexact: Inexact,
            ) -> Outcome<i128> {
                let x = x.to_c();
                match integer {
                    Integer::Int8 => $module::int8::convert(x, rounding, inexact),
                    Integer::Int16 => $module::int16::convert(x, rounding, inexact),
                    Integer::Int32 => $module::int32::convert(x, rounding, inexact),
                    Integer::Int64 => $module::int64::convert(x, rounding, inexact),
                    Integer::UInt8 => $module::uint8::convert(x, rounding, inexact),
                    Integer::UInt16 => $module::uint16::convert(x, rounding, inexact),
                    Integer::UInt32 => $module::uint32::convert(x, rounding, inexact),
                    Integer::UInt64 => $module::uint64::convert(x, rounding, inexact),
                }
            }

            fn from_int32(value: i32, rounding: Rounding) -> Outcome<$bits> {
                $module::from_int32(value, rounding).map(<$bits>::from_c)
            }

            fn from_uint32(value: u32, rounding: Rounding) -> Outcome<$bits> {
                $module::from_uint32(value, rounding).map(<$bits>::from_c)
            }

            fn from_int64(value: i64, rounding: Rounding) -> Outcome<$bits> {
                $module::from_int64(value, rounding).map(<$bits>::from_c)
            }

            fn from_uint64(value: u64, rounding: Rounding) -> Outcome<$bits> {
                $module::from_uint64(value, rounding).map(<$bits>::from_c)
            }

            fn to_dpd(x: $bits) -> $bits {
                <$bits>::from_c(call_1($module::to_dpd, x.to_c()))
            }

            fn from_dpd(x: $bits) -> $bits {
                <$bits>::from_c(call_1($module::from_dpd, x.to_c()))
            }

            fn from_binary32(bits: u32, rounding: Rounding) -> Outcome<$bits> {
                call_1r($module::from_binary32, f32::from_bits(bits), rounding).map(<$bits>::from_c)
            }

            fn from_binary64(bits: u64, rounding: Rounding) -> Outcome<$bits> {
                call_1r($module::from_binary64, f64::from_bits(bits), rounding).map(<$bits>::from_c)
            }

            fn from_binary80(bits: u128, rounding: Rounding) -> Outcome<$bits> {
                let buffer: Binary80 = bits.to_le_bytes();
                let mut flags = 0;
                // SAFETY: the shim reads the 16-byte buffer and writes only
                // the flags.
                let value = unsafe {
                    $module::from_binary80(&raw const buffer, rounding.code(), &raw mut flags)
                };
                Outcome {
                    value: <$bits>::from_c(value),
                    flags: Flags::from_bits(flags),
                }
            }

            fn from_binary128(bits: u128, rounding: Rounding) -> Outcome<$bits> {
                call_1r($module::from_binary128, bits.to_c(), rounding).map(<$bits>::from_c)
            }

            fn to_binary32(x: $bits, rounding: Rounding) -> Outcome<u32> {
                call_1r($module::to_binary32, x.to_c(), rounding).map(f32::to_bits)
            }

            fn to_binary64(x: $bits, rounding: Rounding) -> Outcome<u64> {
                call_1r($module::to_binary64, x.to_c(), rounding).map(f64::to_bits)
            }

            fn to_binary80(x: $bits, rounding: Rounding) -> Outcome<u128> {
                let mut buffer: Binary80 = [0; 16];
                let mut flags = 0;
                // SAFETY: the shim writes the 16-byte buffer and the flags.
                unsafe {
                    $module::to_binary80(&raw mut buffer, x.to_c(), rounding.code(), &raw mut flags)
                };
                Outcome {
                    value: u128::from_le_bytes(buffer),
                    flags: Flags::from_bits(flags),
                }
            }

            fn to_binary128(x: $bits, rounding: Rounding) -> Outcome<u128> {
                call_1r($module::to_binary128, x.to_c(), rounding).map(u128::from_c)
            }

            fn from_string(text: &str, rounding: Rounding) -> Outcome<$bits> {
                let mut text = CString::new(text)
                    .expect("a numeric string has no NUL character")
                    .into_bytes_with_nul();
                let mut flags = 0;
                // SAFETY: the function reads the NUL-terminated string in
                // the buffer and writes the flags. It can also write to the
                // buffer, so the buffer is a mutable copy of the string.
                let value = unsafe {
                    $module::from_string(
                        text.as_mut_ptr().cast::<c_char>(),
                        rounding.code(),
                        &raw mut flags,
                    )
                };
                Outcome {
                    value: <$bits>::from_c(value),
                    flags: Flags::from_bits(flags),
                }
            }

            fn to_string(x: $bits) -> Outcome<String> {
                let mut buffer: [c_char; STRING_CAPACITY] = [0; STRING_CAPACITY];
                let mut flags = 0;
                // SAFETY: the buffer is longer than the longest string, and
                // the function writes only the string and the flags.
                unsafe { $module::to_string(buffer.as_mut_ptr(), x.to_c(), &raw mut flags) };
                // SAFETY: the function terminates the string with a NUL
                // inside the buffer.
                let text = unsafe { CStr::from_ptr(buffer.as_ptr()) };
                Outcome {
                    value: text
                        .to_str()
                        .expect("the library writes ASCII strings")
                        .to_owned(),
                    flags: Flags::from_bits(flags),
                }
            }
        }
    };
}

bid_declarations!(bid32, "32", u32, [rounded, rounded, rounded, rounded]);
bid_declarations!(bid64, "64", u64, [exact, exact, rounded, rounded]);
bid_declarations!(
    bid128,
    "128",
    crate::intel_decimal::ffi::Word128,
    [exact, exact, exact, exact]
);
bid_format!(Bid32, bid32, "32", u32, u32);
bid_format!(Bid64, bid64, "64", u64, u64);
bid_format!(Bid128, bid128, "128", u128, Word128);

/// The C conversions between the widths.
mod widths {
    use super::Word128;

    unsafe extern "C" {
        #[link_name = "__bid32_to_bid64"]
        pub(super) fn bid32_to_bid64(x: u32, flags: *mut u32) -> u64;
        #[link_name = "__bid32_to_bid128"]
        pub(super) fn bid32_to_bid128(x: u32, flags: *mut u32) -> Word128;
        #[link_name = "__bid64_to_bid32"]
        pub(super) fn bid64_to_bid32(x: u64, rounding: u32, flags: *mut u32) -> u32;
        #[link_name = "__bid64_to_bid128"]
        pub(super) fn bid64_to_bid128(x: u64, flags: *mut u32) -> Word128;
        #[link_name = "__bid128_to_bid32"]
        pub(super) fn bid128_to_bid32(x: Word128, rounding: u32, flags: *mut u32) -> u32;
        #[link_name = "__bid128_to_bid64"]
        pub(super) fn bid128_to_bid64(x: Word128, rounding: u32, flags: *mut u32) -> u64;
    }
}

/// `bid32_to_bid64`: the conversion from decimal32 to decimal64.
#[must_use]
pub fn bid32_to_bid64(x: u32) -> Outcome<u64> {
    call_1f(widths::bid32_to_bid64, x)
}

/// `bid32_to_bid128`: the conversion from decimal32 to decimal128.
#[must_use]
pub fn bid32_to_bid128(x: u32) -> Outcome<u128> {
    call_1f(widths::bid32_to_bid128, x).map(u128::from_c)
}

/// `bid64_to_bid32`: the conversion from decimal64 to decimal32.
#[must_use]
pub fn bid64_to_bid32(x: u64, rounding: Rounding) -> Outcome<u32> {
    call_1r(widths::bid64_to_bid32, x, rounding)
}

/// `bid64_to_bid128`: the conversion from decimal64 to decimal128.
#[must_use]
pub fn bid64_to_bid128(x: u64) -> Outcome<u128> {
    call_1f(widths::bid64_to_bid128, x).map(u128::from_c)
}

/// `bid128_to_bid32`: the conversion from decimal128 to decimal32.
#[must_use]
pub fn bid128_to_bid32(x: u128, rounding: Rounding) -> Outcome<u32> {
    call_1r(widths::bid128_to_bid32, x.to_c(), rounding)
}

/// `bid128_to_bid64`: the conversion from decimal128 to decimal64.
#[must_use]
pub fn bid128_to_bid64(x: u128, rounding: Rounding) -> Outcome<u64> {
    call_1r(widths::bid128_to_bid64, x.to_c(), rounding)
}
