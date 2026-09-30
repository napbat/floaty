//! The lists of the small and the wide formats that the oracle tests run,
//! and the special values of a format, [`Specials`]. A new format joins
//! every one of those tests through its list.
//!
//! Each macro invokes a callback macro once for each format of its list. The
//! callback gets the arguments that follow its name first, and then the
//! parameters of the format. A format alias is an identifier of the floaty
//! crate root, such as `F8E4M3Fn`, so that a callback can name the type as
//! `floaty::$alias` and the format as `stringify!($alias)`. The other items
//! have full paths, so the callback keeps its own names.

/// How a format encodes its special values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Specials {
    /// IEEE 754 infinities and NaNs.
    Ieee,
    /// No infinity; the all-ones significand at `emax` is the NaN.
    NoInf,
    /// No infinity and no negative zero; the one NaN is negative.
    Fnuz,
    /// No infinity and no NaN. Every overflow saturates, and a NaN result is
    /// positive zero.
    Finite,
}

/// Invokes `$callback!` once for each FP8 format, with the arguments that
/// follow `$callback`, the alias, the standard, the width, the special
/// values, the seed of the random operands of the format, and the index of
/// the block of the format in the `ml_dtypes` tables. The block index is the
/// position of the format in `FORMATS` of `scripts/generate_fp8_reference.py`.
#[macro_export]
macro_rules! for_each_fp8_format {
    ($callback:ident $(, $argument:tt)* $(,)?) => {
        $callback!(
            $($argument,)*
            F8E4M3Fn,
            $crate::__floaty::Binary<4, $crate::__floaty::NoInf>,
            8,
            $crate::formats::Specials::NoInf,
            1,
            0
        );
        $callback!(
            $($argument,)*
            F8E5M2,
            $crate::__floaty::Binary<5>,
            8,
            $crate::formats::Specials::Ieee,
            2,
            1
        );
        $callback!(
            $($argument,)*
            F8E4M3Fnuz,
            $crate::__floaty::Binary<4, $crate::__floaty::Fnuz>,
            8,
            $crate::formats::Specials::Fnuz,
            3,
            2
        );
        $callback!(
            $($argument,)*
            F8E5M2Fnuz,
            $crate::__floaty::Binary<5, $crate::__floaty::Fnuz>,
            8,
            $crate::formats::Specials::Fnuz,
            4,
            3
        );
        $callback!(
            $($argument,)*
            F8E4M3,
            $crate::__floaty::Binary<4>,
            8,
            $crate::formats::Specials::Ieee,
            8,
            4
        );
        $callback!(
            $($argument,)*
            F8E3M4,
            $crate::__floaty::Binary<3>,
            8,
            $crate::formats::Specials::Ieee,
            9,
            5
        );
        $callback!(
            $($argument,)*
            F8E4M3B11Fnuz,
            $crate::__floaty::Binary<4, $crate::__floaty::B11Fnuz>,
            8,
            $crate::formats::Specials::Fnuz,
            10,
            6
        );
    };
}

/// Invokes `$callback!` once for each MX format of 4 and 6 bits, with the
/// parameters of [`for_each_fp8_format!`]. The block index is the position of
/// the format in `FORMATS` of `scripts/generate_mx_reference.py`.
#[macro_export]
macro_rules! for_each_mx_format {
    ($callback:ident $(, $argument:tt)* $(,)?) => {
        $callback!(
            $($argument,)*
            F4E2M1Fn,
            $crate::__floaty::Binary<2, $crate::__floaty::Finite>,
            4,
            $crate::formats::Specials::Finite,
            5,
            0
        );
        $callback!(
            $($argument,)*
            F6E2M3Fn,
            $crate::__floaty::Binary<2, $crate::__floaty::Finite>,
            6,
            $crate::formats::Specials::Finite,
            6,
            1
        );
        $callback!(
            $($argument,)*
            F6E3M2Fn,
            $crate::__floaty::Binary<3, $crate::__floaty::Finite>,
            6,
            $crate::formats::Specials::Finite,
            7,
            2
        );
    };
}

/// Invokes `$callback!` once for each format of at most 8 bits: every FP8
/// format of [`for_each_fp8_format!`], then every MX format of
/// [`for_each_mx_format!`].
#[macro_export]
macro_rules! for_each_small_format {
    ($callback:ident $(, $argument:tt)* $(,)?) => {
        $crate::for_each_fp8_format!($callback $(, $argument)*);
        $crate::for_each_mx_format!($callback $(, $argument)*);
    };
}

/// Invokes `$callback!` once for each wide IEEE format from binary160 to
/// binary512, with the arguments that follow `$callback`, the alias, the
/// width, the exponent bits, and the number of 64-bit limbs of the encoding.
/// The standard of each format is `floaty::Binary<$exponent_bits>`.
#[macro_export]
macro_rules! for_each_wide_format {
    ($callback:ident $(, $argument:tt)* $(,)?) => {
        $callback!($($argument,)* F160, 160, 16, 3);
        $callback!($($argument,)* F192, 192, 17, 3);
        $callback!($($argument,)* F224, 224, 18, 4);
        $callback!($($argument,)* F256, 256, 19, 4);
        $callback!($($argument,)* F288, 288, 20, 5);
        $callback!($($argument,)* F320, 320, 20, 5);
        $callback!($($argument,)* F352, 352, 21, 6);
        $callback!($($argument,)* F384, 384, 21, 6);
        $callback!($($argument,)* F416, 416, 22, 7);
        $callback!($($argument,)* F448, 448, 22, 7);
        $callback!($($argument,)* F480, 480, 23, 8);
        $callback!($($argument,)* F512, 512, 23, 8);
    };
}
