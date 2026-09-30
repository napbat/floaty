//! Dispatch from a behavior chosen at run time to a mode fixed at compile
//! time.
//!
//! An emulator that follows a control register at run time can select one
//! mode type for each state of the register. Each operation of the mode then
//! runs with the fields of the mode as constants. The macros here make that
//! selection for the tests: each arm of the `match` is a separate compiled
//! copy of the body.
//!
//! The macros name every floaty item by a full path, so the body keeps its
//! own names. `with_sse_mode!`, `with_x87_mode!`, and `with_fpcr_mode!` also
//! define the alias `__FloatyRounded` in the scope of the body.

/// Evaluates `$body` with the type `$mode` bound to the mode `$base` with the
/// rounding direction `$rounding`, a [`floaty::Rounding`].
///
/// # Panics
///
/// Panics for a rounding direction that a later floaty adds.
#[macro_export]
macro_rules! with_rounding_mode {
    ($rounding:expr, $base:ty, $mode:ident => $body:expr) => {
        match $rounding {
            $crate::__floaty::Rounding::TiesToEven => {
                type $mode = $crate::__floaty::mode::Rounded<
                    $base,
                    $crate::__floaty::mode::direction::TiesToEven,
                >;
                $body
            }
            $crate::__floaty::Rounding::TiesToAway => {
                type $mode = $crate::__floaty::mode::Rounded<
                    $base,
                    $crate::__floaty::mode::direction::TiesToAway,
                >;
                $body
            }
            $crate::__floaty::Rounding::TiesTowardZero => {
                type $mode = $crate::__floaty::mode::Rounded<
                    $base,
                    $crate::__floaty::mode::direction::TiesTowardZero,
                >;
                $body
            }
            $crate::__floaty::Rounding::TowardPositive => {
                type $mode = $crate::__floaty::mode::Rounded<
                    $base,
                    $crate::__floaty::mode::direction::TowardPositive,
                >;
                $body
            }
            $crate::__floaty::Rounding::TowardNegative => {
                type $mode = $crate::__floaty::mode::Rounded<
                    $base,
                    $crate::__floaty::mode::direction::TowardNegative,
                >;
                $body
            }
            $crate::__floaty::Rounding::TowardZero => {
                type $mode = $crate::__floaty::mode::Rounded<
                    $base,
                    $crate::__floaty::mode::direction::TowardZero,
                >;
                $body
            }
            $crate::__floaty::Rounding::AwayFromZero => {
                type $mode = $crate::__floaty::mode::Rounded<
                    $base,
                    $crate::__floaty::mode::direction::AwayFromZero,
                >;
                $body
            }
            $crate::__floaty::Rounding::ToOdd => {
                type $mode = $crate::__floaty::mode::Rounded<
                    $base,
                    $crate::__floaty::mode::direction::ToOdd,
                >;
                $body
            }
            _ => panic!("the tests know every rounding direction"),
        }
    };
}

/// Evaluates `$body` with the type `$mode` bound to the SSE mode of an MXCSR
/// state: the rounding direction, FTZ, and DAZ. There are 16 such modes.
///
/// # Panics
///
/// Panics for a rounding direction that a later floaty adds.
#[macro_export]
macro_rules! with_sse_mode {
    ($rounding:expr, $ftz:expr, $daz:expr, $mode:ident => $body:expr) => {
        $crate::with_rounding_mode!($rounding, $crate::__floaty::mode::X86Sse, __FloatyRounded => {
            match ($ftz, $daz) {
                (false, false) => {
                    type $mode = $crate::__floaty::mode::DenormalsAreZero<
                        $crate::__floaty::mode::FlushToZero<
                            __FloatyRounded,
                            $crate::__floaty::mode::switch::Off,
                        >,
                        $crate::__floaty::mode::switch::Off,
                    >;
                    $body
                }
                (true, false) => {
                    type $mode = $crate::__floaty::mode::DenormalsAreZero<
                        $crate::__floaty::mode::FlushToZero<
                            __FloatyRounded,
                            $crate::__floaty::mode::switch::On,
                        >,
                        $crate::__floaty::mode::switch::Off,
                    >;
                    $body
                }
                (false, true) => {
                    type $mode = $crate::__floaty::mode::DenormalsAreZero<
                        $crate::__floaty::mode::FlushToZero<
                            __FloatyRounded,
                            $crate::__floaty::mode::switch::Off,
                        >,
                        $crate::__floaty::mode::switch::On,
                    >;
                    $body
                }
                (true, true) => {
                    type $mode = $crate::__floaty::mode::DenormalsAreZero<
                        $crate::__floaty::mode::FlushToZero<
                            __FloatyRounded,
                            $crate::__floaty::mode::switch::On,
                        >,
                        $crate::__floaty::mode::switch::On,
                    >;
                    $body
                }
            }
        })
    };
}

/// Evaluates `$body` with the type `$mode` bound to the x87 mode of a control
/// word state: the rounding direction and the precision control of 24, 53, or
/// 64 bits. There are 12 such modes.
///
/// # Panics
///
/// Panics for a precision that x87 precision control does not have, and for
/// a rounding direction that a later floaty adds.
#[macro_export]
macro_rules! with_x87_mode {
    ($rounding:expr, $precision:expr, $mode:ident => $body:expr) => {
        $crate::with_rounding_mode!($rounding, $crate::__floaty::mode::X87, __FloatyRounded => {
            match $precision {
                24 => {
                    type $mode = $crate::__floaty::mode::Precision<__FloatyRounded, 24>;
                    $body
                }
                53 => {
                    type $mode = $crate::__floaty::mode::Precision<__FloatyRounded, 53>;
                    $body
                }
                64 => {
                    type $mode = $crate::__floaty::mode::Precision<__FloatyRounded, 64>;
                    $body
                }
                _ => panic!("x87 precision control has 24, 53, and 64 bits"),
            }
        })
    };
}

/// Evaluates `$body` with the type `$mode` bound to the default mode with an
/// Arm FPCR state: the rounding direction, the default-NaN mode that DN sets,
/// and a tininess rule. There are 32 such modes.
///
/// # Panics
///
/// Panics for a rounding direction that a later floaty adds.
#[macro_export]
macro_rules! with_fpcr_mode {
    ($rounding:expr, $default_nan:expr, $tininess:expr, $mode:ident => $body:expr) => {
        $crate::with_rounding_mode!($rounding, $crate::__floaty::mode::Ieee, __FloatyRounded => {
            match ($default_nan, $tininess) {
                (false, $crate::__floaty::env::Tininess::BeforeRounding) => {
                    type $mode = $crate::__floaty::mode::DetectTininess<
                        $crate::__floaty::mode::Propagation<
                            __FloatyRounded,
                            $crate::__floaty::mode::propagation::SignalingFirst,
                        >,
                        $crate::__floaty::mode::tininess::BeforeRounding,
                    >;
                    $body
                }
                (false, $crate::__floaty::env::Tininess::AfterRounding) => {
                    type $mode = $crate::__floaty::mode::DetectTininess<
                        $crate::__floaty::mode::Propagation<
                            __FloatyRounded,
                            $crate::__floaty::mode::propagation::SignalingFirst,
                        >,
                        $crate::__floaty::mode::tininess::AfterRounding,
                    >;
                    $body
                }
                (true, $crate::__floaty::env::Tininess::BeforeRounding) => {
                    type $mode = $crate::__floaty::mode::DetectTininess<
                        $crate::__floaty::mode::Propagation<
                            __FloatyRounded,
                            $crate::__floaty::mode::propagation::DefaultNan,
                        >,
                        $crate::__floaty::mode::tininess::BeforeRounding,
                    >;
                    $body
                }
                (true, $crate::__floaty::env::Tininess::AfterRounding) => {
                    type $mode = $crate::__floaty::mode::DetectTininess<
                        $crate::__floaty::mode::Propagation<
                            __FloatyRounded,
                            $crate::__floaty::mode::propagation::DefaultNan,
                        >,
                        $crate::__floaty::mode::tininess::AfterRounding,
                    >;
                    $body
                }
            }
        })
    };
}
