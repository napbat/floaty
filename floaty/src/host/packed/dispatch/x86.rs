//! The instruction sets of x86 and x86-64 beyond the build, the check of the
//! processor, and the copies of the entry points compiled for each set.
//!
//! - `V3` is x86-64-v3: AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, and MOVBE.
//!   Its forms are the 256-bit forms of `wide` and the VEX forms of the
//!   128-bit chunks, so no legacy SSE encoding runs beside a 256-bit one.
//! - `V4` is x86-64-v4: `V3` and AVX-512F, AVX-512BW, AVX-512CD, AVX-512DQ,
//!   and AVX-512VL. It adds the 512-bit forms of `avx512`.
//!
//! The feature lists name the features of the levels of the x86-64 psABI,
//! so a copy compiles as a build with `-C target-cpu=x86-64-v3` or `v4`
//! compiles. A set runs only where the processor has every feature, and
//! only when the build does not enable every one already: such a build has
//! the forms in `Build`.
//!
//! 32-bit x86 runs the same two sets. A processor with their features runs
//! the same instructions there, in eight vector registers.

use core::sync::atomic::{AtomicU8, Ordering};

use super::super::super::environment::packed::{avx512, wide};
use super::super::super::{Isa, Operation};
use crate::env::Rounding;
use crate::format::internal::MinMax;
use crate::sealed::Sealed;

/// A larger instruction set than the build, which this processor has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Level {
    /// x86-64-v3.
    V3,
    /// x86-64-v4.
    V4,
}

/// `true` when the build enables every feature of `V3`.
const V3_IN_BUILD: bool = cfg!(all(
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "bmi1",
    target_feature = "bmi2",
    target_feature = "f16c",
    target_feature = "fma",
    target_feature = "lzcnt",
    target_feature = "movbe",
));

/// `true` when the build enables every feature of `V4`.
const V4_IN_BUILD: bool = V3_IN_BUILD
    && cfg!(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512cd",
        target_feature = "avx512dq",
        target_feature = "avx512vl",
    ));

/// Returns `true` when the processor has every feature of `V3`.
fn has_v3() -> bool {
    std::arch::is_x86_feature_detected!("avx")
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("f16c")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("movbe")
}

/// Returns `true` when the processor has every feature of `V4`.
fn has_v4() -> bool {
    has_v3()
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512cd")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

/// The instruction set of a processor whose features `selected` checked.
/// Only `selected` makes a value, so the processor that holds one has the
/// features of its set.
#[derive(Clone, Copy, Debug)]
pub struct Selected(Level);

/// The answer of the check of the processor: 0 before the check, then 1 for
/// `Build`, 2 for `V3`, and 3 for `V4`.
static ANSWER: AtomicU8 = AtomicU8::new(0);

/// Returns the larger instruction set that this processor runs, or `None`
/// for `Build`. The first call checks the processor, and `ANSWER` keeps the
/// answer for every later call. Two threads can both check, and both store
/// the same answer.
#[inline]
pub fn selected() -> Option<Selected> {
    if V4_IN_BUILD {
        return None;
    }
    let answer = match ANSWER.load(Ordering::Relaxed) {
        0 => {
            let answer = check();
            ANSWER.store(answer, Ordering::Relaxed);
            answer
        }
        answer => answer,
    };
    match answer {
        2 => Some(Selected(Level::V3)),
        3 => Some(Selected(Level::V4)),
        _ => None,
    }
}

/// Returns the names of the larger instruction sets that this processor
/// can run, as `Selected::name` names them, from the smaller.
#[cfg(feature = "override-host-level")]
pub fn levels() -> &'static [&'static str] {
    match (!V3_IN_BUILD && has_v3(), !V4_IN_BUILD && has_v4()) {
        (true, true) => &["v3", "v4"],
        (false, true) => &["v4"],
        (true, false) => &["v3"],
        (false, false) => &[],
    }
}

/// Returns the answer of `selected` for this processor: 1 for `Build`, 2
/// for `V3`, and 3 for `V4`. The largest set that the processor has and the
/// build lacks runs.
#[cold]
fn check() -> u8 {
    let largest = host_level_override().unwrap_or(3);
    if largest >= 3 && !V4_IN_BUILD && has_v4() {
        3
    } else if largest >= 2 && !V3_IN_BUILD && has_v3() {
        2
    } else {
        1
    }
}

/// Returns the largest instruction set that the environment variable
/// `FLOATY_HOST_LEVEL` allows, as `check` numbers it, so that the tests run
/// each set on one processor: `build`, `v3`, or `v4`.
///
/// # Panics
///
/// Panics for another value, and for a set that the processor does not
/// have. A test that names a set must run in it.
#[cfg(feature = "override-host-level")]
fn host_level_override() -> Option<u8> {
    let value = std::env::var("FLOATY_HOST_LEVEL").ok()?;
    let (largest, present) = match value.as_str() {
        "build" => (1, true),
        "v3" => (2, has_v3()),
        "v4" => (3, has_v4()),
        _ => panic!("FLOATY_HOST_LEVEL names build, v3, or v4, not {value:?}"),
    };
    assert!(
        present,
        "FLOATY_HOST_LEVEL names {value}, which this processor does not have"
    );
    Some(largest)
}

/// Returns `None`: only the feature `override-host-level` reads the
/// environment.
#[cfg(not(feature = "override-host-level"))]
fn host_level_override() -> Option<u8> {
    None
}

/// Defines the copies of the entry points for the instruction set `$isa`,
/// in the module `$module`, compiled with the features `$features`. Each
/// copy is a function that enables the features for its own code, so the
/// generic code of the entry point inlines into it with the forms of
/// `$isa`.
macro_rules! copies {
    ($(#[$doc:meta])* $module:ident, $isa:ident, $features:literal) => {
        $(#[$doc])*
        mod $module {
            use super::super::super::super::{Host, Load, Step, Term};
            use super::super::super::{convert_chunks_on, elementwise, kernel};
            use super::$isa;
            use crate::env::Mode;
            use crate::float::Float;
            use crate::format::Standard;
            use crate::format::internal::MinMax;

            /// `accumulate_on` in the instruction set.
            ///
            /// # Safety
            ///
            /// The processor must have the features of the instruction set.
            #[target_feature(enable = $features)]
            #[inline]
            pub unsafe fn accumulate<const N: usize>(
                x: impl Load,
                y: impl Load,
                term: Term,
                step: Step,
            ) -> Option<u32> {
                kernel::accumulate_on::<$isa, N>(x, y, term, step)
            }

            /// `rows_on` in the instruction set.
            ///
            /// # Safety
            ///
            /// The processor must have the features of the instruction set.
            #[target_feature(enable = $features)]
            #[inline]
            pub unsafe fn rows<const N: usize>(
                rows: impl Load,
                query: impl Load,
                row_count: usize,
                term: Term,
                each: impl FnMut(usize, Option<u32>),
            ) {
                kernel::rows_on::<$isa, N>(rows, query, row_count, term, each);
            }

            /// `store_on` in the instruction set.
            ///
            /// # Safety
            ///
            /// The processor must have the features of the instruction set.
            #[target_feature(enable = $features)]
            #[inline]
            pub unsafe fn store<const N: usize>(
                values: impl Load,
                each: impl FnMut(usize, usize, Option<&[u32]>),
            ) {
                elementwise::store_on::<$isa, N>(values, each);
            }

            /// `to_int_on` in the instruction set.
            ///
            /// # Safety
            ///
            /// The processor must have the features of the instruction set.
            #[target_feature(enable = $features)]
            #[inline]
            pub unsafe fn to_int<const N: usize>(
                values: impl Load,
                each: impl FnMut(usize, usize, Option<&[i32]>),
            ) {
                elementwise::to_int_on::<$isa, N>(values, each);
            }

            /// `reduce_on` in the instruction set.
            ///
            /// # Safety
            ///
            /// The processor must have the features of the instruction set.
            #[target_feature(enable = $features)]
            #[inline]
            pub unsafe fn reduce<const N: usize>(
                values: impl Load,
                operation: MinMax,
            ) -> Option<u32> {
                elementwise::reduce_on::<$isa, N>(values, operation)
            }

            /// `convert_chunks_on` in the instruction set.
            ///
            /// # Safety
            ///
            /// The processor must have the features of the instruction set.
            #[target_feature(enable = $features)]
            #[inline]
            pub unsafe fn convert_chunks<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
                values: &[Float<S, W, M>],
                to: Host,
                each: impl FnMut(usize, Option<[u64; N]>),
            ) {
                convert_chunks_on::<$isa, S, W, M, N>(values, to, each);
            }
        }
    };
}

copies!(
    /// The copies of the entry points in `V3`.
    v3,
    V3,
    "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe"
);
copies!(
    /// The copies of the entry points in `V4`.
    v4,
    V4,
    "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
);

/// Runs the copy of `$function` of the instruction set of `$selected`.
macro_rules! run {
    ($selected:expr, $function:ident::<$($generic:tt),+>($($argument:expr),*)) => {
        match $selected.0 {
            // SAFETY: a `Selected` of `V3` exists only on a processor with
            // the features of `V3`, as `selected` checked.
            Level::V3 => unsafe { v3::$function::<$($generic),+>($($argument),*) },
            // SAFETY: as for `V3`, with the features of `V4`.
            Level::V4 => unsafe { v4::$function::<$($generic),+>($($argument),*) },
        }
    };
}

impl Selected {
    /// Returns the name of the instruction set: `v3` or `v4`.
    #[cfg(feature = "override-host-level")]
    pub fn name(self) -> &'static str {
        match self.0 {
            Level::V3 => "v3",
            Level::V4 => "v4",
        }
    }

    /// Runs `accumulate_on` in the instruction set.
    #[inline]
    pub fn accumulate<const N: usize>(
        self,
        x: impl super::Load,
        y: impl super::Load,
        term: super::Term,
        step: super::Step,
    ) -> Option<u32> {
        run!(self, accumulate::<N>(x, y, term, step))
    }

    /// Runs `rows_on` in the instruction set.
    #[inline]
    pub fn rows<const N: usize>(
        self,
        rows: impl super::Load,
        query: impl super::Load,
        row_count: usize,
        term: super::Term,
        each: impl FnMut(usize, Option<u32>),
    ) {
        run!(self, rows::<N>(rows, query, row_count, term, each));
    }

    /// Runs `store_on` in the instruction set.
    #[inline]
    pub fn store<const N: usize>(
        self,
        values: impl super::Load,
        each: impl FnMut(usize, usize, Option<&[u32]>),
    ) {
        run!(self, store::<N>(values, each));
    }

    /// Runs `to_int_on` in the instruction set.
    #[inline]
    pub fn to_int<const N: usize>(
        self,
        values: impl super::Load,
        each: impl FnMut(usize, usize, Option<&[i32]>),
    ) {
        run!(self, to_int::<N>(values, each));
    }

    /// Runs `reduce_on` in the instruction set.
    #[inline]
    pub fn reduce<const N: usize>(
        self,
        values: impl super::Load,
        operation: MinMax,
    ) -> Option<u32> {
        run!(self, reduce::<N>(values, operation))
    }

    /// Runs `convert_chunks_on` in the instruction set.
    #[inline]
    pub fn convert_chunks<
        S: crate::format::Standard<W>,
        const W: usize,
        M: crate::env::Mode,
        const N: usize,
    >(
        self,
        values: &[crate::float::Float<S, W, M>],
        to: super::Host,
        each: impl FnMut(usize, Option<[u64; N]>),
    ) {
        run!(self, convert_chunks::<S, W, M, N>(values, to, each));
    }
}

/// x86-64-v3, as the module states. Only the copies of `v3` and `v4` name
/// the type, and a copy runs only on a processor with the features of its
/// set, as `Selected` guarantees. So each form below runs only on a
/// processor with AVX, AVX2, F16C, and FMA.
struct V3;

/// x86-64-v4, as the module states. Only the copies of `v4` name the type,
/// so each form below runs only on a processor with the features of `V4`.
struct V4;

impl Sealed for V3 {}
impl Sealed for V4 {}

/// Defines forms of an instruction set that run the forms of the module
/// `$module`, whose features the processor has, as the comment of the type
/// states. A form whose function returns an `Option` passes it on.
macro_rules! forms {
    ($module:ident: $($name:ident($($argument:ident: $type:ty),+) -> $result:ty;)+) => {
        $(
            #[inline]
            fn $name($($argument: $type),+) -> Option<$result> {
                // SAFETY: the processor has the features of the form, as the
                // comment of the type states.
                Some(unsafe { $module::$name($($argument),+) })
            }
        )+
    };
}

impl Isa for V3 {
    const WIDE: bool = true;
    const EXTRA_WIDE: bool = false;
    const HALF: bool = true;
    const INTEGERS: bool = true;

    forms!(wide:
        binary_f32x4(left: [f32; 4], right: [f32; 4], operation: Operation) -> [f32; 4];
        binary_f32x8(left: [f32; 8], right: [f32; 8], operation: Operation) -> [f32; 8];
        mul_add_f32x4(left: [f32; 4], right: [f32; 4], addend: [f32; 4]) -> [f32; 4];
        mul_add_f32x8(left: [f32; 8], right: [f32; 8], addend: [f32; 8]) -> [f32; 8];
        min_max_f32x4(left: [f32; 4], right: [f32; 4], operation: MinMax) -> [f32; 4];
        min_max_f32x8(left: [f32; 8], right: [f32; 8], operation: MinMax) -> [f32; 8];
        to_int_f32x4(value: [f32; 4]) -> [i32; 4];
        to_int_f32x8(value: [f32; 8]) -> [i32; 8];
        from_int_x4(value: [i32; 4]) -> [f32; 4];
        from_int_x8(value: [i32; 8]) -> [f32; 8];
        widen_halves_x4(value: [u16; 4]) -> [f32; 4];
        widen_halves_x8(value: [u16; 8]) -> [f32; 8];
        narrow_halves_x4(value: [f32; 4]) -> [u16; 4];
        narrow_halves_x8(value: [f32; 8]) -> [u16; 8];
        widen_x2(value: [f32; 2]) -> [f64; 2];
        widen_x4(value: [f32; 4]) -> [f64; 4];
        narrow_x2(value: [f64; 2]) -> [f32; 2];
        narrow_x4(value: [f64; 4]) -> [f32; 4];
    );

    #[inline]
    fn round_f32x4(value: [f32; 4], rounding: Rounding) -> Option<[f32; 4]> {
        // SAFETY: the processor has AVX, as the comment of the type states.
        unsafe { wide::round_f32x4(value, rounding) }
    }

    #[inline]
    fn round_f32x8(value: [f32; 8], rounding: Rounding) -> Option<[f32; 8]> {
        // SAFETY: as in `round_f32x4`.
        unsafe { wide::round_f32x8(value, rounding) }
    }

    #[inline]
    fn binary_f32x16(_: [f32; 16], _: [f32; 16], _: Operation) -> Option<[f32; 16]> {
        None
    }

    #[inline]
    fn mul_add_f32x16(_: [f32; 16], _: [f32; 16], _: [f32; 16]) -> Option<[f32; 16]> {
        None
    }

    #[inline]
    fn min_max_f32x16(_: [f32; 16], _: [f32; 16], _: MinMax) -> Option<[f32; 16]> {
        None
    }

    #[inline]
    fn round_f32x16(_: [f32; 16], _: Rounding) -> Option<[f32; 16]> {
        None
    }

    #[inline]
    fn to_int_f32x16(_: [f32; 16]) -> Option<[i32; 16]> {
        None
    }

    #[inline]
    fn from_int_x16(_: [i32; 16]) -> Option<[f32; 16]> {
        None
    }

    #[inline]
    fn widen_halves_x16(_: [u16; 16]) -> Option<[f32; 16]> {
        None
    }

    #[inline]
    fn narrow_halves_x16(_: [f32; 16]) -> Option<[u16; 16]> {
        None
    }
}

/// Defines forms of `V4` that run the forms of `V3`, whose features `V4`
/// has.
macro_rules! forms_of_v3 {
    ($($name:ident($($argument:ident: $type:ty),+) -> $result:ty;)+) => {
        $(
            #[inline]
            fn $name($($argument: $type),+) -> Option<$result> {
                V3::$name($($argument),+)
            }
        )+
    };
}

impl Isa for V4 {
    const WIDE: bool = true;
    const EXTRA_WIDE: bool = true;
    const HALF: bool = true;
    const INTEGERS: bool = true;

    forms_of_v3!(
        binary_f32x4(left: [f32; 4], right: [f32; 4], operation: Operation) -> [f32; 4];
        binary_f32x8(left: [f32; 8], right: [f32; 8], operation: Operation) -> [f32; 8];
        mul_add_f32x4(left: [f32; 4], right: [f32; 4], addend: [f32; 4]) -> [f32; 4];
        mul_add_f32x8(left: [f32; 8], right: [f32; 8], addend: [f32; 8]) -> [f32; 8];
        min_max_f32x4(left: [f32; 4], right: [f32; 4], operation: MinMax) -> [f32; 4];
        min_max_f32x8(left: [f32; 8], right: [f32; 8], operation: MinMax) -> [f32; 8];
        round_f32x4(value: [f32; 4], rounding: Rounding) -> [f32; 4];
        round_f32x8(value: [f32; 8], rounding: Rounding) -> [f32; 8];
        to_int_f32x4(value: [f32; 4]) -> [i32; 4];
        to_int_f32x8(value: [f32; 8]) -> [i32; 8];
        from_int_x4(value: [i32; 4]) -> [f32; 4];
        from_int_x8(value: [i32; 8]) -> [f32; 8];
        widen_halves_x4(value: [u16; 4]) -> [f32; 4];
        widen_halves_x8(value: [u16; 8]) -> [f32; 8];
        narrow_halves_x4(value: [f32; 4]) -> [u16; 4];
        narrow_halves_x8(value: [f32; 8]) -> [u16; 8];
        widen_x2(value: [f32; 2]) -> [f64; 2];
        widen_x4(value: [f32; 4]) -> [f64; 4];
        narrow_x2(value: [f64; 2]) -> [f32; 2];
        narrow_x4(value: [f64; 4]) -> [f32; 4];
    );

    forms!(avx512:
        binary_f32x16(left: [f32; 16], right: [f32; 16], operation: Operation) -> [f32; 16];
        mul_add_f32x16(left: [f32; 16], right: [f32; 16], addend: [f32; 16]) -> [f32; 16];
        min_max_f32x16(left: [f32; 16], right: [f32; 16], operation: MinMax) -> [f32; 16];
        to_int_f32x16(value: [f32; 16]) -> [i32; 16];
        from_int_x16(value: [i32; 16]) -> [f32; 16];
        widen_halves_x16(value: [u16; 16]) -> [f32; 16];
        narrow_halves_x16(value: [f32; 16]) -> [u16; 16];
    );

    #[inline]
    fn round_f32x16(value: [f32; 16], rounding: Rounding) -> Option<[f32; 16]> {
        // SAFETY: the processor has AVX-512F, as the comment of the type
        // states.
        unsafe { avx512::round_f32x16(value, rounding) }
    }
}
