//! The vectors that the slice kernels and the elementwise slice operations
//! of `Lanes` read on the host unit, and the direction of each rounding of
//! a load as a type.

use super::Isa;
use crate::env::Rounding;
use crate::sealed::Sealed;

/// The direction of each rounding of a slice operation on the host unit, as
/// a type. The copy of a loop for `packed::Nearest` rounds to nearest even
/// in the forms of the environment. The copy for `packed::Directed` rounds
/// in its direction in the forms with a rounding control. The two copies
/// share no function, so LLVM inlines each into its own loop, and the copy
/// to nearest even holds no form of the other. With one copy for both, the
/// store of the update `x * keep + y * eta` on 1,280 values took 360 ns
/// against 71 ns in x86-64-v4 on a Ryzen AI Max+ 395.
pub trait Direction: Sealed + Copy {
    /// `true` for `packed::Nearest`.
    const NEAREST: bool;

    /// Returns the direction.
    fn rounding(self) -> Rounding;
}

/// A vector that a slice kernel of `Lanes` reads on the host unit, as the
/// binary32 encodings of its values. A load computes in the instruction set
/// `I`, with its features: `load` and `load_rest` run `load_on` and
/// `load_rest_on` in [`Isa::run`]. Each step of a load that rounds, such as
/// the sum of a view, rounds in the direction `direction` of the mode.
pub trait Load: Copy {
    /// Returns the number of values.
    fn count(self) -> usize;

    /// Returns the binary32 encodings of the `N` values from `start`, or
    /// `None` when the instruction set has no instruction for the
    /// conversion or the direction. The vector holds at least `start + N`
    /// values.
    #[inline]
    fn load<I: Isa, const N: usize>(
        self,
        start: usize,
        direction: impl Direction,
    ) -> Option<[u32; N]> {
        I::run(|| self.load_on::<I, N>(start, direction))
    }

    /// Returns the binary32 encodings of the values from `start`, with +0
    /// in the lanes past the last value, as [`load`](Self::load) does. The
    /// vector holds more than `start` and fewer than `start + N` values.
    #[inline]
    fn load_rest<I: Isa, const N: usize>(
        self,
        start: usize,
        direction: impl Direction,
    ) -> Option<[u32; N]> {
        I::run(|| self.load_rest_on::<I, N>(start, direction))
    }

    /// Computes [`load`](Self::load) in the instruction set `I`.
    fn load_on<I: Isa, const N: usize>(
        self,
        start: usize,
        direction: impl Direction,
    ) -> Option<[u32; N]>;

    /// Computes [`load_rest`](Self::load_rest) in the instruction set `I`.
    fn load_rest_on<I: Isa, const N: usize>(
        self,
        start: usize,
        direction: impl Direction,
    ) -> Option<[u32; N]>;

    /// Returns the `count` values from `start` as a vector of the same kind.
    /// The vector holds at least `start + count` values.
    #[must_use]
    fn part(self, start: usize, count: usize) -> Self;
}
