//! The slice kernels of `Lanes`: a sum, a dot product, a sum of squared
//! differences, and a norm, accumulated in binary32 lanes in a fixed order.

use super::Lanes;
use super::vector::{Single, Vector};
use crate::env::{Behavior, Flags, Mode, Override};
use crate::host::{self, Host, Kind, Step, Term};

/// The slice kernels. Each one reads one or two [`Vector`]s and accumulates
/// `N` binary32 lanes in a fixed order, which gives the same bits on every
/// host:
///
/// 1. Every lane starts at +0.
/// 2. Value `i` of each vector converts to binary32, as [`Vector`] states,
///    and the term of value `i` adds into lane `i % N`, in increasing order
///    of `i`. A separate kernel rounds the term, then rounds `lane + term`.
///    A fused kernel adds the product of the term into the lane in one fused
///    multiply-add, rounded once.
/// 3. While more than one lane is left, lane `j` of the low half becomes
///    lane `j` plus lane `j` of the high half.
///
/// The term of [`sum`](Self::sum) is the value `x` itself. The term of
/// [`dot`](Self::dot) is `x * y`. The term of
/// [`distance_square`](Self::distance_square) is `d * d`, with `d = x - y`
/// rounded first. [`norm`](Self::norm) is the square root of the dot product
/// of a vector with itself. [`dot_rows`](Self::dot_rows) and
/// [`distance_square_rows`](Self::distance_square_rows) compute one kernel
/// for each row of a matrix.
///
/// `N` must be a power of two. `Lanes<F32, 8>` matches eight host
/// accumulators, as a loop that LLVM vectorizes keeps them. A larger `N`
/// keeps more independent sums in flight, for example 32 for four 256-bit
/// registers of AVX2.
///
/// The methods without flags take the packed host path where the build has
/// one: the separate kernels with the binary32 arithmetic path, and the
/// fused kernels with the binary32 fused multiply-add path. The path checks
/// the mode and the environment once for the call, and sends a sum that is
/// a NaN to the engine. A `_with` method always runs the engine, and returns
/// the union of the flags of every step.
///
/// ```
/// use floaty::{BF16, F32, Lanes};
///
/// let x = [0x3F80_0000, 0x4000_0000, 0x4040_0000].map(F32::from_bits); // 1, 2, 3
/// let y = [0x4080_0000, 0x40A0_0000, 0x40C0_0000].map(F32::from_bits); // 4, 5, 6
/// assert_eq!(Lanes::<F32, 2>::dot(&x[..], &y[..]).to_bits(), 0x4200_0000); // 32
/// assert_eq!(Lanes::<F32, 2>::distance_square(&x[..], &y[..]).to_bits(), 0x41D8_0000); // 27
///
/// // bfloat16 values widen to binary32 first.
/// let z = [0x3F80, 0x4000, 0x4040].map(BF16::from_bits);
/// assert_eq!(Lanes::<F32, 2>::dot_fused(&z[..], &z[..]).to_bits(), 0x4160_0000); // 14
/// ```
///
/// # Panics
///
/// The kernels of two vectors panic when the vectors have different counts
/// of values. The kernels of rows panic when the matrix does not hold one
/// row of the length of the query for each result.
impl<M: Mode, const N: usize> Lanes<Single<M>, N> {
    /// Returns the sum of the values of `x` in the order of the kernels,
    /// with the default mode.
    #[must_use]
    #[inline]
    pub fn sum(x: impl Vector) -> Single<M> {
        Self::kernel(x, x, Term::Value, Step::Separate)
    }

    /// Returns the sum of the values of `x` as [`sum`](Self::sum) computes
    /// it, and the flags.
    #[must_use]
    pub fn sum_with(x: impl Vector, behavior: impl Override) -> (Single<M>, Flags) {
        Self::engine(x, x, Term::Value, Step::Separate, behavior.apply::<M>())
    }

    /// Returns the dot product of `x` and `y` in the order of the kernels,
    /// with the product and the sum of each step rounded apart, with the
    /// default mode.
    #[must_use]
    #[inline]
    pub fn dot(x: impl Vector, y: impl Vector) -> Single<M> {
        Self::kernel(x, y, Term::Product, Step::Separate)
    }

    /// Returns the dot product of `x` and `y` as [`dot`](Self::dot) computes
    /// it, and the flags.
    #[must_use]
    pub fn dot_with(x: impl Vector, y: impl Vector, behavior: impl Override) -> (Single<M>, Flags) {
        Self::engine(x, y, Term::Product, Step::Separate, behavior.apply::<M>())
    }

    /// Returns the dot product of `x` and `y` in the order of the kernels,
    /// with each step a fused multiply-add, with the default mode.
    #[must_use]
    #[inline]
    pub fn dot_fused(x: impl Vector, y: impl Vector) -> Single<M> {
        Self::kernel(x, y, Term::Product, Step::Fused)
    }

    /// Returns the dot product of `x` and `y` as
    /// [`dot_fused`](Self::dot_fused) computes it, and the flags.
    #[must_use]
    pub fn dot_fused_with(
        x: impl Vector,
        y: impl Vector,
        behavior: impl Override,
    ) -> (Single<M>, Flags) {
        Self::engine(x, y, Term::Product, Step::Fused, behavior.apply::<M>())
    }

    /// Returns the sum of the squares of the differences of `x` and `y`, the
    /// square of the Euclidean distance, in the order of the kernels, with
    /// each step rounded apart, with the default mode.
    #[must_use]
    #[inline]
    pub fn distance_square(x: impl Vector, y: impl Vector) -> Single<M> {
        Self::kernel(x, y, Term::SquareDifference, Step::Separate)
    }

    /// Returns the square of the distance of `x` and `y` as
    /// [`distance_square`](Self::distance_square) computes it, and the
    /// flags.
    #[must_use]
    pub fn distance_square_with(
        x: impl Vector,
        y: impl Vector,
        behavior: impl Override,
    ) -> (Single<M>, Flags) {
        Self::engine(
            x,
            y,
            Term::SquareDifference,
            Step::Separate,
            behavior.apply::<M>(),
        )
    }

    /// Returns the square of the distance of `x` and `y` in the order of the
    /// kernels, with the difference rounded and then a fused multiply-add of
    /// it with itself, with the default mode.
    #[must_use]
    #[inline]
    pub fn distance_square_fused(x: impl Vector, y: impl Vector) -> Single<M> {
        Self::kernel(x, y, Term::SquareDifference, Step::Fused)
    }

    /// Returns the square of the distance of `x` and `y` as
    /// [`distance_square_fused`](Self::distance_square_fused) computes it,
    /// and the flags.
    #[must_use]
    pub fn distance_square_fused_with(
        x: impl Vector,
        y: impl Vector,
        behavior: impl Override,
    ) -> (Single<M>, Flags) {
        Self::engine(
            x,
            y,
            Term::SquareDifference,
            Step::Fused,
            behavior.apply::<M>(),
        )
    }

    /// Returns the Euclidean norm of `x`: the square root of
    /// [`dot`](Self::dot) of `x` with itself, with the default mode.
    #[must_use]
    #[inline]
    pub fn norm(x: impl Vector) -> Single<M> {
        Self::dot(x, x).sqrt()
    }

    /// Returns the norm of `x` as [`norm`](Self::norm) computes it, and the
    /// flags.
    #[must_use]
    pub fn norm_with(x: impl Vector, behavior: impl Override) -> (Single<M>, Flags) {
        Self::root(x, Step::Separate, behavior.apply::<M>())
    }

    /// Returns the Euclidean norm of `x`: the square root of
    /// [`dot_fused`](Self::dot_fused) of `x` with itself, with the default
    /// mode.
    #[must_use]
    #[inline]
    pub fn norm_fused(x: impl Vector) -> Single<M> {
        Self::dot_fused(x, x).sqrt()
    }

    /// Returns the norm of `x` as [`norm_fused`](Self::norm_fused) computes
    /// it, and the flags.
    #[must_use]
    pub fn norm_fused_with(x: impl Vector, behavior: impl Override) -> (Single<M>, Flags) {
        Self::root(x, Step::Fused, behavior.apply::<M>())
    }

    /// Writes into `out[r]` the [`dot`](Self::dot) product of row `r` of
    /// `rows` with `query`, with the default mode. Row `r` is the values of
    /// `rows` from `r * n` to `r * n + n - 1`, where `n` is the count of
    /// `query`. One check of the environment serves every row.
    ///
    /// ```
    /// use floaty::{F32, Lanes};
    ///
    /// let rows = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let query = [1.0_f32, 1.0];
    /// let mut out = [0.0_f32; 3];
    /// Lanes::<F32, 8>::dot_rows(&rows[..], &query[..], &mut out);
    /// assert_eq!(out.map(f32::to_bits), [3.0_f32, 7.0, 11.0].map(f32::to_bits));
    /// ```
    #[inline]
    pub fn dot_rows<T: From<Single<M>>>(rows: impl Vector, query: impl Vector, out: &mut [T]) {
        Self::rows(rows, query, Term::Product, out);
    }

    /// Writes into `out[r]` the dot product of row `r` of `rows` with
    /// `query`, as [`dot_rows`](Self::dot_rows) computes it, and returns the
    /// union of the flags of every row.
    pub fn dot_rows_with<T: From<Single<M>>>(
        rows: impl Vector,
        query: impl Vector,
        out: &mut [T],
        behavior: impl Override,
    ) -> Flags {
        Self::rows_with(rows, query, Term::Product, out, behavior.apply::<M>())
    }

    /// Writes into `out[r]` the
    /// [`distance_square`](Self::distance_square) of row `r` of `rows` and
    /// `query`, with the default mode, with the rows of
    /// [`dot_rows`](Self::dot_rows).
    #[inline]
    pub fn distance_square_rows<T: From<Single<M>>>(
        rows: impl Vector,
        query: impl Vector,
        out: &mut [T],
    ) {
        Self::rows(rows, query, Term::SquareDifference, out);
    }

    /// Writes into `out[r]` the square of the distance of row `r` of `rows`
    /// and `query`, as [`distance_square_rows`](Self::distance_square_rows)
    /// computes it, and returns the union of the flags of every row.
    pub fn distance_square_rows_with<T: From<Single<M>>>(
        rows: impl Vector,
        query: impl Vector,
        out: &mut [T],
        behavior: impl Override,
    ) -> Flags {
        Self::rows_with(
            rows,
            query,
            Term::SquareDifference,
            out,
            behavior.apply::<M>(),
        )
    }

    /// Returns `true` when this processor has the host path of the kernels
    /// with `step`. The separate kernels take a constant: a call of the
    /// `const fn` at run time stays a call across crates. The fused kernels
    /// also run where the processor has a larger instruction set than the
    /// build, as the check of the processor finds.
    #[inline]
    fn on_host(step: Step) -> bool {
        match step {
            Step::Separate => const { host::packed::available(Host::Single, Kind::Arithmetic) },
            Step::Fused => host::packed::fused_kernels(),
        }
    }

    /// Runs a kernel with the default mode: on the host unit where the build
    /// and the environment allow it, and in the engine otherwise.
    #[inline]
    fn kernel(x: impl Vector, y: impl Vector, term: Term, step: Step) -> Single<M> {
        const { assert!(N.is_power_of_two(), "the lane count is a power of two") };
        assert_eq!(
            x.count(),
            y.count(),
            "the vectors of a kernel have one count"
        );
        if !Self::on_host(step) {
            return Self::engine(x, y, term, step, M::default()).0;
        }
        match host::packed::accumulate::<N>(x, y, term, step, &M::ENV) {
            Some(bits) => Single::from_bits(bits),
            None => Self::engine_out_of_line(x, y, term, step),
        }
    }

    /// Runs a kernel in the engine with the default mode, for a call whose
    /// host path declines. The call stays out of line, so that the host path
    /// inlines into its caller.
    #[cold]
    #[inline(never)]
    fn engine_out_of_line(x: impl Vector, y: impl Vector, term: Term, step: Step) -> Single<M> {
        Self::engine(x, y, term, step, M::default()).0
    }

    /// Runs a kernel in the engine, in the order of the kernels.
    fn engine<B: Behavior>(
        x: impl Vector,
        y: impl Vector,
        term: Term,
        step: Step,
        behavior: B,
    ) -> (Single<M>, Flags) {
        const { assert!(N.is_power_of_two(), "the lane count is a power of two") };
        let count = x.count();
        assert_eq!(count, y.count(), "the vectors of a kernel have one count");
        let mut lanes = [Single::<M>::from_bits(0); N];
        let mut flags = Flags::NONE;
        let mut run = |(value, step_flags): (Single<M>, Flags)| {
            flags |= step_flags;
            value
        };
        for index in 0..count {
            let a = run(x.value_with(index, behavior));
            let lane = &mut lanes[index % N];
            let (left, right) = match term {
                Term::Value => {
                    *lane = run(lane.add_with(a, behavior));
                    continue;
                }
                Term::Product => (a, run(y.value_with(index, behavior))),
                Term::SquareDifference => {
                    let b = run(y.value_with(index, behavior));
                    let difference = run(a.sub_with(b, behavior));
                    (difference, difference)
                }
            };
            *lane = match step {
                Step::Separate => {
                    let product = run(left.mul_with(right, behavior));
                    run(lane.add_with(product, behavior))
                }
                Step::Fused => run(left.mul_add_with(right, *lane, behavior)),
            };
        }
        let mut half = N / 2;
        while half > 0 {
            let (low, high) = lanes.split_at_mut(half);
            for (low, &high) in low.iter_mut().zip(&*high) {
                *low = run(low.add_with(high, behavior));
            }
            half /= 2;
        }
        (lanes[0], flags)
    }

    /// Runs a norm in the engine: the square root of the dot product of `x`
    /// with itself.
    fn root<B: Behavior>(x: impl Vector, step: Step, behavior: B) -> (Single<M>, Flags) {
        let (sum, sum_flags) = Self::engine(x, x, Term::Product, step, behavior);
        let (root, root_flags) = sum.sqrt_with(behavior);
        (root, sum_flags | root_flags)
    }

    /// Checks that `rows` holds one row of the count of `query` for each of
    /// `row_count` results.
    fn check_rows(rows: impl Vector, query: impl Vector, row_count: usize) {
        const { assert!(N.is_power_of_two(), "the lane count is a power of two") };
        let values = query.count().checked_mul(row_count);
        assert_eq!(
            values,
            Some(rows.count()),
            "the matrix holds one row per result"
        );
    }

    /// Runs a separate kernel of `term` on each row with the default mode:
    /// on the host unit where the build and the environment allow it, and in
    /// the engine otherwise.
    #[inline]
    fn rows<T: From<Single<M>>>(rows: impl Vector, query: impl Vector, term: Term, out: &mut [T]) {
        Self::check_rows(rows, query, out.len());
        let count = query.count();
        if Self::on_host(Step::Separate) {
            let done = host::packed::accumulate_rows::<N>(
                rows,
                query,
                out.len(),
                term,
                &M::ENV,
                |index, bits| {
                    out[index] = match bits {
                        Some(bits) => Single::from_bits(bits),
                        None => Self::engine_out_of_line(
                            rows.part(index * count, count),
                            query,
                            term,
                            Step::Separate,
                        ),
                    }
                    .into();
                },
            );
            if done.is_some() {
                return;
            }
        }
        Self::rows_with(rows, query, term, out, M::default());
    }

    /// Runs a separate kernel of `term` on each row in the engine, and
    /// returns the union of the flags.
    fn rows_with<T: From<Single<M>>, B: Behavior>(
        rows: impl Vector,
        query: impl Vector,
        term: Term,
        out: &mut [T],
        behavior: B,
    ) -> Flags {
        Self::check_rows(rows, query, out.len());
        let count = query.count();
        let mut flags = Flags::NONE;
        for (index, result) in out.iter_mut().enumerate() {
            let row = rows.part(index * count, count);
            let (value, row_flags) = Self::engine(row, query, term, Step::Separate, behavior);
            *result = value.into();
            flags |= row_flags;
        }
        flags
    }
}
