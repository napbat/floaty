//! The operations of `Lanes` of the binary formats alone: `log_b`, the
//! algebraic functions, the augmented operations, and the NaN payload
//! operations. Each lane takes the scalar operation.

use super::{Lanes, with_flags};
use crate::env::{Flags, Mode, Override};
use crate::float::{Augmented, Float};
use crate::format::{Binary, Encoding, Standard, Storage, Width};

/// A float of a binary format.
type BinaryFloat<const E: u32, Enc, const W: usize, M> = Float<Binary<E, Enc>, W, M>;

impl<const E: u32, Enc: Encoding, const W: usize, M: Mode, const N: usize>
    Lanes<BinaryFloat<E, Enc, W, M>, N>
where
    Width<W>: Storage,
    Binary<E, Enc>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
{
    /// Applies `operation` to each lane and the integer of its lane.
    fn map_counted(
        self,
        counts: [i64; N],
        mut operation: impl FnMut(BinaryFloat<E, Enc, W, M>, i64) -> BinaryFloat<E, Enc, W, M>,
    ) -> Self {
        Self::new(core::array::from_fn(|index| {
            operation(self.lanes[index], counts[index])
        }))
    }

    /// Applies `operation`, which returns flags, to each lane and the integer
    /// of its lane, and returns the lanes and the union of the flags.
    fn map_counted_with(
        self,
        counts: [i64; N],
        mut operation: impl FnMut(BinaryFloat<E, Enc, W, M>, i64) -> (BinaryFloat<E, Enc, W, M>, Flags),
    ) -> (Self, Flags) {
        let pairs: [_; N] = core::array::from_fn(|index| (self.lanes[index], counts[index]));
        let (lanes, flags) = with_flags(pairs, |(lane, count)| operation(lane, count));
        (Self::new(lanes), flags)
    }

    /// Applies an augmented operation to each pair of lanes, and returns the
    /// heads and the tails as lanes, with the union of the flags.
    fn augmented_with(
        self,
        other: Self,
        operation: impl FnMut(
            BinaryFloat<E, Enc, W, M>,
            BinaryFloat<E, Enc, W, M>,
        ) -> (Augmented<BinaryFloat<E, Enc, W, M>>, Flags),
    ) -> (Augmented<Self>, Flags) {
        let (pairs, flags) = self.pairs_with(other, operation);
        let lanes = Augmented {
            head: Self::new(pairs.map(|pair| pair.head)),
            tail: Self::new(pairs.map(|pair| pair.tail)),
        };
        (lanes, flags)
    }

    /// Returns the exponent of the leading bit of each lane, as
    /// [`Float::log_b`] does.
    #[must_use]
    pub fn log_b(self) -> Self {
        self.map(<BinaryFloat<E, Enc, W, M>>::log_b)
    }

    /// Returns the exponent of the leading bit of each lane, as
    /// [`Float::log_b_with`] does, and the union of the flags.
    #[must_use]
    pub fn log_b_with(self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.map_with(|lane| lane.log_b_with(behavior))
    }

    /// Returns `sqrt(x^2 + y^2)` of each pair of lanes, with the default mode.
    #[must_use]
    pub fn hypot(self, other: Self) -> Self {
        self.zip(other, Float::hypot)
    }

    /// Returns `sqrt(x^2 + y^2)` of each pair of lanes, as
    /// [`Float::hypot_with`] does, and the union of the flags.
    #[must_use]
    pub fn hypot_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.zip_with(other, |x, y| x.hypot_with(y, behavior))
    }

    /// Returns `1 / sqrt(x)` of each lane, with the default mode.
    #[must_use]
    pub fn reciprocal_sqrt(self) -> Self {
        self.map(Float::reciprocal_sqrt)
    }

    /// Returns `1 / sqrt(x)` of each lane, as [`Float::reciprocal_sqrt_with`]
    /// does, and the union of the flags.
    #[must_use]
    pub fn reciprocal_sqrt_with(self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.map_with(|lane| lane.reciprocal_sqrt_with(behavior))
    }

    /// Returns `x^n` of each lane and the `n` of its lane, with the default
    /// mode.
    #[must_use]
    pub fn pown(self, n: [i64; N]) -> Self {
        self.map_counted(n, Float::pown)
    }

    /// Returns `x^n` of each lane and the `n` of its lane, as
    /// [`Float::pown_with`] does, and the union of the flags.
    #[must_use]
    pub fn pown_with(self, n: [i64; N], behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.map_counted_with(n, |lane, n| lane.pown_with(n, behavior))
    }

    /// Returns `x^(1/n)` of each lane and the `n` of its lane, with the
    /// default mode.
    #[must_use]
    pub fn rootn(self, n: [i64; N]) -> Self {
        self.map_counted(n, Float::rootn)
    }

    /// Returns `x^(1/n)` of each lane and the `n` of its lane, as
    /// [`Float::rootn_with`] does, and the union of the flags.
    #[must_use]
    pub fn rootn_with(self, n: [i64; N], behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.map_counted_with(n, |lane, n| lane.rootn_with(n, behavior))
    }

    /// Returns `(1 + x)^n` of each lane and the `n` of its lane, with the
    /// default mode.
    #[must_use]
    pub fn compound(self, n: [i64; N]) -> Self {
        self.map_counted(n, Float::compound)
    }

    /// Returns `(1 + x)^n` of each lane and the `n` of its lane, as
    /// [`Float::compound_with`] does, and the union of the flags.
    #[must_use]
    pub fn compound_with(self, n: [i64; N], behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.map_counted_with(n, |lane, n| lane.compound_with(n, behavior))
    }

    /// Returns `self + other` of each pair of lanes as a head and a tail,
    /// with the default mode.
    #[must_use]
    pub fn augmented_add(self, other: Self) -> Augmented<Self> {
        self.augmented_add_with(other, M::default()).0
    }

    /// Returns `self + other` of each pair of lanes as a head and a tail, as
    /// [`Float::augmented_add_with`] does, and the union of the flags.
    #[must_use]
    pub fn augmented_add_with(
        self,
        other: Self,
        behavior: impl Override,
    ) -> (Augmented<Self>, Flags) {
        let behavior = behavior.apply::<M>();
        self.augmented_with(other, |x, y| x.augmented_add_with(y, behavior))
    }

    /// Returns `self - other` of each pair of lanes as a head and a tail,
    /// with the default mode.
    #[must_use]
    pub fn augmented_sub(self, other: Self) -> Augmented<Self> {
        self.augmented_sub_with(other, M::default()).0
    }

    /// Returns `self - other` of each pair of lanes as a head and a tail, as
    /// [`Float::augmented_sub_with`] does, and the union of the flags.
    #[must_use]
    pub fn augmented_sub_with(
        self,
        other: Self,
        behavior: impl Override,
    ) -> (Augmented<Self>, Flags) {
        let behavior = behavior.apply::<M>();
        self.augmented_with(other, |x, y| x.augmented_sub_with(y, behavior))
    }

    /// Returns `self * other` of each pair of lanes as a head and a tail,
    /// with the default mode.
    #[must_use]
    pub fn augmented_mul(self, other: Self) -> Augmented<Self> {
        self.augmented_mul_with(other, M::default()).0
    }

    /// Returns `self * other` of each pair of lanes as a head and a tail, as
    /// [`Float::augmented_mul_with`] does, and the union of the flags.
    #[must_use]
    pub fn augmented_mul_with(
        self,
        other: Self,
        behavior: impl Override,
    ) -> (Augmented<Self>, Flags) {
        let behavior = behavior.apply::<M>();
        self.augmented_with(other, |x, y| x.augmented_mul_with(y, behavior))
    }

    /// Returns the payload of each lane, as [`Float::payload`] does.
    #[must_use]
    pub fn payload(self) -> Self {
        self.map(<BinaryFloat<E, Enc, W, M>>::payload)
    }

    /// Returns the quiet NaN with the payload of each lane, as
    /// [`Float::from_payload`] does.
    #[must_use]
    pub fn from_payload(payload: Self) -> Self {
        payload.map(<BinaryFloat<E, Enc, W, M>>::from_payload)
    }

    /// Returns the signaling NaN with the payload of each lane, as
    /// [`Float::from_payload_signaling`] does.
    #[must_use]
    pub fn from_payload_signaling(payload: Self) -> Self {
        payload.map(<BinaryFloat<E, Enc, W, M>>::from_payload_signaling)
    }
}
