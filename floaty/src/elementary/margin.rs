//! The measurement of the margins that [`crate::margins`] records, for the
//! test feature `record-margins`.

use super::ball::Ball;
use super::{Target, truncate};
use crate::limbs::Widen;
use crate::margins::{self, EXACT, NEVER, Record, Recording};

/// The recording of one evaluation, from its first ball.
pub(super) struct Recorder {
    /// What the thread records.
    recording: Recording,
    /// The margin of the first ball.
    first: Option<i64>,
}

impl Recorder {
    /// Starts the recording of an evaluation when the thread records, and
    /// measures its first ball.
    pub(super) fn start<W: Widen>(first: Option<&Ball<W>>, target: &Target) -> Option<Self> {
        let recording = margins::recording()?;
        Some(Self {
            recording,
            first: first.map(|ball| measure(ball, target, recording.cap)),
        })
    }

    /// Records an evaluation whose first ball decided, with the margin of
    /// the ball that `second` gives when the thread records both.
    pub(super) fn decided<W: Widen>(
        self,
        second: impl FnOnce() -> Option<Ball<W>>,
        target: &Target,
    ) {
        let second = self
            .recording
            .both
            .then(second)
            .flatten()
            .map(|ball| measure(&ball, target, self.recording.cap));
        margins::store(Record {
            first: self.first,
            second,
        });
    }

    /// Records an evaluation that took its second ball.
    pub(super) fn undecided<W: Widen>(self, second: &Ball<W>, target: &Target) {
        margins::store(Record {
            first: self.first,
            second: Some(measure(second, target, self.recording.cap)),
        });
    }
}

/// Returns the margin of a ball, at most `cap`: the greatest `j` such that
/// the ball with its radius times `2^j` decides the truncation. A larger
/// radius decides no more truncations, so a bisection finds `j`.
fn measure<W: Widen>(ball: &Ball<W>, target: &Target, cap: i64) -> i64 {
    if ball.radius().is_zero() {
        return EXACT;
    }
    let decides = |shift: i64| truncate(&ball.radius_shifted(shift), target).is_some();
    // A radius below the lowest bit of the width, or above the value,
    // changes no decision.
    let bound = i64::from(W::BITS) + 64;
    let high = cap.min(bound);
    if decides(high) {
        return high;
    }
    if !decides(-bound) {
        return NEVER;
    }
    let (mut low, mut high) = (-bound, high);
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if decides(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}
