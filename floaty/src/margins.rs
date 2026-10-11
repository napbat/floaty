//! The margins of the balls of the elementary functions, which only the test
//! feature `record-margins` records.
//!
//! An elementary function evaluates a ball at the first working precision,
//! and at the second when the first does not decide the truncation. When
//! the second does not decide either, the result rounds from the center of
//! the ball. The margin of a ball is the greatest `j` such that the ball with
//! its radius times `2^j` still decides the truncation. So a negative margin
//! at the second precision marks a result that rounds from the center.
//!
//! [`record`] starts or stops the recording on the calling thread, and
//! [`take`] returns the margins of the last evaluation on that thread.

use std::cell::Cell;

/// The margin of a ball of radius zero.
pub const EXACT: i64 = i64::MAX;

/// The margin of a ball that no radius makes decide.
pub const NEVER: i64 = i64::MIN;

/// What the evaluations on a thread record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Recording {
    /// `true` evaluates the ball at the second precision also when the
    /// first decides, to record its margin.
    pub both: bool,
    /// The largest margin to resolve: a larger margin reads as `cap`. A
    /// small cap saves the search for the exact margin.
    pub cap: i64,
}

/// The margins of the balls of one evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    /// The margin of the ball at the first precision, or `None` when the
    /// function gave no ball there.
    pub first: Option<i64>,
    /// The margin of the ball at the second precision, when the evaluation
    /// or [`Recording::both`] evaluated it.
    pub second: Option<i64>,
}

std::thread_local! {
    static RECORDING: Cell<Option<Recording>> = const { Cell::new(None) };
    static LAST: Cell<Option<Record>> = const { Cell::new(None) };
}

/// Starts the recording of the evaluations on this thread, or stops it for
/// `None`, and clears the last record.
pub fn record(recording: Option<Recording>) {
    RECORDING.with(|cell| cell.set(recording));
    LAST.with(|cell| cell.set(None));
}

/// Returns and clears the margins of the last evaluation on this thread
/// that took a ball, or `None` when none did since the last call.
pub fn take() -> Option<Record> {
    LAST.with(Cell::take)
}

/// Returns the recording of this thread.
pub(crate) fn recording() -> Option<Recording> {
    RECORDING.with(Cell::get)
}

/// Keeps the margins of an evaluation.
pub(crate) fn store(record: Record) {
    LAST.with(|cell| cell.set(Some(record)));
}
