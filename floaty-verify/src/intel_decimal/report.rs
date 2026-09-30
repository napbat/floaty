//! The report of the random and the conversion tests against the library:
//! the cases of each group, the failures, the skipped cases, and the flags
//! of the expected outcomes.

use core::fmt;
use std::collections::BTreeMap;

use super::{Flags, Outcome};

/// The key of a group of cases: an operation, and the rule of floaty that
/// decides the expected result where it differs from the result of the
/// library.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Group {
    /// The operation, for example `bid64_add`.
    pub operation: String,
    /// The rule that decides the expected result, or `None` when the
    /// expected result is the result of the library.
    pub rule: Option<&'static str>,
}

impl Group {
    /// Returns the group of `operation` under `rule`.
    #[must_use]
    pub fn new(operation: impl Into<String>, rule: Option<&'static str>) -> Self {
        Self {
            operation: operation.into(),
            rule,
        }
    }
}

/// An operation without a rule: its expected result is the result of the
/// library.
impl From<String> for Group {
    fn from(operation: String) -> Self {
        Self::new(operation, None)
    }
}

impl fmt::Display for Group {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.rule {
            Some(rule) => write!(formatter, "{} by {rule}", self.operation),
            None => formatter.write_str(&self.operation),
        }
    }
}

/// The failures of one group: their count and the first few.
#[derive(Debug, Default)]
struct Failures {
    count: usize,
    examples: Vec<String>,
}

/// The counts of a run.
#[derive(Debug, Default)]
pub struct Report {
    cases: BTreeMap<Group, usize>,
    failures: BTreeMap<Group, Failures>,
    skipped: BTreeMap<&'static str, usize>,
    /// How many expected outcomes raise each library flag, by bit.
    flags: [usize; 6],
}

impl Report {
    /// The number of failures that the report prints for each group.
    const EXAMPLES: usize = 6;

    /// Counts one case of `group`, and compares floaty's outcome `ours` with
    /// the expected outcome. `operands` describes the case in a failure.
    pub fn check<T: Copy + PartialEq + fmt::Debug>(
        &mut self,
        group: impl Into<Group>,
        operands: &dyn Fn() -> String,
        ours: Outcome<T>,
        expected: Outcome<T>,
    ) {
        let group = group.into();
        *self.cases.entry(group.clone()).or_default() += 1;
        for (bit, count) in self.flags.iter_mut().enumerate() {
            *count += usize::from(expected.flags.bits() >> bit & 1 == 1);
        }
        if ours == expected {
            return;
        }
        let failures = self.failures.entry(group).or_default();
        failures.count += 1;
        if failures.examples.len() < Self::EXAMPLES {
            failures.examples.push(format!(
                "{}: floaty {:x?} {:#04x}, expected {:x?} {:#04x}",
                operands(),
                ours.value,
                ours.flags.bits(),
                expected.value,
                expected.flags.bits(),
            ));
        }
    }

    /// Counts one skipped case, for `reason`.
    pub fn skip(&mut self, reason: &'static str) {
        *self.skipped.entry(reason).or_default() += 1;
    }

    /// Prints the counts and the failures, and returns the number of
    /// failures.
    #[must_use]
    pub fn print(&self, title: &str) -> usize {
        let cases: usize = self.cases.values().sum();
        let failed: usize = self.failures.values().map(|failures| failures.count).sum();
        println!("{title}: {cases} cases, {failed} failed");
        let flags: Vec<String> = Flags::NAMES
            .iter()
            .zip(self.flags)
            .map(|(name, count)| format!("{name} {count}"))
            .collect();
        println!("  expected flags: {}", flags.join(", "));
        for (group, count) in &self.cases {
            println!("  cases   {count:7}: {group}");
        }
        for (reason, count) in &self.skipped {
            println!("  skipped {count:7}: {reason}");
        }
        for (group, failures) in &self.failures {
            println!("  FAILED  {:7}: {group}", failures.count);
            for example in &failures.examples {
                println!("    {group} {example}");
            }
        }
        failed
    }

    /// Prints the report, and fails on a failure.
    ///
    /// # Panics
    ///
    /// Panics when a case failed.
    pub fn finish(&self, title: &str) {
        let failed = self.print(title);
        assert_eq!(failed, 0, "{title}: floaty differs from the library");
    }

    /// Asserts the number of cases, the cases that each rule decides, as
    /// `(operation, rule, count)`, and the skipped cases, as `(reason,
    /// count)`. The generators and the test file are pinned, so the counts
    /// are exact.
    ///
    /// # Panics
    ///
    /// Panics when a count differs.
    pub fn assert_counts(
        &self,
        cases: usize,
        rules: &[(&str, &'static str, usize)],
        skipped: &[(&'static str, usize)],
    ) {
        assert_eq!(self.cases.values().sum::<usize>(), cases, "the cases");
        let decided: BTreeMap<Group, usize> = self
            .cases
            .iter()
            .filter(|(group, _)| group.rule.is_some())
            .map(|(group, &count)| (group.clone(), count))
            .collect();
        let expected: BTreeMap<Group, usize> = rules
            .iter()
            .map(|&(operation, rule, count)| (Group::new(operation, Some(rule)), count))
            .collect();
        assert_eq!(decided, expected, "the cases that each rule decides");
        let skipped = BTreeMap::from_iter(skipped.iter().copied());
        assert_eq!(self.skipped, skipped, "the skipped cases");
    }
}
