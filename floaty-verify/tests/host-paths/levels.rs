//! The runs of the host-path tests in each instruction set of the processor.
//!
//! The slice kernels, the elementwise slice operations, and the slice
//! conversion of `Lanes` select an instruction set at run time. One run of
//! the tests checks the set that the processor selects. `every_level` runs
//! this binary again for each other set that the processor can run, with
//! `FLOATY_HOST_LEVEL` set to the set. So each set must give the bits of the
//! engine in every test. Under a cross runner, such as QEMU, the binary runs
//! again under the command in `FLOATY_RUNNER`, which `.cargo/config.toml`
//! has each runner set.

use std::env;
use std::process::Command;

/// The environment variable that selects the largest instruction set.
const LEVEL: &str = "FLOATY_HOST_LEVEL";

/// The environment variable that holds the runner command, split at
/// whitespace, that runs this binary when the host cannot.
const RUNNER: &str = "FLOATY_RUNNER";

/// Runs the other tests in each instruction set that the processor can run,
/// beside the set that it selects.
#[test]
fn every_level() {
    // A run for one set checks only that set.
    if env::var_os(LEVEL).is_some() {
        return;
    }
    let selected = floaty::host_level();
    let binary = env::current_exe().expect("a test runs from its binary");
    for level in floaty::host_levels().filter(|&level| level != selected) {
        let runner = env::var(RUNNER).unwrap_or_default();
        let mut words = runner.split_whitespace();
        let mut command = match words.next() {
            Some(program) => {
                let mut command = Command::new(program);
                command.args(words).arg(&binary);
                command
            }
            None => Command::new(&binary),
        };
        let status = command
            .env(LEVEL, level)
            .args(["--skip", "levels::every_level"])
            .status()
            .expect("the test binary runs again");
        assert!(
            status.success(),
            "the tests fail in the instruction set {level}"
        );
    }
}

/// Checks that `FLOATY_HOST_LEVEL` selects the instruction set that it
/// names. A set that the build enables runs as `build`.
#[test]
fn the_variable_selects_the_level() {
    let Ok(level) = env::var(LEVEL) else {
        return;
    };
    let expected = if floaty::host_levels().any(|name| name == level) {
        level.as_str()
    } else {
        "build"
    };
    assert_eq!(floaty::host_level(), expected);
}
