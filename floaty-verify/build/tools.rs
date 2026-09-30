//! Runs the host tools of the build: the compilers, `ar`, and `make`. Records
//! each finished step in a stamp file.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The compiler variables that the build removes from the environment of
/// `make` and `configure`, so that an exported value cannot replace a pinned
/// setting.
pub(super) const COMPILER_VARIABLES: [&str; 6] =
    ["CFLAGS", "CPPFLAGS", "LDFLAGS", "CC", "CXX", "CXXFLAGS"];

/// Runs a tool that the build needs and returns its standard output. Stops
/// the build with `advice` when the tool cannot run or fails.
pub(super) fn tool_output(program: &str, arguments: &[&str], advice: &str) -> String {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .unwrap_or_else(|error| panic!("{program} cannot run ({error}); {advice}"));
    assert!(
        output.status.success(),
        "`{program} {}` failed; {advice}",
        arguments.join(" ")
    );
    String::from_utf8(output.stdout).expect("the tool writes text")
}

/// Stops the build when `compiler` is not the pinned GCC `version`, whose
/// machine code floaty's `algorithm` algorithm follows. Returns the first
/// line of `--version`, which names the distribution build of GCC.
pub(super) fn check_gcc(compiler: &str, version: &str, algorithm: &str, advice: &str) -> String {
    let found = tool_output(compiler, &["-dumpfullversion"], advice);
    assert!(
        found.trim() == version,
        "{compiler} is GCC {}, not the pinned GCC {version}. floaty's {algorithm} algorithm \
         follows the machine code of the pinned compiler",
        found.trim()
    );
    let banner = tool_output(compiler, &["--version"], advice);
    banner.lines().next().unwrap_or_default().to_owned()
}

/// Runs `step` unless the stamp file holds `key`, then records `key`. The key
/// names every input of the step, so a changed input runs the step again.
pub(super) fn run_once(stamp: &Path, key: &str, step: impl FnOnce()) {
    if fs::read_to_string(stamp).is_ok_and(|recorded| recorded == key) {
        return;
    }
    step();
    fs::write(stamp, key).expect("the stamp file can be written");
}

/// Compiles one source file with `compiler`.
pub(super) fn compile(compiler: &str, source: &Path, flags: &[&str], object: &Path) {
    let status = Command::new(compiler)
        .args(flags)
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(object)
        .status()
        .expect("the compiler can run");
    assert!(
        status.success(),
        "{compiler} cannot compile {}",
        source.display()
    );
}

/// Replaces a static library with the given objects.
pub(super) fn archive(library: &Path, objects: &[PathBuf]) {
    if library.exists() {
        fs::remove_file(library).expect("an old library can be removed");
    }
    let status = Command::new("ar")
        .arg("rcs")
        .arg(library)
        .args(objects)
        .status()
        .expect("ar can run");
    assert!(status.success(), "ar cannot create {}", library.display());
}

/// Copies the files of a build directory, the Makefile and `platform.h`, so
/// that the object files land in `OUT_DIR` and the submodule stays clean.
///
/// A file with the same content is not written again. Every object depends on
/// `platform.h`, so a new modification time would rebuild every object.
pub(super) fn copy_files(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("the build directory can be created");
    for entry in fs::read_dir(from).expect("the platform build directory exists") {
        let path = entry.expect("the build directory can be read").path();
        if !path.is_file() {
            continue;
        }
        let target = to.join(path.file_name().expect("a file has a name"));
        let content = fs::read(&path).expect("the build file can be read");
        if fs::read(&target).ok().as_deref() != Some(content.as_slice()) {
            fs::write(&target, content).expect("the build file can be written");
        }
    }
}

/// Runs `make` in a directory and stops the build when it fails. `make` shares
/// the job slots of cargo through the jobserver in `CARGO_MAKEFLAGS`.
///
/// The Intel makefile sets `CFLAGS` only when the environment does not, so an
/// exported `CFLAGS` replaces its settings. The function removes the compiler
/// variables, so that every reference builds with its pinned settings.
pub(super) fn make(directory: &Path, arguments: &[String]) {
    let mut command = Command::new("make");
    if let Ok(flags) = env::var("CARGO_MAKEFLAGS") {
        command.env("MAKEFLAGS", flags);
    }
    for variable in COMPILER_VARIABLES {
        command.env_remove(variable);
    }
    let status = command
        .arg("-C")
        .arg(directory)
        .args(arguments)
        .status()
        .expect("make can run");
    assert!(status.success(), "make failed in {}", directory.display());
}

/// Copies the member of a static library that has the file name of `object`
/// into `object`, with the `ar` program of the library's target.
pub(super) fn extract_member(ar: &str, library: &Path, object: &Path, advice: &str) {
    let member = object.file_name().expect("an object has a file name");
    let output = Command::new(ar)
        .arg("p")
        .arg(library)
        .arg(member)
        .output()
        .unwrap_or_else(|error| panic!("{ar} cannot run ({error}); {advice}"));
    assert!(
        output.status.success(),
        "{} has no {}",
        library.display(),
        Path::new(member).display()
    );
    fs::write(object, output.stdout).expect("the library member can be written");
}
