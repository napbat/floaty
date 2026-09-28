//! Builds the Berkeley TestFloat generator, `testfloat_gen`, against Berkeley
//! SoftFloat for each NaN specialization that the tests use.
//!
//! The sources are the git submodules under `reference/`. The build needs
//! `make` and `gcc`, because the pinned Makefiles name `gcc`, and it runs on
//! Linux x86-64 hosts only.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The SoftFloat NaN specializations, and the environment variable that
/// names the generator of each one.
const SPECIALIZATIONS: [(&str, &str); 2] = [
    ("ARM-VFPv2", "FLOATY_TESTFLOAT_GEN_ARM"),
    (
        "ARM-VFPv2-defaultNaN",
        "FLOATY_TESTFLOAT_GEN_ARM_DEFAULT_NAN",
    ),
];

/// The SoftFloat and TestFloat build directory for this host.
const PLATFORM: &str = "Linux-x86_64-GCC";

fn main() {
    let manifest =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let softfloat = manifest.join("reference").join("berkeley-softfloat-3");
    let testfloat = manifest.join("reference").join("berkeley-testfloat-3");
    println!("cargo:rerun-if-changed=build.rs");
    for directory in [&softfloat, &testfloat] {
        println!(
            "cargo:rerun-if-changed={}",
            directory.join("source").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            directory.join("build").join(PLATFORM).display()
        );
    }

    // The generator runs on the host, so the check is on the host, not on the
    // target.
    let host = env::var("HOST").expect("cargo sets HOST");
    assert!(
        host.starts_with("x86_64-") && host.contains("-linux-"),
        "floaty-verify builds TestFloat on Linux x86-64 hosts only, not on {host}"
    );
    assert!(
        softfloat.join("source").is_dir() && testfloat.join("source").is_dir(),
        "the TestFloat sources are missing; run `git submodule update --init`"
    );

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    for (specialization, variable) in SPECIALIZATIONS {
        let root = out.join(specialization);
        let library = root.join("softfloat");
        copy_files(&softfloat.join("build").join(PLATFORM), &library);
        make(
            &library,
            &[
                format!("SOURCE_DIR={}", softfloat.join("source").display()),
                format!("SPECIALIZE_TYPE={specialization}"),
                String::from("softfloat.a"),
            ],
        );
        let generator = root.join("testfloat");
        copy_files(&testfloat.join("build").join(PLATFORM), &generator);
        make(
            &generator,
            &[
                format!("SOURCE_DIR={}", testfloat.join("source").display()),
                format!(
                    "SOFTFLOAT_INCLUDE_DIR={}",
                    softfloat.join("source").join("include").display()
                ),
                format!("SOFTFLOAT_LIB={}", library.join("softfloat.a").display()),
                String::from("testfloat_gen"),
            ],
        );
        println!(
            "cargo:rustc-env={variable}={}",
            generator.join("testfloat_gen").display()
        );
    }
}

/// Copies the files of a build directory, the Makefile and `platform.h`, so
/// that the object files land in `OUT_DIR` and the submodule stays clean.
///
/// A file with the same content is not written again. Every object depends on
/// `platform.h`, so a new modification time would rebuild every object.
fn copy_files(from: &Path, to: &Path) {
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
fn make(directory: &Path, arguments: &[String]) {
    let mut command = Command::new("make");
    if let Ok(flags) = env::var("CARGO_MAKEFLAGS") {
        command.env("MAKEFLAGS", flags);
    }
    let status = command
        .arg("-C")
        .arg(directory)
        .args(arguments)
        .status()
        .expect("make can run");
    assert!(status.success(), "make failed in {}", directory.display());
}
