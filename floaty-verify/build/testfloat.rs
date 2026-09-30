//! Berkeley TestFloat's generator, built against Berkeley SoftFloat for each
//! NaN specialization that the tests use.

use std::path::Path;

use crate::tools::{copy_files, make};

/// The SoftFloat NaN specializations, and the environment variable that
/// names the generator of each one.
const SPECIALIZATIONS: [(&str, &str); 4] = [
    ("ARM-VFPv2", "FLOATY_TESTFLOAT_GEN_ARM"),
    (
        "ARM-VFPv2-defaultNaN",
        "FLOATY_TESTFLOAT_GEN_ARM_DEFAULT_NAN",
    ),
    ("8086", "FLOATY_TESTFLOAT_GEN_X87"),
    ("8086-SSE", "FLOATY_TESTFLOAT_GEN_SSE"),
];

/// The SoftFloat and TestFloat build directory for this host.
const PLATFORM: &str = "Linux-x86_64-GCC";

/// Builds `testfloat_gen` for each specialization and names each one in an
/// environment variable of the harness.
pub(super) fn build(manifest: &Path, out: &Path) {
    let softfloat = manifest.join("reference").join("berkeley-softfloat-3");
    let testfloat = manifest.join("reference").join("berkeley-testfloat-3");
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
    assert!(
        softfloat.join("source").is_dir() && testfloat.join("source").is_dir(),
        "the TestFloat sources are missing; run `git submodule update --init`"
    );

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
