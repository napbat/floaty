//! The Intel Decimal Floating-Point Math Library, its tests, and the
//! binary80 shim, from a pinned release archive.

use std::path::Path;

use crate::archive::{Archive, Packing};
use crate::tools::{HOST_CC, compiler_release, make, run_once, shim_library};

/// The Intel Decimal Floating-Point Math Library 2.0 Update 2, under the
/// BSD 3-clause license, with its tests.
const INTEL: Archive = Archive {
    file: "IntelRDFPMathLib20U2.tar.gz",
    packing: Packing::GzipTar,
    url: "https://www.netlib.org/misc/intel/IntelRDFPMathLib20U2.tar.gz",
    sha256: "93c0c78e0989df88f8540bf38d6743734804cef1e40706fd8fe5c6a03f79e173",
    root: Some("IntelRDFPMathLib20U2"),
};

/// The interface of the Intel library: each `make` variable and the
/// preprocessor macro that the makefile sets from it. Every one is 0, so
/// values, the rounding direction, and a pointer to the status flags are
/// function arguments. No global state is shared between tests.
const INTEL_INTERFACE: [(&str, &str); 3] = [
    ("CALL_BY_REF", "DECIMAL_CALL_BY_REFERENCE"),
    ("GLOBAL_RND", "DECIMAL_GLOBAL_ROUNDING"),
    ("GLOBAL_FLAGS", "DECIMAL_GLOBAL_EXCEPTION_FLAGS"),
];

/// The `make` settings of the Intel library after [`INTEL_INTERFACE`], as its
/// `README` describes.
///
/// - The code predates C23, so it compiles as C99. GCC 14 rejects implicit
///   declarations, and the `pow` functions call `abs` without `stdlib.h`.
/// - The library keeps its default optimization level, none. At `-O2`,
///   `bid128_pow` never returns on one of the library's own test cases.
const INTEL_BUILD_SETTINGS: [&str; 2] = [
    "UNCHANGED_BINARY_FLAGS=0",
    "CFLAGS_AUX=-std=gnu99 -Wno-error=implicit-function-declaration -fPIC",
];

/// The preprocessor settings for this host that the Intel makefile gives
/// every source file after the macros of [`INTEL_INTERFACE`].
const INTEL_HOST_DEFINES: [&str; 4] = [
    "-DUSE_COMPILER_F128_TYPE=0",
    "-DUSE_COMPILER_F80_TYPE=0",
    "-DLINUX",
    "-Defi2",
];

/// Returns the `make` settings of the Intel library: the C compiler, the
/// interface, and [`INTEL_BUILD_SETTINGS`].
fn make_settings() -> Vec<String> {
    let interface = INTEL_INTERFACE.map(|(variable, _)| format!("{variable}=0"));
    let build = INTEL_BUILD_SETTINGS.map(String::from);
    [format!("CC={HOST_CC}")]
        .into_iter()
        .chain(interface)
        .chain(build)
        .collect()
}

/// Returns the preprocessor settings that the Intel makefile gives every
/// source file on this host. The binary80 shim includes the library headers
/// with the same settings.
fn defines() -> Vec<String> {
    let interface = INTEL_INTERFACE.map(|(_, name)| format!("-D{name}=0"));
    let host = INTEL_HOST_DEFINES.map(String::from);
    interface.into_iter().chain(host).collect()
}

/// Fetches the Intel library and names its `readtest.in` in
/// `FLOATY_INTEL_READTEST`. Builds the library and the binary80 shim, whose
/// library goes into `libraries`, and links both.
pub(super) fn build(manifest: &Path, out: &Path, libraries: &Path) {
    let shim = manifest.join("shim").join("intel_binary80.c");
    println!("cargo:rerun-if-changed={}", shim.display());
    let intel = INTEL.extract(manifest, &out.join("intel"));
    println!(
        "cargo:rustc-env=FLOATY_INTEL_READTEST={}",
        intel.join("TESTS").join("readtest.in").display()
    );

    let intel_library = intel.join("LIBRARY");
    build_library(&intel_library, libraries);
    build_shim(&shim, &intel_library.join("src"), libraries);

    // The shim calls the Intel library, so it comes first.
    println!("cargo:rustc-link-search=native={}", intel_library.display());
    println!("cargo:rustc-link-lib=static=floaty_binary80");
    println!("cargo:rustc-link-lib=static=bid");
}

/// Builds the Intel library into `libbid.a` in its `LIBRARY` directory. The
/// makefile does not track its settings, so a new setting starts from
/// `make clean`.
fn build_library(library: &Path, libraries: &Path) {
    let settings = make_settings();
    let key = format!(
        "{} {}\n{}",
        INTEL.sha256,
        settings.join(" "),
        compiler_release(HOST_CC)
    );
    run_once(&libraries.join("bid.stamp"), &key, || {
        make(library, &[String::from("clean")]);
        make(library, &settings);
    });
}

/// Compiles the binary80 shim into `libfloaty_binary80.a`. The shim passes
/// the C `long double` values of the Intel library, which Rust has no type
/// for.
fn build_shim(shim: &Path, intel_source: &Path, libraries: &Path) {
    let defines = defines();
    let include = format!("-I{}", intel_source.display());
    let mut flags = vec!["-std=gnu99", "-O2", "-fPIC", include.as_str()];
    flags.extend(defines.iter().map(String::as_str));
    shim_library(
        HOST_CC,
        shim,
        &flags,
        &format!("{} {}", INTEL.sha256, defines.join(" ")),
        &libraries.join("floaty_binary80.stamp"),
        libraries,
        "floaty_binary80",
    );
}
