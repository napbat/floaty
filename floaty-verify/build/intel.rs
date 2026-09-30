//! The Intel Decimal Floating-Point Math Library, its tests, and the
//! binary80 shim, from a pinned release archive.

use std::fs;
use std::path::Path;

use crate::archive::{Archive, Packing, fetch, unpack};
use crate::tools::{archive, compile, make, run_once};

/// The Intel Decimal Floating-Point Math Library 2.0 Update 2, under the
/// BSD 3-clause license, with its tests.
const INTEL: Archive = Archive {
    file: "IntelRDFPMathLib20U2.tar.gz",
    packing: Packing::GzipTar,
    url: "https://www.netlib.org/misc/intel/IntelRDFPMathLib20U2.tar.gz",
    sha256: "93c0c78e0989df88f8540bf38d6743734804cef1e40706fd8fe5c6a03f79e173",
};

/// The `make` settings of the Intel library, as its `README` describes.
///
/// - Values, the rounding direction, and a pointer to the status flags are
///   function arguments. No global state is shared between tests.
/// - The code predates C23, so it compiles as C99. GCC 14 rejects implicit
///   declarations, and the `pow` functions call `abs` without `stdlib.h`.
/// - The library keeps its default optimization level, none. At `-O2`,
///   `bid128_pow` never returns on one of the library's own test cases.
const INTEL_SETTINGS: [&str; 6] = [
    "CC=gcc",
    "CALL_BY_REF=0",
    "GLOBAL_RND=0",
    "GLOBAL_FLAGS=0",
    "UNCHANGED_BINARY_FLAGS=0",
    "CFLAGS_AUX=-std=gnu99 -Wno-error=implicit-function-declaration -fPIC",
];

/// The preprocessor settings that the Intel makefile gives every source
/// file on this host. The binary80 shim includes the library headers with
/// the same settings.
const INTEL_DEFINES: [&str; 7] = [
    "-DDECIMAL_CALL_BY_REFERENCE=0",
    "-DDECIMAL_GLOBAL_ROUNDING=0",
    "-DDECIMAL_GLOBAL_EXCEPTION_FLAGS=0",
    "-DUSE_COMPILER_F128_TYPE=0",
    "-DUSE_COMPILER_F80_TYPE=0",
    "-DLINUX",
    "-Defi2",
];

/// Fetches the Intel library and names its `readtest.in` in
/// `FLOATY_INTEL_READTEST`. Builds the library and the binary80 shim, whose
/// library goes into `libraries`, and links both.
pub(super) fn build(manifest: &Path, out: &Path, libraries: &Path) {
    let shim = manifest.join("shim").join("intel_binary80.c");
    println!("cargo:rerun-if-changed={}", shim.display());
    let downloads = manifest.join("reference").join("downloads");

    let intel = out.join("intel");
    unpack(&fetch(&INTEL, &downloads), &INTEL, &intel);
    let intel = intel.join("IntelRDFPMathLib20U2");
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
    let key = format!("{} {}", INTEL.sha256, INTEL_SETTINGS.join(" "));
    run_once(&libraries.join("bid.stamp"), &key, || {
        make(library, &[String::from("clean")]);
        make(library, &INTEL_SETTINGS.map(String::from));
    });
}

/// Compiles the binary80 shim into `libfloaty_binary80.a`. The shim passes
/// the C `long double` values of the Intel library, which Rust has no type
/// for.
fn build_shim(shim: &Path, intel_source: &Path, libraries: &Path) {
    let content = fs::read_to_string(shim).expect("the binary80 shim can be read");
    let key = format!("{} {}\n{content}", INTEL.sha256, INTEL_DEFINES.join(" "));
    run_once(&libraries.join("floaty_binary80.stamp"), &key, || {
        let include = format!("-I{}", intel_source.display());
        let mut flags = vec!["-std=gnu99", "-O2", "-fPIC", include.as_str()];
        flags.extend(INTEL_DEFINES);
        let object = libraries.join("floaty_binary80.o");
        compile("gcc", shim, &flags, &object);
        archive(&libraries.join("libfloaty_binary80.a"), &[object]);
    });
}
