//! The shim that exposes the NaN payload functions of the host glibc libm.

use std::path::Path;

use crate::tools::{HOST_CC, shim_library};

/// The compiler options of the libm shim. `-fPIC` lets the object link into
/// the position-independent test executables.
const LIBM_FLAGS: [&str; 3] = ["-std=gnu11", "-O2", "-fPIC"];

/// Compiles the shim `shim/libm_payload.c` into `libfloaty_libm.a`, and links
/// it with the libm of the host.
pub(super) fn build(manifest: &Path, out: &Path) {
    let shim = manifest.join("shim").join("libm_payload.c");
    println!("cargo:rerun-if-changed={}", shim.display());
    shim_library(
        HOST_CC,
        &shim,
        &LIBM_FLAGS,
        "the libm payload shim",
        &out.join("floaty_libm.stamp"),
        out,
        "floaty_libm",
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=floaty_libm");
    println!("cargo:rustc-link-lib=dylib=m");
}
