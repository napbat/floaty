//! Builds the C reference libraries of the harness.
//!
//! - Berkeley TestFloat's generator, `testfloat_gen`, against Berkeley
//!   SoftFloat for each NaN specialization that the tests use. The sources
//!   are the git submodules under `reference/`.
//! - decNumber and the Intel Decimal Floating-Point Math Library, and the
//!   decTest vectors, from pinned release archives. The Intel archive also
//!   holds the Intel tests. The build downloads each archive once into
//!   `reference/downloads/` and checks its SHA-256 before it uses it.
//! - The double-double references. The IBM `long double` routines of the
//!   powerpc64le libgcc, and the IBM `long double` functions of the libm of
//!   its glibc 2.43, run in a batch program under `qemu-ppc64le`. QD comes
//!   from a pinned release archive, as the decimal libraries do.
//! - The NaN payload functions of the host glibc libm, through a shim.
//!
//! The build needs `make`, `gcc`, `g++` 15.2.0, `ar`, `objcopy`, `curl`,
//! `tar`, `python3`, `sha256sum`, `powerpc64le-linux-gnu-gcc` 15.2.0, and
//! `qemu-ppc64le`, and it runs on Linux x86-64 hosts only. The pinned
//! Makefiles name `gcc`.
//!
//! Each module builds one reference and holds its pins. `archive` fetches
//! and extracts the pinned release archives, and `tools` runs the host
//! tools.

mod archive;
mod decnumber;
mod ibm_ldouble;
mod intel;
mod libm;
mod mesa;
mod qd;
mod testfloat;
mod tools;

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    println!("cargo:rerun-if-changed=build");

    // The references build only for x86-64. On another target, the harness
    // has only the tests that need no reference library.
    if env::var("CARGO_CFG_TARGET_ARCH").expect("cargo sets CARGO_CFG_TARGET_ARCH") != "x86_64" {
        return;
    }
    // The references run on the host, so the check is on the host, not on
    // the target.
    let host = env::var("HOST").expect("cargo sets HOST");
    assert!(
        host.starts_with("x86_64-") && host.contains("-linux-"),
        "floaty-verify builds its references on Linux x86-64 hosts only, not on {host}"
    );

    testfloat::build(&manifest, &out);

    // The decimal references share one library directory.
    let decimal = out.join("decimal");
    let libraries = decimal.join("lib");
    fs::create_dir_all(&libraries).expect("the library directory can be created");
    println!("cargo:rustc-link-search=native={}", libraries.display());
    intel::build(&manifest, &decimal, &libraries);
    decnumber::build(&manifest, &decimal, &libraries);

    ibm_ldouble::build(&manifest, &out.join("ibm_ldouble"));
    qd::build(&manifest, &out.join("qd"));
    mesa::build(&manifest, &out.join("mesa"));
    libm::build(&manifest, &out.join("libm"));
}
