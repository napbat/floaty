//! Mesa's R11G11B10 conversions, from two pinned release headers, and the
//! shim that exposes them.

use std::fs;
use std::path::Path;

use crate::archive::{Archive, Packing, fetch, unpack};
use crate::tools::{archive, compile, run_once};

/// Mesa's conversions of the unsigned 11-bit and 10-bit floats of
/// R11G11B10, from its release 25.2.0, under the MIT license. They follow
/// `GL_EXT_packed_float`.
const MESA_R11G11B10: Archive = Archive {
    file: "mesa-25.2.0-format_r11g11b10f.h",
    packing: Packing::File("format_r11g11b10f.h"),
    url: "https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-25.2.0/src/util/format_r11g11b10f.h",
    sha256: "b0beabfa138da9c4935227e7d4d392c677007d84b7ec90e776f16ac68a79cf19",
    root: None,
};

/// The rounding functions of Mesa 25.2.0 that the R11G11B10 conversions
/// call, under the MIT license.
const MESA_ROUNDING: Archive = Archive {
    file: "mesa-25.2.0-rounding.h",
    packing: Packing::File("rounding.h"),
    url: "https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-25.2.0/src/util/rounding.h",
    sha256: "4265b083f0424e5243c6a81771e1fc077203d2fa829e43a6caafa26f20fb1a02",
    root: None,
};

/// The compiler options of the Mesa shim. `-fPIC` lets the object link into
/// the position-independent test executables.
const MESA_FLAGS: [&str; 3] = ["-std=gnu11", "-O2", "-fPIC"];

/// Fetches Mesa's R11G11B10 header and the rounding header that it includes,
/// and compiles the shim that exposes its inline functions into
/// `libfloaty_mesa.a`.
pub(super) fn build(manifest: &Path, out: &Path) {
    let shim = manifest.join("shim").join("mesa_r11g11b10.c");
    println!("cargo:rerun-if-changed={}", shim.display());
    let downloads = manifest.join("reference").join("downloads");
    let format = unpack(
        &fetch(&MESA_R11G11B10, &downloads),
        &MESA_R11G11B10,
        &out.join("format"),
    );
    let rounding = unpack(
        &fetch(&MESA_ROUNDING, &downloads),
        &MESA_ROUNDING,
        &out.join("rounding"),
    );
    let content = fs::read_to_string(&shim).expect("the Mesa shim can be read");
    let key = format!(
        "{} {} {}\n{content}",
        MESA_R11G11B10.sha256,
        MESA_ROUNDING.sha256,
        MESA_FLAGS.join(" ")
    );
    run_once(&out.join("floaty_mesa.stamp"), &key, || {
        let format_include = format!("-I{}", format.display());
        let rounding_include = format!("-I{}", rounding.display());
        let mut flags = MESA_FLAGS.to_vec();
        flags.extend([format_include.as_str(), rounding_include.as_str()]);
        let object = out.join("floaty_mesa.o");
        compile("gcc", &shim, &flags, &object);
        archive(&out.join("libfloaty_mesa.a"), &[object]);
    });
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=floaty_mesa");
}
