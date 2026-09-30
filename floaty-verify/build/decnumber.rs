//! decNumber and the decTest vectors, from pinned release archives.

use std::fs;
use std::path::{Path, PathBuf};

use crate::archive::{Archive, Packing, fetch, unpack};
use crate::tools::{archive, compile, run_once};

/// The decTest vectors, version 2.62.
const DECTEST: Archive = Archive {
    file: "dectest.zip",
    packing: Packing::Zip,
    url: "https://speleotrove.com/decimal/dectest.zip",
    sha256: "b70a224cd52e82b7a8150aedac5efa2d0cb3941696fd829bdbe674f9f65c3926",
    root: None,
};

/// decNumber 3.68, under the ICU license.
const DECNUMBER: Archive = Archive {
    file: "decNumber-icu-368.zip",
    packing: Packing::Zip,
    url: "https://speleotrove.com/decimal/decNumber-icu-368.zip",
    sha256: "14ec2cf30b58758493a7661b78b80abfb281652b61a425b85cda83173518fe25",
    root: Some("decNumber"),
};

/// The decNumber sources that the harness links: the context, the
/// arbitrary-precision numbers, and the three fixed-size DPD formats.
const DECNUMBER_SOURCES: [&str; 5] = [
    "decContext",
    "decNumber",
    "decSingle",
    "decDouble",
    "decQuad",
];

/// The compiler options for decNumber. `-fPIC` lets the objects link into
/// the position-independent test executables.
const DECNUMBER_FLAGS: [&str; 3] = ["-std=gnu99", "-O2", "-fPIC"];

/// Fetches the decTest vectors and names them in `FLOATY_DECTEST_DIR`.
/// Fetches decNumber, compiles it into `libraries`, and links it.
pub(super) fn build(manifest: &Path, out: &Path, libraries: &Path) {
    let downloads = manifest.join("reference").join("downloads");

    let dectest = unpack(&fetch(&DECTEST, &downloads), &DECTEST, &out.join("dectest"));
    println!("cargo:rustc-env=FLOATY_DECTEST_DIR={}", dectest.display());

    let decnumber = unpack(
        &fetch(&DECNUMBER, &downloads),
        &DECNUMBER,
        &out.join("decnumber"),
    );
    build_library(&decnumber, libraries);
    println!("cargo:rustc-link-lib=static=decnumber");
}

/// Compiles decNumber into `libdecnumber.a`.
fn build_library(source: &Path, libraries: &Path) {
    let objects = libraries.join("decnumber");
    let key = format!("{} {}", DECNUMBER.sha256, DECNUMBER_FLAGS.join(" "));
    run_once(&objects.with_extension("stamp"), &key, || {
        fs::create_dir_all(&objects).expect("the object directory can be created");
        let objects: Vec<PathBuf> = DECNUMBER_SOURCES
            .iter()
            .map(|name| {
                let object = objects.join(format!("{name}.o"));
                compile(
                    "gcc",
                    &source.join(format!("{name}.c")),
                    &DECNUMBER_FLAGS,
                    &object,
                );
                object
            })
            .collect();
        archive(&libraries.join("libdecnumber.a"), &objects);
    });
}
