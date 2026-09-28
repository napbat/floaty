//! Builds the C reference libraries of the harness.
//!
//! - Berkeley TestFloat's generator, `testfloat_gen`, against Berkeley
//!   SoftFloat for each NaN specialization that the tests use. The sources
//!   are the git submodules under `reference/`.
//! - decNumber and the Intel Decimal Floating-Point Math Library, and the
//!   decTest vectors, from pinned release archives. The Intel archive also
//!   holds the Intel tests. The build downloads each archive once into
//!   `reference/downloads/` and checks its SHA-256 before it uses it.
//!
//! The build needs `make`, `gcc`, `ar`, `curl`, `tar`, `python3`, and
//! `sha256sum`, and it runs on Linux x86-64 hosts only. The pinned Makefiles
//! name `gcc`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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

/// How an archive is packed.
#[derive(Clone, Copy)]
enum Packing {
    /// A zip file. `unzip` is not a dependency of the build; Python's
    /// `zipfile` module is.
    Zip,
    /// A tar file compressed with gzip.
    GzipTar,
}

/// A pinned release archive.
struct Archive {
    /// The file name in the download cache.
    file: &'static str,
    /// How the archive is packed.
    packing: Packing,
    /// Where the release is published.
    url: &'static str,
    /// The SHA-256 of the archive, in lowercase hexadecimal.
    sha256: &'static str,
}

/// The decTest vectors, version 2.62.
const DECTEST: Archive = Archive {
    file: "dectest.zip",
    packing: Packing::Zip,
    url: "https://speleotrove.com/decimal/dectest.zip",
    sha256: "b70a224cd52e82b7a8150aedac5efa2d0cb3941696fd829bdbe674f9f65c3926",
};

/// decNumber 3.68, under the ICU license.
const DECNUMBER: Archive = Archive {
    file: "decNumber-icu-368.zip",
    packing: Packing::Zip,
    url: "https://speleotrove.com/decimal/decNumber-icu-368.zip",
    sha256: "14ec2cf30b58758493a7661b78b80abfb281652b61a425b85cda83173518fe25",
};

/// The Intel Decimal Floating-Point Math Library 2.0 Update 2, under the
/// BSD 3-clause license, with its tests.
const INTEL: Archive = Archive {
    file: "IntelRDFPMathLib20U2.tar.gz",
    packing: Packing::GzipTar,
    url: "https://www.netlib.org/misc/intel/IntelRDFPMathLib20U2.tar.gz",
    sha256: "93c0c78e0989df88f8540bf38d6743734804cef1e40706fd8fe5c6a03f79e173",
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

fn main() {
    let manifest =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    println!("cargo:rerun-if-changed=build.rs");

    // The references run on the host, so the check is on the host, not on
    // the target.
    let host = env::var("HOST").expect("cargo sets HOST");
    assert!(
        host.starts_with("x86_64-") && host.contains("-linux-"),
        "floaty-verify builds its references on Linux x86-64 hosts only, not on {host}"
    );

    build_testfloat(&manifest, &out);
    build_decimal(&manifest, &out.join("decimal"));
}

/// Builds `testfloat_gen` for each specialization and names each one in an
/// environment variable of the harness.
fn build_testfloat(manifest: &Path, out: &Path) {
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

/// Fetches the decimal references, builds decNumber, the Intel library, and
/// the binary80 shim, links them, and names the test data in environment
/// variables of the harness.
fn build_decimal(manifest: &Path, out: &Path) {
    let shim = manifest.join("shim").join("intel_binary80.c");
    println!("cargo:rerun-if-changed={}", shim.display());
    let downloads = manifest.join("reference").join("downloads");

    let dectest = out.join("dectest");
    unpack(&fetch(&DECTEST, &downloads), &DECTEST, &dectest);
    println!("cargo:rustc-env=FLOATY_DECTEST_DIR={}", dectest.display());

    let decnumber = out.join("decnumber");
    unpack(&fetch(&DECNUMBER, &downloads), &DECNUMBER, &decnumber);
    let intel = out.join("intel");
    unpack(&fetch(&INTEL, &downloads), &INTEL, &intel);
    let intel = intel.join("IntelRDFPMathLib20U2");
    println!(
        "cargo:rustc-env=FLOATY_INTEL_READTEST={}",
        intel.join("TESTS").join("readtest.in").display()
    );

    let libraries = out.join("lib");
    fs::create_dir_all(&libraries).expect("the library directory can be created");
    build_decnumber(&decnumber.join("decNumber"), &libraries);
    let intel_library = intel.join("LIBRARY");
    build_intel(&intel_library, &libraries);
    build_shim(&shim, &intel_library.join("src"), &libraries);

    // The shim calls the Intel library, so it comes first.
    println!("cargo:rustc-link-search=native={}", libraries.display());
    println!("cargo:rustc-link-search=native={}", intel_library.display());
    println!("cargo:rustc-link-lib=static=floaty_binary80");
    println!("cargo:rustc-link-lib=static=bid");
    println!("cargo:rustc-link-lib=static=decnumber");
}

/// Returns the cached archive, and downloads it first when the cache does
/// not hold it. Stops the build when the archive does not match its pinned
/// SHA-256.
fn fetch(archive: &Archive, downloads: &Path) -> PathBuf {
    let path = downloads.join(archive.file);
    if path.is_file() {
        let actual = sha256(&path);
        assert!(
            actual == archive.sha256,
            "{} has SHA-256 {actual}, not the pinned {}; delete the file to download it again",
            path.display(),
            archive.sha256
        );
        return path;
    }
    fs::create_dir_all(downloads).expect("the download cache can be created");
    // A partial download never takes the final name, so the cache holds
    // only verified archives.
    let partial = downloads.join(format!("{}.part", archive.file));
    let status = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "5",
            "--retry-all-errors",
            "--output",
        ])
        .arg(&partial)
        .arg(archive.url)
        .status()
        .expect("curl can run");
    assert!(status.success(), "curl cannot download {}", archive.url);
    let actual = sha256(&partial);
    assert!(
        actual == archive.sha256,
        "{} has SHA-256 {actual}, not the pinned {}",
        archive.url,
        archive.sha256
    );
    fs::rename(&partial, &path).expect("the verified download can be renamed");
    path
}

/// Returns the SHA-256 of a file in lowercase hexadecimal.
fn sha256(path: &Path) -> String {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .expect("sha256sum can run");
    assert!(
        output.status.success(),
        "sha256sum failed on {}",
        path.display()
    );
    let text = String::from_utf8(output.stdout).expect("sha256sum writes text");
    text.split_whitespace()
        .next()
        .expect("sha256sum writes the digest first")
        .to_owned()
}

/// Extracts an archive into an empty `destination`, unless the destination
/// already holds this archive.
fn unpack(path: &Path, archive: &Archive, destination: &Path) {
    run_once(&destination.with_extension("stamp"), archive.sha256, || {
        if destination.exists() {
            fs::remove_dir_all(destination).expect("an old extraction can be removed");
        }
        fs::create_dir_all(destination).expect("the extraction directory can be created");
        let status = match archive.packing {
            Packing::Zip => Command::new("python3")
                .args(["-m", "zipfile", "-e"])
                .arg(path)
                .arg(destination)
                .status()
                .expect("python3 can run"),
            Packing::GzipTar => Command::new("tar")
                .arg("-xzf")
                .arg(path)
                .arg("-C")
                .arg(destination)
                .status()
                .expect("tar can run"),
        };
        assert!(status.success(), "{} cannot be extracted", path.display());
    });
}

/// Compiles decNumber into `libdecnumber.a`.
fn build_decnumber(source: &Path, libraries: &Path) {
    let objects = libraries.join("decnumber");
    let key = format!("{} {}", DECNUMBER.sha256, DECNUMBER_FLAGS.join(" "));
    run_once(&objects.with_extension("stamp"), &key, || {
        fs::create_dir_all(&objects).expect("the object directory can be created");
        let objects: Vec<PathBuf> = DECNUMBER_SOURCES
            .iter()
            .map(|name| {
                let object = objects.join(format!("{name}.o"));
                compile(&source.join(format!("{name}.c")), &DECNUMBER_FLAGS, &object);
                object
            })
            .collect();
        archive(&libraries.join("libdecnumber.a"), &objects);
    });
}

/// Builds the Intel library into `libbid.a` in its `LIBRARY` directory. The
/// makefile does not track its settings, so a new setting starts from
/// `make clean`.
fn build_intel(library: &Path, libraries: &Path) {
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
        compile(shim, &flags, &object);
        archive(&libraries.join("libfloaty_binary80.a"), &[object]);
    });
}

/// Runs `step` unless the stamp file holds `key`, then records `key`. The key
/// names every input of the step, so a changed input runs the step again.
fn run_once(stamp: &Path, key: &str, step: impl FnOnce()) {
    if fs::read_to_string(stamp).is_ok_and(|recorded| recorded == key) {
        return;
    }
    step();
    fs::write(stamp, key).expect("the stamp file can be written");
}

/// Compiles one C source file with `gcc`.
fn compile(source: &Path, flags: &[&str], object: &Path) {
    let status = Command::new("gcc")
        .args(flags)
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(object)
        .status()
        .expect("gcc can run");
    assert!(status.success(), "gcc cannot compile {}", source.display());
}

/// Replaces a static library with the given objects.
fn archive(library: &Path, objects: &[PathBuf]) {
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
///
/// The Intel makefile sets `CFLAGS` only when the environment does not, so an
/// exported `CFLAGS` replaces its settings. The function removes the compiler
/// variables, so that every reference builds with its pinned settings.
fn make(directory: &Path, arguments: &[String]) {
    let mut command = Command::new("make");
    if let Ok(flags) = env::var("CARGO_MAKEFLAGS") {
        command.env("MAKEFLAGS", flags);
    }
    for variable in ["CFLAGS", "CPPFLAGS", "LDFLAGS", "CC"] {
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
