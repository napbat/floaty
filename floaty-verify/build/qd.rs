//! QD and its shim, from a pinned release archive, with the checks of the
//! machine code that floaty's `Qd` algorithm follows.

use std::fs;
use std::path::Path;
use std::process::Command;

use crate::archive::{Archive, Packing, sha256};
use crate::tools::{COMPILER_VARIABLES, check_gcc, extract_member, make, run_once, shim_library};

/// QD 2.3.24, under the BSD-LBNL license.
const QD: Archive = Archive {
    file: "qd-2.3.24.tar.gz",
    packing: Packing::GzipTar,
    url: "https://www.davidhbailey.com/dhbsoftware/qd-2.3.24.tar.gz",
    sha256: "a47b6c73f86e6421e86a883568dd08e299b20e36c11a99bdfbe50e01bde60e38",
    root: Some("qd-2.3.24"),
};

/// The `configure` options of QD. The first three select the arithmetic
/// that floaty's `Qd` follows.
///
/// - `--enable-ieee-add` selects the addition with the IEEE-style error
///   bound, and `--disable-sloppy-div` selects the accurate division.
/// - `--enable-fma=c99` computes the error of a product with the C `fma`.
/// - `--with-pic` lets the library link into the position-independent test
///   executables. The other options skip parts that the harness does not
///   use.
const QD_CONFIGURE: [&str; 6] = [
    "--enable-ieee-add",
    "--disable-sloppy-div",
    "--enable-fma=c99",
    "--disable-fortran",
    "--disable-shared",
    "--with-pic",
];

/// The C++ compiler of QD and of its shim.
const QD_CXX: &str = "g++";

/// The pinned GCC release of the C++ compiler of QD.
const QD_CXX_VERSION: &str = "15.2.0";

/// The C++ compiler options of QD, whose machine code floaty's `Qd`
/// follows. The shim compiles the inline operators of QD, so it uses the
/// same options.
const QD_CXXFLAGS: [&str; 2] = ["-O2", "-ffp-contract=off"];

/// The lines of `include/qd/qd_config.h` that the pinned configuration
/// gives. The build checks each one after `configure`.
const QD_CONFIG_LINES: [&str; 4] = [
    "#define QD_IEEE_ADD 1",
    "/* #undef QD_SLOPPY_DIV */",
    "#define QD_FMA(x,y,z) fma(x,y,z)",
    "#define QD_FMS(x,y,z) fma(x,y,-z)",
];

/// The sections of the shim object that floaty's `Qd` algorithm follows: the
/// code of `run`, which inlines the addition, subtraction, and multiplication
/// of QD, the code of `run_remainder`, which inlines `drem`, the code of
/// `dd_real::accurate_div`, and their constants.
const QD_SHIM_SECTIONS: [&str; 3] = [
    ".text",
    ".text._ZN7dd_real12accurate_divERKS_S1_",
    ".rodata.cst16",
];

/// The SHA-256 of [`QD_SHIM_SECTIONS`] in the shim object, one section after
/// the other.
const QD_SHIM_SHA256: &str = "d6346bd1dc44447732adb1db9c5776b23e206fbb52f512177e8e865b6356c583";

/// The SHA-256 of [`QD_SHIM_SECTIONS`] in `qd_unary.cpp`'s object.
/// The square follows the inspected inline steps. The inverse calls the
/// same `dd_real::accurate_div` as the arithmetic shim. The integer power
/// and the library root call `npwr` and `nroot` of `dd_real.o`, whose code
/// [`QD_LIBRARY_SHA256`] pins. The seeded root inlines the source of
/// `nroot` with the seed as an operand.
const QD_UNARY_SHA256: &str = "84d0d502e2452ee74d40ae4a05188f9611e65ca375442a863ae3be6587e54816";

/// The sections of `dd_real.o` in `libqd.a` that floaty's `Qd` algorithm
/// follows: the code of the square root, `fmod`, `npwr`, and `nroot`, and
/// their constants. The `accurate_div` that `npwr` and `nroot` call has the
/// bytes of the shim's pinned copy.
const QD_LIBRARY_SECTIONS: [&str; 3] = [".text", ".rodata.cst8", ".rodata.cst16"];

/// The SHA-256 of [`QD_LIBRARY_SECTIONS`] in `dd_real.o`, one section after
/// the other.
const QD_LIBRARY_SHA256: &str = "0bcc95f145495cde281ad918b8aa6f1a841eeb879c4049bfcad234edb643dab1";

/// Fetches and builds QD, compiles its shim, and links both. Stops the build
/// when the C++ compiler is not the pinned release, or when it makes other
/// machine code than floaty's `Qd` algorithm follows.
pub(super) fn build(manifest: &Path, out: &Path) {
    let shim = manifest.join("shim").join("qd_shim.cpp");
    println!("cargo:rerun-if-changed={}", shim.display());
    let unary = manifest.join("shim").join("qd_unary.cpp");
    println!("cargo:rerun-if-changed={}", unary.display());
    let compiler = check_gcc(QD_CXX, QD_CXX_VERSION, "Qd", "install the g++ package");
    let source = out.join("source");
    let root = QD.extract(manifest, &source);

    // configure records the compiler and its options in the Makefiles, so
    // a new setting starts from `make clean`. The stamp lives in the
    // extraction, so a new extraction builds the library again.
    let settings = format!(
        "{} {} CXX={QD_CXX} CXXFLAGS={}\n{compiler}",
        QD.sha256,
        QD_CONFIGURE.join(" "),
        QD_CXXFLAGS.join(" ")
    );
    run_once(&source.join("library.stamp"), &settings, || {
        configure(&root);
        make(&root.join("src"), &[String::from("clean")]);
        make(&root.join("src"), &[]);
    });

    let libraries = out.join("lib");
    let include = format!("-I{}", root.join("include").display());
    let mut flags = QD_CXXFLAGS.to_vec();
    flags.extend(["-fPIC", include.as_str()]);
    let shim_object = shim_library(
        QD_CXX,
        &shim,
        &flags,
        &settings,
        &out.join("shim.stamp"),
        &libraries,
        "floaty_qd",
    );
    check_machine_code(&shim_object, &QD_SHIM_SECTIONS, QD_SHIM_SHA256);
    let unary_object = shim_library(
        QD_CXX,
        &unary,
        &flags,
        &settings,
        &out.join("unary.stamp"),
        &libraries,
        "floaty_qd_unary",
    );
    check_machine_code(&unary_object, &QD_SHIM_SECTIONS, QD_UNARY_SHA256);
    let library = root.join("src").join(".libs").join("libqd.a");
    let object = out.join("dd_real.o");
    extract_member("ar", &library, &object, "install the binutils package");
    check_machine_code(&object, &QD_LIBRARY_SECTIONS, QD_LIBRARY_SHA256);

    // The shim calls QD, and both call the C++ standard library.
    println!("cargo:rustc-link-search=native={}", libraries.display());
    println!(
        "cargo:rustc-link-search=native={}",
        root.join("src").join(".libs").display()
    );
    println!("cargo:rustc-link-lib=static=floaty_qd");
    println!("cargo:rustc-link-lib=static=floaty_qd_unary");
    println!("cargo:rustc-link-lib=static=qd");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// Runs the QD `configure` script with the pinned options, and checks that
/// `include/qd/qd_config.h` holds the pinned configuration.
fn configure(root: &Path) {
    let mut command = Command::new("./configure");
    for variable in COMPILER_VARIABLES {
        command.env_remove(variable);
    }
    let status = command
        .current_dir(root)
        .args(QD_CONFIGURE)
        .arg(format!("CXX={QD_CXX}"))
        .arg(format!("CXXFLAGS={}", QD_CXXFLAGS.join(" ")))
        .status()
        .expect("the QD configure script can run");
    assert!(
        status.success(),
        "QD configure failed in {}",
        root.display()
    );

    let config = root.join("include").join("qd").join("qd_config.h");
    let text = fs::read_to_string(&config).expect("configure writes qd_config.h");
    for expected in QD_CONFIG_LINES {
        assert!(
            text.lines().any(|line| line.trim() == expected),
            "{} does not hold `{expected}`",
            config.display()
        );
    }
}

/// Stops the build when the given sections of a QD object do not have the
/// pinned SHA-256. The sections hold the machine code and its constants, and
/// not the metadata of the object, such as the `.comment` of the compiler.
/// The digest does not cover the relocations. The compiler pin and the
/// pinned sources fix them.
fn check_machine_code(object: &Path, sections: &[&str], expected: &str) {
    let mut content = Vec::new();
    let dump = object.with_extension("section");
    for section in sections {
        let status = Command::new("objcopy")
            .args(["-O", "binary"])
            .arg(format!("--only-section={section}"))
            .arg(object)
            .arg(&dump)
            .status()
            .unwrap_or_else(|error| {
                panic!("objcopy cannot run ({error}); install the binutils package")
            });
        assert!(
            status.success(),
            "objcopy cannot read {section} of {}",
            object.display()
        );
        // objcopy writes an empty file for a section that the object does
        // not have.
        let bytes = fs::read(&dump).expect("objcopy writes the section");
        assert!(
            !bytes.is_empty(),
            "{} has no section {section}. floaty's Qd algorithm follows the pinned machine code",
            object.display()
        );
        content.extend(bytes);
    }
    fs::write(&dump, content).expect("the sections can be written");
    let digest = sha256(&dump);
    assert!(
        digest == expected,
        "the sections {sections:?} of {} have SHA-256 {digest}, not the pinned {expected}. \
         floaty's Qd algorithm follows the pinned machine code. A change to the shim or to the \
         compiler needs a new transcription",
        object.display()
    );
}
