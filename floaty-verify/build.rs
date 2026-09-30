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
//!   powerpc64le libgcc run in a batch program under `qemu-ppc64le`. QD
//!   comes from a pinned release archive, as the decimal libraries do.
//!
//! The build needs `make`, `gcc`, `g++` 15.2.0, `ar`, `objcopy`, `curl`,
//! `tar`, `python3`, `sha256sum`, `powerpc64le-linux-gnu-gcc` 15.2.0, and
//! `qemu-ppc64le`, and it runs on Linux x86-64 hosts only. The pinned
//! Makefiles name `gcc`.

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
    /// One source file, which the extraction copies under this name.
    File(&'static str),
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

/// The compiler variables that the build removes from the environment of
/// `make` and `configure`, so that an exported value cannot replace a pinned
/// setting.
const COMPILER_VARIABLES: [&str; 6] = ["CFLAGS", "CPPFLAGS", "LDFLAGS", "CC", "CXX", "CXXFLAGS"];

/// The cross compiler of the libgcc reference.
const POWERPC_GCC: &str = "powerpc64le-linux-gnu-gcc";

/// The pinned GCC release of the libgcc reference.
const POWERPC_GCC_VERSION: &str = "15.2.0";

/// The SHA-256 of `ibm-ldouble.o` in the pinned libgcc: its machine code,
/// its constants, and its relocations. floaty's `Gcc` algorithm follows this
/// object one instruction at a time, so a libgcc that compiles the routines
/// differently is another reference.
const IBM_LDOUBLE_SHA256: &str = "c484948ee6c0e1a9b7b4a54f31820707afbbb90ce33564aee121e0f1082d154c";

/// The pinned QEMU release, which executes the libgcc reference and gives
/// its flags.
const QEMU_VERSION: &str = "10.2.1";

/// The emulator that runs the libgcc reference.
const QEMU_POWERPC: &str = "qemu-ppc64le";

/// The compiler options of the libgcc batch program.
///
/// - `-frounding-math` stops the compiler from assuming the default
///   rounding direction.
/// - `-static` links the C library into the program, so QEMU needs no
///   PowerPC system root.
const IBM_LDOUBLE_FLAGS: [&str; 5] = ["-std=gnu11", "-O2", "-frounding-math", "-Wall", "-static"];

/// QD 2.3.24, under the BSD-LBNL license.
const QD: Archive = Archive {
    file: "qd-2.3.24.tar.gz",
    packing: Packing::GzipTar,
    url: "https://www.davidhbailey.com/dhbsoftware/qd-2.3.24.tar.gz",
    sha256: "a47b6c73f86e6421e86a883568dd08e299b20e36c11a99bdfbe50e01bde60e38",
};

/// Mesa's conversions of the unsigned 11-bit and 10-bit floats of
/// R11G11B10, from its release 25.2.0, under the MIT license. They follow
/// `GL_EXT_packed_float`.
const MESA_R11G11B10: Archive = Archive {
    file: "mesa-25.2.0-format_r11g11b10f.h",
    packing: Packing::File("format_r11g11b10f.h"),
    url: "https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-25.2.0/src/util/format_r11g11b10f.h",
    sha256: "b0beabfa138da9c4935227e7d4d392c677007d84b7ec90e776f16ac68a79cf19",
};

/// The rounding functions of Mesa 25.2.0 that the R11G11B10 conversions
/// call, under the MIT license.
const MESA_ROUNDING: Archive = Archive {
    file: "mesa-25.2.0-rounding.h",
    packing: Packing::File("rounding.h"),
    url: "https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-25.2.0/src/util/rounding.h",
    sha256: "4265b083f0424e5243c6a81771e1fc077203d2fa829e43a6caafa26f20fb1a02",
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
/// of QD, the code of `dd_real::accurate_div`, and their constants.
const QD_SHIM_SECTIONS: [&str; 3] = [
    ".text",
    ".text._ZN7dd_real12accurate_divERKS_S1_",
    ".rodata.cst16",
];

/// The SHA-256 of [`QD_SHIM_SECTIONS`] in the shim object, one section after
/// the other.
const QD_SHIM_SHA256: &str = "8014d42e4610cf75870b2aed54270a083f5a5ed6f35efc0ad4528907983a26bd";

/// The sections of `dd_real.o` in `libqd.a` that floaty's `Qd` algorithm
/// follows: the code of the square root and its constants.
const QD_LIBRARY_SECTIONS: [&str; 3] = [".text", ".rodata.cst8", ".rodata.cst16"];

/// The SHA-256 of [`QD_LIBRARY_SECTIONS`] in `dd_real.o`, one section after
/// the other.
const QD_LIBRARY_SHA256: &str = "0bcc95f145495cde281ad918b8aa6f1a841eeb879c4049bfcad234edb643dab1";

fn main() {
    let manifest =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    println!("cargo:rerun-if-changed=build.rs");

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

    build_testfloat(&manifest, &out);
    build_decimal(&manifest, &out.join("decimal"));
    build_ibm_ldouble(&manifest, &out.join("ibm_ldouble"));
    build_qd(&manifest, &out.join("qd"));
    build_mesa(&manifest, &out.join("mesa"));
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

/// Builds the libgcc batch program for powerpc64le, and names it in
/// `FLOATY_IBM_LDOUBLE`. Stops the build when the cross compiler is not the
/// pinned release, or when QEMU cannot run.
fn build_ibm_ldouble(manifest: &Path, out: &Path) {
    let source = manifest.join("shim").join("ibm_ldouble.c");
    println!("cargo:rerun-if-changed={}", source.display());
    let cross_advice = "install the gcc-powerpc64le-linux-gnu package";
    let version = tool_output(POWERPC_GCC, &["-dumpfullversion"], cross_advice);
    assert!(
        version.trim() == POWERPC_GCC_VERSION,
        "{POWERPC_GCC} is GCC {}, not the pinned GCC {POWERPC_GCC_VERSION}. floaty's Gcc \
         algorithm follows the machine code of the pinned compiler",
        version.trim()
    );
    // The first line of `--version` names the distribution build of GCC.
    let compiler = tool_output(POWERPC_GCC, &["--version"], cross_advice);
    let compiler = compiler.lines().next().unwrap_or_default();
    let libgcc = tool_output(POWERPC_GCC, &["-print-libgcc-file-name"], cross_advice);
    let libgcc = PathBuf::from(libgcc.trim());
    println!("cargo:rerun-if-changed={}", libgcc.display());
    check_ibm_ldouble(&libgcc, out, cross_advice);
    let emulator = tool_output(
        QEMU_POWERPC,
        &["--version"],
        "install the qemu-user package",
    );
    let emulator = emulator.lines().next().unwrap_or_default();
    assert!(
        emulator.starts_with(&format!("qemu-ppc64le version {QEMU_VERSION} ")),
        "{emulator:?} is not the pinned QEMU {QEMU_VERSION}. QEMU executes the libgcc reference \
         and gives its flags"
    );
    // The tests check the release again, because QEMU can change after the
    // build.
    println!("cargo:rustc-env=FLOATY_QEMU_VERSION={QEMU_VERSION}");

    let program = out.join("ibm_ldouble");
    let content = fs::read_to_string(&source).expect("the libgcc batch program can be read");
    let key = format!(
        "{compiler}\n{} {}\n{content}",
        sha256(&libgcc),
        IBM_LDOUBLE_FLAGS.join(" ")
    );
    run_once(&out.with_extension("stamp"), &key, || {
        fs::create_dir_all(out).expect("the program directory can be created");
        let status = Command::new(POWERPC_GCC)
            .args(IBM_LDOUBLE_FLAGS)
            .arg(&source)
            .arg("-lm")
            .arg("-o")
            .arg(&program)
            .status()
            .expect("the cross compiler can run");
        assert!(
            status.success(),
            "{POWERPC_GCC} cannot build {}",
            source.display()
        );
    });
    println!("cargo:rustc-env=FLOATY_IBM_LDOUBLE={}", program.display());
}

/// Stops the build when `ibm-ldouble.o` in `libgcc` differs from the pinned
/// object.
fn check_ibm_ldouble(libgcc: &Path, out: &Path, advice: &str) {
    fs::create_dir_all(out).expect("the program directory can be created");
    let object = out.join("ibm-ldouble.o");
    extract_member("powerpc64le-linux-gnu-ar", libgcc, &object, advice);
    let digest = sha256(&object);
    assert!(
        digest == IBM_LDOUBLE_SHA256,
        "ibm-ldouble.o in {} has SHA-256 {digest}, not the pinned {IBM_LDOUBLE_SHA256}. floaty's \
         Gcc algorithm follows the pinned object",
        libgcc.display()
    );
}

/// Fetches and builds QD, compiles its shim, and links both. Stops the build
/// when the C++ compiler is not the pinned release, or when it makes other
/// machine code than floaty's `Qd` algorithm follows.
fn build_qd(manifest: &Path, out: &Path) {
    let shim = manifest.join("shim").join("qd_shim.cpp");
    println!("cargo:rerun-if-changed={}", shim.display());
    let advice = "install the g++ package";
    let version = tool_output(QD_CXX, &["-dumpfullversion"], advice);
    assert!(
        version.trim() == QD_CXX_VERSION,
        "{QD_CXX} is GCC {}, not the pinned GCC {QD_CXX_VERSION}. floaty's Qd algorithm \
         follows the machine code of the pinned compiler",
        version.trim()
    );
    let compiler = tool_output(QD_CXX, &["--version"], advice);
    let compiler = compiler.lines().next().unwrap_or_default().to_owned();
    let downloads = manifest.join("reference").join("downloads");
    let source = out.join("source");
    unpack(&fetch(&QD, &downloads), &QD, &source);
    let root = source.join("qd-2.3.24");

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
        configure_qd(&root);
        make(&root.join("src"), &[String::from("clean")]);
        make(&root.join("src"), &[]);
    });

    let libraries = out.join("lib");
    let content = fs::read_to_string(&shim).expect("the QD shim can be read");
    run_once(
        &out.join("shim.stamp"),
        &format!("{settings}\n{content}"),
        || {
            fs::create_dir_all(&libraries).expect("the library directory can be created");
            let include = format!("-I{}", root.join("include").display());
            let mut flags = QD_CXXFLAGS.to_vec();
            flags.extend(["-fPIC", include.as_str()]);
            let object = libraries.join("floaty_qd.o");
            compile(QD_CXX, &shim, &flags, &object);
            archive(&libraries.join("libfloaty_qd.a"), &[object]);
        },
    );
    check_machine_code(
        &libraries.join("floaty_qd.o"),
        &QD_SHIM_SECTIONS,
        QD_SHIM_SHA256,
    );
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
    println!("cargo:rustc-link-lib=static=qd");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// Runs the QD `configure` script with the pinned options, and checks that
/// `include/qd/qd_config.h` holds the pinned configuration.
fn configure_qd(root: &Path) {
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

/// Copies the member of a static library that has the file name of `object`
/// into `object`, with the `ar` program of the library's target.
fn extract_member(ar: &str, library: &Path, object: &Path, advice: &str) {
    let member = object.file_name().expect("an object has a file name");
    let output = Command::new(ar)
        .arg("p")
        .arg(library)
        .arg(member)
        .output()
        .unwrap_or_else(|error| panic!("{ar} cannot run ({error}); {advice}"));
    assert!(
        output.status.success(),
        "{} has no {}",
        library.display(),
        Path::new(member).display()
    );
    fs::write(object, output.stdout).expect("the library member can be written");
}

/// Runs a tool that the build needs and returns its standard output. Stops
/// the build with `advice` when the tool cannot run or fails.
fn tool_output(program: &str, arguments: &[&str], advice: &str) -> String {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .unwrap_or_else(|error| panic!("{program} cannot run ({error}); {advice}"));
    assert!(
        output.status.success(),
        "`{program} {}` failed; {advice}",
        arguments.join(" ")
    );
    String::from_utf8(output.stdout).expect("the tool writes text")
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
            Packing::File(name) => {
                fs::copy(path, destination.join(name)).expect("the source file can be copied");
                return;
            }
            Packing::Zip => Command::new("python3")
                .args(["-m", "zipfile", "-e"])
                .arg(path)
                .arg(destination)
                .status()
                .expect("python3 can run"),
            // The QD archive holds macOS extended headers, which GNU tar
            // ignores with a warning.
            Packing::GzipTar => Command::new("tar")
                .arg("--warning=no-unknown-keyword")
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
        compile("gcc", shim, &flags, &object);
        archive(&libraries.join("libfloaty_binary80.a"), &[object]);
    });
}

/// Fetches Mesa's R11G11B10 header and the rounding header that it includes,
/// and compiles the shim that exposes its inline functions into
/// `libfloaty_mesa.a`.
fn build_mesa(manifest: &Path, out: &Path) {
    let shim = manifest.join("shim").join("mesa_r11g11b10.c");
    println!("cargo:rerun-if-changed={}", shim.display());
    let downloads = manifest.join("reference").join("downloads");
    let format = out.join("format");
    unpack(
        &fetch(&MESA_R11G11B10, &downloads),
        &MESA_R11G11B10,
        &format,
    );
    let rounding = out.join("rounding");
    unpack(
        &fetch(&MESA_ROUNDING, &downloads),
        &MESA_ROUNDING,
        &rounding,
    );
    let content = fs::read_to_string(&shim).expect("the Mesa shim can be read");
    let key = format!(
        "{} {}\n{content}",
        MESA_R11G11B10.sha256, MESA_ROUNDING.sha256
    );
    run_once(&out.join("floaty_mesa.stamp"), &key, || {
        let format_include = format!("-I{}", format.display());
        let rounding_include = format!("-I{}", rounding.display());
        let flags = [
            "-std=gnu11",
            "-O2",
            "-fPIC",
            format_include.as_str(),
            rounding_include.as_str(),
        ];
        let object = out.join("floaty_mesa.o");
        compile("gcc", &shim, &flags, &object);
        archive(&out.join("libfloaty_mesa.a"), &[object]);
    });
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=floaty_mesa");
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

/// Compiles one source file with `compiler`.
fn compile(compiler: &str, source: &Path, flags: &[&str], object: &Path) {
    let status = Command::new(compiler)
        .args(flags)
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(object)
        .status()
        .expect("the compiler can run");
    assert!(
        status.success(),
        "{compiler} cannot compile {}",
        source.display()
    );
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
    for variable in COMPILER_VARIABLES {
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
