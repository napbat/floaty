//! The libgcc and glibc reference of the IBM `long double` format: a batch
//! program for powerpc64le that runs under QEMU.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::archive::sha256;
use crate::tools::{check_gcc, extract_member, run_once, tool_output};

/// The cross compiler of the libgcc reference.
const POWERPC_GCC: &str = "powerpc64le-linux-gnu-gcc";

/// The `ar` program of the cross compiler.
const POWERPC_AR: &str = "powerpc64le-linux-gnu-ar";

/// The pinned GCC release of the libgcc reference.
const POWERPC_GCC_VERSION: &str = "15.2.0";

/// The member of the pinned libgcc that floaty's `Gcc` algorithm follows,
/// with its SHA-256: `ibm-ldouble.o`, its machine code, its constants, and
/// its relocations. The algorithm follows this object one instruction at a
/// time, so a libgcc that compiles the routines differently is another
/// reference.
const LIBGCC_OBJECTS: [(&str, &str); 1] = [(
    "ibm-ldouble.o",
    "c484948ee6c0e1a9b7b4a54f31820707afbbb90ce33564aee121e0f1082d154c",
)];

/// The glibc 2.43 release of the pinned cross C library, package
/// `libc6-dev-ppc64el-cross` 2.43-2ubuntu2cross1.
const POWERPC_GLIBC_VERSION: &str = "2.43";

/// The members of the libm of the pinned cross C library that floaty's `Gcc`
/// algorithm follows or that the tests compare with, with the SHA-256 of
/// each: the IBM `long double` functions and their wrappers, and the
/// functions that they call: the binary64 square root of `sqrtl`,
/// `__ieee754_ilogbl`, `__roundeven`, and `__issignalingl`. A libm that
/// compiles them differently is another reference.
const LIBM_OBJECTS: [(&str, &str); 37] = [
    (
        "s_fmal.o",
        "86f2b7a97b10d189a036cbc16aed4ac2b1326c79441d5deb1c7ee65d027790bb",
    ),
    (
        "e_sqrt.o",
        "3df6f82e52adb28b0292a161dbf49dd70ea11fb2786bf058cca3be324ba0a1ff",
    ),
    (
        "e_sqrtl.o",
        "93e65984ea765c0d070f8c8fb925f474109ed451741973ef5134be6188ae9a06",
    ),
    (
        "w_sqrtl.o",
        "08bfb93344686fbe21dd823e2f27a4f534ee65ab0cee91067e0008ac0bbc140b",
    ),
    (
        "s_nextupl.o",
        "e4ab4853d982442b4c7d94e5dbe790a47248a6c3099b5d2336a91789dfd04c6f",
    ),
    (
        "s_nextdownl.o",
        "1c2dcae944237f7fd0457fabbe9d1a5b02b8a841a331f74330dcb7e2160dd9c7",
    ),
    (
        "e_fmodl.o",
        "6614b7e1677d7a9993cf9aed3ddbda50e798c102ba07c88b88b9d45e47945a82",
    ),
    (
        "w_fmodl.o",
        "07ea01960cfef47de1fc7487b0fa688016f72ea2a6d42a99f6900341ef63a271",
    ),
    (
        "e_remainderl.o",
        "ad495d9d171dd5afee6d701855c12cf227bb771c2c03f5755a29ec7d69841857",
    ),
    (
        "w_remainderl.o",
        "97d5572ffc7b1aa156e7a8f66c51fa43ac815d0c253ee45e67ba514b3f106b2d",
    ),
    (
        "s_iscanonicall.o",
        "6999e392e8489a1266bfb704537b7270b56dfeded4fbca11626f6d71586f1c41",
    ),
    (
        "s_logbl.o",
        "7b7b6eb98fd16da3a261f21fb56865e5f2fd365e3dea67a28b037b74114a6b38",
    ),
    (
        "s_floorl.o",
        "f26bca78e7c1d480ea2d6b8b97c267fcbde06736d71340eea0cf863fe3416c8b",
    ),
    (
        "s_ceill.o",
        "71189e58256432edd14d7beb9901f66766abd9ccc5b227f45d3437117ec86e22",
    ),
    (
        "s_truncl.o",
        "f6fa639f374e64a2f96b1d9aafcb4d28fe9d214d3ddbeaa401225f867fd3239d",
    ),
    (
        "s_roundl.o",
        "b4f2b9af39abf92a9c79c16833b180a5ea94e6a47103a861186c985cf7a7c3f2",
    ),
    (
        "s_roundevenl.o",
        "2d74a9961bc8851c3b3b7bd74e20eefc3976de4d9e0855e1fc06e340e4c9be9d",
    ),
    (
        "s_roundeven.o",
        "b13411839c09784079a2f4d7a7d66898f47e9a58c1bf2d7fb772305a5939a45d",
    ),
    (
        "s_rintl.o",
        "f60850e36ef7184ff05068c7fb6da7c865ef9d96a614a41a1a13d189c1ebcd49",
    ),
    (
        "s_nearbyintl.o",
        "5c4828b2b69801e4284c49f1beeb748673e27e57f703f4a91b8dd6542186951d",
    ),
    (
        "w_ilogbl.o",
        "f0a35a08c4288a370a57b94caa9354298f94e51b5bab669b00f70dad89ce8b18",
    ),
    (
        "e_ilogbl.o",
        "2f5f54f9551853b00d9a62df8cbbc4d98dd920ef60e4fca5bb4a899eb6dde960",
    ),
    (
        "s_fmaxl.o",
        "fb19d3aa42364c074469a1a5f5ba76a78cd0bfbb6e94737324ddacf62ebeac4f",
    ),
    (
        "s_fminl.o",
        "8ace4980f56bc8d931123d5cc51ea72be684d4b9366ab37bcd2d8e6ca3baa37b",
    ),
    (
        "s_fmaximuml.o",
        "ed86efec25c514ddb2b253d90ce7b635e3b5f06312b5dcd266143787cdc2c595",
    ),
    (
        "s_fminimuml.o",
        "ee77622a8a01c3d93b7312baa5a925ab9b8eeaf647931863c6878ee1afb0bed3",
    ),
    (
        "s_fmaximum_numl.o",
        "31b6300601576dd4ea41ede3219dc487a2e8164a7e51334a8e736ef6c7d0191d",
    ),
    (
        "s_fminimum_numl.o",
        "dde145c8e8751f37ffe4efc9b87a111f77c8747f05400d21cc1b266078072fd6",
    ),
    (
        "s_fmaximum_magl.o",
        "2d25b11d0c13cb77103bb0c966e9c36ad3e026aaf07c7cb5dc004d672d619aac",
    ),
    (
        "s_fminimum_magl.o",
        "18dac34a7e3136aba33a908ba6cdfc17b9989218daefe0aab3c4c422dcd69ac5",
    ),
    (
        "s_fmaximum_mag_numl.o",
        "0db6db3aac1b1503105e07e794cdd17cc6edc2cc0efe3a2d3fb2f91e3a832175",
    ),
    (
        "s_fminimum_mag_numl.o",
        "8ad6efcb6dada173fd18ebfd1e85cbf4bda23ffeff9488b3b6b6ea8920117cf7",
    ),
    (
        "s_issignalingl.o",
        "00279a808057ee7461772f805ce21eb7221087d68022ed0a061838d6aa8421fe",
    ),
    (
        "s_totalorderl.o",
        "30b02da1080f576f83ac414a36f3166cde3a2f56e2a37d65d1a2ddceed4d0367",
    ),
    (
        "s_totalordermagl.o",
        "5114638e7887a49455a2855f405a0c29e3fc5fc1c454c73e49b40a19904f8154",
    ),
    (
        "s_llrintl.o",
        "20a0895fd8a2122e71af63822f0067ea06cb1aeca35d6509046fe619236f043f",
    ),
    (
        "s_lroundl.o",
        "e2c9201f74d9ad9107e6370abc72d73fe8e9852243ba8bf7dce228e06e43fcd1",
    ),
];

/// The members of the static C library of the pinned cross C library that
/// the reference uses, with the SHA-256 of each: `__frexp`, `__scalbn`, and
/// `qsort`, which `fmal` calls, and whose merge sort decides the order of
/// the partial products, and `copysignl` and `scalbnl`, which glibc builds
/// into the C library.
const LIBC_OBJECTS: [(&str, &str); 5] = [
    (
        "s_frexp.o",
        "70e35774763ee598004b6519a81f13d23d3b488df034e68c6d5704c9e0389ae2",
    ),
    (
        "s_scalbn.o",
        "89b76328c96755555e6bf0403435ef0398302cb0777d15fd2efca2a6253fdd92",
    ),
    (
        "qsort.o",
        "db6193fca885d8dd3cc8165f61cab7c13aea384825b632ee32978f53b74258f7",
    ),
    (
        "s_copysignl.o",
        "3943d3488f6e7f96824930b35de16868bafb0af39333fd485b07fa945367869d",
    ),
    (
        "s_scalbnl.o",
        "5dc21726e5a005f45e831b780cd295a007f856e01f7c077aa3ea1cf3b60f5b59",
    ),
];

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

/// The libraries that the batch program links after its source: the libm of
/// the pinned C library.
const IBM_LDOUBLE_LIBRARIES: [&str; 1] = ["-lm"];

/// Builds the libgcc batch program for powerpc64le, and names it in
/// `FLOATY_IBM_LDOUBLE`. Stops the build when the cross compiler is not the
/// pinned release, or when QEMU cannot run.
pub(super) fn build(manifest: &Path, out: &Path) {
    let source = manifest.join("shim").join("ibm_ldouble.c");
    println!("cargo:rerun-if-changed={}", source.display());
    let cross_advice = "install the gcc-powerpc64le-linux-gnu package";
    let compiler = check_gcc(POWERPC_GCC, POWERPC_GCC_VERSION, "Gcc", cross_advice);
    let libgcc = tool_output(POWERPC_GCC, &["-print-libgcc-file-name"], cross_advice);
    let libgcc = PathBuf::from(libgcc.trim());
    println!("cargo:rerun-if-changed={}", libgcc.display());
    let libgcc_release = format!("libgcc {POWERPC_GCC_VERSION}");
    check_members(&libgcc, &LIBGCC_OBJECTS, &libgcc_release, out, cross_advice);
    let glibc_release = format!("glibc {POWERPC_GLIBC_VERSION}");
    let math_library = tool_output(POWERPC_GCC, &["-print-file-name=libm.a"], cross_advice);
    let math_library = PathBuf::from(math_library.trim());
    println!("cargo:rerun-if-changed={}", math_library.display());
    check_members(
        &math_library,
        &LIBM_OBJECTS,
        &glibc_release,
        out,
        cross_advice,
    );
    let c_library = tool_output(POWERPC_GCC, &["-print-file-name=libc.a"], cross_advice);
    let c_library = PathBuf::from(c_library.trim());
    println!("cargo:rerun-if-changed={}", c_library.display());
    check_members(&c_library, &LIBC_OBJECTS, &glibc_release, out, cross_advice);
    let emulator = tool_output(
        QEMU_POWERPC,
        &["--version"],
        "install the qemu-user package",
    );
    let emulator = emulator.lines().next().unwrap_or_default();
    assert!(
        emulator.starts_with(&format!("{QEMU_POWERPC} version {QEMU_VERSION} ")),
        "{emulator:?} is not the pinned QEMU {QEMU_VERSION}. QEMU executes the libgcc reference \
         and gives its flags"
    );
    // The tests run the same emulator, and check the release again, because
    // QEMU can change after the build.
    println!("cargo:rustc-env=FLOATY_QEMU={QEMU_POWERPC}");
    println!("cargo:rustc-env=FLOATY_QEMU_VERSION={QEMU_VERSION}");

    let program = out.join("ibm_ldouble");
    let content = fs::read_to_string(&source).expect("the libgcc batch program can be read");
    let key = format!(
        "{compiler}\n{} {} {} {} {}\n{content}",
        sha256(&libgcc),
        sha256(&math_library),
        sha256(&c_library),
        IBM_LDOUBLE_FLAGS.join(" "),
        IBM_LDOUBLE_LIBRARIES.join(" ")
    );
    run_once(&out.with_extension("stamp"), &key, || {
        fs::create_dir_all(out).expect("the program directory can be created");
        let status = Command::new(POWERPC_GCC)
            .args(IBM_LDOUBLE_FLAGS)
            .arg(&source)
            .args(IBM_LDOUBLE_LIBRARIES)
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

/// Stops the build when a member of a static library of `release` that
/// floaty's `Gcc` algorithm follows differs from the pinned object.
fn check_members(
    archive: &Path,
    members: &[(&str, &str)],
    release: &str,
    out: &Path,
    advice: &str,
) {
    fs::create_dir_all(out).expect("the program directory can be created");
    for &(member, pinned) in members {
        let object = out.join(member);
        extract_member(POWERPC_AR, archive, &object, advice);
        let digest = sha256(&object);
        assert!(
            digest == pinned,
            "{member} in {} has SHA-256 {digest}, not the pinned {pinned} of {release}. floaty's \
             Gcc algorithm follows the pinned object",
            archive.display()
        );
    }
}
