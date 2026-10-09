//! `host_path`: a type reports `Ready` exactly where its host paths run, and
//! otherwise the first thing that sends its operations to the engine. The
//! format must have a host kind in the build, as the host path table of the
//! README lists. The mode must ask for no field that the host unit does not
//! give, as floaty documents. The environment of the host unit must be its
//! default: the test sets each control of MXCSR, FPCR, or the FPC register
//! that the host-path tests set, and expects `Environment` under each.

use floaty::env::{EnvField, HostPath};
use floaty::mode::direction::{TiesToAway, TowardZero};
use floaty::mode::switch::On;
use floaty::mode::{DenormalsAreZero, FlushToZero, Ieee, Libgcc, Precision, Rounded, X87};
use floaty::{BF16, Binary, D64Bid, F8E4M3, F16, F32, F64, F80, F128, Float};

/// The binary32 type of the mode `M`.
type Single<M> = Float<Binary<8>, 32, M>;

/// Returns `ready` in a build with host paths, and `Unavailable` in a build
/// without them.
fn in_host_builds(ready: HostPath) -> HostPath {
    if cfg!(floaty_engine_only) {
        HostPath::Unavailable
    } else {
        ready
    }
}

#[test]
fn each_format_reports_whether_the_build_has_its_host_kind() {
    let ready = in_host_builds(HostPath::Ready);
    assert_eq!(F16::host_path(), ready);
    assert_eq!(BF16::host_path(), ready);
    assert_eq!(F32::host_path(), ready);
    assert_eq!(F64::host_path(), ready);
    // x87 extended runs on the x87 unit of x86, and binary128 on the binary
    // floating-point unit of s390x.
    let x87 = if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
        ready
    } else {
        HostPath::Unavailable
    };
    assert_eq!(F80::host_path(), x87);
    let quad = if cfg!(target_arch = "s390x") {
        ready
    } else {
        HostPath::Unavailable
    };
    assert_eq!(F128::host_path(), quad);
    assert_eq!(F8E4M3::host_path(), HostPath::Unavailable);
    assert_eq!(D64Bid::host_path(), HostPath::Unavailable);
}

#[test]
fn a_mode_names_the_field_that_takes_the_engine() {
    let mode = |field| in_host_builds(HostPath::Mode(field));
    assert_eq!(
        Single::<Rounded<Ieee, TowardZero>>::host_path(),
        mode(EnvField::Rounding)
    );
    assert_eq!(
        Single::<Rounded<Ieee, TiesToAway>>::host_path(),
        mode(EnvField::Rounding)
    );
    assert_eq!(
        Single::<FlushToZero<Ieee, On>>::host_path(),
        mode(EnvField::FlushToZero)
    );
    assert_eq!(
        Single::<DenormalsAreZero<Ieee, On>>::host_path(),
        mode(EnvField::DenormalsAreZero)
    );
    assert_eq!(
        Single::<Precision<Ieee, 11>>::host_path(),
        mode(EnvField::Precision)
    );
    // The NaN rule and the tininess rule change no result that is not a NaN,
    // and a precision limit at the precision of the format changes none.
    let ready = in_host_builds(HostPath::Ready);
    assert_eq!(Single::<Libgcc>::host_path(), ready);
    assert_eq!(Single::<Precision<Ieee, 24>>::host_path(), ready);
    assert_eq!(Float::<Binary<11>, 64, X87>::host_path(), ready);
}

/// Returns `Environment` in a build with host paths, and `Unavailable` in a
/// build without them.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "s390x"))]
fn environment() -> HostPath {
    in_host_builds(HostPath::Environment)
}

/// Checks that the host kinds of the SSE unit report `Environment` under
/// each MXCSR control of the host-path tests but the default.
#[cfg(target_arch = "x86_64")]
#[test]
fn a_control_of_mxcsr_takes_the_engine() {
    use floaty_verify::x86::{MXCSR_HOST_PATH_CONTROLS, MXCSR_MASKED, with_mxcsr};

    for control in MXCSR_HOST_PATH_CONTROLS {
        let expected = if control == MXCSR_MASKED {
            in_host_builds(HostPath::Ready)
        } else {
            environment()
        };
        let reported = with_mxcsr(control, || {
            [
                F16::host_path(),
                BF16::host_path(),
                F32::host_path(),
                F64::host_path(),
            ]
        });
        assert_eq!(reported, [expected; 4], "MXCSR {control:#x}");
    }
}

/// Checks that x87 extended reports `Environment` under each x87 control
/// word of the host-path tests but the default, which rounds to nearest at
/// the 64-bit precision with every exception masked. A precision below 64
/// bits keeps the x87 arithmetic in the engine.
#[cfg(target_arch = "x86_64")]
#[test]
fn a_control_word_of_the_x87_unit_takes_the_engine() {
    use floaty_verify::x86::{
        X87_FULL_PRECISION, X87_HOST_PATH_CONTROLS, X87_MASKED, with_control_word,
    };

    for control in X87_HOST_PATH_CONTROLS {
        let expected = if control == X87_MASKED | X87_FULL_PRECISION {
            in_host_builds(HostPath::Ready)
        } else {
            environment()
        };
        let reported = with_control_word(control, F80::host_path);
        assert_eq!(reported, expected, "x87 control word {control:#x}");
    }
}

/// Checks that each setting of FPCR that the processor holds reports
/// `Environment`, and that a setting that does not hold reports `Ready`.
//
// Conflict: the FPCR settings of the host-path tests set each trap enable,
// and FPCR in the Arm Architecture Registers, DDI 0601, lets a processor
// that does not support a trap make its enable bit read as zero and ignore
// writes. Under QEMU 10.2.1 IOE, bit 8, reads back as zero after a write of
// 0x100, so FPCR holds the default, and the host paths run. The test reads
// FPCR back, expects `Environment` exactly for the settings that hold, and
// requires that some setting holds, so that it checks the path.
#[cfg(target_arch = "aarch64")]
#[test]
fn a_setting_of_fpcr_takes_the_engine() {
    use floaty_verify::aarch64::{FPCR_SETTINGS, fpcr, with_fpcr};

    let mut held = 0;
    for control in FPCR_SETTINGS {
        let (holds, reported) = with_fpcr(control, || {
            (fpcr() != 0, [F16::host_path(), F32::host_path()])
        });
        let expected = if holds {
            held += 1;
            environment()
        } else {
            in_host_builds(HostPath::Ready)
        };
        assert_eq!(reported, [expected; 2], "FPCR {control:#x}");
    }
    assert!(held > 0, "some setting of FPCR holds");
}

/// Checks that each setting of the FPC register reports `Environment`.
#[cfg(target_arch = "s390x")]
#[test]
fn a_setting_of_fpc_takes_the_engine() {
    use floaty_verify::s390x::{FPC_SETTINGS, with_fpc};

    for control in FPC_SETTINGS {
        let reported = with_fpc(control, || {
            [F16::host_path(), F32::host_path(), F128::host_path()]
        });
        assert_eq!(reported, [environment(); 3], "FPC {control:#x}");
    }
}
