//! Whether a host path may run: the fields of the mode that the host unit
//! gives, and the environment of the unit, which a path reads for each
//! operation, or once for the lanes of one operation through [`Unit`].
//! Every host path, and `host_path`, take the one rule of `incompatible`.

use super::super::environment::{self, default_environment};
use super::super::{Host, Kind, available};
use crate::env::{Env, EnvField, HostPath, Rounding};

/// Returns the first field of `env` that the host unit does not give for a
/// format of `precision` bits, or `None` when the unit gives the results of
/// `env`.
#[inline]
const fn incompatible(env: &Env, precision: u32) -> Option<EnvField> {
    // Every field is named, so a new field of `Env` needs a decision here.
    // Tininess changes only flags, the NaN rule applies only to a NaN result,
    // and the total order applies to no arithmetic. Saturation changes the
    // result of an overflow, which the host unit carries to an infinity.
    let Env {
        rounding,
        flush_to_zero,
        denormals_are_zero,
        tininess: _,
        nan: _,
        precision: limit,
        saturate,
        total_order: _,
    } = *env;
    if !matches!(rounding, Rounding::TiesToEven) {
        return Some(EnvField::Rounding);
    }
    if flush_to_zero {
        return Some(EnvField::FlushToZero);
    }
    if denormals_are_zero {
        return Some(EnvField::DenormalsAreZero);
    }
    if let Some(limit) = limit
        && limit.get() < precision
    {
        return Some(EnvField::Precision);
    }
    if saturate {
        return Some(EnvField::Saturate);
    }
    None
}

/// Returns `true` when the host unit gives the results of `env` for a format
/// of `precision` bits.
#[inline]
pub(super) const fn compatible(env: &Env, precision: u32) -> bool {
    incompatible(env, precision).is_none()
}

/// Returns whether the host paths of a format of the host kind `host` and of
/// `precision` bits run in the mode `env`, with the environment of the host
/// unit of this thread, as [`HostPath`] states. The environment is the one
/// that the arithmetic needs, which for x87 extended includes the 64-bit
/// precision.
#[must_use]
pub fn host_path(host: Host, env: &Env, precision: u32) -> HostPath {
    let kinds = [
        Kind::Arithmetic,
        Kind::SquareRoot,
        Kind::FusedMultiplyAdd,
        Kind::RoundToIntegral,
        Kind::ToInt,
        Kind::FromInt,
        Kind::Comparison,
        Kind::Remainder,
    ];
    if !kinds.into_iter().any(|kind| available(host, kind)) {
        return HostPath::Unavailable;
    }
    if let Some(field) = incompatible(env, precision) {
        return HostPath::Mode(field);
    }
    if ready_for_arithmetic(host, env, precision) {
        HostPath::Ready
    } else {
        HostPath::Environment
    }
}

/// How a scalar host path learns whether the floating-point environment of
/// its unit is the default. A host path reads the environment for each
/// operation by default. A `Lanes` operation that runs a scalar host path in
/// each lane reads it once for all lanes: no code between the lanes changes
/// the environment, and `STMXCSR` takes about 20 cycles on a Ryzen AI Max+
/// 395, more than the operation. The x87 unit reads its control word for
/// each operation either way, because that read is fast.
#[derive(Clone, Copy, Debug)]
pub enum Unit {
    /// The host path reads the environment of its unit.
    Read,
    /// One read found the environment of the unit of every host kind but x87
    /// extended: `true` when it is the default.
    Known(bool),
}

impl Unit {
    /// Reads once the environment of the unit that computes the host kind
    /// `host`, for the scalar host paths of the lanes of one operation. x87
    /// extended and a kind without a host unit keep [`Unit::Read`].
    #[must_use]
    #[inline]
    pub fn read(host: Host) -> Self {
        match host {
            Host::None | Host::Extended => Self::Read,
            Host::Single | Host::Double | Host::Half | Host::BFloat | Host::Quad => {
                Self::Known(default_environment())
            }
        }
    }

    /// Returns `true` when the environment of the unit of every host kind
    /// but x87 extended is the default, as one read found it, or from a read
    /// now.
    #[inline]
    pub(super) fn default_environment(self) -> bool {
        match self {
            Self::Read => default_environment(),
            Self::Known(default) => default,
        }
    }
}

/// Returns the host kind whose unit converts the host kind `from` to `to`:
/// the x87 unit converts to and from x87 extended precision.
#[must_use]
#[inline]
pub const fn conversion_unit(from: Host, to: Host) -> Host {
    if matches!(from, Host::Extended) {
        from
    } else {
        to
    }
}

/// Returns `true` when the mode and the environment of the host unit that
/// computes the host kind `host` allow a host path for a format of
/// `precision` bits. The x87 unit computes x87 extended precision, and the
/// SSE unit, the AArch64 unit, or the s390x unit computes the other kinds.
#[inline]
pub(in crate::host) fn ready_for(host: Host, env: &Env, precision: u32) -> bool {
    ready_for_in(host, env, precision, Unit::Read)
}

/// Returns `true` when `ready_for` allows the path, with the environment of
/// the unit from `unit`.
#[inline]
pub(super) fn ready_for_in(host: Host, env: &Env, precision: u32, unit: Unit) -> bool {
    let unit = match host {
        Host::Extended => environment::x87_environment(),
        Host::None | Host::Half | Host::BFloat | Host::Single | Host::Double | Host::Quad => {
            unit.default_environment()
        }
    };
    compatible(env, precision) && unit
}

/// Returns the rounding direction of `env` when `ready_for` allows a host
/// path of `host` for a format of `precision` bits in `env` rounded to
/// nearest even, and the direction is one of IEEE 754 that the instructions
/// of a path can take from their encoding: to nearest even, toward +∞,
/// toward -∞, or toward zero. Returns `None` otherwise. The packed paths of
/// the slice operations round in the other three directions only in the
/// forms with a rounding control, as `packed::dispatch::direction` checks.
#[inline]
pub(in crate::host) fn ready_in_direction(
    host: Host,
    env: &Env,
    precision: u32,
) -> Option<Rounding> {
    let direction = matches!(
        env.rounding,
        Rounding::TiesToEven
            | Rounding::TowardPositive
            | Rounding::TowardNegative
            | Rounding::TowardZero
    );
    let nearest = env.with_rounding(Rounding::TiesToEven);
    (direction && ready_for(host, &nearest, precision)).then_some(env.rounding)
}

/// Returns `true` when `ready_for` allows a host path whose instructions
/// round to the precision of the unit: `+`, `-`, `*`, `/`, and `sqrt`. The
/// x87 unit must then compute at the 64-bit precision too, which its other
/// paths do not need, as `x87_environment` states.
#[inline]
pub(in crate::host) fn ready_for_arithmetic(host: Host, env: &Env, precision: u32) -> bool {
    ready_for_arithmetic_in(host, env, precision, Unit::Read)
}

/// Returns `true` when `ready_for_arithmetic` allows the path, with the
/// environment of the unit from `unit`.
#[inline]
pub(super) fn ready_for_arithmetic_in(host: Host, env: &Env, precision: u32, unit: Unit) -> bool {
    match host {
        Host::Extended => compatible(env, precision) && environment::x87_full_precision(),
        Host::None | Host::Half | Host::BFloat | Host::Single | Host::Double | Host::Quad => {
            ready_for_in(host, env, precision, unit)
        }
    }
}

/// Returns `true` when the mode and the environment of the host unit allow a
/// rounding to an integral value in the direction of `env`. The instructions
/// take the direction in their encoding, not from the environment, so every
/// other field must allow a host path, and the environment must round to
/// nearest even.
#[inline]
pub(in crate::host) fn ready_for_integral(host: Host, env: &Env, precision: u32) -> bool {
    ready_for_integral_in(host, env, precision, Unit::Read)
}

/// Returns `true` when `ready_for_integral` allows the path, with the
/// environment of the unit from `unit`.
#[inline]
pub(super) fn ready_for_integral_in(host: Host, env: &Env, precision: u32, unit: Unit) -> bool {
    ready_for_in(
        host,
        &env.with_rounding(Rounding::TiesToEven),
        precision,
        unit,
    )
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;

    use super::incompatible;
    use crate::env::{Env, EnvField, Rounding};

    #[test]
    fn only_a_behavior_that_the_host_matches_is_compatible() {
        assert_eq!(incompatible(&Env::IEEE, 53), None);
        assert_eq!(incompatible(&Env::X86_SSE, 24), None);
        assert_eq!(incompatible(&Env::X87, 53), None);
        let precision = Some(EnvField::Precision);
        assert_eq!(incompatible(&Env::X87, 113), precision);
        let toward_zero = Env::IEEE.with_rounding(Rounding::TowardZero);
        assert_eq!(incompatible(&toward_zero, 24), Some(EnvField::Rounding));
        let flush = Env::IEEE.with_flush_to_zero(true);
        assert_eq!(incompatible(&flush, 24), Some(EnvField::FlushToZero));
        let daz = Env::IEEE.with_denormals_are_zero(true);
        assert_eq!(incompatible(&daz, 24), Some(EnvField::DenormalsAreZero));
        // The host unit carries an overflow to an infinity.
        let saturate = Env::IEEE.with_saturate(true);
        assert_eq!(incompatible(&saturate, 24), Some(EnvField::Saturate));
        let limited = Env::IEEE.with_precision(NonZeroU32::new(52));
        assert_eq!(incompatible(&limited, 53), precision);
    }
}
