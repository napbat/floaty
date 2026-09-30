//! The modes: named behaviors for [`Float`](crate::Float) types.
//!
//! The modes live in this module because the encodings already use the names
//! [`Ieee`](crate::Ieee) and [`X87`](crate::X87) at the crate root. A preset
//! mode gives a type the behavior of one processor unit:
//!
//! ```
//! use floaty::{Binary, Float, X87, mode};
//!
//! // x87 extended precision with the behavior of the x87 unit after FNINIT.
//! type Register = Float<Binary<15, X87>, 80, mode::X87>;
//!
//! let infinity = Register::from_bits(0x7FFF_8000_0000_0000_0000);
//! // inf - inf is invalid and gives the negative floating-point indefinite.
//! assert_eq!((infinity - infinity).to_bits(), 0xFFFF_C000_0000_0000_0000);
//! ```
//!
//! A reference mode gives a double-double algorithm the behavior under which
//! it matches its reference, and each algorithm takes its reference mode by
//! default: [`Libgcc`] for [`Gcc`](crate::Gcc), and [`X86Sse`] for
//! [`Qd`](crate::Qd).
//!
//! A combinator makes a mode from another mode and one changed field, at
//! compile time. The fields that processors change at run time each have a
//! combinator: [`Rounded`], [`FlushToZero`], [`DenormalsAreZero`],
//! [`Precision`], [`FullPrecision`], [`Propagation`], and [`DetectTininess`].
//! An emulator can select one mode for each state of a control register, and
//! run the operations of that mode with the fields as constants:
//!
//! ```
//! use floaty::mode::direction::TowardZero;
//! use floaty::mode::switch::On;
//! use floaty::mode::{FlushToZero, Rounded, X86Sse};
//! use floaty::{Binary, Float, Rounding};
//!
//! // MXCSR with round toward zero and FTZ set.
//! type Truncating = FlushToZero<Rounded<X86Sse, TowardZero>, On>;
//! type Double = Float<Binary<11>, 64, Truncating>;
//!
//! assert_eq!(Double::ENV.rounding, Rounding::TowardZero);
//! assert!(Double::ENV.flush_to_zero);
//! // The same bits under another mode.
//! let one = Double::from_bits(0x3FF0_0000_0000_0000);
//! assert_eq!(one.with_mode::<X86Sse>().to_bits(), one.to_bits());
//! ```

use core::marker::PhantomData;
use core::num::NonZeroU32;

use super::{
    Env, FusedNanOrder, InvalidProduct, Mode, NanPropagation, NanRule, Rounding, Tininess,
};
use crate::sealed::Sealed;

/// The IEEE 754 default behavior, [`Env::IEEE`]. It is the default mode of
/// every [`Float`](crate::Float) type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ieee;

impl Sealed for Ieee {}
impl Mode for Ieee {
    const ENV: Env = Env::IEEE;
}

/// The x86 SSE preset, [`Env::X86_SSE`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct X86Sse;

impl Sealed for X86Sse {}
impl Mode for X86Sse {
    const ENV: Env = Env::X86_SSE;
}

/// The x87 preset, [`Env::X87`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct X87;

impl Sealed for X87 {}
impl Mode for X87 {
    const ENV: Env = Env::X87;
}

/// The behavior under which [`Gcc`](crate::Gcc) matches its references:
/// libgcc and the libm of glibc 2.43 for powerpc64le, as `qemu-ppc64le`
/// 10.2.1 runs them. It is the default mode of `DoubleDouble<Gcc>`.
///
/// The behavior is [`Env::IEEE`] with tininess before rounding and the NaN
/// rule of the PowerPC target of QEMU: the
/// [`FirstOperand`](NanPropagation::FirstOperand) propagation rule with a
/// positive default NaN, the [`AddendSecond`](FusedNanOrder::AddendSecond)
/// fused NaN order, and the
/// [`SignalsAndYieldsToNan`](InvalidProduct::SignalsAndYieldsToNan) invalid
/// product rule. The mode is not a preset of a processor: `floaty-verify`
/// confirms it against libgcc and glibc under QEMU, not on POWER hardware.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Libgcc;

impl Sealed for Libgcc {}
impl Mode for Libgcc {
    const ENV: Env = Env::IEEE.with_tininess(Tininess::BeforeRounding).with_nan(
        NanRule::new(NanPropagation::FirstOperand)
            .with_fused_order(FusedNanOrder::AddendSecond)
            .with_invalid_product(InvalidProduct::SignalsAndYieldsToNan),
    );
}

/// The mode `M` with the rounding direction `R`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rounded<M, R>(PhantomData<(M, R)>);

impl<M: Mode, R: Direction> Sealed for Rounded<M, R> {}
impl<M: Mode, R: Direction> Mode for Rounded<M, R> {
    const ENV: Env = M::ENV.with_rounding(R::ROUNDING);
}

/// The mode `M` with flush-to-zero set by `S`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FlushToZero<M, S>(PhantomData<(M, S)>);

impl<M: Mode, S: Switch> Sealed for FlushToZero<M, S> {}
impl<M: Mode, S: Switch> Mode for FlushToZero<M, S> {
    const ENV: Env = M::ENV.with_flush_to_zero(S::ON);
}

/// The mode `M` with denormals-are-zero set by `S`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DenormalsAreZero<M, S>(PhantomData<(M, S)>);

impl<M: Mode, S: Switch> Sealed for DenormalsAreZero<M, S> {}
impl<M: Mode, S: Switch> Mode for DenormalsAreZero<M, S> {
    const ENV: Env = M::ENV.with_denormals_are_zero(S::ON);
}

/// The mode `M` with a precision limit of `DIGITS` digits of the radix, as x87
/// precision control sets. A limit above the precision of the format has no
/// effect. A limit of zero digits fails to compile where a program uses the
/// behavior of the mode:
///
/// ```compile_fail,E0080
/// use floaty::env::Mode;
/// use floaty::mode::{Ieee, Precision};
///
/// let _ = <Precision<Ieee, 0> as Mode>::ENV;
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Precision<M, const DIGITS: u32>(PhantomData<M>);

impl<M: Mode, const DIGITS: u32> Sealed for Precision<M, DIGITS> {}
impl<M: Mode, const DIGITS: u32> Mode for Precision<M, DIGITS> {
    const ENV: Env = match NonZeroU32::new(DIGITS) {
        Some(limit) => M::ENV.with_precision(Some(limit)),
        None => panic!("a precision limit has at least one digit"),
    };
}

/// The mode `M` without a precision limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FullPrecision<M>(PhantomData<M>);

impl<M: Mode> Sealed for FullPrecision<M> {}
impl<M: Mode> Mode for FullPrecision<M> {
    const ENV: Env = M::ENV.with_precision(None);
}

/// The mode `M` with the NaN propagation rule `P`, from [`propagation`]. The
/// other fields of the NaN rule stay. Arm switches its propagation rule to the
/// default NaN with FPCR.DN:
///
/// ```
/// use floaty::env::{Mode, NanPropagation};
/// use floaty::mode::{Ieee, Propagation, propagation};
///
/// type DefaultNan = Propagation<Ieee, propagation::DefaultNan>;
/// assert_eq!(DefaultNan::ENV.nan.propagation, NanPropagation::DefaultNan);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Propagation<M, P>(PhantomData<(M, P)>);

impl<M: Mode, P: PropagationRule> Sealed for Propagation<M, P> {}
impl<M: Mode, P: PropagationRule> Mode for Propagation<M, P> {
    const ENV: Env = M::ENV.with_nan(M::ENV.nan.with_propagation(P::PROPAGATION));
}

/// The mode `M` with the tininess rule `T`, from [`tininess`]. Arm selects
/// tininess after rounding with FPCR.AH.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DetectTininess<M, T>(PhantomData<(M, T)>);

impl<M: Mode, T: TininessRule> Sealed for DetectTininess<M, T> {}
impl<M: Mode, T: TininessRule> Mode for DetectTininess<M, T> {
    const ENV: Env = M::ENV.with_tininess(T::TININESS);
}

/// A NaN propagation rule as a type, for [`Propagation`]. The trait is
/// sealed. The rules are in [`propagation`].
pub trait PropagationRule: Sealed + Copy + Default + 'static {
    /// The rule.
    const PROPAGATION: NanPropagation;
}

/// The NaN propagation rules as types, one for each [`NanPropagation`].
pub mod propagation {
    use super::PropagationRule;
    use crate::env::NanPropagation;
    use crate::sealed::Sealed;

    macro_rules! propagation {
        ($($name:ident),*) => {
            $(
                #[doc = concat!("[`NanPropagation::", stringify!($name), "`] as a type.")]
                #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
                pub struct $name;

                impl Sealed for $name {}
                impl PropagationRule for $name {
                    const PROPAGATION: NanPropagation = NanPropagation::$name;
                }
            )*
        };
    }

    propagation!(SignalingFirst, FirstOperand, LargerSignificand, DefaultNan);
}

/// A tininess rule as a type, for [`DetectTininess`]. The trait is sealed.
/// The rules are in [`tininess`].
pub trait TininessRule: Sealed + Copy + Default + 'static {
    /// The rule.
    const TININESS: Tininess;
}

/// The tininess rules as types, one for each [`Tininess`].
pub mod tininess {
    use super::TininessRule;
    use crate::env::Tininess;
    use crate::sealed::Sealed;

    macro_rules! tininess {
        ($($name:ident),*) => {
            $(
                #[doc = concat!("[`Tininess::", stringify!($name), "`] as a type.")]
                #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
                pub struct $name;

                impl Sealed for $name {}
                impl TininessRule for $name {
                    const TININESS: Tininess = Tininess::$name;
                }
            )*
        };
    }

    tininess!(BeforeRounding, AfterRounding);
}

/// A rounding direction as a type, for [`Rounded`]. The trait is sealed. The
/// directions are in [`direction`].
pub trait Direction: Sealed + Copy + Default + 'static {
    /// The direction.
    const ROUNDING: Rounding;
}

/// The rounding directions as types, one for each [`Rounding`].
pub mod direction {
    use super::Direction;
    use crate::env::Rounding;
    use crate::sealed::Sealed;

    macro_rules! direction {
        ($($name:ident),*) => {
            $(
                #[doc = concat!("[`Rounding::", stringify!($name), "`] as a type.")]
                #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
                pub struct $name;

                impl Sealed for $name {}
                impl Direction for $name {
                    const ROUNDING: Rounding = Rounding::$name;
                }
            )*
        };
    }

    direction!(
        TiesToEven,
        TiesToAway,
        TiesTowardZero,
        TowardPositive,
        TowardNegative,
        TowardZero,
        AwayFromZero,
        ToOdd
    );
}

/// A setting that is on or off, as a type, for [`FlushToZero`] and
/// [`DenormalsAreZero`]. The trait is sealed. The settings are in [`switch`].
pub trait Switch: Sealed + Copy + Default + 'static {
    /// `true` when the setting is on.
    const ON: bool;
}

/// The two settings of a [`Switch`].
pub mod switch {
    use super::Switch;
    use crate::sealed::Sealed;

    /// The setting is on.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct On;

    impl Sealed for On {}
    impl Switch for On {
        const ON: bool = true;
    }

    /// The setting is off.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct Off;

    impl Sealed for Off {}
    impl Switch for Off {
        const ON: bool = false;
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;

    use super::direction::{TowardNegative, TowardZero};
    use super::switch::{Off, On};
    use super::{
        DenormalsAreZero, DetectTininess, FlushToZero, FullPrecision, Ieee, Precision, Propagation,
        Rounded, X86Sse, X87, propagation, tininess,
    };
    use crate::env::{Env, Mode, NanPropagation, Rounding, Tininess};

    #[test]
    fn each_combinator_changes_one_field() {
        assert_eq!(
            <Rounded<X86Sse, TowardZero>>::ENV,
            Env::X86_SSE.with_rounding(Rounding::TowardZero)
        );
        assert_eq!(
            <FlushToZero<Ieee, On>>::ENV,
            Env::IEEE.with_flush_to_zero(true)
        );
        assert_eq!(
            <DenormalsAreZero<FlushToZero<X86Sse, On>, On>>::ENV,
            Env::X86_SSE
                .with_flush_to_zero(true)
                .with_denormals_are_zero(true)
        );
        assert_eq!(
            <Precision<X87, 24>>::ENV,
            Env::X87.with_precision(NonZeroU32::new(24))
        );
        assert_eq!(<FullPrecision<X87>>::ENV, Env::X87.with_precision(None));
        assert_eq!(
            <Propagation<X86Sse, propagation::LargerSignificand>>::ENV,
            Env::X86_SSE.with_nan(
                Env::X86_SSE
                    .nan
                    .with_propagation(NanPropagation::LargerSignificand)
            )
        );
        assert_eq!(
            <DetectTininess<Ieee, tininess::BeforeRounding>>::ENV,
            Env::IEEE.with_tininess(Tininess::BeforeRounding)
        );
    }

    #[test]
    fn each_propagation_type_names_its_rule() {
        let rules = [
            <Propagation<Ieee, propagation::SignalingFirst>>::ENV.nan,
            <Propagation<Ieee, propagation::FirstOperand>>::ENV.nan,
            <Propagation<Ieee, propagation::LargerSignificand>>::ENV.nan,
            <Propagation<Ieee, propagation::DefaultNan>>::ENV.nan,
        ];
        let expected = [
            NanPropagation::SignalingFirst,
            NanPropagation::FirstOperand,
            NanPropagation::LargerSignificand,
            NanPropagation::DefaultNan,
        ];
        for (rule, propagation) in rules.into_iter().zip(expected) {
            assert_eq!(rule, Env::IEEE.nan.with_propagation(propagation));
        }
        assert_eq!(
            <DetectTininess<Ieee, tininess::AfterRounding>>::ENV,
            Env::IEEE
        );
    }

    #[test]
    fn a_later_combinator_replaces_an_earlier_field() {
        type Twice = Rounded<Rounded<Ieee, TowardZero>, TowardNegative>;
        assert_eq!(Twice::ENV.rounding, Rounding::TowardNegative);
        assert_eq!(<FlushToZero<FlushToZero<Ieee, On>, Off>>::ENV, Env::IEEE);
    }
}
