use super::{Host, Kind, Operation, Unit, available, convertible};
use crate::env::Env;
use crate::float::Float;
use crate::format::internal::{LimbConversion, MinMax};
use crate::format::{Binary, Standard, X87};
use crate::limbs::Limbs;

/// The host kinds, as conversion destinations.
const HOSTS: [Host; 6] = [
    Host::Half,
    Host::BFloat,
    Host::Single,
    Host::Double,
    Host::Extended,
    Host::Quad,
];

/// Asserts that each host path of the format `S` that `available`,
/// `convertible`, `packed::available`, or `packed::convertible` claims
/// gives a result. The operands 3, 2, and 1 are exact in every format,
/// and no path declines them by value.
fn claimed_paths_give_results<S: Standard<W>, const W: usize>() {
    let env = Env::IEEE;
    let host = S::HOST;
    // One read for every scalar path, as the lanes of one `Lanes`
    // operation take it.
    let unit = Unit::read(host);
    let value = |integer: i64| Float::<S, W>::from_int(integer);
    let (three, two, one) = (value(3), value(2), value(1));
    let (left_bits, right_bits, addend_bits) = (three.to_bits(), two.to_bits(), one.to_bits());
    let add = Operation::Add;
    let minimum = MinMax::Minimum;
    let scalar = [
        (
            Kind::Arithmetic,
            super::binary::<S, W>(left_bits, right_bits, add, &env, unit).is_some(),
        ),
        (
            Kind::SquareRoot,
            super::sqrt::<S, W>(left_bits, &env, unit).is_some(),
        ),
        (
            Kind::FusedMultiplyAdd,
            super::mul_add::<S, W>(left_bits, right_bits, addend_bits, &env, unit).is_some(),
        ),
        (
            Kind::RoundToIntegral,
            super::round_to_integral::<S, W>(left_bits, &env, unit).is_some(),
        ),
        (
            Kind::ToInt,
            super::to_int::<S, W>(left_bits, &env, unit).is_some(),
        ),
        (
            Kind::FromInt,
            super::from_int::<S, W>(3, &env, unit).is_some(),
        ),
        (
            Kind::Comparison,
            super::compare::<S, W>(left_bits, right_bits, &env, unit).is_some()
                && super::min_max::<S, W>(left_bits, right_bits, minimum, &env, unit).is_some(),
        ),
        (
            Kind::Remainder,
            super::remainder::<S, W>(left_bits, right_bits, &env, unit).is_some(),
        ),
    ];
    for (kind, taken) in scalar {
        assert!(
            !available(host, kind) || taken,
            "{host:?} {kind:?}: the claimed path gives no result"
        );
    }
    let (left, right, addend) = ([three; 8], [two; 8], [one; 8]);
    let packed = [
        (
            Kind::Arithmetic,
            super::packed::binary(&left, &right, Operation::Add, &env).is_some(),
        ),
        (Kind::SquareRoot, super::packed::sqrt(&left, &env).is_some()),
        (
            Kind::FusedMultiplyAdd,
            super::packed::mul_add(&left, &right, &addend, &env).is_some(),
        ),
        (
            Kind::RoundToIntegral,
            super::packed::round_to_integral(&left, &env).is_some(),
        ),
        (
            Kind::ToInt,
            super::packed::to_int_i32(&left, &env).is_some(),
        ),
        (
            Kind::Comparison,
            super::packed::compare(&left, &right, &env).is_some()
                && super::packed::min_max(&left, &right, MinMax::Minimum, &env).is_some(),
        ),
    ];
    for (kind, taken) in packed {
        assert!(
            !super::packed::available(host, kind) || taken,
            "{host:?} {kind:?}: the claimed packed path gives no result"
        );
    }
    let limbs = left_bits.to_limbs();
    let encoding = [limbs.limb(0), limbs.limb(1)];
    for to in HOSTS {
        assert!(
            !convertible(host, to)
                || super::convert(host, to, encoding, &env, Unit::Read).is_some(),
            "{host:?} to {to:?}: the claimed conversion gives no result"
        );
        assert!(
            !super::packed::convertible(host, to)
                || super::packed::convert(&left, to, &env).is_some(),
            "{host:?} to {to:?}: the claimed packed conversion gives no result"
        );
    }
}

#[test]
fn every_claimed_host_path_gives_a_result() {
    claimed_paths_give_results::<Binary<5>, 16>();
    claimed_paths_give_results::<Binary<8>, 16>();
    claimed_paths_give_results::<Binary<8>, 32>();
    claimed_paths_give_results::<Binary<11>, 64>();
    claimed_paths_give_results::<Binary<15, X87>, 80>();
    claimed_paths_give_results::<Binary<15>, 128>();
}
