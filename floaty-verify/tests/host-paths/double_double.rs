//! The operators and `sqrt` of the double-double algorithms.

use floaty::{DoubleDouble, F64, Gcc, Qd};
use floaty_verify::random::SplitMix64;

use super::operands::double_double_pairs;

/// Checks that the operators and `sqrt` of a double-double algorithm give the
/// results of its `_with` methods under the default mode.
fn double_double_operators_match<Alg: floaty::Algorithm>(pairs: &[(F64, F64)]) {
    let value = |&(hi, lo): &(F64, F64)| DoubleDouble::<Alg>::from_parts(hi, lo);
    let bits = |value: DoubleDouble<Alg>| (value.hi().to_bits(), value.lo().to_bits());
    for (first, second) in pairs.iter().zip(pairs.iter().rev()) {
        let (x, y) = (value(first), value(second));
        let env = floaty::Env::IEEE;
        let context = format!("{x:?} {y:?}");
        assert_eq!(bits(x + y), bits(x.add_with(y, env).0), "{context} +");
        assert_eq!(bits(x - y), bits(x.sub_with(y, env).0), "{context} -");
        assert_eq!(bits(x * y), bits(x.mul_with(y, env).0), "{context} *");
        assert_eq!(bits(x / y), bits(x.div_with(y, env).0), "{context} /");
        assert_eq!(bits(x.sqrt()), bits(x.sqrt_with(env).0), "{context} sqrt");
    }
}

#[test]
fn double_double_operators_give_the_default_mode_results() {
    let mut random = SplitMix64::new(0xDD00);
    let pairs = double_double_pairs(&mut random);
    double_double_operators_match::<Qd>(&pairs);
    double_double_operators_match::<Gcc>(&pairs);
}
