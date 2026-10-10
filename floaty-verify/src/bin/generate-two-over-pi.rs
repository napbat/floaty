//! Writes the bits of 2/pi that the argument reduction of `sin`, `cos`, and
//! `tan` in floaty reads: `floor(2/pi * 2^N)` with `N = 2^22 + 8192`, as
//! `N / 8` bytes, the most significant first.
//!
//! Run it from the workspace root:
//!
//! ```text
//! cargo run -p floaty-verify --release --bin generate-two-over-pi -- floaty/src/elementary/two_over_pi.bin
//! ```
//!
//! MPFR bounds 2/pi from below and from above at `N + 64` bits. Both bounds
//! give the same `N` bits, so the bits are exact.

/// The number of bits of 2/pi after the point.
#[cfg(target_arch = "x86_64")]
const BITS: u32 = (1 << 22) + 8192;

#[cfg(target_arch = "x86_64")]
fn main() {
    use rug::float::{Constant, Round};
    use rug::integer::Order;
    use rug::{Float, Integer};

    let path = std::env::args()
        .nth(1)
        .expect("the first argument names the output file");
    let precision = BITS + 64;
    let bound = |round: Round, other: Round| {
        let pi = Float::with_val_round(precision, Constant::Pi, other).0;
        let quotient = Float::with_val_round(precision, 2 / &pi, round).0 << BITS;
        quotient
            .to_integer_round(Round::Down)
            .expect("2/pi is finite")
            .0
    };
    let low: Integer = bound(Round::Down, Round::Up);
    let high: Integer = bound(Round::Up, Round::Down);
    assert_eq!(low, high, "the bounds of 2/pi give the same bits");
    let mut bytes = low.to_digits::<u8>(Order::MsfBe);
    let length = usize::try_from(BITS / 8).expect("the length fits a usize");
    assert_eq!(bytes.len(), length, "2/pi lies in [1/2, 1)");
    bytes.truncate(length);
    std::fs::write(&path, bytes).expect("the output file is writable");
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("the generator needs MPFR, which floaty-verify builds only for x86-64");
    std::process::exit(1);
}
