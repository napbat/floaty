//! The densely packed decimal declet: three decimal digits in 10 bits, as
//! IEEE 754-2019 Tables 3.3 and 3.4 define it.
//!
//! A declet has the bits `p q r s t u v w x y`, with `p` the most significant.
//! Of the 1024 declets, 1000 are canonical. The other 24 decode to a digit
//! value that a canonical declet also encodes.

/// Returns the declet of a value from 0 to 999, as IEEE 754-2019 Table 3.4
/// encodes it.
const fn encode(value: u16) -> u16 {
    let (high, middle, low) = (value / 100, value / 10 % 10, value % 10);
    // A digit of 8 or 9 is large; its low bit is the only one that varies.
    // `u16::from` is not callable in a constant; a `bool` cast is lossless.
    let large = (((high >= 8) as u16) << 2) | (((middle >= 8) as u16) << 1) | (low >= 8) as u16;
    let (d, h, m) = (high & 1, middle & 1, low & 1);
    let (bcd, fgh, jkm) = (high & 7, middle & 7, low & 7);
    let (fg, jk) = ((middle >> 1) & 3, (low >> 1) & 3);
    let (pqr, stu, wxy) = match large {
        0b000 => (bcd, fgh, jkm),
        0b001 => (bcd, fgh, 0b1000 | m),
        0b010 => (bcd, (jk << 1) | h, 0b1010 | m),
        0b011 => (bcd, 0b100 | h, 0b1110 | m),
        0b100 => ((jk << 1) | d, fgh, 0b1100 | m),
        0b101 => ((fg << 1) | d, 0b010 | h, 0b1110 | m),
        0b110 => ((jk << 1) | d, h, 0b1110 | m),
        _ => (d, 0b110 | h, 0b1110 | m),
    };
    // `wxy` above also holds `v`, the indicator bit, as its bit 3.
    (pqr << 7) | (stu << 4) | wxy
}

/// Returns the value from 0 to 999 of a declet, as IEEE 754-2019 Table 3.3
/// decodes it.
const fn decode(declet: u16) -> u16 {
    let (pqr, stu, wxy) = ((declet >> 7) & 7, (declet >> 4) & 7, declet & 7);
    let (pq, r) = (pqr >> 1, pqr & 1);
    let (st, u) = (stu >> 1, stu & 1);
    let (wx, y) = (wxy >> 1, wxy & 1);
    let (high, middle, low) = if (declet >> 3) & 1 == 0 {
        (pqr, stu, wxy)
    } else {
        match (wx, st) {
            (0b00, _) => (pqr, stu, 8 + y),
            (0b01, _) => (pqr, 8 + u, (st << 1) | y),
            (0b10, _) => (8 + r, stu, (pq << 1) | y),
            (_, 0b00) => (8 + r, 8 + u, (pq << 1) | y),
            (_, 0b01) => (8 + r, (pq << 1) | u, 8 + y),
            (_, 0b10) => (pqr, 8 + u, 8 + y),
            _ => (8 + r, 8 + u, 8 + y),
        }
    };
    high * 100 + middle * 10 + low
}

/// The declet of each value from 0 to 999.
pub const DECLETS: [u16; 1000] = {
    let mut table = [0; 1000];
    let mut value: u16 = 0;
    // `usize::from` is not callable in a constant; the cast widens.
    while value < 1000 {
        table[value as usize] = encode(value);
        value += 1;
    }
    table
};

/// The value of each of the 1024 declets.
pub const VALUES: [u16; 1024] = {
    let mut table = [0; 1024];
    let mut declet: u16 = 0;
    // `usize::from` is not callable in a constant; the cast widens.
    while declet < 1024 {
        table[declet as usize] = decode(declet);
        declet += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::{DECLETS, VALUES};

    #[test]
    fn every_value_round_trips_and_24_declets_are_not_canonical() {
        for (value, &declet) in DECLETS.iter().enumerate() {
            assert_eq!(usize::from(VALUES[usize::from(declet)]), value);
        }
        let canonical = VALUES
            .iter()
            .enumerate()
            .filter(|&(declet, &value)| usize::from(DECLETS[usize::from(value)]) == declet)
            .count();
        assert_eq!(canonical, 1000);
        // Spot values: 9 sets the indicator bit, 999 is 0x0FF, and the
        // non-canonical 0x3FF reads as 999.
        assert_eq!(DECLETS[0], 0);
        assert_eq!(DECLETS[9], 0b00_0000_1001);
        assert_eq!(DECLETS[999], 0b00_1111_1111);
        assert_eq!(VALUES[0b11_1111_1111], 999);
    }
}
