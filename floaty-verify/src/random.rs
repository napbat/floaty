//! Seeded random numbers for reproducible test inputs.

/// A seeded `SplitMix64` generator.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Makes a generator from a seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Returns the next 128 random bits.
    pub fn next_u128(&mut self) -> u128 {
        (u128::from(self.next_u64()) << 64) | u128::from(self.next_u64())
    }

    /// Returns a random value below `bound`, from the next 64 random bits.
    ///
    /// # Panics
    ///
    /// Panics when `bound` is zero.
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }

    /// Returns a random value below `bound`, from the next 128 random bits.
    ///
    /// # Panics
    ///
    /// Panics when `bound` is zero.
    pub fn below_u128(&mut self, bound: u128) -> u128 {
        self.next_u128() % bound
    }

    /// Returns `true` or `false` with even odds, from the lowest of the next
    /// 64 random bits.
    pub fn coin_flip(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}
