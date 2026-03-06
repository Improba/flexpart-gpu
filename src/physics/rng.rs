//! Philox counter-based RNG CPU reference implementation.
//!
//! This is a deterministic implementation of Philox4x32-10 intended to be
//! paired with WGSL kernels for reproducible GPU/CPU parity tests.

/// 128-bit Philox counter as 4 x 32-bit lanes (lane 0 is least significant).
pub type PhiloxCounter = [u32; 4];
/// 64-bit Philox key as 2 x 32-bit lanes.
pub type PhiloxKey = [u32; 2];

/// Number of rounds for the standard Philox4x32 generator.
pub const PHILOX_ROUNDS: u32 = 10;

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;
const U32_TO_UNIT_SCALE_24BIT: f32 = 1.0 / 16_777_216.0;

#[allow(clippy::cast_possible_truncation)]
#[inline]
fn mul_hi_lo_u32(a: u32, b: u32) -> (u32, u32) {
    let product = u64::from(a) * u64::from(b);
    ((product >> 32) as u32, product as u32)
}

#[inline]
fn philox_round(counter: PhiloxCounter, key: PhiloxKey) -> PhiloxCounter {
    let (hi0, lo0) = mul_hi_lo_u32(PHILOX_M0, counter[0]);
    let (hi1, lo1) = mul_hi_lo_u32(PHILOX_M1, counter[2]);
    [
        hi1 ^ counter[1] ^ key[0],
        lo1,
        hi0 ^ counter[3] ^ key[1],
        lo0,
    ]
}

#[inline]
fn raise_key(key: PhiloxKey) -> PhiloxKey {
    [
        key[0].wrapping_add(PHILOX_W0),
        key[1].wrapping_add(PHILOX_W1),
    ]
}

/// Stateless Philox4x32 with configurable round count.
#[must_use]
pub fn philox4x32_with_rounds(
    mut counter: PhiloxCounter,
    mut key: PhiloxKey,
    rounds: u32,
) -> PhiloxCounter {
    for _ in 0..rounds {
        counter = philox_round(counter, key);
        key = raise_key(key);
    }
    counter
}

/// Stateless Philox4x32-10 block generation.
#[must_use]
pub fn philox4x32(counter: PhiloxCounter, key: PhiloxKey) -> PhiloxCounter {
    philox4x32_with_rounds(counter, key, PHILOX_ROUNDS)
}

/// Add `offset` blocks to a Philox counter (128-bit wrapping arithmetic).
#[must_use]
pub fn philox_counter_add(counter: PhiloxCounter, offset: u64) -> PhiloxCounter {
    let [mut c0, mut c1, mut c2, mut c3] = counter;
    let low = offset as u32;
    let high = (offset >> 32) as u32;

    let (next0, carry0) = c0.overflowing_add(low);
    c0 = next0;
    let (next1, carry1a) = c1.overflowing_add(high);
    let (next1b, carry1b) = next1.overflowing_add(u32::from(carry0));
    c1 = next1b;

    let carry1 = carry1a || carry1b;
    let (next2, carry2) = c2.overflowing_add(u32::from(carry1));
    c2 = next2;
    let (next3, _) = c3.overflowing_add(u32::from(carry2));
    c3 = next3;

    [c0, c1, c2, c3]
}

/// Convert one `u32` to a reproducible uniform random value in `[0, 1)`.
///
/// Uses the high 24 bits so conversion is exactly representable in `f32`.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn u32_to_uniform01(value: u32) -> f32 {
    ((value >> 8) as f32) * U32_TO_UNIT_SCALE_24BIT
}

/// Convert a full Philox block to 4 uniform random values in `[0, 1)`.
#[must_use]
pub fn philox4x32_uniforms(counter: PhiloxCounter, key: PhiloxKey) -> [f32; 4] {
    let block = philox4x32(counter, key);
    [
        u32_to_uniform01(block[0]),
        u32_to_uniform01(block[1]),
        u32_to_uniform01(block[2]),
        u32_to_uniform01(block[3]),
    ]
}

/// Stateful deterministic Philox RNG stream.
#[derive(Debug, Clone)]
pub struct PhiloxRng {
    key: PhiloxKey,
    counter: PhiloxCounter,
    cached_block: PhiloxCounter,
    cached_index: usize,
}

impl PhiloxRng {
    /// Create a stream from an explicit key and counter.
    #[must_use]
    pub const fn new(key: PhiloxKey, counter: PhiloxCounter) -> Self {
        Self {
            key,
            counter,
            cached_block: [0; 4],
            cached_index: 4,
        }
    }

    fn refill(&mut self) {
        self.cached_block = philox4x32(self.counter, self.key);
        self.cached_index = 0;
        self.counter = philox_counter_add(self.counter, 1);
    }

    /// Next deterministic `u32` from the stream.
    #[must_use]
    pub fn next_u32(&mut self) -> u32 {
        if self.cached_index >= self.cached_block.len() {
            self.refill();
        }

        let value = self.cached_block[self.cached_index];
        self.cached_index += 1;
        value
    }

    /// Next deterministic uniform random in `[0, 1)`.
    #[must_use]
    pub fn next_uniform(&mut self) -> f32 {
        u32_to_uniform01(self.next_u32())
    }

    /// Next 4 uniforms from the stream.
    #[must_use]
    pub fn next_uniform4(&mut self) -> [f32; 4] {
        [
            self.next_uniform(),
            self.next_uniform(),
            self.next_uniform(),
            self.next_uniform(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_zero_vector_matches_reference_block() {
        let counter = [0, 0, 0, 0];
        let key = [0, 0];
        let block = philox4x32(counter, key);
        assert_eq!(block, [0x6627_E8D5, 0xE169_C58D, 0xBC57_AC4C, 0x9B00_DBD8]);
    }

    #[test]
    fn philox_stream_is_deterministic_for_fixed_seed() {
        let key = [0xDECA_FBAD, 0x1234_5678];
        let counter = [0, 1, 2, 3];
        let mut rng_a = PhiloxRng::new(key, counter);
        let mut rng_b = PhiloxRng::new(key, counter);

        for _ in 0..64 {
            assert_eq!(rng_a.next_u32(), rng_b.next_u32());
        }
    }

    #[test]
    fn philox_uniform_distribution_mean_is_near_half() {
        let mut rng = PhiloxRng::new([0xA409_3822, 0x299F_31D0], [0x243F_6A88, 0x85A3_08D3, 0, 0]);
        let sample_count = 200_000_u32;
        let mut sum = 0.0_f64;

        for _ in 0..sample_count {
            let value = rng.next_uniform();
            assert!((0.0..1.0).contains(&value));
            sum += f64::from(value);
        }

        let mean = sum / f64::from(sample_count);
        assert!(
            (mean - 0.5).abs() < 0.01,
            "mean should be close to 0.5, got {mean}"
        );
    }

    #[test]
    fn philox_counter_add_handles_carry() {
        let counter = [u32::MAX, u32::MAX, 7, 9];
        let next = philox_counter_add(counter, 1);
        assert_eq!(next, [0, 0, 8, 9]);
    }
}
