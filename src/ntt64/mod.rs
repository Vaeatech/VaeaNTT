//! # 64-bit NTT Pipeline
//!
//! High-performance NTT for 60–62 bit NTT-friendly primes, compatible with
//! SEAL, OpenFHE, and general FHE libraries.
//!
//! ## Modules
//!
//! - [`arith`] — Modular arithmetic (Barrett, Montgomery, add/sub)
//! - [`prime`] — Prime generation and primality testing
//! - [`context`] — NTT context with precomputed twiddle tables

pub mod arith;
pub mod context;
pub mod prime;

// Re-exports for convenience
pub use arith::{
    from_montgomery, mod_add, mod_inv, mod_mul_barrett, mod_mul_mont, mod_pow, mod_sub,
    to_montgomery, Ntt64Arith, PRIME_60_1, PRIME_60_2, PRIME_60_3, PRIME_62_1, PRIME_SEAL,
};

pub use prime::{find_primitive_root, generate_primes_60, is_prime};

pub use context::Ntt64Context;
