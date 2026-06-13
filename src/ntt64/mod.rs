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
pub mod prime;
pub mod context;

// Re-exports for convenience
pub use arith::{
    Ntt64Arith,
    mod_add, mod_sub,
    mod_mul_barrett, mod_mul_mont,
    to_montgomery, from_montgomery,
    mod_pow, mod_inv,
    PRIME_60_1, PRIME_SEAL, PRIME_62_1, PRIME_60_2, PRIME_60_3,
};

pub use prime::{
    is_prime,
    generate_primes_60,
    find_primitive_root,
};

pub use context::Ntt64Context;
