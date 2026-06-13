//! # VaeaNTT — High-Performance Number Theoretic Transforms
//!
//! Production-quality NTT engine for post-quantum cryptography.
//! ARM NEON native with scalar fallback.
//!
//! ## Pipelines
//!
//! - [`ntt32`] — 28-bit primes (< 2²⁸), ultra-fast on ARM NEON
//! - [`ntt64`] — 60-62 bit primes, compatible with SEAL/OpenFHE/FHE
//! - [`poly`] — Polynomials over Z_q\[X\]/(X^N+1)
//! - [`rns`] — Multi-prime CRT (Residue Number System)
//!
//! ## `no_std` Support
//!
//! This crate is `no_std` compatible (requires `alloc`).
//! Enable the `std` feature (on by default) for `std::error::Error` impl.

#![no_std]
#![warn(missing_docs)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

/// Errors returned by NTT context construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NttError {
    /// N must be a power of 2 >= 2.
    InvalidSize(usize),
    /// q must be prime.
    NotPrime(u64),
    /// q must satisfy q ≡ 1 (mod 2N) for NTT.
    NotNttFriendly {
        /// The modulus that failed the NTT-friendly check.
        q: u64,
        /// The polynomial size N.
        n: usize,
    },
    /// q must be < 2^28 for the 32-bit pipeline.
    PrimeTooLarge(u64),
}

impl core::fmt::Display for NttError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NttError::InvalidSize(n) => write!(f, "N={n} must be a power of 2 >= 2"),
            NttError::NotPrime(q) => write!(f, "q={q} is not prime"),
            NttError::NotNttFriendly { q, n } => {
                write!(f, "q={q} does not satisfy q ≡ 1 (mod {})", 2 * n)
            }
            NttError::PrimeTooLarge(q) => write!(f, "q={q} must be < 2^28"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NttError {}

pub mod ntt32;
pub mod ntt64;
pub mod poly;
pub mod rns;

#[cfg(feature = "ffi")]
pub mod ffi;
