// Copyright (C) 2024-2026 Vaea SAS
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of VaeaNTT.
//
// VaeaNTT is free software: you can redistribute it and/or modify it under
// the terms of the GNU Affero General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// VaeaNTT is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public
// License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with VaeaNTT. If not, see <https://www.gnu.org/licenses/>.


//! # 28-bit NTT-Friendly Prime Generation
//!
//! Generates primes of the form `k · 2N + 1` that are < 2^28,
//! and finds primitive 2N-th roots of unity in Z_q*.
//!
//! These primes satisfy `q ≡ 1 (mod 2N)`, the necessary condition for
//! the existence of a 2N-th root of unity — required by negacyclic NTT.

use super::arith::{mod_inv_32, mod_pow_32};
use alloc::vec;
use alloc::vec::Vec;

// ===========================================================================
// Primality testing
// ===========================================================================

/// Trial-division primality test, sufficient for numbers < 2^28.
///
/// For n < 2^28 ≈ 268M, √n < 16384, so the loop is very fast.
pub fn is_prime_32(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    let mut d = 5u32;
    while d.saturating_mul(d) <= n {
        if n.is_multiple_of(d) || n.is_multiple_of(d + 2) {
            return false;
        }
        d += 6;
    }
    true
}

// ===========================================================================
// NTT-friendly prime generation
// ===========================================================================

/// Generates NTT-friendly 28-bit primes of the form `k · 2N + 1`.
///
/// These primes satisfy `q ≡ 1 (mod 2N)`, the necessary condition for
/// the existence of a 2N-th root of unity in Z_q*.
///
/// Primes are returned in descending order (largest first) to maximize
/// the coefficient space for CKKS-like schemes.
///
/// # Arguments
/// - `poly_degree` — polynomial size N (must be a power of 2)
/// - `count` — number of primes to generate
///
/// # Panics
/// If fewer than `count` valid primes exist below 2^28.
pub fn generate_primes_28(poly_degree: usize, count: usize) -> Vec<u32> {
    let two_n = (2 * poly_degree) as u64;
    let mut primes = Vec::with_capacity(count);

    // k_max = (2^28 - 1) / (2N)
    let k_max = ((1u64 << 28) - 1) / two_n;

    // Start from largest k for primes close to 2^28
    for k in (1..=k_max).rev() {
        let candidate = k * two_n + 1;
        debug_assert!(candidate < (1u64 << 28));
        if is_prime_32(candidate as u32) {
            primes.push(candidate as u32);
            if primes.len() == count {
                break;
            }
        }
    }

    assert!(
        primes.len() == count,
        "Cannot find {count} 28-bit primes for N={poly_degree}, found only {}",
        primes.len()
    );

    primes
}

// ===========================================================================
// Primitive root finding
// ===========================================================================

/// Trial-division factorization (32-bit).
///
/// Returns the distinct prime factors of `n`.
pub(crate) fn small_factor_32(mut n: u32) -> Vec<u32> {
    let mut factors = Vec::new();

    if n.is_multiple_of(2) {
        factors.push(2);
        while n.is_multiple_of(2) {
            n /= 2;
        }
    }

    let mut d = 3u32;
    while d.saturating_mul(d) <= n {
        if n.is_multiple_of(d) {
            factors.push(d);
            while n.is_multiple_of(d) {
                n /= d;
            }
        }
        d += 2;
    }

    if n > 1 {
        factors.push(n);
    }

    factors
}

/// Finds a generator of Z_q* by trial (32-bit).
///
/// g is a generator iff `g^((q-1)/p) ≠ 1` for every prime factor p of q-1.
fn find_generator_32(q: u32, prime_factors: &[u32]) -> u32 {
    let q_minus_1 = q - 1;

    for g in 2..q {
        let mut is_generator = true;
        for &p in prime_factors {
            let exp = q_minus_1 / p;
            if mod_pow_32(g, exp, q) == 1 {
                is_generator = false;
                break;
            }
        }
        if is_generator {
            return g;
        }
    }

    panic!("No generator found for q={q} — this should never happen");
}

/// Finds a primitive 2N-th root of unity modulo q (32-bit).
///
/// Returns ψ such that:
/// - `ψ^(2N) ≡ 1 (mod q)`
/// - `ψ^N ≡ -1 (mod q)`, i.e. `ψ^N ≡ q - 1`
///
/// # Algorithm
/// 1. Factorize q - 1
/// 2. Find a generator g of Z_q* by trial
/// 3. Compute ψ = g^((q-1)/(2N))
///
/// # Panics
/// If `q` does not satisfy `q ≡ 1 (mod 2N)`.
pub fn find_primitive_root(n: usize, q: u32) -> u32 {
    let two_n = 2 * n as u32;
    assert!(
        (q - 1).is_multiple_of(two_n),
        "find_primitive_root: q={q} does not satisfy q ≡ 1 (mod 2N={})",
        two_n
    );

    let q_minus_1 = q - 1;
    let prime_factors = small_factor_32(q_minus_1);

    // Find a generator g of Z_q*
    let g = find_generator_32(q, &prime_factors);

    // ψ = g^((q-1)/(2N)) is a primitive 2N-th root
    let exp = q_minus_1 / two_n;
    let psi = mod_pow_32(g, exp, q);

    // Safety checks
    debug_assert_eq!(mod_pow_32(psi, two_n, q), 1, "ψ^(2N) ≠ 1: not a 2N-th root");
    debug_assert_eq!(
        mod_pow_32(psi, n as u32, q),
        q - 1,
        "ψ^N ≠ -1: not a PRIMITIVE 2N-th root"
    );

    psi
}

// ===========================================================================
// Bit-reversal utility
// ===========================================================================

/// Reverses the `bits` least-significant bits of `x`.
#[inline]
pub(crate) fn bit_reverse(x: u32, bits: u32) -> u32 {
    x.reverse_bits() >> (32 - bits)
}

// ===========================================================================
// Internal context for root power precomputation
// ===========================================================================

/// Internal NTT context for a single 28-bit prime.
///
/// Stores precomputed twiddle factors (root powers) in Longa-Naehrig ordering
/// for sequential access during the butterfly operations.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct NttRootTable {
    /// Polynomial size (power of 2)
    pub n: usize,
    /// log2(n)
    pub log_n: u32,
    /// Prime < 2^28
    pub q: u32,
    /// Forward root powers in Longa-Naehrig ordering:
    /// `root_powers[m + i]` = twiddle factor for group i at layer m.
    pub root_powers: Vec<u32>,
    /// Inverse root powers for INTT
    pub inv_root_powers: Vec<u32>,
    /// N^{-1} mod q — normalization factor for INTT
    pub n_inv: u32,
}

impl NttRootTable {
    /// Creates a new root table for a 28-bit prime.
    ///
    /// # Panics
    /// - If `n` is not a power of 2 ≥ 2
    /// - If `q` is not < 2^28
    /// - If `(q - 1)` is not divisible by `2N`
    pub fn new(n: usize, q: u32) -> Self {
        assert!(n >= 2 && n.is_power_of_two(), "N must be a power of 2 >= 2");
        assert!(q < (1u32 << 28), "q={q} must be < 2^28");
        let log_n = n.trailing_zeros();

        assert!(
            (q - 1).is_multiple_of(2 * n as u32),
            "q={q} does not satisfy q ≡ 1 (mod 2N={})",
            2 * n
        );

        // Find the primitive 2N-th root
        let psi = find_primitive_root(n, q);
        let psi_inv = mod_inv_32(psi, q);
        let n_inv = mod_inv_32(n as u32, q);

        // Precompute twiddle factors (Longa-Naehrig ordering)
        // root_powers[i] = psi^{bit_reverse(i, log_n)} for i = 0..N
        let mut root_powers = vec![0u32; n];
        let mut inv_root_powers = vec![0u32; n];

        for i in 0..n {
            let exp = bit_reverse(i as u32, log_n);
            root_powers[i] = mod_pow_32(psi, exp, q);
            inv_root_powers[i] = mod_pow_32(psi_inv, exp, q);
        }

        Self {
            n,
            log_n,
            q,
            root_powers,
            inv_root_powers,
            n_inv,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime_32() {
        assert!(!is_prime_32(0));
        assert!(!is_prime_32(1));
        assert!(is_prime_32(2));
        assert!(is_prime_32(3));
        assert!(!is_prime_32(4));
        assert!(is_prime_32(5));
        assert!(is_prime_32(7));
        assert!(!is_prime_32(9));
        assert!(is_prime_32(13));
        assert!(is_prime_32(268_435_399));
    }

    #[test]
    fn test_generate_primes_28() {
        for &n in &[16, 64, 1024, 4096] {
            let primes = generate_primes_28(n, 5);
            assert_eq!(primes.len(), 5);

            for &p in &primes {
                assert!(p < (1u32 << 28), "Prime {p} >= 2^28");
                assert!(is_prime_32(p), "{p} is not prime");
                assert_eq!(
                    (p - 1) % (2 * n as u32),
                    0,
                    "Prime {p} is not NTT-friendly for N={n}"
                );
            }

            // Check uniqueness
            for i in 0..primes.len() {
                for j in (i + 1)..primes.len() {
                    assert_ne!(primes[i], primes[j], "Duplicate primes");
                }
            }
        }
    }

    #[test]
    fn test_find_primitive_root() {
        for &n in &[16, 64, 1024] {
            let q = generate_primes_28(n, 1)[0];
            let psi = find_primitive_root(n, q);

            assert_eq!(
                mod_pow_32(psi, 2 * n as u32, q),
                1,
                "ψ^(2N) ≠ 1 for N={n}, q={q}"
            );
            assert_eq!(
                mod_pow_32(psi, n as u32, q),
                q - 1,
                "ψ^N ≠ -1 for N={n}, q={q}"
            );
        }
    }
}
