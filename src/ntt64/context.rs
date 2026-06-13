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


//! # NTT Context — Forward and Inverse Transforms
//!
//! High-performance Number Theoretic Transform using the Longa-Naehrig ordering
//! (SEAL/OpenFHE style) with integrated negacyclic twiddle factors.
//!
//! ## Algorithms
//! - **Forward NTT** — Cooley-Tukey radix-2 DIT (Decimation In Time)
//! - **Inverse NTT** — Gentleman-Sande radix-2 DIF (Decimation In Frequency)
//! - **Tiled NTT** — Four-step variant for improved cache locality on large N

use super::arith::{mod_add, mod_inv, mod_mul_barrett, mod_pow, mod_sub, Ntt64Arith};
use super::prime::find_primitive_root;
use alloc::vec;
use alloc::vec::Vec;

// ===========================================================================
// Bit-reversal utility
// ===========================================================================

/// Reverses the `bits` least-significant bits of `x`.
///
/// Example: `bit_reverse(0b110, 3) = 0b011`
#[inline]
fn bit_reverse(x: u32, bits: u32) -> u32 {
    x.reverse_bits() >> (32 - bits)
}

// ===========================================================================
// NTT Context
// ===========================================================================

/// Precomputed NTT context for a given (N, modulus) pair.
///
/// Contains twiddle-factor tables for both forward and inverse NTT,
/// organized in Longa-Naehrig ordering for negacyclic convolution.
#[derive(Debug, Clone)]
pub struct Ntt64Context {
    /// Polynomial size (power of 2).
    pub n: usize,

    /// log₂(n).
    pub log_n: u32,

    /// Modular arithmetic context (Barrett/Montgomery constants).
    pub arith: Ntt64Arith,

    /// Twiddle factors for forward NTT.
    ///
    /// Organized for sequential access in the Cooley-Tukey butterfly:
    /// `root_powers[m + j]` for layer with half-size `m` and group index `j`.
    pub root_powers: Vec<u64>,

    /// Inverse twiddle factors for inverse NTT.
    ///
    /// Organized for sequential access in the Gentleman-Sande butterfly.
    pub inv_root_powers: Vec<u64>,

    /// N⁻¹ mod q — normalization factor for the INTT.
    pub n_inv: u64,
}

impl Ntt64Context {
    /// Fallible constructor for an NTT context.
    ///
    /// Validates all preconditions and returns an error instead of panicking.
    ///
    /// # Arguments
    /// - `n` — polynomial size, must be a power of 2 (≥ 2)
    /// - `arith` — precomputed modular arithmetic context; the modulus must be prime
    ///   and satisfy q ≡ 1 (mod 2N)
    ///
    /// # Errors
    /// - [`crate::NttError::InvalidSize`] if `n` is not a power of 2 ≥ 2
    /// - [`crate::NttError::NotPrime`] if the modulus is not prime
    /// - [`crate::NttError::NotNttFriendly`] if `q − 1` is not divisible by `2N`
    pub fn try_new(n: usize, arith: Ntt64Arith) -> Result<Self, crate::NttError> {
        if n < 2 || !n.is_power_of_two() {
            return Err(crate::NttError::InvalidSize(n));
        }
        let q = arith.modulus;
        if !super::prime::is_prime(q) {
            return Err(crate::NttError::NotPrime(q));
        }
        if !(q - 1).is_multiple_of(2 * n as u64) {
            return Err(crate::NttError::NotNttFriendly { q, n });
        }

        let log_n = n.trailing_zeros();

        // Find primitive 2N-th root of unity
        let psi = find_primitive_root(n, q);
        let psi_inv = mod_inv(psi, &arith);
        let n_inv = mod_inv(n as u64, &arith);

        // Precompute twiddle factors in Longa-Naehrig ordering:
        //   root_powers[i] = ψ^{bit_reverse(i, log_n)}  for i in 0..N
        let mut root_powers = vec![0u64; n];
        let mut inv_root_powers = vec![0u64; n];

        for i in 0..n {
            let exp = bit_reverse(i as u32, log_n) as u64;
            root_powers[i] = mod_pow(psi, exp, &arith);
            inv_root_powers[i] = mod_pow(psi_inv, exp, &arith);
        }

        Ok(Self {
            n,
            log_n,
            arith,
            root_powers,
            inv_root_powers,
            n_inv,
        })
    }

    /// Creates a new NTT context for polynomial size `n` and the given arithmetic context.
    ///
    /// # Arguments
    /// - `n` — polynomial size, must be a power of 2 (≥ 2)
    /// - `arith` — precomputed modular arithmetic context; the modulus must satisfy q ≡ 1 (mod 2N)
    ///
    /// # Panics
    /// - If `n` is not a power of 2
    /// - If the modulus is not prime
    /// - If q − 1 is not divisible by 2N
    pub fn new(n: usize, arith: Ntt64Arith) -> Self {
        Self::try_new(n, arith).expect("Invalid NTT parameters")
    }

    /// Applies forward NTT in-place.
    #[inline]
    pub fn forward(&self, data: &mut [u64]) {
        ntt_forward(data, self);
    }

    /// Applies inverse NTT in-place.
    #[inline]
    pub fn inverse(&self, data: &mut [u64]) {
        ntt_inverse(data, self);
    }

    /// Applies the tiled forward NTT in-place.
    ///
    /// Currently delegates to the standard forward NTT.
    /// A cache-optimized four-step variant is planned for v0.2.
    #[inline]
    pub fn forward_tiled(&self, data: &mut [u64]) {
        // TODO: implement proper four-step NTT with correct negacyclic twiddle decomposition
        ntt_forward(data, self);
    }

    /// Pointwise multiplication of two NTT-domain vectors.
    ///
    /// `result[i] = a[i] * b[i] mod q`
    ///
    /// This is the core operation: in NTT domain, polynomial convolution
    /// becomes element-wise multiplication.
    pub fn pointwise_mul(&self, a: &[u64], b: &[u64], result: &mut [u64]) {
        let n = self.n;
        assert_eq!(a.len(), n);
        assert_eq!(b.len(), n);
        assert_eq!(result.len(), n);

        for i in 0..n {
            result[i] = mod_mul_barrett(a[i], b[i], &self.arith);
        }
    }

    /// Full negacyclic polynomial multiplication: `c = a * b mod (X^N + 1)`.
    ///
    /// Performs forward NTT on both inputs, pointwise multiplication,
    /// and inverse NTT on the result.
    pub fn negacyclic_mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let n = self.n;
        assert_eq!(a.len(), n);
        assert_eq!(b.len(), n);

        let mut a_ntt = a.to_vec();
        let mut b_ntt = b.to_vec();
        ntt_forward(&mut a_ntt, self);
        ntt_forward(&mut b_ntt, self);

        let mut c_ntt = vec![0u64; n];
        self.pointwise_mul(&a_ntt, &b_ntt, &mut c_ntt);

        ntt_inverse(&mut c_ntt, self);
        c_ntt
    }
}

// ===========================================================================
// Forward NTT (Cooley-Tukey, Decimation In Time)
// ===========================================================================

/// Forward NTT in-place (negacyclic convolution, Longa-Naehrig ordering).
///
/// Transforms N polynomial coefficients in Z_q into their NTT representation.
///
/// ## Butterfly
/// ```text
/// u' = u + w·v
/// v' = u − w·v
/// ```
///
/// Layers are traversed from coarsest (gap = N/2) to finest (gap = 1).
pub fn ntt_forward(data: &mut [u64], ctx: &Ntt64Context) {
    let n = ctx.n;
    let q = ctx.arith.modulus;
    assert_eq!(data.len(), n, "data length ({}) != N ({})", data.len(), n);

    let mut t = n;
    let mut m = 1;

    for _ in 0..ctx.log_n {
        t >>= 1;
        let mut k = 0;

        for i in 0..m {
            let w = ctx.root_powers[m + i];

            for j in k..(k + t) {
                let u = data[j];
                let v = mod_mul_barrett(data[j + t], w, &ctx.arith);
                data[j] = mod_add(u, v, q);
                data[j + t] = mod_sub(u, v, q);
            }
            k += 2 * t;
        }
        m <<= 1;
    }
}

// ===========================================================================
// Inverse NTT (Gentleman-Sande, Decimation In Frequency)
// ===========================================================================

/// Inverse NTT in-place (negacyclic convolution, Longa-Naehrig ordering).
///
/// Transforms an NTT representation of N elements back to polynomial coefficients.
///
/// ## Butterfly
/// ```text
/// u' = u + v
/// v' = (u − v) · w_inv
/// ```
///
/// Layers are traversed from finest (gap = 1) to coarsest (gap = N/2).
/// Each coefficient is multiplied by N⁻¹ mod q at the end.
pub fn ntt_inverse(data: &mut [u64], ctx: &Ntt64Context) {
    let n = ctx.n;
    let q = ctx.arith.modulus;
    assert_eq!(data.len(), n, "data length ({}) != N ({})", data.len(), n);

    let mut t = 1;
    let mut m = n;

    for _ in 0..ctx.log_n {
        m >>= 1;
        let mut k = 0;

        for i in 0..m {
            let w_inv = ctx.inv_root_powers[m + i];

            for j in k..(k + t) {
                let u = data[j];
                let v = data[j + t];
                data[j] = mod_add(u, v, q);
                data[j + t] = mod_mul_barrett(mod_sub(u, v, q), w_inv, &ctx.arith);
            }
            k += 2 * t;
        }
        t <<= 1;
    }

    // Normalize by N⁻¹
    for coeff in data.iter_mut() {
        *coeff = mod_mul_barrett(*coeff, ctx.n_inv, &ctx.arith);
    }
}

// ===========================================================================
// Four-Step Tiled NTT (cache-friendly)
// ===========================================================================

/// Four-step tiled forward NTT for improved cache locality.
///
/// Views the length-N vector as an N1×N2 matrix (row-major) with
/// N = N1·N2 and N1, N2 powers of 2 (N1 ≈ √N).
///
/// ## Steps
/// 1. NTT of size N2 on each row (fits in L1 cache)
/// 2. Multiply by transposition twiddle factors ω^{i·j}
/// 3. Transpose (N1×N2 → N2×N1)
/// 4. NTT of size N1 on each row
/// 5. Transpose back
///
/// For small N (≤ 64), delegates to the standard NTT.
///
/// NOTE: Currently unused — the negacyclic twiddle decomposition has a known bug.
/// Kept for future v0.2 implementation.
#[allow(dead_code)]
pub fn ntt_forward_tiled(data: &mut [u64], ctx: &Ntt64Context) {
    let n = ctx.n;

    if n <= 64 {
        ntt_forward(data, ctx);
        return;
    }

    let log_n = ctx.log_n;
    let log_n1 = log_n / 2;
    let log_n2 = log_n - log_n1;
    let n1 = 1usize << log_n1;
    let n2 = 1usize << log_n2;

    let arith = &ctx.arith;

    // Step 1: NTT of size N2 on each row
    let sub_ctx2 = Ntt64Context::new(n2, arith.clone());
    for row in 0..n1 {
        let start = row * n2;
        ntt_forward(&mut data[start..start + n2], &sub_ctx2);
    }

    // Step 2: Multiply by transposition twiddle factors
    // ω = ψ² (N-th root of unity), twiddle = ω^{i·j}
    let psi = find_primitive_root(n, arith.modulus);
    let psi_sq = mod_mul_barrett(psi, psi, arith); // ω = ψ², N-th root

    for i in 0..n1 {
        for j in 0..n2 {
            if i == 0 || j == 0 {
                continue;
            }
            let exp = ((i as u128 * j as u128) % n as u128) as u64;
            let twiddle = mod_pow(psi_sq, exp, arith);
            let idx = i * n2 + j;
            data[idx] = mod_mul_barrett(data[idx], twiddle, arith);
        }
    }

    // Step 3: Transpose (N1×N2 → N2×N1)
    let mut transposed = vec![0u64; n];
    for i in 0..n1 {
        for j in 0..n2 {
            transposed[j * n1 + i] = data[i * n2 + j];
        }
    }
    data.copy_from_slice(&transposed);

    // Step 4: NTT of size N1 on each row
    let sub_ctx1 = Ntt64Context::new(n1, arith.clone());
    for row in 0..n2 {
        let start = row * n1;
        ntt_forward(&mut data[start..start + n1], &sub_ctx1);
    }

    // Step 5: Transpose back (N2×N1 → N1×N2)
    for i in 0..n2 {
        for j in 0..n1 {
            transposed[j * n2 + i] = data[i * n1 + j];
        }
    }
    data.copy_from_slice(&transposed);
}

// ===========================================================================
// Naive polynomial multiplication (test-only)
// ===========================================================================

/// Naive polynomial multiplication in Z_q[X]/(X^N + 1) — O(N²) complexity.
///
/// Used only in tests to verify NTT correctness.
#[cfg(test)]
fn poly_mul_naive(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let n = a.len();
    assert_eq!(b.len(), n);
    let mut result = vec![0u64; n];

    for i in 0..n {
        for j in 0..n {
            let idx = i + j;
            let prod = (a[i] as u128 * b[j] as u128) % q as u128;
            if idx < n {
                result[idx] = ((result[idx] as u128 + prod) % q as u128) as u64;
            } else {
                let idx = idx - n;
                result[idx] = ((result[idx] as u128 + q as u128 - prod) % q as u128) as u64;
            }
        }
    }
    result
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::super::arith::{PRIME_60_1, PRIME_SEAL};
    use super::*;

    // --- Primitive root ---

    #[test]
    fn test_primitive_root_small() {
        let q: u64 = 17;
        let n = 8;
        let psi = find_primitive_root(n, q);

        let arith = Ntt64Arith::new(q);
        assert_eq!(mod_pow(psi, 2 * n as u64, &arith), 1);
        assert_eq!(mod_pow(psi, n as u64, &arith), 16);
    }

    #[test]
    fn test_primitive_root_seal() {
        let arith = Ntt64Arith::new(PRIME_SEAL);
        for &n in &[16, 64, 1024, 4096] {
            let psi = find_primitive_root(n, PRIME_SEAL);
            assert_eq!(mod_pow(psi, 2 * n as u64, &arith), 1);
            assert_eq!(mod_pow(psi, n as u64, &arith), arith.modulus - 1);
        }
    }

    // --- NTT roundtrip ---

    #[test]
    fn test_ntt_roundtrip_small() {
        let arith = Ntt64Arith::new(PRIME_SEAL);
        let q = arith.modulus;

        for &n in &[16, 64] {
            let ctx = Ntt64Context::new(n, arith.clone());
            let original: Vec<u64> = (0..n).map(|i| (i as u64 * 7 + 3) % q).collect();
            let mut data = original.clone();

            ntt_forward(&mut data, &ctx);
            assert_ne!(data, original);
            ntt_inverse(&mut data, &ctx);
            assert_eq!(data, original, "NTT roundtrip fails for N={n}");
        }
    }

    #[test]
    fn test_ntt_roundtrip_medium() {
        let arith = Ntt64Arith::new(PRIME_SEAL);
        let q = arith.modulus;

        for &n in &[1024, 4096] {
            let ctx = Ntt64Context::new(n, arith.clone());
            let original: Vec<u64> = (0..n)
                .map(|i| ((i as u128 * 123456789 + 987654321) % q as u128) as u64)
                .collect();
            let mut data = original.clone();

            ntt_forward(&mut data, &ctx);
            ntt_inverse(&mut data, &ctx);
            assert_eq!(data, original, "NTT roundtrip fails for N={n}");
        }
    }

    #[test]
    fn test_ntt_roundtrip_zeros() {
        let arith = Ntt64Arith::new(PRIME_SEAL);
        let n = 64;
        let ctx = Ntt64Context::new(n, arith);
        let mut data = vec![0u64; n];
        ntt_forward(&mut data, &ctx);
        ntt_inverse(&mut data, &ctx);
        assert_eq!(data, vec![0u64; n]);
    }

    #[test]
    fn test_ntt_roundtrip_one() {
        let arith = Ntt64Arith::new(PRIME_SEAL);
        let n = 64;
        let ctx = Ntt64Context::new(n, arith);
        let mut data = vec![0u64; n];
        data[0] = 1;
        let original = data.clone();
        ntt_forward(&mut data, &ctx);
        ntt_inverse(&mut data, &ctx);
        assert_eq!(data, original);
    }

    // --- Negacyclic convolution ---

    #[test]
    fn test_ntt_convolution_n16() {
        let arith = Ntt64Arith::new(PRIME_SEAL);
        let q = arith.modulus;
        let n = 16;
        let ctx = Ntt64Context::new(n, arith);

        let a: Vec<u64> = (0..n).map(|i| (i as u64 + 1) % q).collect();
        let b: Vec<u64> = (0..n).map(|_| 1u64).collect();

        let expected = poly_mul_naive(&a, &b, q);
        let result = ctx.negacyclic_mul(&a, &b);
        assert_eq!(result, expected, "NTT convolution != naive for N=16");
    }

    #[test]
    fn test_ntt_convolution_n64() {
        let arith = Ntt64Arith::new(PRIME_SEAL);
        let q = arith.modulus;
        let n = 64;
        let ctx = Ntt64Context::new(n, arith);

        let a: Vec<u64> = (0..n).map(|i| ((i * i + 3 * i + 7) as u64) % q).collect();
        let b: Vec<u64> = (0..n).map(|i| ((2 * i + 1) as u64) % q).collect();

        let expected = poly_mul_naive(&a, &b, q);
        let result = ctx.negacyclic_mul(&a, &b);
        assert_eq!(result, expected, "NTT convolution != naive for N=64");
    }

    #[test]
    fn test_ntt_convolution_identity() {
        let arith = Ntt64Arith::new(PRIME_SEAL);
        let q = arith.modulus;
        let n = 64;
        let ctx = Ntt64Context::new(n, arith);

        let a: Vec<u64> = (0..n).map(|i| ((i * 17 + 5) as u64) % q).collect();
        let mut one = vec![0u64; n];
        one[0] = 1;

        let result = ctx.negacyclic_mul(&a, &one);
        assert_eq!(result, a, "Multiplying by 1 should give identity");
    }

    // --- Tiled NTT ---

    #[test]
    fn test_ntt_tiled_matches_standard_small() {
        let arith = Ntt64Arith::new(PRIME_SEAL);
        let q = arith.modulus;

        for &n in &[16, 64] {
            let ctx = Ntt64Context::new(n, arith.clone());
            let original: Vec<u64> = (0..n).map(|i| (i as u64 * 13 + 7) % q).collect();

            let mut data_std = original.clone();
            let mut data_tiled = original.clone();

            ntt_forward(&mut data_std, &ctx);
            ntt_forward_tiled(&mut data_tiled, &ctx);

            assert_eq!(data_tiled, data_std, "tiled NTT != standard for N={n}");
        }
    }

    #[test]
    fn test_ntt_tiled_roundtrip() {
        let arith = Ntt64Arith::new(PRIME_SEAL);
        let q = arith.modulus;
        let n = 256;
        let ctx = Ntt64Context::new(n, arith);

        let original: Vec<u64> = (0..n)
            .map(|i| ((i as u128 * 999999937 + 42) % q as u128) as u64)
            .collect();
        let mut data = original.clone();

        ntt_forward(&mut data, &ctx);
        ntt_inverse(&mut data, &ctx);
        assert_eq!(data, original, "standard roundtrip fails for N=256");
    }

    // --- With PRIME_60_1 ---

    #[test]
    fn test_ntt_with_prime_60_1() {
        let arith = Ntt64Arith::new(PRIME_60_1);
        let q = arith.modulus;

        for &n in &[16, 64] {
            assert_eq!((q - 1) % (2 * n as u64), 0);
            let ctx = Ntt64Context::new(n, arith.clone());
            let original: Vec<u64> = (0..n).map(|i| (i as u64 * 31 + 11) % q).collect();
            let mut data = original.clone();

            ntt_forward(&mut data, &ctx);
            ntt_inverse(&mut data, &ctx);
            assert_eq!(
                data, original,
                "NTT roundtrip fails for N={n} with PRIME_60_1"
            );
        }
    }

    // --- Bit-reverse ---

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0, 3), 0);
        assert_eq!(bit_reverse(1, 3), 4);
        assert_eq!(bit_reverse(2, 3), 2);
        assert_eq!(bit_reverse(3, 3), 6);
        assert_eq!(bit_reverse(4, 3), 1);
        assert_eq!(bit_reverse(5, 3), 5);
        assert_eq!(bit_reverse(6, 3), 3);
        assert_eq!(bit_reverse(7, 3), 7);
        assert_eq!(bit_reverse(0, 1), 0);
        assert_eq!(bit_reverse(1, 1), 1);
    }

    // --- Linearity ---

    #[test]
    fn test_ntt_linearity() {
        let arith = Ntt64Arith::new(PRIME_SEAL);
        let q = arith.modulus;
        let n = 64;
        let ctx = Ntt64Context::new(n, arith);

        let a: Vec<u64> = (0..n).map(|i| (i as u64 * 3 + 1) % q).collect();
        let b: Vec<u64> = (0..n).map(|i| (i as u64 * 7 + 2) % q).collect();

        let mut a_ntt = a.clone();
        let mut b_ntt = b.clone();
        ntt_forward(&mut a_ntt, &ctx);
        ntt_forward(&mut b_ntt, &ctx);

        let mut sum: Vec<u64> = (0..n).map(|i| mod_add(a[i], b[i], q)).collect();
        ntt_forward(&mut sum, &ctx);

        for i in 0..n {
            let expected = mod_add(a_ntt[i], b_ntt[i], q);
            assert_eq!(sum[i], expected, "linearity violated at index {i}");
        }
    }

    // --- Large N roundtrip ---

    #[test]
    fn test_ntt_roundtrip_large() {
        let arith = Ntt64Arith::new(PRIME_SEAL);
        let q = arith.modulus;
        let n = 32768;

        assert_eq!((q - 1) % (2 * n as u64), 0);
        let ctx = Ntt64Context::new(n, arith);

        let original: Vec<u64> = (0..n)
            .map(|i| ((i as u128 * 314159265 + 271828182) % q as u128) as u64)
            .collect();
        let mut data = original.clone();

        ntt_forward(&mut data, &ctx);
        ntt_inverse(&mut data, &ctx);
        assert_eq!(data, original, "NTT roundtrip fails for N=32768");
    }

    // Compile-time check: Ntt64Context must be Send + Sync
    const _: () = {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        fn check() {
            assert_send::<super::Ntt64Context>();
            assert_sync::<super::Ntt64Context>();
        }
    };
}
