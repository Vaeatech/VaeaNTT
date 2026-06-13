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


//! Integration tests — cross-validate NTT32 pipeline against naive O(N²) computation
//! and stress-test with randomized inputs.

use vaea_ntt::ntt32::{generate_primes_28, Ntt32Context};

/// Naive O(N²) negacyclic convolution reference implementation.
/// Computes c = a * b in Z_q[X]/(X^N + 1).
fn naive_negacyclic_mul(a: &[u32], b: &[u32], q: u32) -> Vec<u32> {
    let n = a.len();
    assert_eq!(b.len(), n);
    let mut result = vec![0u64; n];

    for i in 0..n {
        for j in 0..n {
            let prod = a[i] as u64 * b[j] as u64;
            if i + j < n {
                result[i + j] += prod;
            } else {
                // X^N = -1 in the negacyclic ring
                let idx = i + j - n;
                result[idx] += q as u64 * q as u64; // ensure no underflow
                result[idx] -= prod;
            }
        }
    }

    result.iter().map(|&x| (x % q as u64) as u32).collect()
}

#[test]
fn test_ntt32_vs_naive_n16() {
    let n = 16;
    let q = generate_primes_28(n, 1)[0];
    let ctx = Ntt32Context::new(n, q);

    let a: Vec<u32> = (0..n).map(|i| (i as u32 * 7 + 3) % q).collect();
    let b: Vec<u32> = (0..n).map(|i| (i as u32 * 13 + 5) % q).collect();

    let ntt_result = ctx.negacyclic_mul(&a, &b);
    let naive_result = naive_negacyclic_mul(&a, &b, q);

    assert_eq!(
        ntt_result, naive_result,
        "NTT32 vs naive mismatch for N={n}"
    );
}

#[test]
fn test_ntt32_vs_naive_n64() {
    let n = 64;
    let q = generate_primes_28(n, 1)[0];
    let ctx = Ntt32Context::new(n, q);

    let a: Vec<u32> = (0..n).map(|i| (i as u32 * 41 + 17) % q).collect();
    let b: Vec<u32> = (0..n).map(|i| (i as u32 * 59 + 23) % q).collect();

    let ntt_result = ctx.negacyclic_mul(&a, &b);
    let naive_result = naive_negacyclic_mul(&a, &b, q);

    assert_eq!(
        ntt_result, naive_result,
        "NTT32 vs naive mismatch for N={n}"
    );
}

#[test]
fn test_ntt32_vs_naive_n256() {
    let n = 256;
    let q = generate_primes_28(n, 1)[0];
    let ctx = Ntt32Context::new(n, q);

    let a: Vec<u32> = (0..n).map(|i| (i as u32 * 101 + 37) % q).collect();
    let b: Vec<u32> = (0..n).map(|i| (i as u32 * 97 + 53) % q).collect();

    let ntt_result = ctx.negacyclic_mul(&a, &b);
    let naive_result = naive_negacyclic_mul(&a, &b, q);

    assert_eq!(
        ntt_result, naive_result,
        "NTT32 vs naive mismatch for N={n}"
    );
}

#[test]
fn test_ntt32_linearity() {
    // NTT(a + b) = NTT(a) + NTT(b) (in NTT domain)
    let n = 256;
    let q = generate_primes_28(n, 1)[0];
    let ctx = Ntt32Context::new(n, q);

    let a: Vec<u32> = (0..n).map(|i| (i as u32 * 17 + 3) % q).collect();
    let b: Vec<u32> = (0..n).map(|i| (i as u32 * 31 + 7) % q).collect();

    // NTT(a + b)
    let ab_sum: Vec<u32> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((x as u64 + y as u64) % q as u64) as u32)
        .collect();
    let mut ntt_sum = ab_sum;
    ctx.forward(&mut ntt_sum);

    // NTT(a) + NTT(b)
    let mut ntt_a = a;
    let mut ntt_b = b;
    ctx.forward(&mut ntt_a);
    ctx.forward(&mut ntt_b);
    let sum_ntt: Vec<u32> = ntt_a
        .iter()
        .zip(ntt_b.iter())
        .map(|(&x, &y)| ((x as u64 + y as u64) % q as u64) as u32)
        .collect();

    assert_eq!(ntt_sum, sum_ntt, "NTT linearity violated for N={n}");
}

#[test]
fn test_ntt32_multiple_primes_same_n() {
    // Verify roundtrip works with different primes for the same N
    let n = 1024;
    let primes = generate_primes_28(n, 5);

    for (idx, &q) in primes.iter().enumerate() {
        let ctx = Ntt32Context::new(n, q);
        let original: Vec<u32> = (0..n)
            .map(|i| ((i as u64 * 41 + 7) % q as u64) as u32)
            .collect();
        let mut data = original.clone();

        ctx.forward(&mut data);
        ctx.inverse(&mut data);

        assert_eq!(data, original, "Roundtrip failed for prime[{idx}]={q}");
    }
}

#[test]
fn test_ntt32_max_values() {
    // Edge case: all coefficients = q - 1 (maximum value)
    let n = 64;
    let q = generate_primes_28(n, 1)[0];
    let ctx = Ntt32Context::new(n, q);
    let original = vec![q - 1; n];
    let mut data = original.clone();

    ctx.forward(&mut data);
    ctx.inverse(&mut data);
    assert_eq!(data, original, "Max-value roundtrip failed");
}

#[test]
fn test_ntt32_alternating_pattern() {
    // Stress test: alternating [0, q-1, 0, q-1, ...]
    let n = 128;
    let q = generate_primes_28(n, 1)[0];
    let ctx = Ntt32Context::new(n, q);
    let original: Vec<u32> = (0..n).map(|i| if i % 2 == 0 { 0 } else { q - 1 }).collect();
    let mut data = original.clone();

    ctx.forward(&mut data);
    ctx.inverse(&mut data);
    assert_eq!(data, original, "Alternating pattern roundtrip failed");
}

#[test]
fn test_ntt32_commutativity() {
    // a * b == b * a
    let n = 64;
    let q = generate_primes_28(n, 1)[0];
    let ctx = Ntt32Context::new(n, q);

    let a: Vec<u32> = (0..n).map(|i| (i as u32 * 23 + 5) % q).collect();
    let b: Vec<u32> = (0..n).map(|i| (i as u32 * 47 + 11) % q).collect();

    let ab = ctx.negacyclic_mul(&a, &b);
    let ba = ctx.negacyclic_mul(&b, &a);
    assert_eq!(ab, ba, "Multiplication not commutative");
}

#[test]
fn test_ntt32_associativity() {
    // (a * b) * c == a * (b * c)
    let n = 64;
    let q = generate_primes_28(n, 1)[0];
    let ctx = Ntt32Context::new(n, q);

    let a: Vec<u32> = (0..n).map(|i| (i as u32 * 7 + 1) % q).collect();
    let b: Vec<u32> = (0..n).map(|i| (i as u32 * 11 + 2) % q).collect();
    let c: Vec<u32> = (0..n).map(|i| (i as u32 * 13 + 3) % q).collect();

    let ab = ctx.negacyclic_mul(&a, &b);
    let ab_c = ctx.negacyclic_mul(&ab, &c);

    let bc = ctx.negacyclic_mul(&b, &c);
    let a_bc = ctx.negacyclic_mul(&a, &bc);

    assert_eq!(ab_c, a_bc, "Multiplication not associative");
}
