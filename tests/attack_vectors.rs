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


//! Attack vector tests for VaeaNTT — Day 1 security validation

use vaea_ntt::ntt32::Ntt32Context;
use vaea_ntt::NttError;

// === 1. Boundary attacks ===

#[test]
fn attack_zero_polynomial() {
    let ctx = Ntt32Context::new(256, 8380417);
    let mut data = vec![0u32; 256];
    ctx.forward(&mut data);
    ctx.inverse(&mut data);
    assert!(data.iter().all(|&x| x == 0), "zero poly roundtrip failed");
}

#[test]
fn attack_max_coefficient() {
    // Coefficients at q-1 (maximum valid value)
    let q = 8380417u32;
    let ctx = Ntt32Context::new(256, q);
    let mut data = vec![q - 1; 256];
    let original = data.clone();
    ctx.forward(&mut data);
    ctx.inverse(&mut data);
    assert_eq!(data, original, "max coefficient roundtrip failed");
}

#[test]
fn attack_single_nonzero() {
    // Only one coefficient is non-zero (sparse polynomial)
    let q = 8380417u32;
    let ctx = Ntt32Context::new(256, q);
    for pos in [0, 1, 127, 255] {
        let mut data = vec![0u32; 256];
        data[pos] = 42;
        let original = data.clone();
        ctx.forward(&mut data);
        ctx.inverse(&mut data);
        assert_eq!(data, original, "sparse poly roundtrip failed at pos {pos}");
    }
}

// === 2. Invalid parameter attacks ===

#[test]
fn attack_non_power_of_two() {
    let result = Ntt32Context::try_new(255, 8380417);
    assert!(result.is_err(), "should reject non-power-of-2 size");
}

#[test]
fn attack_non_prime_modulus() {
    let result = Ntt32Context::try_new(256, 100);
    assert!(result.is_err(), "should reject non-prime modulus");
}

#[test]
fn attack_non_ntt_friendly() {
    // 17 is prime but 17 mod (2*256) != 1
    let result = Ntt32Context::try_new(256, 17);
    assert!(result.is_err(), "should reject non-NTT-friendly prime");
}

#[test]
fn attack_prime_too_large() {
    // 2^28 = 268435456 — too large
    let result = Ntt32Context::try_new(256, 268435457);
    assert!(result.is_err(), "should reject prime >= 2^28");
}

#[test]
fn attack_size_one() {
    // N=1 is not valid (need at least 2)
    let result = Ntt32Context::try_new(1, 3);
    assert!(result.is_err(), "should reject N=1");
}

// === 3. Algebraic correctness attacks ===

#[test]
fn attack_negacyclic_identity() {
    // Multiply by [1, 0, 0, ...] should return the same polynomial
    let q = 8380417u32;
    let ctx = Ntt32Context::new(256, q);
    let a: Vec<u32> = (1..=256).map(|i| i % q).collect();
    let mut identity = vec![0u32; 256];
    identity[0] = 1;
    let result = ctx.negacyclic_mul(&a, &identity);
    assert_eq!(result, a, "multiplication by identity failed");
}

#[test]
fn attack_negacyclic_commutativity() {
    // a * b == b * a
    let q = 8380417u32;
    let ctx = Ntt32Context::new(256, q);
    let a: Vec<u32> = (0..256).map(|i| (i * 7 + 3) as u32 % q).collect();
    let b: Vec<u32> = (0..256).map(|i| (i * 13 + 5) as u32 % q).collect();
    let ab = ctx.negacyclic_mul(&a, &b);
    let ba = ctx.negacyclic_mul(&b, &a);
    assert_eq!(ab, ba, "negacyclic multiplication not commutative");
}

#[test]
fn attack_negacyclic_x_times_x() {
    // X * X = X^2 in Z_q[X]/(X^N+1)
    // X = [0, 1, 0, 0, ...]
    // X^2 = [0, 0, 1, 0, ...]
    let q = 8380417u32;
    let n = 256;
    let ctx = Ntt32Context::new(n, q);
    let mut x = vec![0u32; n];
    x[1] = 1;
    let result = ctx.negacyclic_mul(&x, &x);
    let mut expected = vec![0u32; n];
    expected[2] = 1;
    assert_eq!(result, expected, "X * X != X^2");
}

#[test]
fn attack_negacyclic_xn_wrap() {
    // X^(N-1) * X = X^N = -1 mod (X^N+1)
    // X^(N-1) = [0, 0, ..., 0, 1] (coeff N-1 = 1)
    // Result should be [q-1, 0, 0, ..., 0] = -1
    let q = 8380417u32;
    let n = 256;
    let ctx = Ntt32Context::new(n, q);
    let mut x = vec![0u32; n];
    x[1] = 1;
    let mut xn_minus_1 = vec![0u32; n];
    xn_minus_1[n - 1] = 1;
    let result = ctx.negacyclic_mul(&xn_minus_1, &x);
    let mut expected = vec![0u32; n];
    expected[0] = q - 1; // -1 mod q
    assert_eq!(result, expected, "X^(N-1) * X != -1 mod (X^N+1)");
}

// === 4. All NIST PQ primes ===

#[test]
fn attack_all_pq_primes() {
    // Note: ML-KEM q=3329 requires N=128 (not 256) because
    // 3329-1 = 3328, and 2*256=512 does not divide 3328.
    // 2*128=256 divides 3328 = 256*13. ✓
    let primes: &[(u32, usize, &str)] = &[
        (8380417, 256, "ML-DSA"),
        (3329, 128, "ML-KEM"),
        (12289, 512, "Falcon-512"),
        (12289, 1024, "Falcon-1024"),
    ];
    for &(q, n, name) in primes {
        let ctx = Ntt32Context::new(n, q);
        let data: Vec<u32> = (0..n).map(|i| (i as u32 * 37 + 1) % q).collect();
        let mut buf = data.clone();
        ctx.forward(&mut buf);
        // Verify it's different from input (not identity)
        assert_ne!(buf, data, "{name}: forward NTT produced no change");
        ctx.inverse(&mut buf);
        assert_eq!(buf, data, "{name}: roundtrip failed");
    }
}

// === 5. Stress: large sizes ===

#[test]
fn attack_large_n_4096() {
    // ML-DSA q with N=4096
    // 8380416 / (2*4096) = 8380416 / 8192 = 1023 ✓
    let q = 8380417u32;
    let n = 4096;
    let ctx = Ntt32Context::new(n, q);
    let data: Vec<u32> = (0..n).map(|i| (i as u32) % q).collect();
    let mut buf = data.clone();
    ctx.forward(&mut buf);
    ctx.inverse(&mut buf);
    assert_eq!(buf, data, "N=4096 roundtrip failed");
}
