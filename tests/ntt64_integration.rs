//! Integration tests for the 64-bit NTT pipeline.
//! Cross-validates against naive O(N²) reference and tests algebraic properties.

use vaea_ntt::ntt64::{
    generate_primes_60, mod_add, Ntt64Arith, Ntt64Context, PRIME_60_1, PRIME_SEAL,
};

/// Naive O(N²) negacyclic convolution for u64 primes.
fn naive_negacyclic_mul_64(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let n = a.len();
    assert_eq!(b.len(), n);
    let mut result = vec![0u128; n];

    for i in 0..n {
        for j in 0..n {
            let prod = a[i] as u128 * b[j] as u128;
            if i + j < n {
                result[i + j] += prod;
            } else {
                let idx = i + j - n;
                result[idx] += q as u128 * q as u128;
                result[idx] -= prod;
            }
        }
    }

    result.iter().map(|&x| (x % q as u128) as u64).collect()
}

#[test]
fn test_ntt64_vs_naive_n16() {
    let n = 16;
    let primes = generate_primes_60(n, 60, 1);
    let arith = Ntt64Arith::new(primes[0]);
    let ctx = Ntt64Context::new(n, arith);

    let a: Vec<u64> = (0..n)
        .map(|i| ((i as u128 * 7 + 3) % primes[0] as u128) as u64)
        .collect();
    let b: Vec<u64> = (0..n)
        .map(|i| ((i as u128 * 13 + 5) % primes[0] as u128) as u64)
        .collect();

    let ntt_result = ctx.negacyclic_mul(&a, &b);
    let naive_result = naive_negacyclic_mul_64(&a, &b, primes[0]);

    assert_eq!(
        ntt_result, naive_result,
        "NTT64 vs naive mismatch for N={n}"
    );
}

#[test]
fn test_ntt64_vs_naive_n64() {
    let n = 64;
    let primes = generate_primes_60(n, 60, 1);
    let arith = Ntt64Arith::new(primes[0]);
    let ctx = Ntt64Context::new(n, arith);

    let a: Vec<u64> = (0..n)
        .map(|i| ((i as u128 * 41 + 17) % primes[0] as u128) as u64)
        .collect();
    let b: Vec<u64> = (0..n)
        .map(|i| ((i as u128 * 59 + 23) % primes[0] as u128) as u64)
        .collect();

    let ntt_result = ctx.negacyclic_mul(&a, &b);
    let naive_result = naive_negacyclic_mul_64(&a, &b, primes[0]);

    assert_eq!(
        ntt_result, naive_result,
        "NTT64 vs naive mismatch for N={n}"
    );
}

#[test]
fn test_ntt64_seal_prime_roundtrip() {
    let n = 4096;
    let arith = Ntt64Arith::new(PRIME_SEAL);
    let ctx = Ntt64Context::new(n, arith);

    let original: Vec<u64> = (0..n)
        .map(|i| ((i as u128 * 314159 + 271828) % PRIME_SEAL as u128) as u64)
        .collect();
    let mut data = original.clone();

    ctx.forward(&mut data);
    assert_ne!(data, original, "Forward NTT did nothing");
    ctx.inverse(&mut data);
    assert_eq!(data, original, "SEAL prime roundtrip failed for N={n}");
}

#[test]
fn test_ntt64_prime60_1_roundtrip() {
    let n = 1024;
    let arith = Ntt64Arith::new(PRIME_60_1);
    let ctx = Ntt64Context::new(n, arith);

    let original: Vec<u64> = (0..n)
        .map(|i| ((i as u128 * 271828) % PRIME_60_1 as u128) as u64)
        .collect();
    let mut data = original.clone();

    ctx.forward(&mut data);
    ctx.inverse(&mut data);
    assert_eq!(data, original, "PRIME_60_1 roundtrip failed");
}

#[test]
fn test_ntt64_linearity() {
    let n = 256;
    let primes = generate_primes_60(n, 60, 1);
    let q = primes[0];
    let arith = Ntt64Arith::new(q);
    let ctx = Ntt64Context::new(n, arith);

    let a: Vec<u64> = (0..n)
        .map(|i| ((i as u128 * 17 + 3) % q as u128) as u64)
        .collect();
    let b: Vec<u64> = (0..n)
        .map(|i| ((i as u128 * 31 + 7) % q as u128) as u64)
        .collect();

    // NTT(a + b)
    let ab_sum: Vec<u64> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| mod_add(x, y, q))
        .collect();
    let mut ntt_sum = ab_sum;
    ctx.forward(&mut ntt_sum);

    // NTT(a) + NTT(b)
    let mut ntt_a = a;
    let mut ntt_b = b;
    ctx.forward(&mut ntt_a);
    ctx.forward(&mut ntt_b);
    let sum_ntt: Vec<u64> = ntt_a
        .iter()
        .zip(ntt_b.iter())
        .map(|(&x, &y)| mod_add(x, y, q))
        .collect();

    assert_eq!(ntt_sum, sum_ntt, "NTT64 linearity violated");
}

#[test]
fn test_ntt64_tiled_matches_standard() {
    // Tiled NTT must produce the same result as standard NTT
    let n = 4096;
    let primes = generate_primes_60(n, 60, 1);
    let arith = Ntt64Arith::new(primes[0]);
    let ctx = Ntt64Context::new(n, arith);

    let original: Vec<u64> = (0..n)
        .map(|i| ((i as u128 * 41 + 7) % primes[0] as u128) as u64)
        .collect();

    let mut standard = original.clone();
    let mut tiled = original.clone();

    ctx.forward(&mut standard);
    ctx.forward_tiled(&mut tiled);

    assert_eq!(
        standard, tiled,
        "Tiled NTT does not match standard for N={n}"
    );
}

#[test]
fn test_ntt64_multiple_primes_same_n() {
    let n = 1024;
    let primes = generate_primes_60(n, 60, 3);

    for (idx, &q) in primes.iter().enumerate() {
        let arith = Ntt64Arith::new(q);
        let ctx = Ntt64Context::new(n, arith);
        let original: Vec<u64> = (0..n)
            .map(|i| ((i as u128 * 41 + 7) % q as u128) as u64)
            .collect();
        let mut data = original.clone();

        ctx.forward(&mut data);
        ctx.inverse(&mut data);

        assert_eq!(
            data, original,
            "Roundtrip failed for 60-bit prime[{idx}]={q}"
        );
    }
}

#[test]
fn test_ntt64_commutativity() {
    let n = 64;
    let primes = generate_primes_60(n, 60, 1);
    let arith = Ntt64Arith::new(primes[0]);
    let ctx = Ntt64Context::new(n, arith);

    let a: Vec<u64> = (0..n)
        .map(|i| ((i as u128 * 23 + 5) % primes[0] as u128) as u64)
        .collect();
    let b: Vec<u64> = (0..n)
        .map(|i| ((i as u128 * 47 + 11) % primes[0] as u128) as u64)
        .collect();

    let ab = ctx.negacyclic_mul(&a, &b);
    let ba = ctx.negacyclic_mul(&b, &a);
    assert_eq!(ab, ba, "64-bit multiplication not commutative");
}
