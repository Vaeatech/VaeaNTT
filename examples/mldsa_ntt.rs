//! Example: ML-DSA NTT (Dilithium signature scheme)
//!
//! Demonstrates using VaeaNTT for the NTT operation at the heart of ML-DSA (FIPS 204).
//! ML-DSA uses q = 8380417 (23-bit prime), N = 256.
//!
//! Run: `cargo run --example mldsa_ntt`

use vaea_ntt::ntt32::{Ntt32Context, generate_primes_28};

fn main() {
    // ---------------------------------------------------------------
    // ML-DSA parameters (FIPS 204)
    // ---------------------------------------------------------------
    const Q: u32 = 8_380_417; // 2^23 - 2^13 + 1
    const N: usize = 256;

    println!("=== VaeaNTT — ML-DSA (Dilithium) NTT Demo ===\n");
    println!("Parameters: q = {Q}, N = {N}");
    println!("q = 2^23 - 2^13 + 1 = {Q} ({} bits)", 32 - Q.leading_zeros());

    // Create NTT context — validates q is prime and NTT-friendly
    let ctx = Ntt32Context::new(N, Q);

    // ---------------------------------------------------------------
    // Simulate a polynomial from ML-DSA (e.g., a coefficient of s1)
    // ---------------------------------------------------------------
    let mut poly: Vec<u32> = (0..N)
        .map(|i| ((i as u64 * 1753 + 42) % Q as u64) as u32) // deterministic test data
        .collect();

    let original = poly.clone();
    println!("\nFirst 8 coefficients: {:?}", &poly[..8]);

    // ---------------------------------------------------------------
    // Forward NTT
    // ---------------------------------------------------------------
    ctx.forward(&mut poly);
    println!("After NTT (first 8):  {:?}", &poly[..8]);

    // ---------------------------------------------------------------
    // Inverse NTT — should recover original
    // ---------------------------------------------------------------
    ctx.inverse(&mut poly);
    println!("After INTT (first 8): {:?}", &poly[..8]);

    assert_eq!(poly, original, "Roundtrip failed!");
    println!("\n✅ NTT roundtrip verified for ML-DSA parameters.");

    // ---------------------------------------------------------------
    // Negacyclic polynomial multiplication (core of ML-DSA signing)
    // ---------------------------------------------------------------
    let a: Vec<u32> = (0..N).map(|i| ((i as u64 * 17 + 5) % Q as u64) as u32).collect();
    let mut one = vec![0u32; N];
    one[0] = 1; // multiplicative identity

    let result = ctx.negacyclic_mul(&a, &one);
    assert_eq!(result, a, "Multiply by 1 should be identity!");
    println!("✅ Negacyclic multiplication verified (a × 1 = a).");

    // ---------------------------------------------------------------
    // Zero-alloc API (production usage)
    // ---------------------------------------------------------------
    let mut a_buf = a.clone();
    let mut b_buf = one.clone();
    let mut out = vec![0u32; N];

    ctx.negacyclic_mul_into(&mut a_buf, &mut b_buf, &mut out);
    assert_eq!(out, a, "Zero-alloc multiply should match!");
    println!("✅ Zero-allocation API verified.");

    // ---------------------------------------------------------------
    // Also works with other PQ primes
    // ---------------------------------------------------------------
    println!("\n--- Other Post-Quantum Standards ---");

    // ML-KEM (Kyber): q=3329, N=128
    let ctx_kem = Ntt32Context::new(128, 3329);
    let mut d = vec![1u32; 128];
    ctx_kem.forward(&mut d);
    ctx_kem.inverse(&mut d);
    assert!(d.iter().all(|&x| x == 1));
    println!("✅ ML-KEM  (q=3329, N=128): roundtrip OK");

    // Falcon-512: q=12289, N=512
    let ctx_falcon = Ntt32Context::new(512, 12289);
    let mut d = vec![42u32; 512];
    ctx_falcon.forward(&mut d);
    ctx_falcon.inverse(&mut d);
    assert!(d.iter().all(|&x| x == 42));
    println!("✅ Falcon  (q=12289, N=512): roundtrip OK");

    // FHE CRT prime (28-bit)
    let fhe_primes = generate_primes_28(4096, 1);
    let ctx_fhe = Ntt32Context::new(4096, fhe_primes[0]);
    let mut d: Vec<u32> = (0..4096).map(|i| i as u32 % fhe_primes[0]).collect();
    let orig = d.clone();
    ctx_fhe.forward(&mut d);
    ctx_fhe.inverse(&mut d);
    assert_eq!(d, orig);
    println!("✅ FHE CRT (q={}, N=4096): roundtrip OK", fhe_primes[0]);

    println!("\n🎯 VaeaNTT handles ALL NIST PQ standards with a single API.");
}
