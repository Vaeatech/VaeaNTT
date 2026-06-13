//! Benchmark VaeaNTT with actual NIST Post-Quantum primes.
//!
//! ML-KEM (FIPS 203): q = 3329, NTT size 128 (incomplete NTT to degree-2)
//! ML-DSA (FIPS 204): q = 8380417, NTT size 256
//! Falcon:            q = 12289, NTT size 512/1024

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box};
use vaea_ntt::ntt32::Ntt32Context;

/// Post-quantum NTT parameters
const PQ_PARAMS: &[(&str, usize, u32)] = &[
    // ML-KEM: q=3329, 128-point NTT (Kyber uses incomplete NTT from 256 to 128 pairs)
    // 3329 ≡ 1 (mod 256), so 128-point negacyclic NTT works
    ("ML-KEM/q=3329/N=128", 128, 3329),

    // ML-DSA: q=8380417, 256-point NTT
    // 8380417 ≡ 1 (mod 512), full 256-point negacyclic NTT
    ("ML-DSA/q=8380417/N=256", 256, 8380417),

    // Falcon-512: q=12289, 512-point NTT
    // 12289 ≡ 1 (mod 1024), full 512-point negacyclic NTT
    ("Falcon-512/q=12289/N=512", 512, 12289),

    // Falcon-1024: q=12289, 1024-point NTT
    // 12289 ≡ 1 (mod 2048), full 1024-point negacyclic NTT
    ("Falcon-1024/q=12289/N=1024", 1024, 12289),
];

/// Also benchmark with our best 28-bit prime for comparison
const VAEA_PARAMS: &[(&str, usize)] = &[
    ("VaeaNTT-28bit/N=128", 128),
    ("VaeaNTT-28bit/N=256", 256),
    ("VaeaNTT-28bit/N=512", 512),
    ("VaeaNTT-28bit/N=1024", 1024),
];

fn bench_pq_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_forward_ntt");
    group.sample_size(300);

    // PQ standard primes
    for &(name, n, q) in PQ_PARAMS {
        let ctx = Ntt32Context::new(n, q);
        let data: Vec<u32> = (0..n).map(|i| i as u32 % q).collect();

        group.bench_with_input(BenchmarkId::new("standard", name), &n, |b, _| {
            let mut d = data.clone();
            b.iter(|| {
                ctx.forward(black_box(&mut d));
            });
        });
    }

    // Our best 28-bit primes for same N
    for &(name, n) in VAEA_PARAMS {
        let q = vaea_ntt::ntt32::generate_primes_28(n, 1)[0];
        let ctx = Ntt32Context::new(n, q);
        let data: Vec<u32> = (0..n).map(|i| i as u32 % q).collect();

        group.bench_with_input(BenchmarkId::new("28bit-best", name), &n, |b, _| {
            let mut d = data.clone();
            b.iter(|| {
                ctx.forward(black_box(&mut d));
            });
        });
    }

    group.finish();
}

fn bench_pq_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_inverse_ntt");
    group.sample_size(300);

    for &(name, n, q) in PQ_PARAMS {
        let ctx = Ntt32Context::new(n, q);
        let mut data: Vec<u32> = (0..n).map(|i| i as u32 % q).collect();
        ctx.forward(&mut data);

        group.bench_with_input(BenchmarkId::new("standard", name), &n, |b, _| {
            let mut d = data.clone();
            b.iter(|| {
                ctx.inverse(black_box(&mut d));
            });
        });
    }

    group.finish();
}

fn bench_pq_negacyclic_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_negacyclic_mul");
    group.sample_size(200);

    for &(name, n, q) in PQ_PARAMS {
        let ctx = Ntt32Context::new(n, q);
        let a: Vec<u32> = (0..n).map(|i| ((i as u64 * 17 + 3) % q as u64) as u32).collect();
        let b: Vec<u32> = (0..n).map(|i| ((i as u64 * 31 + 7) % q as u64) as u32).collect();
        let mut a_buf = a.clone();
        let mut b_buf = b.clone();
        let mut result = vec![0u32; n];

        group.bench_with_input(BenchmarkId::new("zero-alloc", name), &n, |bench, _| {
            bench.iter(|| {
                a_buf.copy_from_slice(&a);
                b_buf.copy_from_slice(&b);
                ctx.negacyclic_mul_into(
                    black_box(&mut a_buf),
                    black_box(&mut b_buf),
                    black_box(&mut result),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_pq_forward, bench_pq_inverse, bench_pq_negacyclic_mul);
criterion_main!(benches);
