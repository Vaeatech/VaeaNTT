use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box};
use vaea_ntt::ntt32::{Ntt32Context, generate_primes_28};

fn bench_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt32_forward");
    for &n in &[64, 256, 1024, 4096, 8192, 16384, 32768] {
        let q = generate_primes_28(n, 1)[0];
        let ctx = Ntt32Context::new(n, q);
        let data_orig: Vec<u32> = (0..n).map(|i| ((i as u64 * 41 + 7) % q as u64) as u32).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut data = data_orig.clone();
            b.iter(|| {
                ctx.forward(black_box(&mut data));
            });
        });
    }
    group.finish();
}

fn bench_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt32_inverse");
    for &n in &[64, 256, 1024, 4096, 8192, 16384, 32768] {
        let q = generate_primes_28(n, 1)[0];
        let ctx = Ntt32Context::new(n, q);
        let mut data: Vec<u32> = (0..n).map(|i| ((i as u64 * 41 + 7) % q as u64) as u32).collect();
        ctx.forward(&mut data); // start in NTT domain

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut d = data.clone();
            b.iter(|| {
                ctx.inverse(black_box(&mut d));
            });
        });
    }
    group.finish();
}

fn bench_inverse_lazy(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt32_inverse_lazy");
    for &n in &[256, 1024, 4096, 8192, 32768] {
        let q = generate_primes_28(n, 1)[0];
        let ctx = Ntt32Context::new(n, q);
        let mut data: Vec<u32> = (0..n).map(|i| ((i as u64 * 41 + 7) % q as u64) as u32).collect();
        ctx.forward(&mut data);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut d = data.clone();
            b.iter(|| {
                ctx.inverse_lazy(black_box(&mut d));
            });
        });
    }
    group.finish();
}

fn bench_negacyclic_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt32_negacyclic_mul");
    for &n in &[256, 1024, 4096, 8192, 32768] {
        let q = generate_primes_28(n, 1)[0];
        let ctx = Ntt32Context::new(n, q);
        let a: Vec<u32> = (0..n).map(|i| ((i as u64 * 17 + 3) % q as u64) as u32).collect();
        let b: Vec<u32> = (0..n).map(|i| ((i as u64 * 31 + 11) % q as u64) as u32).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                black_box(ctx.negacyclic_mul(black_box(&a), black_box(&b)));
            });
        });
    }
    group.finish();
}

fn bench_negacyclic_mul_zero_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt32_negacyclic_mul_zero_alloc");
    for &n in &[256, 1024, 4096, 8192, 32768] {
        let q = generate_primes_28(n, 1)[0];
        let ctx = Ntt32Context::new(n, q);
        let a_orig: Vec<u32> = (0..n).map(|i| ((i as u64 * 17 + 3) % q as u64) as u32).collect();
        let b_orig: Vec<u32> = (0..n).map(|i| ((i as u64 * 31 + 11) % q as u64) as u32).collect();

        // Pre-allocate all buffers outside the loop
        let mut a_buf = a_orig.clone();
        let mut b_buf = b_orig.clone();
        let mut result = vec![0u32; n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                // Refill from originals (simulates real usage where data changes)
                a_buf.copy_from_slice(&a_orig);
                b_buf.copy_from_slice(&b_orig);
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

fn bench_pointwise_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt32_pointwise_mul");
    for &n in &[1024, 4096, 8192, 32768] {
        let q = generate_primes_28(n, 1)[0];
        let ctx = Ntt32Context::new(n, q);
        let a: Vec<u32> = (0..n).map(|i| ((i as u64 * 17 + 3) % q as u64) as u32).collect();
        let b: Vec<u32> = (0..n).map(|i| ((i as u64 * 31 + 11) % q as u64) as u32).collect();
        let mut result = vec![0u32; n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                ctx.pointwise_mul(black_box(&a), black_box(&b), black_box(&mut result));
            });
        });
    }
    group.finish();
}

fn bench_context_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt32_context_creation");
    for &n in &[256, 1024, 4096, 32768] {
        let q = generate_primes_28(n, 1)[0];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                black_box(Ntt32Context::new(n, q));
            });
        });
    }
    group.finish();
}

criterion_group!(benches,
    bench_forward,
    bench_inverse,
    bench_inverse_lazy,
    bench_negacyclic_mul,
    bench_negacyclic_mul_zero_alloc,
    bench_pointwise_mul,
    bench_context_creation,
);
criterion_main!(benches);
