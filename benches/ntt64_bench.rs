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


use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vaea_ntt::ntt64::{generate_primes_60, Ntt64Arith, Ntt64Context, PRIME_SEAL};

fn bench_forward_64(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt64_forward");
    for &n in &[64, 256, 1024, 4096, 8192, 32768] {
        let primes = generate_primes_60(n, 60, 1);
        let arith = Ntt64Arith::new(primes[0]);
        let ctx = Ntt64Context::new(n, arith);
        let data_orig: Vec<u64> = (0..n)
            .map(|i| ((i as u128 * 314159 + 271828) % primes[0] as u128) as u64)
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut data = data_orig.clone();
            b.iter(|| {
                ctx.forward(black_box(&mut data));
            });
        });
    }
    group.finish();
}

fn bench_inverse_64(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt64_inverse");
    for &n in &[64, 256, 1024, 4096, 8192, 32768] {
        let primes = generate_primes_60(n, 60, 1);
        let arith = Ntt64Arith::new(primes[0]);
        let ctx = Ntt64Context::new(n, arith);
        let mut data: Vec<u64> = (0..n)
            .map(|i| ((i as u128 * 314159 + 271828) % primes[0] as u128) as u64)
            .collect();
        ctx.forward(&mut data);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut d = data.clone();
            b.iter(|| {
                ctx.inverse(black_box(&mut d));
            });
        });
    }
    group.finish();
}

fn bench_tiled_64(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt64_forward_tiled");
    for &n in &[1024, 4096, 8192, 32768] {
        let primes = generate_primes_60(n, 60, 1);
        let arith = Ntt64Arith::new(primes[0]);
        let ctx = Ntt64Context::new(n, arith);
        let data_orig: Vec<u64> = (0..n)
            .map(|i| ((i as u128 * 314159 + 271828) % primes[0] as u128) as u64)
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut data = data_orig.clone();
            b.iter(|| {
                ctx.forward_tiled(black_box(&mut data));
            });
        });
    }
    group.finish();
}

fn bench_negacyclic_mul_64(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt64_negacyclic_mul");
    for &n in &[256, 1024, 4096, 8192] {
        let primes = generate_primes_60(n, 60, 1);
        let arith = Ntt64Arith::new(primes[0]);
        let ctx = Ntt64Context::new(n, arith);
        let a: Vec<u64> = (0..n)
            .map(|i| ((i as u128 * 17 + 3) % primes[0] as u128) as u64)
            .collect();
        let b_data: Vec<u64> = (0..n)
            .map(|i| ((i as u128 * 31 + 11) % primes[0] as u128) as u64)
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                black_box(ctx.negacyclic_mul(black_box(&a), black_box(&b_data)));
            });
        });
    }
    group.finish();
}

fn bench_seal_prime(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt64_seal_prime");
    let arith = Ntt64Arith::new(PRIME_SEAL);
    for &n in &[1024, 4096, 32768] {
        let ctx = Ntt64Context::new(n, arith.clone());
        let data_orig: Vec<u64> = (0..n)
            .map(|i| ((i as u128 * 314159) % PRIME_SEAL as u128) as u64)
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut data = data_orig.clone();
            b.iter(|| {
                ctx.forward(black_box(&mut data));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_forward_64,
    bench_inverse_64,
    bench_tiled_64,
    bench_negacyclic_mul_64,
    bench_seal_prime,
);
criterion_main!(benches);
