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


//! Benchmark: VaeaNTT vs libcrux-ml-kem NEON implementation
//!
//! This benchmark compares our NTT kernel against libcrux's NEON-optimized
//! ML-KEM implementation to answer: "Are we state-of-the-art NEON, or just
//! faster than a non-NEON competitor?"

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

// ---------------------------------------------------------------------------
// 1. VaeaNTT pure NTT kernel (our implementation)
// ---------------------------------------------------------------------------
fn bench_vaea_ntt_kernel(c: &mut Criterion) {
    use vaea_ntt::ntt32::Ntt32Context;

    let mut group = c.benchmark_group("1_ntt_kernel_N256_q3329");
    group.sample_size(300);

    // ML-KEM parameters: q=3329, but N=128 for NTT (q-1=3328, 3328/256=13)
    let q_kem = 3329u32;
    let n_kem = 128;
    let ctx_kem = Ntt32Context::new(n_kem, q_kem);

    // ML-DSA parameters: q=8380417, N=256
    let q_dsa = 8380417u32;
    let n_dsa = 256;
    let ctx_dsa = Ntt32Context::new(n_dsa, q_dsa);

    // Forward NTT kernel only
    group.bench_function("vaea/forward/ML-KEM/N=128", |b| {
        let mut data: Vec<u32> = (0..n_kem).map(|i| (i as u32 * 37 + 1) % q_kem).collect();
        b.iter(|| {
            ctx_kem.forward(&mut data);
        });
    });

    group.bench_function("vaea/forward/ML-DSA/N=256", |b| {
        let mut data: Vec<u32> = (0..n_dsa).map(|i| (i as u32 * 37 + 1) % q_dsa).collect();
        b.iter(|| {
            ctx_dsa.forward(&mut data);
        });
    });

    // Inverse NTT kernel only
    group.bench_function("vaea/inverse/ML-KEM/N=128", |b| {
        let mut data: Vec<u32> = (0..n_kem).map(|i| (i as u32 * 37 + 1) % q_kem).collect();
        ctx_kem.forward(&mut data);
        b.iter(|| {
            ctx_kem.inverse(&mut data);
        });
    });

    group.bench_function("vaea/inverse/ML-DSA/N=256", |b| {
        let mut data: Vec<u32> = (0..n_dsa).map(|i| (i as u32 * 37 + 1) % q_dsa).collect();
        ctx_dsa.forward(&mut data);
        b.iter(|| {
            ctx_dsa.inverse(&mut data);
        });
    });

    // Full negacyclic mul (forward + pointwise + inverse)
    group.bench_function("vaea/negacyclic_mul/ML-KEM/N=128", |b| {
        let a: Vec<u32> = (0..n_kem).map(|i| (i as u32 * 7 + 3) % q_kem).collect();
        let identity = {
            let mut v = vec![0u32; n_kem];
            v[0] = 1;
            v
        };
        b.iter(|| {
            ctx_kem.negacyclic_mul(&a, &identity)
        });
    });

    group.bench_function("vaea/negacyclic_mul/ML-DSA/N=256", |b| {
        let a: Vec<u32> = (0..n_dsa).map(|i| (i as u32 * 7 + 3) % q_dsa).collect();
        let identity = {
            let mut v = vec![0u32; n_dsa];
            v[0] = 1;
            v
        };
        b.iter(|| {
            ctx_dsa.negacyclic_mul(&a, &identity)
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. libcrux-ml-kem: full keygen (NTT-dominated operation)
// ---------------------------------------------------------------------------
fn bench_libcrux_mlkem(c: &mut Criterion) {
    use libcrux_ml_kem::mlkem768;

    let mut group = c.benchmark_group("2_libcrux_mlkem768_ops");
    group.sample_size(200);

    // Key generation (NTT-dominated: 3 NTTs + matrix operations)
    group.bench_function("libcrux/keygen/ML-KEM-768", |b| {
        let mut rng = rand::thread_rng();
        b.iter(|| {
            let mut seed = [0u8; 64];
            rand::RngCore::fill_bytes(&mut rng, &mut seed);
            mlkem768::generate_key_pair(seed)
        });
    });

    // Encapsulation
    group.bench_function("libcrux/encaps/ML-KEM-768", |b| {
        let mut rng = rand::thread_rng();
        let mut seed = [0u8; 64];
        rand::RngCore::fill_bytes(&mut rng, &mut seed);
        let (_, pk) = mlkem768::generate_key_pair(seed);
        b.iter(|| {
            let mut eseed = [0u8; 32];
            rand::RngCore::fill_bytes(&mut rng, &mut eseed);
            mlkem768::encapsulate(&pk, eseed)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_vaea_ntt_kernel, bench_libcrux_mlkem);
criterion_main!(benches);
