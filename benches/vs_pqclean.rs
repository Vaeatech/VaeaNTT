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


//! # VaeaNTT vs mlkem-native (PQCP) — Definitive NTT Benchmark
//!
//! Compares VaeaNTT NEON intrinsics against:
//! - mlkem-native: SLOTHY-superoptimized aarch64 assembly (THE state of the art)
//! - PQClean: hand-written aarch64 assembly (predecessor)
//!
//! Parameters: q=3329, N=256 (Kyber/ML-KEM)
//!
//! mlkem-native uses int16_t (8 elements/NEON register, 7 NTT stages)
//! VaeaNTT uses u32 (4 elements/NEON register, 8 NTT stages)

use criterion::{criterion_group, criterion_main, Criterion, black_box};
use vaea_ntt::ntt32::Ntt32Context;

// PQClean aarch64 assembly (predecessor)
extern "C" {
    fn PQCLEAN_MLKEM768_AARCH64_ntt(r: *mut i16);
    fn PQCLEAN_MLKEM768_AARCH64_invntt(r: *mut i16);
}

// mlkem-native SLOTHY-optimized assembly (current state of the art)
extern "C" {
    fn mlkem_native_ntt(p: *mut i16);
    fn mlkem_native_intt(p: *mut i16);
}

fn bench_vs_reference(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_N256_reference");
    group.sample_size(500);

    // =========================================================================
    // 1. mlkem-native (SLOTHY-superoptimized assembly) — THE reference
    // =========================================================================
    {
        let mut poly = [0i16; 256];
        for i in 0..256 { poly[i] = (i as i16 * 17 + 5) % 3329; }

        group.bench_function("1_mlkem-native-SLOTHY/forward", |b| {
            b.iter(|| {
                unsafe { mlkem_native_ntt(black_box(poly.as_mut_ptr())) };
                black_box(&poly);
            })
        });

        group.bench_function("1_mlkem-native-SLOTHY/inverse", |b| {
            b.iter(|| {
                unsafe { mlkem_native_intt(black_box(poly.as_mut_ptr())) };
                black_box(&poly);
            })
        });
    }

    // =========================================================================
    // 2. PQClean aarch64 assembly (predecessor)
    // =========================================================================
    {
        let mut poly = [0i16; 256];
        for i in 0..256 { poly[i] = (i as i16 * 17 + 5) % 3329; }

        group.bench_function("2_pqclean-asm/forward", |b| {
            b.iter(|| {
                unsafe { PQCLEAN_MLKEM768_AARCH64_ntt(black_box(poly.as_mut_ptr())) };
                black_box(&poly);
            })
        });
    }

    // =========================================================================
    // 3. VaeaNTT — Falcon prime (q=12289, closest to Kyber's small prime)
    // =========================================================================
    {
        let ctx = Ntt32Context::new(256, 12289);
        let mut poly = vec![0u32; 256];
        for i in 0..256 { poly[i] = ((i as u64 * 17 + 5) % 12289) as u32; }

        group.bench_function("3_vaea-ntt-u32-q12289/forward", |b| {
            b.iter(|| {
                ctx.forward(black_box(&mut poly));
                black_box(&poly);
            })
        });

        group.bench_function("3_vaea-ntt-u32-q12289/inverse", |b| {
            b.iter(|| {
                ctx.inverse(black_box(&mut poly));
                black_box(&poly);
            })
        });
    }

    // =========================================================================
    // 4. VaeaNTT — ML-DSA prime (q=8380417, 23-bit)
    // =========================================================================
    {
        let ctx = Ntt32Context::new(256, 8380417);
        let mut poly = vec![0u32; 256];
        for i in 0..256 { poly[i] = ((i as u64 * 17 + 5) % 8380417) as u32; }

        group.bench_function("4_vaea-ntt-u32-qMLDSA/forward", |b| {
            b.iter(|| {
                ctx.forward(black_box(&mut poly));
                black_box(&poly);
            })
        });
    }

    // =========================================================================
    // 5. VaeaNTT — 28-bit prime (production FHE/ZK config)
    // =========================================================================
    {
        let q = vaea_ntt::ntt32::generate_primes_28(256, 1)[0];
        let ctx = Ntt32Context::new(256, q);
        let mut poly = vec![0u32; 256];
        for i in 0..256 { poly[i] = ((i as u64 * 17 + 5) % q as u64) as u32; }

        group.bench_function("5_vaea-ntt-u32-q28bit/forward", |b| {
            b.iter(|| {
                ctx.forward(black_box(&mut poly));
                black_box(&poly);
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_vs_reference);
criterion_main!(benches);
