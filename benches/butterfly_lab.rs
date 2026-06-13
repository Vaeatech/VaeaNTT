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

#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    dead_code,
    clippy::needless_range_loop
)]
//! # VaeaNTT Butterfly Lab — Recherche d'une invention NEON
//!
//! 4 variantes de butterfly à tester :
//! 1. BASELINE: Shoup actuel (vmull 2-lane + eager reduction)
//! 2. VQDMULH: Shoup via vqdmulhq_s32 (4-lane, 1 instruction)
//! 3. FUSED: vqdmulhq + skip add/sub reduction (batch reduce)
//! 4. RADIX4: Merge 2 stages, amortize memory access

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ===========================================================================
// Approach 1: BASELINE — Current VaeaNTT Shoup (reference)
// ===========================================================================
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn shoup_mul_baseline(
    a: uint32x4_t,
    w: uint32x4_t,
    w_shoup: uint32x4_t,
    q: uint32x4_t,
) -> uint32x4_t {
    // High 32 bits of a * w_shoup (needs 2 vmull for 4 lanes)
    let a_lo = vget_low_u32(a);
    let a_hi = vget_high_u32(a);
    let ws_lo = vget_low_u32(w_shoup);
    let ws_hi = vget_high_u32(w_shoup);

    let prod_lo = vmull_u32(a_lo, ws_lo); // 2 lanes → 64-bit
    let prod_hi = vmull_u32(a_hi, ws_hi); // 2 lanes → 64-bit

    let q_hat = vcombine_u32(vshrn_n_u64::<32>(prod_lo), vshrn_n_u64::<32>(prod_hi));

    // r = a*w - q_hat*q
    let t = vmulq_u32(a, w);
    let qhat_q = vmulq_u32(q_hat, q);
    let r = vsubq_u32(t, qhat_q);

    // Conditional reduce [0, 2q) → [0, q)
    let mask = vcgeq_u32(r, q);
    vsubq_u32(r, vandq_u32(mask, q))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn butterfly_baseline(
    u: uint32x4_t,
    v: uint32x4_t,
    w: uint32x4_t,
    w_shoup: uint32x4_t,
    q: uint32x4_t,
) -> (uint32x4_t, uint32x4_t) {
    let t = shoup_mul_baseline(v, w, w_shoup, q);

    // mod_add: u + t, reduce to [0, q)
    let sum = vaddq_u32(u, t);
    let mask_add = vcgeq_u32(sum, q);
    let u_out = vsubq_u32(sum, vandq_u32(mask_add, q));

    // mod_sub: u - t, keep in [0, q)
    let mask_sub = vcltq_u32(u, t);
    let diff = vaddq_u32(vsubq_u32(u, t), vandq_u32(mask_sub, q));

    (u_out, diff)
}

// ===========================================================================
// Approach 2: VQDMULH — Single-instruction quotient estimate
// ===========================================================================
// Key insight: vqdmulhq_s32(a, b) = floor(2*a*b / 2^32) in 1 NEON instruction
// on ALL 4 lanes simultaneously.
//
// For Shoup: precompute w_qmulh = floor(w * 2^31 / q) instead of floor(w * 2^32 / q)
// Then: q_hat = vqdmulhq_s32(a, w_qmulh) ≈ floor(a*w/q)
//
// For 28-bit q: the lost bit of precision still gives exact quotient.

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn shoup_mul_vqdmulh(
    a: uint32x4_t,
    w: uint32x4_t,
    w_qmulh: int32x4_t, // floor(w * 2^31 / q) — precomputed
    q: uint32x4_t,
) -> uint32x4_t {
    // q_hat ≈ floor(a*w/q) — ONE instruction for 4 lanes!
    let a_signed = vreinterpretq_s32_u32(a);
    let q_hat = vqdmulhq_s32(a_signed, w_qmulh);

    // r = a*w - q_hat*q
    let t = vmulq_u32(a, w);
    let qhat_q = vmulq_u32(vreinterpretq_u32_s32(q_hat), q);
    let r = vsubq_u32(t, qhat_q);

    // Conditional reduce [0, 2q) → [0, q)
    let mask = vcgeq_u32(r, q);
    vsubq_u32(r, vandq_u32(mask, q))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn butterfly_vqdmulh(
    u: uint32x4_t,
    v: uint32x4_t,
    w: uint32x4_t,
    w_qmulh: int32x4_t,
    q: uint32x4_t,
) -> (uint32x4_t, uint32x4_t) {
    let t = shoup_mul_vqdmulh(v, w, w_qmulh, q);

    // Same add/sub as baseline
    let sum = vaddq_u32(u, t);
    let mask_add = vcgeq_u32(sum, q);
    let u_out = vsubq_u32(sum, vandq_u32(mask_add, q));

    let mask_sub = vcltq_u32(u, t);
    let diff = vaddq_u32(vsubq_u32(u, t), vandq_u32(mask_sub, q));

    (u_out, diff)
}

// ===========================================================================
// Approach 3: FUSED — vqdmulh + deferred reduction
// ===========================================================================
// Outputs in [0, 2q) instead of [0, q).
// Skip add/sub conditional reductions entirely.
// Only reduce when feeding into the next shoup_mul (which needs [0, q)).
//
// For 28-bit q: 2q < 2^29 < 2^31, so everything fits in u32.

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn shoup_mul_vqdmulh_lazy(
    a: uint32x4_t, // input in [0, 2q)
    w: uint32x4_t,
    w_qmulh: int32x4_t,
    q: uint32x4_t,
) -> uint32x4_t {
    // First reduce a from [0, 2q) → [0, q) for Shoup precision
    let mask_a = vcgeq_u32(a, q);
    let a_reduced = vsubq_u32(a, vandq_u32(mask_a, q));

    // Then vqdmulh Shoup — 1 instruction for quotient
    let a_signed = vreinterpretq_s32_u32(a_reduced);
    let q_hat = vqdmulhq_s32(a_signed, w_qmulh);

    let t = vmulq_u32(a_reduced, w);
    let qhat_q = vmulq_u32(vreinterpretq_u32_s32(q_hat), q);
    let r = vsubq_u32(t, qhat_q);

    // Reduce to [0, q)
    let mask = vcgeq_u32(r, q);
    vsubq_u32(r, vandq_u32(mask, q))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn butterfly_fused(
    u: uint32x4_t, // input in [0, 2q)
    v: uint32x4_t, // input in [0, 2q)
    w: uint32x4_t,
    w_qmulh: int32x4_t,
    q: uint32x4_t,
    two_q: uint32x4_t,
) -> (uint32x4_t, uint32x4_t) {
    // t = v * w mod q, handling v ∈ [0, 2q)
    let t = shoup_mul_vqdmulh_lazy(v, w, w_qmulh, q);
    // t ∈ [0, q)

    // u + t: u ∈ [0, 2q), t ∈ [0, q) → sum ∈ [0, 3q)
    // Reduce to [0, 2q) with one conditional sub of q
    let sum = vaddq_u32(u, t);
    let mask_add = vcgeq_u32(sum, two_q);
    let u_out = vsubq_u32(sum, vandq_u32(mask_add, q)); // [0, 2q)

    // u - t + q: u ∈ [0, 2q), t ∈ [0, q) → u-t ∈ (-q, 2q) → u-t+q ∈ (0, 3q)
    // Just add q to avoid underflow, then reduce to [0, 2q)
    let diff_raw = vaddq_u32(vsubq_u32(u, t), q);
    let mask_sub = vcgeq_u32(diff_raw, two_q);
    let v_out = vsubq_u32(diff_raw, vandq_u32(mask_sub, q)); // [0, 2q)

    (u_out, v_out) // outputs in [0, 2q)!
}

// ===========================================================================
// Approach 4: SIGNED — work in [-q/2, q/2), no conditional reductions
// ===========================================================================
// The ultimate trick: signed arithmetic eliminates ALL conditional branches
// from add/sub. Only the modular multiply needs reduction.
//
// In [-q/2, q/2): u + t ∈ [-q, q) — fits in i32 for 28-bit q!
// No conditional reduction needed for add/sub at ALL.

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn shoup_mul_signed(
    a: int32x4_t,       // input in [-q, q) (relaxed from [-q/2, q/2))
    w: int32x4_t,       // twiddle factor (signed)
    w_qmulh: int32x4_t, // floor(w * 2^31 / q)
    q: int32x4_t,
) -> int32x4_t {
    // q_hat ≈ floor(a*w/q)
    let q_hat = vqdmulhq_s32(a, w_qmulh);

    // r = a*w - q_hat*q
    let t = vmulq_s32(a, w);
    let qhat_q = vmulq_s32(q_hat, q);
    vsubq_s32(t, qhat_q)
    // Result in approximately [-q, q) — may need correction of ±q
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn butterfly_signed(
    u: int32x4_t, // input in [-q, q)
    v: int32x4_t, // input in [-q, q)
    w: int32x4_t,
    w_qmulh: int32x4_t,
    q: int32x4_t,
) -> (int32x4_t, int32x4_t) {
    // t = v * w mod q — with potential ±q error
    let t_raw = shoup_mul_signed(v, w, w_qmulh, q);

    // Correct t to [-q, q): add q if < -q, sub q if >= q
    let neg_q = vnegq_s32(q);
    let too_low = vcltq_s32(t_raw, neg_q);
    let too_high = vcgeq_s32(t_raw, q);
    let t = vaddq_s32(
        vsubq_s32(t_raw, vandq_s32(vreinterpretq_s32_u32(too_high), q)),
        vandq_s32(vreinterpretq_s32_u32(too_low), q),
    );

    // Add/sub — NO conditional reduction needed!
    // u ∈ [-q, q), t ∈ [-q, q) → u+t ∈ [-2q, 2q), u-t ∈ [-2q, 2q)
    // These fit in i32 for 28-bit q (2q < 2^29 < 2^31)
    let u_out = vaddq_s32(u, t); // Just ADD. No reduction. No branches.
    let v_out = vsubq_s32(u, t); // Just SUB. No reduction. No branches.

    (u_out, v_out)
    // Outputs in [-2q, 2q) — need reduction before next Shoup
    // But can defer for multiple stages!
}

// ===========================================================================
// Benchmarks
// ===========================================================================

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn precompute_w_qmulh(w: u32, q: u32) -> i32 {
    // floor(w * 2^31 / q)
    ((w as u64 * (1u64 << 31)) / q as u64) as i32
}

fn precompute_w_shoup(w: u32, q: u32) -> u32 {
    // floor(w * 2^32 / q)
    ((w as u64 * (1u64 << 32)) / q as u64) as u32
}

#[cfg(target_arch = "aarch64")]
fn bench_butterfly_variants(c: &mut Criterion) {
    use std::arch::aarch64::*;

    let q_val = 8380417u32; // ML-DSA prime
    let w_val = 1234567u32 % q_val;
    let w_shoup_val = precompute_w_shoup(w_val, q_val);
    let w_qmulh_val = precompute_w_qmulh(w_val, q_val);

    // Verify correctness first
    let test_a = 7654321u32 % q_val;
    let expected = ((test_a as u64 * w_val as u64) % q_val as u64) as u32;

    unsafe {
        let a = vdupq_n_u32(test_a);
        let w = vdupq_n_u32(w_val);
        let w_shoup = vdupq_n_u32(w_shoup_val);
        let w_qmulh = vdupq_n_s32(w_qmulh_val);
        let q = vdupq_n_u32(q_val);

        // Verify baseline
        let r1 = shoup_mul_baseline(a, w, w_shoup, q);
        let r1_val = vgetq_lane_u32::<0>(r1);
        assert_eq!(
            r1_val, expected,
            "Baseline Shoup failed: {} != {}",
            r1_val, expected
        );

        // Verify vqdmulh
        let r2 = shoup_mul_vqdmulh(a, w, w_qmulh, q);
        let r2_val = vgetq_lane_u32::<0>(r2);
        assert_eq!(
            r2_val, expected,
            "vqdmulh Shoup failed: {} != {}",
            r2_val, expected
        );

        println!(
            "✅ Both variants produce correct results for a={}, w={}, q={}",
            test_a, w_val, q_val
        );
        println!(
            "   Expected: {}, Baseline: {}, vqdmulh: {}",
            expected, r1_val, r2_val
        );
    }

    let mut group = c.benchmark_group("butterfly_variants");
    group.sample_size(500);

    // Benchmark: isolated modular multiply (the hot path)
    group.bench_function("1_shoup_mul_baseline", |b| unsafe {
        let mut a = vdupq_n_u32(test_a);
        let w = vdupq_n_u32(w_val);
        let w_shoup = vdupq_n_u32(w_shoup_val);
        let q = vdupq_n_u32(q_val);
        b.iter(|| {
            a = shoup_mul_baseline(black_box(a), w, w_shoup, q);
            a
        });
    });

    group.bench_function("2_shoup_mul_vqdmulh", |b| unsafe {
        let mut a = vdupq_n_u32(test_a);
        let w = vdupq_n_u32(w_val);
        let w_qmulh = vdupq_n_s32(w_qmulh_val);
        let q = vdupq_n_u32(q_val);
        b.iter(|| {
            a = shoup_mul_vqdmulh(black_box(a), w, w_qmulh, q);
            a
        });
    });

    // Benchmark: full butterfly
    group.bench_function("3_butterfly_baseline", |b| unsafe {
        let mut u = vdupq_n_u32(test_a);
        let mut v = vdupq_n_u32((test_a * 3) % q_val);
        let w = vdupq_n_u32(w_val);
        let w_shoup = vdupq_n_u32(w_shoup_val);
        let q = vdupq_n_u32(q_val);
        b.iter(|| {
            let (u2, v2) = butterfly_baseline(black_box(u), black_box(v), w, w_shoup, q);
            u = u2;
            v = v2;
            (u, v)
        });
    });

    group.bench_function("4_butterfly_vqdmulh", |b| unsafe {
        let mut u = vdupq_n_u32(test_a);
        let mut v = vdupq_n_u32((test_a * 3) % q_val);
        let w = vdupq_n_u32(w_val);
        let w_qmulh = vdupq_n_s32(w_qmulh_val);
        let q = vdupq_n_u32(q_val);
        b.iter(|| {
            let (u2, v2) = butterfly_vqdmulh(black_box(u), black_box(v), w, w_qmulh, q);
            u = u2;
            v = v2;
            (u, v)
        });
    });

    group.bench_function("5_butterfly_fused", |b| unsafe {
        let mut u = vdupq_n_u32(test_a);
        let mut v = vdupq_n_u32((test_a * 3) % q_val);
        let w = vdupq_n_u32(w_val);
        let w_qmulh = vdupq_n_s32(w_qmulh_val);
        let q = vdupq_n_u32(q_val);
        let two_q = vdupq_n_u32(2 * q_val);
        b.iter(|| {
            let (u2, v2) = butterfly_fused(black_box(u), black_box(v), w, w_qmulh, q, two_q);
            u = u2;
            v = v2;
            (u, v)
        });
    });

    group.bench_function("6_butterfly_signed", |b| unsafe {
        let mut u = vdupq_n_s32(test_a as i32);
        let mut v = vdupq_n_s32(((test_a * 3) % q_val) as i32);
        let w = vdupq_n_s32(w_val as i32);
        let w_qmulh = vdupq_n_s32(w_qmulh_val);
        let q = vdupq_n_s32(q_val as i32);
        b.iter(|| {
            let (u2, v2) = butterfly_signed(black_box(u), black_box(v), w, w_qmulh, q);
            u = u2;
            v = v2;
            (u, v)
        });
    });

    group.finish();
}

#[cfg(not(target_arch = "aarch64"))]
fn bench_butterfly_variants(c: &mut Criterion) {
    eprintln!("This benchmark requires aarch64 (ARM NEON)");
}

criterion_group!(benches, bench_butterfly_variants);
criterion_main!(benches);
