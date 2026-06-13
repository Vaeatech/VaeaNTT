//! # Scalar NTT — Shoup + Harvey Butterfly
//!
//! Scalar (non-SIMD) NTT implementation using Shoup's precomputed quotient
//! trick for division-free modular multiplication, combined with Harvey's
//! lazy butterfly to minimize conditional reductions.
//!
//! All branches are **constant-time** (branchless) using wrapping arithmetic.

use super::arith::{mod_add_28, mod_sub_28};

// ===========================================================================
// Shoup multiplication (precomputed quotient)
// ===========================================================================

/// Precomputes the Shoup quotient for a twiddle factor w.
///
/// `w_shoup = floor(w · 2^32 / q)`
///
/// For w < q < 2^28, we have `w · 2^32 < 2^60` which fits in u64.
#[inline(always)]
pub fn compute_shoup(w: u32, q: u32) -> u32 {
    debug_assert!(w < q, "compute_shoup: w={w} >= q={q}");
    debug_assert!(q < (1u32 << 28), "compute_shoup: q={q} >= 2^28");
    (((w as u64) << 32) / q as u64) as u32
}

/// Branchless Shoup modular multiplication: computes `v × w mod q` without division.
///
/// Uses the precomputed quotient `w_shoup = floor(w · 2^32 / q)` to estimate
/// the division quotient, then corrects with at most one subtraction.
///
/// # Preconditions
/// - `v < q`, `w < q`
/// - `w_shoup = floor(w · 2^32 / q)`
/// - `q < 2^28`
///
/// # Postcondition
/// - result ∈ [0, q)
#[inline(always)]
pub fn shoup_mul(v: u32, w: u32, w_shoup: u32, q: u32) -> u32 {
    // Step 1: full product v × w (< 2^56 for 28-bit)
    let t = v as u64 * w as u64;

    // Step 2: estimate quotient via precomputed Shoup value
    // q_hat ≈ v * w / q — the >> 32 extracts the high word (UMULH on ARM)
    let q_hat = ((v as u64 * w_shoup as u64) >> 32) as u32;

    // Step 3: residue r = t - q_hat * q
    // We work modulo 2^32 (wrapping) since the final result < 2q < 2^29
    let r = (t as u32).wrapping_sub(q_hat.wrapping_mul(q));

    // Branchless correction: at most one subtraction since q_hat error ≤ 1
    let mask = ((r >= q) as u32).wrapping_neg();
    r.wrapping_sub(q & mask)
}

// ===========================================================================
// Harvey Butterfly (lazy reduction)
// ===========================================================================

/// Harvey butterfly for NTT forward (Cooley-Tukey DIT) with lazy reduction.
///
/// # Input
/// - `u, v ∈ [0, 2q)`
/// - `w`: twiddle factor, `w ∈ [0, q)`
/// - `w_shoup`: precomputed Shoup quotient
///
/// # Output
/// - `u', v' ∈ [0, 2q)`
///
/// ```text
/// u' = u + w·v mod q   (lazy, in [0, 2q))
/// v' = u - w·v mod q   (lazy, in [0, 2q))
/// ```
///
/// All reductions are **branchless**.
#[inline(always)]
fn harvey_butterfly_ct(
    u: u32, v: u32,
    w: u32, w_shoup: u32,
    q: u32, two_q: u32,
) -> (u32, u32) {
    // Reduce v from [0, 2q) to [0, q) for Shoup multiplication — branchless
    let v_ge_q = ((v >= q) as u32).wrapping_neg();
    let v_red = v.wrapping_sub(q & v_ge_q);

    // wv = w * v_red mod q ∈ [0, q) — exact Shoup multiplication
    let wv = shoup_mul(v_red, w, w_shoup, q);

    // Lazy addition: u + wv ∈ [0, 3q), reduce to [0, 2q) — branchless
    let u_new = u + wv; // u < 2q, wv < q → u_new < 3q
    let u_ge_2q = ((u_new >= two_q) as u32).wrapping_neg();
    let u_new = u_new.wrapping_sub(two_q & u_ge_2q);

    // Lazy subtraction: u - wv + 2q ∈ (0, 4q), reduce to [0, 2q) — branchless
    let v_new = u + two_q - wv; // always >= 0 since 2q > wv and u >= 0
    let v_ge_2q = ((v_new >= two_q) as u32).wrapping_neg();
    let v_new = v_new.wrapping_sub(two_q & v_ge_2q);

    (u_new, v_new)
}

/// Harvey butterfly for NTT inverse (Gentleman-Sande DIF) with lazy reduction.
///
/// # Input
/// - `u, v ∈ [0, 2q)`
/// - `w_inv`: inverse twiddle factor, `w_inv ∈ [0, q)`
/// - `w_inv_shoup`: precomputed Shoup quotient
///
/// # Output
/// - `u', v' ∈ [0, 2q)`
///
/// ```text
/// u' = u + v            (lazy, in [0, 2q))
/// v' = (u - v) · w_inv  (lazy, in [0, 2q))
/// ```
///
/// All reductions are **branchless**.
#[inline(always)]
fn harvey_butterfly_gs(
    u: u32, v: u32,
    w_inv: u32, w_inv_shoup: u32,
    q: u32, two_q: u32,
) -> (u32, u32) {
    // Lazy addition: u + v ∈ [0, 4q), reduce to [0, 2q) — branchless
    let u_new = u + v;
    let u_ge_2q = ((u_new >= two_q) as u32).wrapping_neg();
    let u_new = u_new.wrapping_sub(two_q & u_ge_2q);

    // Difference: u - v + 2q ∈ (0, 4q), reduce to [0, 2q) — branchless
    let diff = u + two_q - v;
    let d_ge_2q = ((diff >= two_q) as u32).wrapping_neg();
    let diff = diff.wrapping_sub(two_q & d_ge_2q);

    // Reduce diff from [0, 2q) to [0, q) before Shoup — branchless
    let diff_ge_q = ((diff >= q) as u32).wrapping_neg();
    let diff_red = diff.wrapping_sub(q & diff_ge_q);

    // v' = diff * w_inv mod q ∈ [0, q) ⊂ [0, 2q)
    let v_new = shoup_mul(diff_red, w_inv, w_inv_shoup, q);

    (u_new, v_new)
}

// ===========================================================================
// Scalar NTT Forward (Cooley-Tukey DIT) — Shoup exact
// ===========================================================================

/// Scalar NTT forward in-place with Shoup (Cooley-Tukey DIT).
///
/// Uses `shoup_mul` instead of `mod_mul_28` to avoid hardware division.
/// All conditional reductions are branchless.
pub fn ntt_forward_scalar(data: &mut [u32], ctx: &super::context::Ntt32Context) {
    let n = ctx.n;
    let q = ctx.q;
    assert_eq!(
        data.len(), n,
        "Data length ({}) does not match N ({})",
        data.len(), n
    );

    let mut t = n;
    let mut m = 1;

    for _ in 0..ctx.log_n {
        t >>= 1;
        let mut k = 0;

        for i in 0..m {
            let w = ctx.root_powers[m + i];
            let w_shoup = ctx.root_powers_shoup[m + i];

            for j in k..(k + t) {
                let u = data[j];
                let v = shoup_mul(data[j + t], w, w_shoup, q);
                data[j] = mod_add_28(u, v, q);
                data[j + t] = mod_sub_28(u, v, q);
            }
            k += 2 * t;
        }
        m <<= 1;
    }
}

// ===========================================================================
// Scalar NTT Inverse (Gentleman-Sande DIF) — Shoup exact
// ===========================================================================

/// Scalar NTT inverse in-place with Shoup (Gentleman-Sande DIF).
///
/// Uses `shoup_mul` for twiddle multiplications.
/// Includes final normalization by N^{-1} mod q.
/// All conditional reductions are branchless.
pub fn ntt_inverse_scalar(data: &mut [u32], ctx: &super::context::Ntt32Context) {
    let n = ctx.n;
    let q = ctx.q;
    assert_eq!(
        data.len(), n,
        "Data length ({}) does not match N ({})",
        data.len(), n
    );

    let mut t = 1;
    let mut m = n;

    for _ in 0..ctx.log_n {
        m >>= 1;
        let mut k = 0;

        for i in 0..m {
            let w_inv = ctx.inv_root_powers[m + i];
            let w_inv_shoup = ctx.inv_root_powers_shoup[m + i];

            for j in k..(k + t) {
                let u = data[j];
                let v = data[j + t];
                data[j] = mod_add_28(u, v, q);
                let diff = mod_sub_28(u, v, q);
                data[j + t] = shoup_mul(diff, w_inv, w_inv_shoup, q);
            }
            k += 2 * t;
        }
        t <<= 1;
    }

    // Normalization by N^{-1} via Shoup
    let n_inv = ctx.n_inv;
    let n_inv_shoup = ctx.n_inv_shoup;
    for x in data.iter_mut() {
        *x = shoup_mul(*x, n_inv, n_inv_shoup, q);
    }
}

/// NTT inverse (Gentleman-Sande DIF) WITHOUT N^{-1} normalization.
///
/// Identical to [`ntt_inverse_scalar`] but skips the final N^{-1} pass.
/// Output values are scaled by N relative to the true INTT.
pub fn ntt_inverse_scalar_lazy(data: &mut [u32], ctx: &super::context::Ntt32Context) {
    let n = ctx.n;
    let q = ctx.q;
    assert_eq!(
        data.len(), n,
        "Data length ({}) does not match N ({})",
        data.len(), n
    );

    let mut t = 1;
    let mut m = n;

    for _ in 0..ctx.log_n {
        m >>= 1;
        let mut k = 0;

        for i in 0..m {
            let w_inv = ctx.inv_root_powers[m + i];
            let w_inv_shoup = ctx.inv_root_powers_shoup[m + i];

            for j in k..(k + t) {
                let u = data[j];
                let v = data[j + t];
                data[j] = mod_add_28(u, v, q);
                let diff = mod_sub_28(u, v, q);
                data[j + t] = shoup_mul(diff, w_inv, w_inv_shoup, q);
            }
            k += 2 * t;
        }
        t <<= 1;
    }
    // No normalization — caller is responsible
}

// ===========================================================================
// Harvey variants (lazy reduction throughout)
// ===========================================================================

/// NTT forward with Shoup + Harvey lazy butterfly (scalar).
///
/// Intermediate values are kept in [0, 2q) instead of [0, q).
/// This eliminates conditional branches in the butterfly add/sub.
/// A final reduction brings each coefficient back to [0, q).
/// All reductions are branchless.
pub fn forward_harvey(data: &mut [u32], ctx: &super::context::Ntt32Context) {
    let n = ctx.n;
    let q = ctx.q;
    let two_q = ctx.two_q;
    assert_eq!(
        data.len(), n,
        "Data length ({}) does not match N ({})",
        data.len(), n
    );

    let mut t = n;
    let mut m = 1;

    for _ in 0..ctx.log_n {
        t >>= 1;
        let mut k = 0;

        for i in 0..m {
            let w = ctx.root_powers[m + i];
            let w_shoup = ctx.root_powers_shoup[m + i];

            for j in k..(k + t) {
                let (u_new, v_new) = harvey_butterfly_ct(
                    data[j], data[j + t],
                    w, w_shoup,
                    q, two_q,
                );
                data[j] = u_new;
                data[j + t] = v_new;
            }
            k += 2 * t;
        }
        m <<= 1;
    }

    // Final reduction: bring from [0, 2q) to [0, q) — branchless
    for x in data.iter_mut() {
        let mask = ((*x >= q) as u32).wrapping_neg();
        *x = x.wrapping_sub(q & mask);
    }
}

/// NTT inverse with Shoup + Harvey lazy butterfly (scalar).
///
/// Intermediate values are kept in [0, 2q).
/// The final normalization by N^{-1} and reduction are fused.
/// All reductions are branchless.
pub fn inverse_harvey(data: &mut [u32], ctx: &super::context::Ntt32Context) {
    let n = ctx.n;
    let q = ctx.q;
    let two_q = ctx.two_q;
    assert_eq!(
        data.len(), n,
        "Data length ({}) does not match N ({})",
        data.len(), n
    );

    let mut t = 1;
    let mut m = n;

    for _ in 0..ctx.log_n {
        m >>= 1;
        let mut k = 0;

        for i in 0..m {
            let w_inv = ctx.inv_root_powers[m + i];
            let w_inv_shoup = ctx.inv_root_powers_shoup[m + i];

            for j in k..(k + t) {
                let (u_new, v_new) = harvey_butterfly_gs(
                    data[j], data[j + t],
                    w_inv, w_inv_shoup,
                    q, two_q,
                );
                data[j] = u_new;
                data[j + t] = v_new;
            }
            k += 2 * t;
        }
        t <<= 1;
    }

    // Reduce from [0, 2q) to [0, q) + normalize by N^{-1} — branchless
    let n_inv = ctx.n_inv;
    let n_inv_shoup = ctx.n_inv_shoup;
    for x in data.iter_mut() {
        let mask = ((*x >= q) as u32).wrapping_neg();
        *x = x.wrapping_sub(q & mask);
        *x = shoup_mul(*x, n_inv, n_inv_shoup, q);
    }
}

// ===========================================================================
// Pointwise multiplication
// ===========================================================================

/// Pointwise multiplication of two vectors in the NTT domain.
///
/// `result[i] = a[i] · b[i] mod q`
///
/// Since `b` changes every call, Shoup precomputation is not beneficial here;
/// we use direct u64 modular reduction instead.
pub fn ntt_pointwise_mul_scalar(a: &[u32], b: &[u32], result: &mut [u32], q: u32, n: usize) {
    assert_eq!(a.len(), n);
    assert_eq!(b.len(), n);
    assert_eq!(result.len(), n);

    for i in 0..n {
        result[i] = ((a[i] as u64 * b[i] as u64) % q as u64) as u32;
    }
}
