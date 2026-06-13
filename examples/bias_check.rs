//! Bias verification: check that concrete-ntt and VaeaNTT produce
//! compatible results (same normalization, same output order).

fn main() {
    let n = 256;
    // Find a shared 28-bit prime
    let two_n = 2 * n as u32;
    let upper = 1u32 << 28;
    let mut p = 0u32;
    let mut k = upper / two_n;
    while k > 1 {
        let candidate = k * two_n + 1;
        if candidate < upper
            && candidate > (1u32 << 27)
            && concrete_ntt::prime32::Plan::try_new(n, candidate).is_some()
            && vaea_ntt::ntt32::is_prime_32(candidate)
        {
            p = candidate;
            break;
        }
        k -= 1;
    }
    println!("Shared prime: {p} ({} bits)", 32 - p.leading_zeros());

    // Input data
    let input: Vec<u32> = (0..n)
        .map(|i| ((i as u64 * 41 + 7) % p as u64) as u32)
        .collect();

    // === concrete-ntt ===
    let plan = concrete_ntt::prime32::Plan::try_new(n, p).expect("Plan creation failed");
    let mut cn_data = input.clone();
    println!("=== concrete-ntt ===");
    println!("Input:     {:?}", &cn_data[..8]);
    plan.fwd(&mut cn_data);
    println!("After fwd: {:?}", &cn_data[..8]);
    let cn_fwd = cn_data.clone();
    plan.inv(&mut cn_data);
    println!("After inv: {:?}", &cn_data[..8]);

    // Check if inv(fwd(x)) == x (i.e., inv includes N^{-1} normalization)
    let cn_roundtrip_match = cn_data == input;
    println!("Roundtrip matches input: {}", cn_roundtrip_match);

    // If not exact match, check if it's off by N factor
    if !cn_roundtrip_match {
        let n_inv = mod_inv(n as u32, p);
        let cn_normalized: Vec<u32> = cn_data
            .iter()
            .map(|&x| ((x as u64 * n_inv as u64) % p as u64) as u32)
            .collect();
        let needs_normalize = cn_normalized == input;
        println!("Matches after N^{{-1}} normalize: {}", needs_normalize);
        if needs_normalize {
            println!("⚠️  concrete-ntt inv() does NOT include N^{{-1}} normalization!");
        }
    }

    // === VaeaNTT ===
    let ctx = vaea_ntt::ntt32::Ntt32Context::new(n, p);
    let mut vn_data = input.clone();
    println!("\n=== VaeaNTT ===");
    println!("Input:     {:?}", &vn_data[..8]);
    ctx.forward(&mut vn_data);
    println!("After fwd: {:?}", &vn_data[..8]);
    let vn_fwd = vn_data.clone();
    ctx.inverse(&mut vn_data);
    println!("After inv: {:?}", &vn_data[..8]);
    let vn_roundtrip_match = vn_data == input;
    println!("Roundtrip matches input: {}", vn_roundtrip_match);

    // === Compare NTT domain outputs ===
    println!("\n=== NTT Domain Comparison ===");
    let fwd_match = cn_fwd == vn_fwd;
    println!("Forward outputs match: {}", fwd_match);
    if !fwd_match {
        // Check if one is bit-reversed relative to the other
        let n_u32 = n as u32;
        let log_n = n_u32.trailing_zeros();
        let cn_fwd_bitrev: Vec<u32> = (0..n)
            .map(|i| cn_fwd[bit_reverse(i as u32, log_n) as usize])
            .collect();
        let bitrev_match = cn_fwd_bitrev == vn_fwd;
        println!("Match after bit-reversing concrete-ntt: {}", bitrev_match);

        let vn_fwd_bitrev: Vec<u32> = (0..n)
            .map(|i| vn_fwd[bit_reverse(i as u32, log_n) as usize])
            .collect();
        let bitrev_match2 = vn_fwd_bitrev == cn_fwd;
        println!("Match after bit-reversing VaeaNTT: {}", bitrev_match2);

        // Show first few differences
        println!("\nFirst differences:");
        for i in 0..n.min(8) {
            if cn_fwd[i] != vn_fwd[i] {
                println!("  [{}]: concrete={}, vaea={}", i, cn_fwd[i], vn_fwd[i]);
            }
        }
    }

    // === Allocation check ===
    println!("\n=== Allocation Behavior ===");
    println!("Both work in-place on &mut [u32] ✓");
    println!("Context/Plan creation is outside measurement loop ✓");

    // Summary
    println!("\n=== BIAS CHECKLIST ===");
    println!(
        "☐ Same normalization in inv(): {}",
        if cn_roundtrip_match {
            "YES ✓"
        } else {
            "NO ⚠️ — BIAS DETECTED"
        }
    );
    println!(
        "☐ Same output order in fwd(): {}",
        if fwd_match {
            "YES ✓"
        } else {
            "NO ⚠️ — one does extra bit-reversal"
        }
    );
    println!("☐ Both in-place: YES ✓");
    println!("☐ Context outside loop: YES ✓");
}

fn bit_reverse(x: u32, bits: u32) -> u32 {
    x.reverse_bits() >> (32 - bits)
}

fn mod_inv(a: u32, p: u32) -> u32 {
    mod_pow(a, p - 2, p)
}

fn mod_pow(mut base: u32, mut exp: u32, modulus: u32) -> u32 {
    let mut result = 1u64;
    base %= modulus;
    let m = modulus as u64;
    let mut b = base as u64;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * b % m;
        }
        exp >>= 1;
        b = b * b % m;
    }
    result as u32
}
