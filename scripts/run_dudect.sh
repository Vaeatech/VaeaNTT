#!/bin/bash
# =============================================================================
# DudeCT constant-time analysis for VaeaNTT
# =============================================================================
#
# This script runs the constant-time verification tests using the DudeCT
# methodology (Welch's t-test on timing distributions).
#
# Two modes are available:
#   1. Automated test (bounded, ~2-3 min): runs as #[ignore] test
#   2. Interactive dudect (infinite loop): runs the dudect-bencher example
#
# Usage:
#   ./scripts/run_dudect.sh          # Run the automated bounded test
#   ./scripts/run_dudect.sh --full   # Run the full interactive dudect example
# =============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       VaeaNTT — DudeCT Constant-Time Verification          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ "$1" = "--full" ]; then
    echo "Mode: Interactive DudeCT (dudect-bencher)"
    echo "  Press Ctrl+C to stop and see results."
    echo ""
    echo "Running: cargo run --release --example dudect_ntt"
    echo "---"
    cargo run --release --example dudect_ntt
else
    echo "Mode: Automated bounded test (~2-3 minutes)"
    echo "  Testing: forward(), inverse(), negacyclic_mul_into()"
    echo "  Prime: q = 8380417 (ML-DSA/FIPS 204), N = 256"
    echo "  Measurements: 500K per test, threshold |t| < 4.5"
    echo ""
    echo "Running: cargo test --release --test constant_time -- --ignored"
    echo "---"
    cargo test --release --test constant_time -- --ignored --nocapture
    echo ""
    echo "---"
    echo "✅ All constant-time tests passed!"
fi
