"""
NIST SP 800-22 Randomness Tests (subset)
=========================================

Tests implemented:
  1. Frequency (Monobit) Test
  2. Block Frequency Test
  3. Runs Test
  4. Longest Run of Ones in a Block
  5. Serial Test
  6. Approximate Entropy Test
  7. Cumulative Sums Test
  8. Discrete Fourier Transform (Spectral) Test

Each test returns (statistic, p_value, pass/fail at alpha=0.01).

Usage:
    python nist_tests.py [--source {ode,nkg,both}]

Outputs:
    results/nist_report.json
    results/nist_report.txt
"""

import json
import math
import os
import sys

import numpy as np
from scipy.special import gammaincc, erfc
from scipy.fft import fft

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3   = os.path.join(_HERE, "..", "lsdpak-v3-rgb")
_RES  = os.path.join(_HERE, "results")
os.makedirs(_RES, exist_ok=True)
sys.path.insert(0, _V3)
sys.path.insert(0, _HERE)

ALPHA = 0.01


# =============================================================================
# BIT SEQUENCE HELPERS
# =============================================================================

def bytes_to_bits(data: np.ndarray) -> np.ndarray:
    """Convert uint8 array to binary {0,1} array."""
    return np.unpackbits(data.ravel().astype(np.uint8))


# =============================================================================
# TEST 1: FREQUENCY (MONOBIT) TEST
# =============================================================================

def frequency_test(bits: np.ndarray) -> dict:
    n = len(bits)
    s = np.sum(2 * bits.astype(int) - 1)
    s_obs = abs(s) / math.sqrt(n)
    p_val = erfc(s_obs / math.sqrt(2))
    return {"test": "Frequency (Monobit)", "statistic": float(s_obs),
            "p_value": float(p_val), "pass": bool(p_val >= ALPHA)}


# =============================================================================
# TEST 2: BLOCK FREQUENCY TEST
# =============================================================================

def block_frequency_test(bits: np.ndarray, M: int = 128) -> dict:
    n = len(bits)
    N = n // M
    if N == 0:
        return {"test": "Block Frequency", "statistic": 0, "p_value": 0, "pass": False}
    blocks = bits[:N * M].reshape(N, M)
    proportions = blocks.mean(axis=1)
    chi2 = 4 * M * np.sum((proportions - 0.5) ** 2)
    p_val = float(gammaincc(N / 2.0, chi2 / 2.0))
    return {"test": "Block Frequency", "statistic": float(chi2),
            "p_value": p_val, "pass": bool(p_val >= ALPHA)}


# =============================================================================
# TEST 3: RUNS TEST
# =============================================================================

def runs_test(bits: np.ndarray) -> dict:
    n = len(bits)
    pi = bits.mean()
    if abs(pi - 0.5) >= (2.0 / math.sqrt(n)):
        return {"test": "Runs", "statistic": 0, "p_value": 0.0, "pass": False}
    v_obs = 1 + np.sum(bits[:-1] != bits[1:])
    num = abs(v_obs - 2 * n * pi * (1 - pi))
    den = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    p_val = float(erfc(num / den))
    return {"test": "Runs", "statistic": float(v_obs),
            "p_value": p_val, "pass": bool(p_val >= ALPHA)}


# =============================================================================
# TEST 4: LONGEST RUN OF ONES IN A BLOCK
# =============================================================================

def longest_run_test(bits: np.ndarray) -> dict:
    n = len(bits)
    if n < 6272:
        M, K = 8, 3
        v_values = [1, 2, 3, 4]
        pi_k = [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        M, K = 128, 5
        v_values = [4, 5, 6, 7, 8, 9]
        pi_k = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        M, K = 10000, 6
        v_values = [10, 11, 12, 13, 14, 15, 16]
        pi_k = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    N = n // M
    if N == 0:
        return {"test": "Longest Run", "statistic": 0, "p_value": 0, "pass": False}
    blocks = bits[:N * M].reshape(N, M)

    # Find longest run of ones in each block
    longest = np.zeros(N, dtype=int)
    for i in range(N):
        max_run = cur_run = 0
        for b in blocks[i]:
            if b == 1:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        longest[i] = max_run

    # Count frequencies in each category
    v_freq = np.zeros(len(v_values))
    for i, lr in enumerate(longest):
        if lr <= v_values[0]:
            v_freq[0] += 1
        elif lr >= v_values[-1]:
            v_freq[-1] += 1
        else:
            for j in range(1, len(v_values)):
                if lr == v_values[j]:
                    v_freq[j] += 1
                    break

    chi2 = sum((v_freq[i] - N * pi_k[i]) ** 2 / (N * pi_k[i])
               for i in range(len(pi_k)))
    p_val = float(gammaincc(K / 2.0, chi2 / 2.0))
    return {"test": "Longest Run of Ones", "statistic": float(chi2),
            "p_value": p_val, "pass": bool(p_val >= ALPHA)}


# =============================================================================
# TEST 5: SERIAL TEST
# =============================================================================

def serial_test(bits: np.ndarray, m: int = 4) -> dict:
    n = len(bits)
    augmented = np.concatenate([bits, bits[:m-1]])

    def psi_sq(mm):
        if mm == 0:
            return 0
        counts = {}
        for i in range(n):
            pattern = tuple(augmented[i:i+mm])
            counts[pattern] = counts.get(pattern, 0) + 1
        return (2**mm / n) * sum(v**2 for v in counts.values()) - n

    psi_m   = psi_sq(m)
    psi_m1  = psi_sq(m - 1)
    psi_m2  = psi_sq(m - 2) if m >= 2 else 0

    delta1 = psi_m - psi_m1
    delta2 = psi_m - 2 * psi_m1 + psi_m2

    p1 = float(gammaincc(2**(m-2), delta1 / 2.0))
    p2 = float(gammaincc(2**(m-3), delta2 / 2.0)) if m >= 3 else 1.0

    return {"test": "Serial", "statistic": float(delta1),
            "p_value": p1, "p_value2": p2,
            "pass": bool(p1 >= ALPHA and p2 >= ALPHA)}


# =============================================================================
# TEST 6: APPROXIMATE ENTROPY TEST
# =============================================================================

def approximate_entropy_test(bits: np.ndarray, m: int = 4) -> dict:
    n = len(bits)

    def phi(mm):
        if mm == 0:
            return 0
        augmented = np.concatenate([bits, bits[:mm-1]])
        counts = {}
        for i in range(n):
            pattern = tuple(augmented[i:i+mm])
            counts[pattern] = counts.get(pattern, 0) + 1
        total = sum(counts.values())
        return sum((c / total) * math.log(c / total) for c in counts.values())

    phi_m  = phi(m)
    phi_m1 = phi(m + 1)
    apen   = phi_m - phi_m1
    chi2   = 2 * n * (math.log(2) - apen)
    p_val  = float(gammaincc(2**(m-1), chi2 / 2.0))

    return {"test": "Approximate Entropy", "statistic": float(chi2),
            "p_value": p_val, "pass": bool(p_val >= ALPHA)}


# =============================================================================
# TEST 7: CUMULATIVE SUMS TEST
# =============================================================================

def cumulative_sums_test(bits: np.ndarray) -> dict:
    n = len(bits)
    x = 2 * bits.astype(int) - 1
    S_fwd = np.cumsum(x)
    S_bwd = np.cumsum(x[::-1])

    z_fwd = float(np.max(np.abs(S_fwd)))
    z_bwd = float(np.max(np.abs(S_bwd)))

    def p_value(z):
        sm = 0.0
        sqn = math.sqrt(n)
        start = int((-n / z + 1) / 4)
        end   = int((n / z - 1) / 4) + 1
        for k in range(start, end + 1):
            sm += (erfc(((4*k+1)*z) / (math.sqrt(2*n)))
                 - erfc(((4*k-1)*z) / (math.sqrt(2*n))))
        return max(0.0, min(1.0, 1.0 - sm))

    p_fwd = p_value(z_fwd)
    p_bwd = p_value(z_bwd)

    return {"test": "Cumulative Sums",
            "stat_fwd": z_fwd, "stat_bwd": z_bwd,
            "p_fwd": p_fwd, "p_bwd": p_bwd,
            "pass": bool(p_fwd >= ALPHA and p_bwd >= ALPHA)}


# =============================================================================
# TEST 8: DISCRETE FOURIER TRANSFORM (SPECTRAL) TEST
# =============================================================================

def dft_test(bits: np.ndarray) -> dict:
    n = len(bits)
    x = 2 * bits.astype(float) - 1
    S = np.abs(fft(x))[:n // 2]
    T = math.sqrt(2.995732274 * n)   # threshold = sqrt(log(1/0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = float(np.sum(S < T))
    d  = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    p_val = float(erfc(abs(d) / math.sqrt(2)))
    return {"test": "DFT (Spectral)", "statistic": float(d),
            "p_value": p_val, "pass": bool(p_val >= ALPHA)}


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests(data: np.ndarray, label: str = "") -> list:
    bits = bytes_to_bits(data)
    n = len(bits)
    print(f"  [{label}] {n} bits ({n//8} bytes)")

    # Use subset for expensive tests
    if n > 1_000_000:
        bits_short = bits[:1_000_000]
    else:
        bits_short = bits

    results = []
    tests = [
        ("Frequency",        lambda: frequency_test(bits)),
        ("Block Frequency",  lambda: block_frequency_test(bits)),
        ("Runs",             lambda: runs_test(bits)),
        ("Longest Run",      lambda: longest_run_test(bits)),
        ("Serial",           lambda: serial_test(bits_short, m=4)),
        ("Approx. Entropy",  lambda: approximate_entropy_test(bits_short, m=4)),
        ("Cumulative Sums",  lambda: cumulative_sums_test(bits)),
        ("DFT (Spectral)",   lambda: dft_test(bits)),
    ]

    for name, fn in tests:
        try:
            r = fn()
            results.append(r)
            status = "PASS" if r["pass"] else "FAIL"
            p = r.get("p_value", r.get("p_fwd", 0))
            print(f"    {name:25s}  p={p:.6f}  [{status}]")
        except Exception as e:
            results.append({"test": name, "error": str(e), "pass": False})
            print(f"    {name:25s}  [ERROR: {e}]")

    passed = sum(1 for r in results if r["pass"])
    print(f"    => {passed}/{len(results)} tests passed")
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    from lsdpak_v3_rgb import (
        load_image_rgb, adaptive_ics_rgb, build_keystream,
        image_hash_bytes_rgb, IMG_SIZE,
    )

    print("\nNIST SP 800-22 Randomness Tests")
    print("=" * 60)

    # Load first image
    img = load_image_rgb("aerial")
    hb  = image_hash_bytes_rgb(img)
    ics = adaptive_ics_rgb(img)

    report = {}

    # --- ODE keystream (Lorenz + Rossler) ---
    print("\n--- ODE Keystream (4D Lorenz + Rossler) ---")
    ode_keys = []
    for c in range(3):
        ks = build_keystream(ics[c], IMG_SIZE)
        ode_keys.append(ks)
    ode_flat = np.concatenate([k.ravel() for k in ode_keys])
    report["ODE"] = run_all_tests(ode_flat, "ODE (aerial, 3ch)")

    # --- NKG keystream ---
    try:
        import torch
        from lsdpak_dl import NKG, DEVICE
        _CKPT = os.path.join(_HERE, "checkpoints")

        nkg_path = os.path.join(_CKPT, "nkg_gan.pt")
        if not os.path.exists(nkg_path):
            nkg_path = os.path.join(_CKPT, "nkg_pretrained.pt")

        if os.path.exists(nkg_path):
            print(f"\n--- NKG Keystream (from {os.path.basename(nkg_path)}) ---")
            nkg = NKG().to(DEVICE)
            nkg.load_state_dict(torch.load(nkg_path, map_location=DEVICE))
            nkg.eval()

            nkg_keys = nkg.generate_channel_keystreams(hb, IMG_SIZE)
            nkg_flat = np.concatenate([k.ravel() for k in nkg_keys])
            report["NKG"] = run_all_tests(nkg_flat, "NKG (aerial, 3ch)")
        else:
            print("\n[INFO] No NKG checkpoint found. Run train.py first.")
    except ImportError:
        print("\n[INFO] PyTorch not available. Skipping NKG tests.")

    # --- Random baseline (control) ---
    print("\n--- Random Baseline (np.random) ---")
    rand_data = np.random.randint(0, 256, size=3*IMG_SIZE*IMG_SIZE, dtype=np.uint8)
    report["Random"] = run_all_tests(rand_data, "Random baseline")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for src, results in report.items():
        passed = sum(1 for r in results if r["pass"])
        total  = len(results)
        print(f"  {src:20s}  {passed}/{total} passed")

    # Save
    out_json = os.path.join(_RES, "nist_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved -> {out_json}")

    out_txt = os.path.join(_RES, "nist_report.txt")
    with open(out_txt, "w") as f:
        for src, results in report.items():
            f.write(f"\n=== {src} ===\n")
            for r in results:
                p = r.get("p_value", r.get("p_fwd", "N/A"))
                status = "PASS" if r["pass"] else "FAIL"
                f.write(f"  {r['test']:30s}  p={p}  [{status}]\n")
            passed = sum(1 for r in results if r["pass"])
            f.write(f"  => {passed}/{len(results)} passed\n")
    print(f"  Saved -> {out_txt}")


if __name__ == "__main__":
    main()
