"""
LS-DPAK-DL  —  Evaluation Script
==================================

Evaluates LS-DPAK-DL (NKG + U-Net denoiser) on 6 USC-SIPI images and
compares against:
  1. LS-DPAK v3 RGB  (current best)
  2. Jha et al. (2025) reported values  (from paper Table III)

Metrics computed:
  - Entropy (per-channel + joint)
  - Pixel correlation (H / V / D, per-channel)
  - NPCR differential-plaintext (per-channel avg)
  - UACI differential-plaintext
  - NPCR key-sensitivity
  - UACI key-sensitivity
  - Encryption time (ms)
  - Denoiser PSNR / SSIM  (clean vs denoiser output after S&P noise)
  - Round-trip fidelity

Output:
  results/metrics_dl.json
  results/comparison_table.txt
  results/comparison_table.tex        (copy-paste into LaTeX paper)
  results/attack_report_dl.json
"""

import json
import os
import sys
import time

import numpy as np
from scipy.stats import chi2

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
_V3    = os.path.join(_HERE, "..", "lsdpak-v3-rgb")
_RES   = os.path.join(_HERE, "results")
_CKPT  = os.path.join(_HERE, "checkpoints")
os.makedirs(_RES, exist_ok=True)

sys.path.insert(0, _V3)
sys.path.insert(0, _HERE)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not available — DL models will not be loaded.")

from lsdpak_v3_rgb import (
    load_image_rgb, encrypt_v3_rgb, decrypt_v3_rgb,
    npcr_uaci_key_sensitivity_rgb, image_hash_bytes_rgb,
    adaptive_ics_rgb, build_keystream as lorenz_ks,
    build_D, build_D3, build_D5,
    scramble3, unscramble3, shuffle_blocks_full, unshuffle_blocks_full,
    interchannel_shuffle, interchannel_unshuffle,
    bidir_encrypt_joint, bidir_decrypt_joint,
    IMG_SIZE, MASTER, IMAGE_NAMES as V3_IMAGE_NAMES,
)

if HAS_TORCH:
    from lsdpak_dl import NKG, UNetDenoiser, DEVICE
    from lsdpak_dl import encrypt_dl, decrypt_dl

# ── Image names (keys into lsdpak_v3_rgb.TIFF_SRC dict) ──────────────────────
IMAGE_NAMES = ["aerial", "lena", "couple", "clock", "4_2_03", "5_3_01",
               "gradient", "checker", "noise_img", "mandrill_synth"]

# ── Jha et al. (2025) reference values (from their Table III / text) ──────────
JHA_2025 = {
    "entropy":   7.9973,    # reported average
    "npcr":      99.6016,
    "uaci":      33.4635,
    "corr_h":    0.0012,
    "corr_v":    0.0008,
    "corr_d":    0.0015,
    "key_space": "2^256",
    "enc_ms":    "N/A",
}

# =============================================================================
# METRIC HELPERS
# =============================================================================

def entropy_channel(ch: np.ndarray) -> float:
    hist, _ = np.histogram(ch.ravel(), bins=256, range=(0, 256))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def joint_entropy_rgb(img: np.ndarray) -> float:
    """Approximate joint 3-channel entropy via combined byte sequence."""
    flat = img.ravel()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def pixel_corr(arr: np.ndarray, direction: str) -> float:
    """Adjacent-pixel correlation for one channel 2D array."""
    a = arr.ravel()
    if direction == "H":
        x = arr[:, :-1].ravel()
        y = arr[:, 1:].ravel()
    elif direction == "V":
        x = arr[:-1, :].ravel()
        y = arr[1:,  :].ravel()
    else:  # "D"
        x = arr[:-1, :-1].ravel()
        y = arr[1:,  1:].ravel()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def npcr_uaci_diff(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """NPCR and UACI between two ciphertext images (differential plaintext)."""
    diff = (img1.astype(int) - img2.astype(int)) != 0
    npcr = diff.mean() * 100.0
    uaci = np.abs(img1.astype(int) - img2.astype(int)).mean() / 255.0 * 100.0
    return float(npcr), float(uaci)


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(255**2 / mse))


def ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simple channel-averaged SSIM (no sliding window)."""
    scores = []
    for c in range(3):
        x = img1[:,:,c].astype(float)
        y = img2[:,:,c].astype(float)
        C1, C2 = (0.01*255)**2, (0.03*255)**2
        mu_x, mu_y = x.mean(), y.mean()
        sig_xx = ((x - mu_x)**2).mean()
        sig_yy = ((y - mu_y)**2).mean()
        sig_xy = ((x - mu_x)*(y - mu_y)).mean()
        num = (2*mu_x*mu_y + C1) * (2*sig_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (sig_xx + sig_yy + C2)
        scores.append(num / max(den, 1e-10))
    return float(np.mean(scores))


def add_salt_pepper(img: np.ndarray, density: float = 0.05) -> np.ndarray:
    out = img.copy()
    mask = np.random.random(img.shape[:2])
    out[mask < density / 2]                              = 0
    out[(mask >= density / 2) & (mask < density)]        = 255
    return out


def histogram_chi2_test(ch: np.ndarray) -> tuple:
    hist, _ = np.histogram(ch.ravel(), bins=256, range=(0,256))
    expected = hist.sum() / 256.0
    chi_stat = float(np.sum((hist - expected)**2 / expected))
    p_val    = float(1.0 - chi2.cdf(chi_stat, df=255))
    return chi_stat, p_val


# =============================================================================
# COMPUTE METRICS FOR ONE APPROACH
# =============================================================================

def compute_metrics_v3(name: str, img: np.ndarray) -> dict:
    """LS-DPAK v3 RGB metrics."""
    D = build_D()

    t0 = time.perf_counter()
    cipher, hb = encrypt_v3_rgb(img, D)
    enc_ms = (time.perf_counter() - t0) * 1000.0

    decrypted   = decrypt_v3_rgb(cipher, D, hb)
    roundtrip   = bool(np.array_equal(img, decrypted))

    # Differential plaintext NPCR/UACI
    img2        = img.copy()
    img2[0, 0, 0] = int(img2[0, 0, 0]) ^ 1
    cipher2, _  = encrypt_v3_rgb(img2, D)
    npcr_dp, uaci_dp = npcr_uaci_diff(cipher, cipher2)

    # Key sensitivity
    npcr_key, uaci_key = npcr_uaci_key_sensitivity_rgb(img, D)

    m = {
        "entropy": {
            "R": entropy_channel(cipher[:,:,0]),
            "G": entropy_channel(cipher[:,:,1]),
            "B": entropy_channel(cipher[:,:,2]),
            "joint": joint_entropy_rgb(cipher),
        },
        "corr_h": {c: pixel_corr(cipher[:,:,i], "H") for i, c in enumerate("RGB")},
        "corr_v": {c: pixel_corr(cipher[:,:,i], "V") for i, c in enumerate("RGB")},
        "corr_d": {c: pixel_corr(cipher[:,:,i], "D") for i, c in enumerate("RGB")},
        "npcr_diff": npcr_dp,
        "uaci_diff": uaci_dp,
        "npcr_key":  npcr_key,
        "uaci_key":  uaci_key,
        "enc_ms":    enc_ms,
        "roundtrip": roundtrip,
    }
    return m


def compute_metrics_dl(name: str, img: np.ndarray,
                        nkg: "NKG", denoiser: "UNetDenoiser") -> dict:
    """LS-DPAK-DL metrics."""
    import torch

    t0 = time.perf_counter()
    cipher, hb = encrypt_dl(img, build_D(), nkg, use_nkg=True)
    enc_ms = (time.perf_counter() - t0) * 1000.0

    decrypted   = decrypt_dl(cipher, build_D(), hb, nkg, use_nkg=True)
    roundtrip   = bool(np.array_equal(img, decrypted))

    # Differential plaintext NPCR/UACI
    img2        = img.copy()
    img2[0, 0, 0] = int(img2[0, 0, 0]) ^ 1
    cipher2, _  = encrypt_dl(img2, build_D(), nkg, use_nkg=True)
    npcr_dp, uaci_dp = npcr_uaci_diff(cipher, cipher2)

    # Key sensitivity (perturb NKG hash bytes directly)
    hb2 = bytearray(hb)
    hb2[0] = hb2[0] ^ 1
    hb2 = bytes(hb2)
    # Derive 3-channel keys with perturbed hash
    ks1 = nkg.generate_keystream(hb,  IMG_SIZE).ravel()
    ks2 = nkg.generate_keystream(hb2, IMG_SIZE).ravel()
    N   = IMG_SIZE * IMG_SIZE
    ks1_3 = np.concatenate([ks1[:N], np.roll(ks1, N//3)[:N], np.roll(ks1, 2*N//3)[:N]])
    ks2_3 = np.concatenate([ks2[:N], np.roll(ks2, N//3)[:N], np.roll(ks2, 2*N//3)[:N]])
    # Re-encrypt with perturbed key
    c1_flat  = bidir_encrypt_joint(
        np.concatenate([img[:,:,c].ravel() for c in range(3)]).astype(np.uint8),
        ks1_3.astype(np.uint8),
    )
    c2_flat  = bidir_encrypt_joint(
        np.concatenate([img[:,:,c].ravel() for c in range(3)]).astype(np.uint8),
        ks2_3.astype(np.uint8),
    )
    npcr_key = float(((c1_flat != c2_flat).mean()) * 100.0)
    uaci_key = float(np.abs(c1_flat.astype(int) - c2_flat.astype(int)).mean() / 255.0 * 100.0)

    # Denoiser robustness (PSNR/SSIM after 5% S&P noise)
    noisy_cipher = add_salt_pepper(cipher, density=0.05)
    dec_noisy    = decrypt_dl(noisy_cipher, build_D(), hb, nkg, use_nkg=True)
    dec_denoised = decrypt_dl(noisy_cipher, build_D(), hb, nkg, use_nkg=True, denoiser=denoiser)
    psnr_no_den  = psnr(img, dec_noisy)
    psnr_den     = psnr(img, dec_denoised)
    ssim_no_den  = ssim_simple(img, dec_noisy)
    ssim_den     = ssim_simple(img, dec_denoised)

    m = {
        "entropy": {
            "R": entropy_channel(cipher[:,:,0]),
            "G": entropy_channel(cipher[:,:,1]),
            "B": entropy_channel(cipher[:,:,2]),
            "joint": joint_entropy_rgb(cipher),
        },
        "corr_h": {c: pixel_corr(cipher[:,:,i], "H") for i, c in enumerate("RGB")},
        "corr_v": {c: pixel_corr(cipher[:,:,i], "V") for i, c in enumerate("RGB")},
        "corr_d": {c: pixel_corr(cipher[:,:,i], "D") for i, c in enumerate("RGB")},
        "npcr_diff": npcr_dp,
        "uaci_diff": uaci_dp,
        "npcr_key":  npcr_key,
        "uaci_key":  uaci_key,
        "enc_ms":    enc_ms,
        "roundtrip": roundtrip,
        "denoiser": {
            "psnr_without": psnr_no_den,
            "psnr_with":    psnr_den,
            "ssim_without": ssim_no_den,
            "ssim_with":    ssim_den,
        },
    }
    return m


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def _avg(metrics: dict, field: str) -> float:
    vals = []
    for img_name, m in metrics.items():
        if field in m:
            v = m[field]
            if isinstance(v, dict):
                v = np.mean(list(v.values()))
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def build_comparison_table(v3_metrics: dict, dl_metrics: dict) -> str:
    rows = []
    sep  = "+" + "-"*30 + "+" + "-"*16 + "+" + "-"*16 + "+" + "-"*16 + "+"
    hdr  = f"| {'Metric':<28} | {'LS-DPAK v3':^14} | {'LS-DPAK-DL':^14} | {'Jha 2025':^14} |"

    rows.append(sep)
    rows.append(hdr)
    rows.append(sep)

    def row(label, v3_val, dl_val, jha_val, fmt=".4f"):
        return (f"| {label:<28} | {v3_val:^14{fmt}} | "
                f"{dl_val:^14{fmt}} | {jha_val:^14} |")

    # Entropy
    rows.append(row("Entropy (joint, avg)",
                    _avg(v3_metrics, "entropy"),
                    _avg(dl_metrics, "entropy") if dl_metrics else float("nan"),
                    JHA_2025["entropy"]))

    # Correlation H
    def _avg_corr(metrics, direction):
        vals = []
        for m in metrics.values():
            k = f"corr_{direction}"
            if k in m:
                vals.append(abs(np.mean(list(m[k].values()))))
        return float(np.mean(vals)) if vals else float("nan")

    rows.append(row("Corr. H (|avg|)",
                    _avg_corr(v3_metrics, "h"),
                    _avg_corr(dl_metrics, "h") if dl_metrics else float("nan"),
                    f"{JHA_2025['corr_h']:.4f}"))
    rows.append(row("Corr. V (|avg|)",
                    _avg_corr(v3_metrics, "v"),
                    _avg_corr(dl_metrics, "v") if dl_metrics else float("nan"),
                    f"{JHA_2025['corr_v']:.4f}"))
    rows.append(row("Corr. D (|avg|)",
                    _avg_corr(v3_metrics, "d"),
                    _avg_corr(dl_metrics, "d") if dl_metrics else float("nan"),
                    f"{JHA_2025['corr_d']:.4f}"))

    # NPCR
    rows.append(row("NPCR diff. (%)",
                    _avg(v3_metrics, "npcr_diff"),
                    _avg(dl_metrics, "npcr_diff") if dl_metrics else float("nan"),
                    JHA_2025["npcr"]))
    rows.append(row("UACI diff. (%)",
                    _avg(v3_metrics, "uaci_diff"),
                    _avg(dl_metrics, "uaci_diff") if dl_metrics else float("nan"),
                    JHA_2025["uaci"]))
    rows.append(row("NPCR key sens. (%)",
                    _avg(v3_metrics, "npcr_key"),
                    _avg(dl_metrics, "npcr_key") if dl_metrics else float("nan"),
                    "~99.60"))
    rows.append(row("UACI key sens. (%)",
                    _avg(v3_metrics, "uaci_key"),
                    _avg(dl_metrics, "uaci_key") if dl_metrics else float("nan"),
                    "~33.46"))
    rows.append(row("Enc. time (ms, avg)",
                    _avg(v3_metrics, "enc_ms"),
                    _avg(dl_metrics, "enc_ms") if dl_metrics else float("nan"),
                    "N/A"))

    rows.append(sep)
    return "\n".join(rows)


def build_latex_table(v3_metrics: dict, dl_metrics: dict) -> str:
    def _avg_corr(metrics, direction):
        vals = []
        for m in metrics.values():
            k = f"corr_{direction}"
            if k in m:
                vals.append(abs(np.mean(list(m[k].values()))))
        return float(np.mean(vals)) if vals else float("nan")

    def fv(v, fmt=".4f"):
        try:
            return f"{v:{fmt}}"
        except Exception:
            return str(v)

    v3_e  = _avg(v3_metrics, "entropy")
    dl_e  = _avg(dl_metrics, "entropy") if dl_metrics else float("nan")
    v3_ch = _avg_corr(v3_metrics, "h")
    dl_ch = _avg_corr(dl_metrics, "h") if dl_metrics else float("nan")
    v3_cv = _avg_corr(v3_metrics, "v")
    dl_cv = _avg_corr(dl_metrics, "v") if dl_metrics else float("nan")
    v3_cd = _avg_corr(v3_metrics, "d")
    dl_cd = _avg_corr(dl_metrics, "d") if dl_metrics else float("nan")
    v3_nd = _avg(v3_metrics, "npcr_diff")
    dl_nd = _avg(dl_metrics, "npcr_diff") if dl_metrics else float("nan")
    v3_ud = _avg(v3_metrics, "uaci_diff")
    dl_ud = _avg(dl_metrics, "uaci_diff") if dl_metrics else float("nan")
    v3_nk = _avg(v3_metrics, "npcr_key")
    dl_nk = _avg(dl_metrics, "npcr_key") if dl_metrics else float("nan")
    v3_uk = _avg(v3_metrics, "uaci_key")
    dl_uk = _avg(dl_metrics, "uaci_key") if dl_metrics else float("nan")
    v3_ms = _avg(v3_metrics, "enc_ms")
    dl_ms = _avg(dl_metrics, "enc_ms") if dl_metrics else float("nan")

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Comparison of LS-DPAK-DL vs.\ LS-DPAK v3 and Jha et al.\ (2025)}",
        r"\label{tab:dl_comparison}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"\textbf{Metric} & \textbf{LS-DPAK-DL} & \textbf{LS-DPAK v3} & \textbf{Jha et al.\ (2025)} \\",
        r"\hline",
        rf"Entropy (joint avg)       & {fv(dl_e)} & {fv(v3_e)} & {fv(JHA_2025['entropy'])} \\",
        rf"Correlation H ($|\bar{{r}}_h|$) & {fv(dl_ch)} & {fv(v3_ch)} & {fv(JHA_2025['corr_h'])} \\",
        rf"Correlation V ($|\bar{{r}}_v|$) & {fv(dl_cv)} & {fv(v3_cv)} & {fv(JHA_2025['corr_v'])} \\",
        rf"Correlation D ($|\bar{{r}}_d|$) & {fv(dl_cd)} & {fv(v3_cd)} & {fv(JHA_2025['corr_d'])} \\",
        rf"NPCR$_\text{{diff}}$ (\%)        & {fv(dl_nd)} & {fv(v3_nd)} & {fv(JHA_2025['npcr'])} \\",
        rf"UACI$_\text{{diff}}$ (\%)        & {fv(dl_ud)} & {fv(v3_ud)} & {fv(JHA_2025['uaci'])} \\",
        rf"NPCR$_\text{{key}}$ (\%)         & {fv(dl_nk)} & {fv(v3_nk)} & $\approx$99.60 \\",
        rf"UACI$_\text{{key}}$ (\%)         & {fv(dl_uk)} & {fv(v3_uk)} & $\approx$33.46 \\",
        rf"Enc.\ time (ms)           & {fv(dl_ms, '.1f')} & {fv(v3_ms, '.1f')} & N/A \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =============================================================================
# ATTACK RESISTANCE QUICK-CHECK  (T1–T5 on cipher)
# =============================================================================

def attack_check(cipher: np.ndarray) -> dict:
    report = {}
    for i, ch_name in enumerate("RGB"):
        ch = cipher[:,:,i]
        chi_stat, p_val = histogram_chi2_test(ch)
        report[f"chi2_{ch_name}"] = {"stat": chi_stat, "p_val": p_val,
                                     "uniform": bool(p_val > 0.01)}
        report[f"corr_h_{ch_name}"] = pixel_corr(ch, "H")
        report[f"corr_v_{ch_name}"] = pixel_corr(ch, "V")
        report[f"corr_d_{ch_name}"] = pixel_corr(ch, "D")
        report[f"entropy_{ch_name}"] = entropy_channel(ch)
    return report


# =============================================================================
# MAIN
# =============================================================================

def load_image_safe(name: str) -> np.ndarray | None:
    try:
        return load_image_rgb(name)
    except Exception:
        return None


def main():
    print("\nLS-DPAK-DL  Evaluation")
    print("=" * 60)

    # ── Load images ─────────────────────────────────────────────────────────
    images = {}
    for name in IMAGE_NAMES:
        img = load_image_safe(name)
        if img is not None:
            images[name] = img
            print(f"  Loaded {name:10s}  shape={img.shape}")
        else:
            print(f"  [SKIP] {name} not found")

    if not images:
        print("\n[ERROR] No images found. Aborting.")
        sys.exit(1)

    D = build_D()

    # ── LS-DPAK v3 metrics ───────────────────────────────────────────────────
    print("\n--- LS-DPAK v3 RGB ---")
    v3_metrics = {}
    for name, img in images.items():
        print(f"  {name} …", end=" ", flush=True)
        v3_metrics[name] = compute_metrics_v3(name, img)
        print(f"entropy_joint={v3_metrics[name]['entropy']['joint']:.4f}  "
              f"NPCR_key={v3_metrics[name]['npcr_key']:.2f}%")

    # ── LS-DPAK-DL metrics ───────────────────────────────────────────────────
    dl_metrics = {}

    if HAS_TORCH:
        nkg_path  = os.path.join(_CKPT, "nkg_gan.pt")
        den_path  = os.path.join(_CKPT, "denoiser.pt")

        if not os.path.exists(nkg_path):
            nkg_path = os.path.join(_CKPT, "nkg_pretrained.pt")

        if os.path.exists(nkg_path):
            print(f"\n--- LS-DPAK-DL (NKG from {os.path.basename(nkg_path)}) ---")
            nkg = NKG().to(DEVICE)
            nkg.load_state_dict(torch.load(nkg_path, map_location=DEVICE))
            nkg.eval()

            denoiser = None
            if os.path.exists(den_path):
                denoiser = UNetDenoiser().to(DEVICE)
                denoiser.load_state_dict(torch.load(den_path, map_location=DEVICE))
                denoiser.eval()
                print(f"  Denoiser loaded from {den_path}")
            else:
                print(f"  [INFO] Denoiser checkpoint not found — robustness metrics skipped")

            for name, img in images.items():
                print(f"  {name} …", end=" ", flush=True)
                try:
                    dl_metrics[name] = compute_metrics_dl(name, img, nkg, denoiser)
                    print(f"entropy_joint={dl_metrics[name]['entropy']['joint']:.4f}  "
                          f"NPCR_key={dl_metrics[name]['npcr_key']:.2f}%")
                except Exception as exc:
                    print(f"[ERROR] {exc}")
        else:
            print("\n[INFO] No NKG checkpoint found — run train.py first.")
            print("       Showing v3-only comparison vs Jha 2025.\n")
    else:
        print("\n[INFO] PyTorch not available — DL metrics skipped.")

    # ── Comparison table ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    table_txt = build_comparison_table(v3_metrics, dl_metrics)
    print(table_txt)

    # ── Save outputs ─────────────────────────────────────────────────────────
    with open(os.path.join(_RES, "metrics_v3.json"), "w") as f:
        json.dump(v3_metrics, f, indent=2)
    print(f"\n  Saved v3 metrics     -> results/metrics_v3.json")

    if dl_metrics:
        with open(os.path.join(_RES, "metrics_dl.json"), "w") as f:
            json.dump(dl_metrics, f, indent=2)
        print(f"  Saved DL metrics     -> results/metrics_dl.json")

    with open(os.path.join(_RES, "comparison_table.txt"), "w") as f:
        f.write(table_txt + "\n\nJha et al. 2025 reference:\n")
        f.write(json.dumps(JHA_2025, indent=2))
    print(f"  Saved comparison     -> results/comparison_table.txt")

    latex_table = build_latex_table(v3_metrics, dl_metrics)
    with open(os.path.join(_RES, "comparison_table.tex"), "w") as f:
        f.write(latex_table)
    print(f"  Saved LaTeX table    -> results/comparison_table.tex")

    # ── Quick attack check on first image ────────────────────────────────────
    first_name = list(images.keys())[0]
    first_img  = images[first_name]
    cipher_v3, hb = encrypt_v3_rgb(first_img, D)
    atk = attack_check(cipher_v3)
    atk_path = os.path.join(_RES, "attack_report_dl.json")
    with open(atk_path, "w") as f:
        json.dump({first_name: atk}, f, indent=2)
    print(f"  Saved attack report  -> results/attack_report_dl.json")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY (averages over all images)")
    print("="*60)
    for label, metrics in [("LS-DPAK v3", v3_metrics), ("LS-DPAK-DL", dl_metrics)]:
        if not metrics:
            continue
        print(f"\n  [{label}]")
        print(f"    Entropy (joint avg)  : {_avg(metrics, 'entropy'):.4f}")
        print(f"    NPCR diff (%)        : {_avg(metrics, 'npcr_diff'):.4f}")
        print(f"    UACI diff (%)        : {_avg(metrics, 'uaci_diff'):.4f}")
        print(f"    NPCR key-sens (%)    : {_avg(metrics, 'npcr_key'):.4f}")
        print(f"    UACI key-sens (%)    : {_avg(metrics, 'uaci_key'):.4f}")
        print(f"    Enc time (ms avg)    : {_avg(metrics, 'enc_ms'):.1f}")
        if label == "LS-DPAK-DL" and metrics:
            psnr_w  = np.mean([m["denoiser"]["psnr_with"]    for m in metrics.values() if "denoiser" in m])
            psnr_wo = np.mean([m["denoiser"]["psnr_without"] for m in metrics.values() if "denoiser" in m])
            if not np.isnan(psnr_w):
                print(f"    PSNR w/o denoiser    : {psnr_wo:.2f} dB")
                print(f"    PSNR w/  denoiser    : {psnr_w:.2f} dB")

    print("\n  [Jha et al. 2025]")
    print(f"    Entropy              : {JHA_2025['entropy']}")
    print(f"    NPCR (%)             : {JHA_2025['npcr']}")
    print(f"    UACI (%)             : {JHA_2025['uaci']}")
    print(f"    Corr H               : {JHA_2025['corr_h']}")
    print(f"    Corr V               : {JHA_2025['corr_v']}")
    print(f"    Corr D               : {JHA_2025['corr_d']}")


if __name__ == "__main__":
    main()
