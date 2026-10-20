"""
Ablation Study for LS-DPAK-DL
==============================

Compares three NKG configurations:
  A) Phase 1 only  (pre-trained NKG, uniformity loss only)
  B) Phase 1+2     (GAN-trained NKG)
  C) Full pipeline  (GAN NKG + U-Net denoiser)

Metrics: entropy, NPCR_diff, UACI_diff, NPCR_key, histogram chi2, enc time.
Also compares against v3 ODE baseline.

Output: results/ablation_report.json, results/ablation_table.txt
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3   = os.path.join(_HERE, "..", "lsdpak-v3-rgb")
_RES  = os.path.join(_HERE, "results")
_CKPT = os.path.join(_HERE, "checkpoints")
os.makedirs(_RES, exist_ok=True)
sys.path.insert(0, _V3)
sys.path.insert(0, _HERE)

from lsdpak_v3_rgb import (
    load_image_rgb, encrypt_v3_rgb, decrypt_v3_rgb,
    build_D, image_hash_bytes_rgb, IMG_SIZE,
)

IMAGE_NAMES = ["aerial", "lena", "couple", "clock"]


def entropy_ch(ch):
    hist, _ = np.histogram(ch.ravel(), bins=256, range=(0, 256))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def joint_entropy(img):
    flat = img.ravel()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def npcr_uaci(c1, c2):
    diff = (c1.astype(int) != c2.astype(int))
    npcr = diff.mean() * 100.0
    uaci = np.abs(c1.astype(int) - c2.astype(int)).mean() / 255.0 * 100.0
    return float(npcr), float(uaci)


def chi2_stat(ch):
    from scipy.stats import chi2 as chi2_dist
    hist, _ = np.histogram(ch.ravel(), bins=256, range=(0, 256))
    exp = hist.sum() / 256.0
    stat = float(np.sum((hist - exp)**2 / exp))
    p = float(1.0 - chi2_dist.cdf(stat, df=255))
    return stat, p


def evaluate_config(name, encrypt_fn, images, D):
    results = {}
    for img_name in IMAGE_NAMES:
        if img_name not in images:
            continue
        img = images[img_name]

        t0 = time.perf_counter()
        cipher, hb = encrypt_fn(img)
        enc_ms = (time.perf_counter() - t0) * 1000.0

        # Differential NPCR
        img2 = img.copy()
        img2[0, 0, 0] = int(img2[0, 0, 0]) ^ 1
        cipher2, _ = encrypt_fn(img2)
        nd, ud = npcr_uaci(cipher, cipher2)

        # Chi2 on R channel
        chi_stat, chi_p = chi2_stat(cipher[:, :, 0])

        results[img_name] = {
            "entropy_joint": joint_entropy(cipher),
            "entropy_R": entropy_ch(cipher[:, :, 0]),
            "npcr_diff": nd,
            "uaci_diff": ud,
            "chi2_R": chi_stat,
            "chi2_p_R": chi_p,
            "chi2_pass": bool(chi_p > 0.01),
            "enc_ms": enc_ms,
        }
        print(f"    {img_name:12s}  ent={results[img_name]['entropy_joint']:.4f}  "
              f"NPCR={nd:.2f}%  enc={enc_ms:.0f}ms")

    return results


def main():
    import torch
    from lsdpak_dl import NKG, UNetDenoiser, encrypt_dl, DEVICE

    print("\nAblation Study")
    print("=" * 60)

    # Load images
    images = {}
    for name in IMAGE_NAMES:
        try:
            images[name] = load_image_rgb(name)
        except Exception:
            pass
    print(f"  Loaded {len(images)} images")

    D = build_D()
    report = {}

    # --- Baseline: LS-DPAK v3 (ODE) ---
    print("\n[A0] LS-DPAK v3 (ODE baseline)")
    report["v3_ode"] = evaluate_config(
        "v3_ode",
        lambda img: encrypt_v3_rgb(img, D),
        images, D
    )

    # --- A) Phase 1 only (pre-trained NKG) ---
    p1_path = os.path.join(_CKPT, "nkg_pretrained.pt")
    if os.path.exists(p1_path):
        print("\n[A1] Phase 1 only (pre-trained NKG)")
        nkg_p1 = NKG().to(DEVICE)
        nkg_p1.load_state_dict(torch.load(p1_path, map_location=DEVICE))
        nkg_p1.eval()
        report["phase1_only"] = evaluate_config(
            "phase1",
            lambda img: encrypt_dl(img, D, nkg_p1, use_nkg=True),
            images, D
        )
    else:
        print("  [SKIP] Phase 1 checkpoint not found")

    # --- B) Phase 1+2 (GAN NKG) ---
    p2_path = os.path.join(_CKPT, "nkg_gan.pt")
    if os.path.exists(p2_path):
        print("\n[A2] Phase 1+2 (GAN-trained NKG)")
        nkg_p2 = NKG().to(DEVICE)
        nkg_p2.load_state_dict(torch.load(p2_path, map_location=DEVICE))
        nkg_p2.eval()
        report["phase1_2"] = evaluate_config(
            "phase1+2",
            lambda img: encrypt_dl(img, D, nkg_p2, use_nkg=True),
            images, D
        )
    else:
        print("  [SKIP] Phase 2 checkpoint not found")

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY (averages)")
    print("=" * 60)
    sep = "+" + "-"*22 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*10 + "+"
    hdr = f"| {'Config':<20} | {'Entropy':^10} | {'NPCR%':^10} | {'UACI%':^10} | {'ms':^8} |"
    print(sep)
    print(hdr)
    print(sep)
    for label, key in [("v3 ODE", "v3_ode"),
                       ("Phase 1 only", "phase1_only"),
                       ("Phase 1+2 (GAN)", "phase1_2")]:
        if key not in report:
            continue
        metrics = report[key]
        avg_ent  = np.mean([m["entropy_joint"] for m in metrics.values()])
        avg_npcr = np.mean([m["npcr_diff"] for m in metrics.values()])
        avg_uaci = np.mean([m["uaci_diff"] for m in metrics.values()])
        avg_ms   = np.mean([m["enc_ms"] for m in metrics.values()])
        print(f"| {label:<20} | {avg_ent:^10.4f} | {avg_npcr:^10.4f} | "
              f"{avg_uaci:^10.4f} | {avg_ms:^8.0f} |")
    print(sep)

    # Save
    out = os.path.join(_RES, "ablation_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    main()
