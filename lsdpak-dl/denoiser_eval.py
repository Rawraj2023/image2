"""
Denoiser Evaluation Across Multiple Noise Levels and Types
============================================================

Tests:
  - Salt-and-pepper noise: 1%, 5%, 10%, 15%
  - Gaussian noise: sigma = 10, 25, 50
  - Center crop: 1/16, 1/4

PSNR and SSIM measured with and without denoiser.

Output: results/denoiser_eval.json, results/denoiser_table.txt
"""

import json
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3   = os.path.join(_HERE, "..", "lsdpak-v3-rgb")
_RES  = os.path.join(_HERE, "results")
_CKPT = os.path.join(_HERE, "checkpoints")
os.makedirs(_RES, exist_ok=True)
sys.path.insert(0, _V3)
sys.path.insert(0, _HERE)

from lsdpak_v3_rgb import (
    load_image_rgb, build_D, IMG_SIZE,
)

IMAGE_NAMES = ["aerial", "lena", "couple", "clock", "4_2_03", "5_3_01"]


def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(255**2 / mse))


def ssim_simple(img1, img2):
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


def add_sp(img, density):
    out = img.copy()
    mask = np.random.random(img.shape[:2])
    out[mask < density / 2] = 0
    out[(mask >= density / 2) & (mask < density)] = 255
    return out


def add_gaussian(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


def add_crop(img, frac):
    out = img.copy()
    H, W = img.shape[:2]
    h = int(H * np.sqrt(frac))
    w = int(W * np.sqrt(frac))
    r = (H - h) // 2
    c = (W - w) // 2
    out[r:r+h, c:c+w, :] = 128
    return out


def main():
    import torch
    from lsdpak_dl import NKG, UNetDenoiser, decrypt_dl, DEVICE

    print("\nDenoiser Evaluation")
    print("=" * 60)

    # Load models
    nkg_path = os.path.join(_CKPT, "nkg_gan.pt")
    if not os.path.exists(nkg_path):
        nkg_path = os.path.join(_CKPT, "nkg_pretrained.pt")
    den_path = os.path.join(_CKPT, "denoiser.pt")

    if not os.path.exists(nkg_path) or not os.path.exists(den_path):
        print("[ERROR] Need nkg_gan.pt and denoiser.pt. Run train.py first.")
        return

    from lsdpak_dl import encrypt_dl

    nkg = NKG().to(DEVICE)
    nkg.load_state_dict(torch.load(nkg_path, map_location=DEVICE))
    nkg.eval()

    denoiser = UNetDenoiser().to(DEVICE)
    denoiser.load_state_dict(torch.load(den_path, map_location=DEVICE))
    denoiser.eval()

    D = build_D()

    # Load images
    images = {}
    for name in IMAGE_NAMES:
        try:
            images[name] = load_image_rgb(name)
        except Exception:
            pass
    print(f"  Loaded {len(images)} images")

    # Define corruption scenarios
    scenarios = [
        ("S&P 1%",   lambda c: add_sp(c, 0.01)),
        ("S&P 5%",   lambda c: add_sp(c, 0.05)),
        ("S&P 10%",  lambda c: add_sp(c, 0.10)),
        ("S&P 15%",  lambda c: add_sp(c, 0.15)),
        ("Gauss s10", lambda c: add_gaussian(c, 10)),
        ("Gauss s25", lambda c: add_gaussian(c, 25)),
        ("Gauss s50", lambda c: add_gaussian(c, 50)),
        ("Crop 1/16", lambda c: add_crop(c, 1/16)),
        ("Crop 1/4",  lambda c: add_crop(c, 1/4)),
    ]

    report = {}

    for scenario_name, corrupt_fn in scenarios:
        print(f"\n--- {scenario_name} ---")
        scenario_results = {}

        for img_name, img in images.items():
            # Encrypt with DL pipeline (NKG keystream)
            cipher, hb = encrypt_dl(img, D, nkg, use_nkg=True)

            # Corrupt ciphertext
            noisy_cipher = corrupt_fn(cipher)

            # Decrypt without denoiser
            dec_raw = decrypt_dl(noisy_cipher, D, hb, nkg, use_nkg=True)
            p_raw = psnr(img, dec_raw)
            s_raw = ssim_simple(img, dec_raw)

            # Decrypt with denoiser
            dec_den = decrypt_dl(noisy_cipher, D, hb, nkg, use_nkg=True,
                                 denoiser=denoiser)
            p_den = psnr(img, dec_den)
            s_den = ssim_simple(img, dec_den)

            scenario_results[img_name] = {
                "psnr_without": p_raw,
                "psnr_with": p_den,
                "ssim_without": s_raw,
                "ssim_with": s_den,
                "psnr_gain": p_den - p_raw,
                "ssim_gain": s_den - s_raw,
            }

        # Average
        avg_p_raw = np.mean([r["psnr_without"] for r in scenario_results.values()])
        avg_p_den = np.mean([r["psnr_with"] for r in scenario_results.values()])
        avg_s_raw = np.mean([r["ssim_without"] for r in scenario_results.values()])
        avg_s_den = np.mean([r["ssim_with"] for r in scenario_results.values()])
        print(f"  Avg PSNR: {avg_p_raw:.2f} -> {avg_p_den:.2f} dB "
              f"(+{avg_p_den - avg_p_raw:.2f})")
        print(f"  Avg SSIM: {avg_s_raw:.3f} -> {avg_s_den:.3f} "
              f"(+{avg_s_den - avg_s_raw:.3f})")

        report[scenario_name] = {
            "per_image": scenario_results,
            "avg_psnr_without": avg_p_raw,
            "avg_psnr_with": avg_p_den,
            "avg_ssim_without": avg_s_raw,
            "avg_ssim_with": avg_s_den,
            "avg_psnr_gain": avg_p_den - avg_p_raw,
            "avg_ssim_gain": avg_s_den - avg_s_raw,
        }

    # Print summary table
    print("\n" + "=" * 70)
    print("DENOISER SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Scenario':<14} {'PSNR w/o':>10} {'PSNR w/':>10} {'Gain':>8} "
          f"{'SSIM w/o':>10} {'SSIM w/':>10} {'Gain':>8}")
    print("-" * 70)
    for sc_name, sc_data in report.items():
        print(f"{sc_name:<14} {sc_data['avg_psnr_without']:>10.2f} "
              f"{sc_data['avg_psnr_with']:>10.2f} "
              f"{sc_data['avg_psnr_gain']:>+8.2f} "
              f"{sc_data['avg_ssim_without']:>10.3f} "
              f"{sc_data['avg_ssim_with']:>10.3f} "
              f"{sc_data['avg_ssim_gain']:>+8.3f}")

    # Save
    out = os.path.join(_RES, "denoiser_eval.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    main()
