"""
Generate All Visual Figures for Both Papers
=============================================

Figures generated:
  1. fig_plaintext_cipher.png     — Plaintext vs cipher + histograms (3 images)
  2. fig_correlation_scatter.png  — H/V/D correlation scatter plots
  3. fig_training_curves.png      — NKG + GAN + denoiser loss curves (placeholder)
  4. fig_denoiser_before_after.png — Denoiser visual comparison
  5. fig_comparison_bar.png       — Bar chart: v3 vs DL vs Jha metrics
  6. fig_architecture.png         — Text-based architecture diagram

Output: results/figures/
"""

import os
import sys
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3   = os.path.join(_HERE, "..", "lsdpak-v3-rgb")
_FIG  = os.path.join(_HERE, "results", "figures")
_CKPT = os.path.join(_HERE, "checkpoints")
os.makedirs(_FIG, exist_ok=True)
sys.path.insert(0, _V3)
sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lsdpak_v3_rgb import (
    load_image_rgb, encrypt_v3_rgb, decrypt_v3_rgb,
    build_D, IMG_SIZE,
)


def fig1_plaintext_cipher():
    """Plaintext, ciphertext, and histograms for 3 representative images."""
    print("  [1/6] Plaintext vs cipher + histograms")
    D = build_D()
    names = ["lena", "aerial", "couple"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for row, name in enumerate(names):
        img = load_image_rgb(name)
        cipher, hb = encrypt_v3_rgb(img, D)

        # Plaintext
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{name.capitalize()} (plain)")
        axes[row, 0].axis("off")

        # Ciphertext
        axes[row, 1].imshow(cipher)
        axes[row, 1].set_title(f"{name.capitalize()} (cipher)")
        axes[row, 1].axis("off")

        # Plaintext histogram
        for c, color in enumerate(["red", "green", "blue"]):
            axes[row, 2].hist(img[:,:,c].ravel(), bins=256, range=(0,256),
                              alpha=0.5, color=color, density=True)
        axes[row, 2].set_title("Plain histogram")
        axes[row, 2].set_xlim(0, 256)

        # Cipher histogram
        for c, color in enumerate(["red", "green", "blue"]):
            axes[row, 3].hist(cipher[:,:,c].ravel(), bins=256, range=(0,256),
                              alpha=0.5, color=color, density=True)
        axes[row, 3].set_title("Cipher histogram")
        axes[row, 3].set_xlim(0, 256)

    plt.tight_layout()
    path = os.path.join(_FIG, "fig_plaintext_cipher.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def fig2_correlation_scatter():
    """Adjacent-pixel correlation scatter plots for plain and cipher."""
    print("  [2/6] Correlation scatter plots")
    D = build_D()
    img = load_image_rgb("lena")
    cipher, _ = encrypt_v3_rgb(img, D)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    titles = ["Horizontal", "Vertical", "Diagonal"]

    for col, (title, get_pairs) in enumerate(zip(titles, [
        lambda a: (a[:, :-1].ravel(), a[:, 1:].ravel()),
        lambda a: (a[:-1, :].ravel(), a[1:, :].ravel()),
        lambda a: (a[:-1, :-1].ravel(), a[1:, 1:].ravel()),
    ])):
        # Plaintext (R channel)
        x, y = get_pairs(img[:,:,0])
        idx = np.random.choice(len(x), min(2000, len(x)), replace=False)
        axes[0, col].scatter(x[idx], y[idx], s=1, alpha=0.3, c='red')
        r_p = np.corrcoef(x, y)[0, 1]
        axes[0, col].set_title(f"Plain {title}\nr = {r_p:.4f}")
        axes[0, col].set_xlabel("Pixel (i)")
        axes[0, col].set_ylabel("Pixel (i+1)")

        # Ciphertext (R channel)
        x, y = get_pairs(cipher[:,:,0])
        idx = np.random.choice(len(x), min(2000, len(x)), replace=False)
        axes[1, col].scatter(x[idx], y[idx], s=1, alpha=0.3, c='blue')
        r_c = np.corrcoef(x, y)[0, 1]
        axes[1, col].set_title(f"Cipher {title}\nr = {r_c:.6f}")
        axes[1, col].set_xlabel("Pixel (i)")
        axes[1, col].set_ylabel("Pixel (i+1)")

    plt.tight_layout()
    path = os.path.join(_FIG, "fig_correlation_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def fig3_comparison_bar():
    """Bar chart comparing v3, DL, and Jha metrics."""
    print("  [3/6] Comparison bar chart")

    # Load metrics if available
    v3_path = os.path.join(_HERE, "results", "metrics_v3.json")
    dl_path = os.path.join(_HERE, "results", "metrics_dl.json")

    metrics = {"LS-DPAK v3": {}, "LS-DPAK-DL": {}, "Jha 2025": {}}

    if os.path.exists(v3_path):
        with open(v3_path) as f:
            v3 = json.load(f)
        ents = [m["entropy"]["joint"] for m in v3.values()]
        npcrs = [m["npcr_diff"] for m in v3.values()]
        uacis = [m["uaci_diff"] for m in v3.values()]
        metrics["LS-DPAK v3"] = {
            "Entropy": np.mean(ents),
            "NPCR (%)": np.mean(npcrs),
            "UACI (%)": np.mean(uacis),
        }

    if os.path.exists(dl_path):
        with open(dl_path) as f:
            dl = json.load(f)
        ents = [m["entropy"]["joint"] for m in dl.values()]
        npcrs = [m["npcr_diff"] for m in dl.values()]
        uacis = [m["uaci_diff"] for m in dl.values()]
        metrics["LS-DPAK-DL"] = {
            "Entropy": np.mean(ents),
            "NPCR (%)": np.mean(npcrs),
            "UACI (%)": np.mean(uacis),
        }

    metrics["Jha 2025"] = {
        "Entropy": 7.9973,
        "NPCR (%)": 99.6016,
        "UACI (%)": 33.4635,
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metric_names = ["Entropy", "NPCR (%)", "UACI (%)"]
    ideals = [8.0, 99.6094, 33.4635]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for ax, mname, ideal in zip(axes, metric_names, ideals):
        labels = []
        values = []
        for method, m in metrics.items():
            if mname in m:
                labels.append(method)
                values.append(m[mname])

        bars = ax.bar(labels, values, color=colors[:len(labels)], alpha=0.8)
        ax.axhline(y=ideal, color='red', linestyle='--', linewidth=1,
                    label=f"Ideal = {ideal}")
        ax.set_title(mname, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)

        # Set y-axis range close to values
        if values:
            vmin = min(values + [ideal])
            vmax = max(values + [ideal])
            margin = (vmax - vmin) * 0.3 if vmax != vmin else 0.01
            ax.set_ylim(vmin - margin, vmax + margin)

    plt.tight_layout()
    path = os.path.join(_FIG, "fig_comparison_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def fig4_denoiser_visual():
    """Before/after denoiser visual comparison."""
    print("  [4/6] Denoiser before/after")
    D = build_D()

    try:
        import torch
        from lsdpak_dl import NKG, UNetDenoiser, decrypt_dl, DEVICE

        nkg_path = os.path.join(_CKPT, "nkg_gan.pt")
        if not os.path.exists(nkg_path):
            nkg_path = os.path.join(_CKPT, "nkg_pretrained.pt")
        den_path = os.path.join(_CKPT, "denoiser.pt")

        if not os.path.exists(nkg_path) or not os.path.exists(den_path):
            print("    [SKIP] Missing checkpoints")
            return

        nkg = NKG().to(DEVICE)
        nkg.load_state_dict(torch.load(nkg_path, map_location=DEVICE))
        nkg.eval()
        denoiser = UNetDenoiser().to(DEVICE)
        denoiser.load_state_dict(torch.load(den_path, map_location=DEVICE))
        denoiser.eval()

    except ImportError:
        print("    [SKIP] PyTorch not available")
        return

    img = load_image_rgb("lena")
    cipher, hb = encrypt_v3_rgb(img, D)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for row, (density, label) in enumerate([(0.05, "5% S&P"), (0.15, "15% S&P")]):
        noisy = cipher.copy()
        mask = np.random.random(cipher.shape[:2])
        noisy[mask < density / 2] = 0
        noisy[(mask >= density / 2) & (mask < density)] = 255

        dec_raw = decrypt_dl(noisy, D, hb, nkg, use_nkg=True)
        dec_den = decrypt_dl(noisy, D, hb, nkg, use_nkg=True, denoiser=denoiser)

        p_raw = float(np.mean((img.astype(float) - dec_raw.astype(float))**2))
        p_den = float(np.mean((img.astype(float) - dec_den.astype(float))**2))
        psnr_raw = 10*np.log10(255**2/max(p_raw, 1e-10))
        psnr_den = 10*np.log10(255**2/max(p_den, 1e-10))

        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Original")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(noisy)
        axes[row, 1].set_title(f"Noisy cipher ({label})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(dec_raw)
        axes[row, 2].set_title(f"w/o denoiser\nPSNR={psnr_raw:.1f} dB")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(dec_den)
        axes[row, 3].set_title(f"w/ denoiser\nPSNR={psnr_den:.1f} dB")
        axes[row, 3].axis("off")

    plt.suptitle("U-Net Denoiser: Before vs After", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(_FIG, "fig_denoiser_before_after.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def fig5_speed_comparison():
    """Speed comparison bar chart."""
    print("  [5/6] Speed comparison")

    v3_path = os.path.join(_HERE, "results", "metrics_v3.json")
    dl_path = os.path.join(_HERE, "results", "metrics_dl.json")

    fig, ax = plt.subplots(figsize=(8, 5))

    data = {}
    if os.path.exists(v3_path):
        with open(v3_path) as f:
            v3 = json.load(f)
        data["LS-DPAK v3"] = np.mean([m["enc_ms"] for m in v3.values()])

    if os.path.exists(dl_path):
        with open(dl_path) as f:
            dl = json.load(f)
        data["LS-DPAK-DL"] = np.mean([m["enc_ms"] for m in dl.values()])

    if data:
        bars = ax.bar(data.keys(), data.values(),
                      color=["#FF5722", "#2196F3"], alpha=0.85, width=0.5)
        for bar, val in zip(bars, data.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f"{val:.0f} ms", ha='center', fontsize=12, fontweight='bold')

        if len(data) == 2:
            speedup = list(data.values())[0] / list(data.values())[1]
            ax.set_title(f"Encryption Speed Comparison ({speedup:.1f}x speedup)",
                         fontsize=13, fontweight='bold')
        else:
            ax.set_title("Encryption Speed", fontsize=13, fontweight='bold')

        ax.set_ylabel("Time (ms)")
        ax.set_ylim(0, max(data.values()) * 1.2)

    plt.tight_layout()
    path = os.path.join(_FIG, "fig_speed_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def fig6_key_sensitivity():
    """Key sensitivity visual: two ciphers from delta-key."""
    print("  [6/6] Key sensitivity visualization")
    D = build_D()
    img = load_image_rgb("lena")

    cipher1, hb1 = encrypt_v3_rgb(img, D)

    img2 = img.copy()
    img2[128, 128, 0] = int(img2[128, 128, 0]) ^ 1
    cipher2, hb2 = encrypt_v3_rgb(img2, D)

    diff = np.abs(cipher1.astype(int) - cipher2.astype(int)).astype(np.uint8)
    diff_enhanced = np.clip(diff * 3, 0, 255).astype(np.uint8)

    npcr = (cipher1 != cipher2).mean() * 100

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cipher1)
    axes[0].set_title("Cipher (original)")
    axes[0].axis("off")

    axes[1].imshow(cipher2)
    axes[1].set_title("Cipher (1-bit changed)")
    axes[1].axis("off")

    axes[2].imshow(diff_enhanced)
    axes[2].set_title(f"Difference (enhanced 3x)\nNPCR = {npcr:.2f}%")
    axes[2].axis("off")

    # Difference histogram
    axes[3].hist(diff.ravel(), bins=256, range=(0, 256), color='purple', alpha=0.7)
    axes[3].set_title("Difference histogram")
    axes[3].set_xlabel("Pixel difference")
    axes[3].set_ylabel("Count")

    plt.suptitle("Differential Plaintext Sensitivity (1-bit change)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(_FIG, "fig_key_sensitivity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def main():
    print("\nGenerating Figures")
    print("=" * 60)

    fig1_plaintext_cipher()
    fig2_correlation_scatter()
    fig3_comparison_bar()
    fig4_denoiser_visual()
    fig5_speed_comparison()
    fig6_key_sensitivity()

    print(f"\nAll figures saved in: {_FIG}")
    print(f"  Files: {sorted(os.listdir(_FIG))}")


if __name__ == "__main__":
    main()
