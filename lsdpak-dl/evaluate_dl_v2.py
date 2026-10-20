"""
LS-DPAK-DL v2 — Two-Axis Diffusion for the Neural-Keystream Cipher
=====================================================================

Same motivation as lsdpak-v4-2d-diffusion/lsdpak_v4.py: encrypt_dl()
(lsdpak_dl.py) inherits v3's bidir_encrypt_joint row-major diffusion
unchanged, so it has the same directional mixing asymmetry (horizontal
neighbours are chain-adjacent, vertical/diagonal neighbours are ~H
chain-steps apart) regardless of whether the keystream is chaotic or
neural.

Fix, mirrored from v4: add a second, independent diffusion pass along
the column axis. For the DL cipher this is *cheaper* than for the
chaotic version -- no second 90k-step integration is needed. The NKG
is a deterministic function of a 21-byte hash; feeding it a second,
hash-chained input (h2 = SHA-256(hb)) with the SAME already-trained
weights yields an independent-looking second keystream at the cost of
one more (fast) forward pass, not a second training run.

This script does not retrain anything -- it loads the existing
checkpoints/nkg_gan.pt (or nkg_pretrained.pt) exactly as evaluate_dl.py
does, and adds the second diffusion pass on top.
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

import torch

from lsdpak_v3_rgb import (
    load_image_rgb, image_hash_bytes_rgb,
    build_D, build_D5, scramble3, unscramble3,
    shuffle_blocks_full, unshuffle_blocks_full,
    interchannel_shuffle, interchannel_unshuffle,
    bidir_encrypt_joint, bidir_decrypt_joint,
    IMG_SIZE, hashlib,
)
from lsdpak_dl import NKG, DEVICE

IMAGE_NAMES = ["aerial", "lena", "couple", "clock", "4_2_03", "5_3_01"]


def second_round_hb(hb):
    """Hash-chain the transmitted metadata for an independent 2nd-pass input."""
    return hashlib.sha256(bytes(hb)).digest()[:21]


def encrypt_dl_2axis(img_rgb, D, nkg, ics_hb_override=None):
    H, W = img_rgb.shape[:2]
    n = D.shape[0]
    band = H // n
    D5 = build_D5(n)

    hb = ics_hb_override if ics_hb_override is not None else image_hash_bytes_rgb(img_rgb)
    keys1 = nkg.generate_channel_keystreams(hb, H)

    shuffled = []
    for ch in range(3):
        sc = scramble3(img_rgb[:, :, ch], D)
        sh = shuffle_blocks_full(sc)
        shuffled.append(sh)
    R_sh, G_sh, B_sh = interchannel_shuffle(shuffled[0], shuffled[1], shuffled[2], D5, band)

    # Pass 1: row-major (identical to lsdpak_dl.encrypt_dl)
    flat_arr = np.concatenate([R_sh.ravel(), G_sh.ravel(), B_sh.ravel()])
    flat_key1 = np.concatenate([keys1[0].ravel(), keys1[1].ravel(), keys1[2].ravel()])
    row_cipher_flat = bidir_encrypt_joint(flat_arr, flat_key1)
    N = H * W
    row_cipher = np.stack([row_cipher_flat[0:N].reshape(H, W),
                            row_cipher_flat[N:2 * N].reshape(H, W),
                            row_cipher_flat[2 * N:3 * N].reshape(H, W)], axis=2)

    # Pass 2: column-major, independent NKG input (hash-chained), same weights
    hb2 = second_round_hb(hb)
    keys2 = nkg.generate_channel_keystreams(hb2, H)
    row_cipher_T = row_cipher.transpose(1, 0, 2)
    keys2_T = [k.T for k in keys2]
    flat_arr2 = np.concatenate([row_cipher_T[:, :, 0].ravel(),
                                 row_cipher_T[:, :, 1].ravel(),
                                 row_cipher_T[:, :, 2].ravel()])
    flat_key2 = np.concatenate([keys2_T[0].ravel(), keys2_T[1].ravel(), keys2_T[2].ravel()])
    col_cipher_flat = bidir_encrypt_joint(flat_arr2, flat_key2)
    col_cipher_T = np.stack([col_cipher_flat[0:N].reshape(W, H),
                              col_cipher_flat[N:2 * N].reshape(W, H),
                              col_cipher_flat[2 * N:3 * N].reshape(W, H)], axis=2)
    cipher = col_cipher_T.transpose(1, 0, 2).astype(np.uint8)
    return cipher, hb


def decrypt_dl_2axis(cipher, D, hb, nkg):
    H, W = cipher.shape[:2]
    n = D.shape[0]
    band = H // n
    D5 = build_D5(n)
    N = H * W

    keys1 = nkg.generate_channel_keystreams(hb, H)
    hb2 = second_round_hb(hb)
    keys2 = nkg.generate_channel_keystreams(hb2, H)
    keys2_T = [k.T for k in keys2]

    cipher_T = cipher.transpose(1, 0, 2)
    flat_cipher2 = np.concatenate([cipher_T[:, :, 0].ravel(),
                                    cipher_T[:, :, 1].ravel(),
                                    cipher_T[:, :, 2].ravel()])
    flat_key2 = np.concatenate([keys2_T[0].ravel(), keys2_T[1].ravel(), keys2_T[2].ravel()])
    row_cipher_flat_T = bidir_decrypt_joint(flat_cipher2, flat_key2)
    row_cipher_T = np.stack([row_cipher_flat_T[0:N].reshape(W, H),
                              row_cipher_flat_T[N:2 * N].reshape(W, H),
                              row_cipher_flat_T[2 * N:3 * N].reshape(W, H)], axis=2)
    row_cipher = row_cipher_T.transpose(1, 0, 2)

    flat_cipher1 = np.concatenate([row_cipher[:, :, 0].ravel(),
                                    row_cipher[:, :, 1].ravel(),
                                    row_cipher[:, :, 2].ravel()])
    flat_key1 = np.concatenate([keys1[0].ravel(), keys1[1].ravel(), keys1[2].ravel()])
    plain_flat = bidir_decrypt_joint(flat_cipher1, flat_key1)

    R_sh = plain_flat[0:N].reshape(H, W).astype(np.uint8)
    G_sh = plain_flat[N:2 * N].reshape(H, W).astype(np.uint8)
    B_sh = plain_flat[2 * N:3 * N].reshape(H, W).astype(np.uint8)
    R_sh, G_sh, B_sh = interchannel_unshuffle(R_sh, G_sh, B_sh, D5, band)

    result = np.empty((H, W, 3), np.uint8)
    for ch_i, sh_ch in enumerate([R_sh, G_sh, B_sh]):
        ush = unshuffle_blocks_full(sh_ch)
        result[:, :, ch_i] = unscramble3(ush, D)
    return result


def entropy_channel(ch):
    hist, _ = np.histogram(ch.ravel(), bins=256, range=(0, 256))
    p = hist / hist.sum(); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def joint_entropy_rgb(img):
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    p = hist / hist.sum(); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def pixel_corr(arr, direction):
    if direction == "H":
        x, y = arr[:, :-1].ravel(), arr[:, 1:].ravel()
    elif direction == "V":
        x, y = arr[:-1, :].ravel(), arr[1:, :].ravel()
    else:
        x, y = arr[:-1, :-1].ravel(), arr[1:, 1:].ravel()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])


def npcr_uaci_diff(img1, img2):
    diff = (img1.astype(int) - img2.astype(int)) != 0
    npcr = diff.mean() * 100.0
    uaci = np.abs(img1.astype(int) - img2.astype(int)).mean() / 255.0 * 100.0
    return float(npcr), float(uaci)


def main():
    D = build_D()
    nkg_path = os.path.join(_CKPT, "nkg_gan.pt")
    if not os.path.exists(nkg_path):
        nkg_path = os.path.join(_CKPT, "nkg_pretrained.pt")
    nkg = NKG().to(DEVICE)
    nkg.load_state_dict(torch.load(nkg_path, map_location=DEVICE))
    nkg.eval()
    print(f"Loaded NKG from {os.path.basename(nkg_path)} (no retraining)")

    all_metrics = {}
    print(f"{'Image':<10} {'E_jnt':>7} {'NPCR_diff':>10} {'UACI_diff':>10} "
          f"{'maxCorr':>8} {'EncMs':>8}")
    print("-" * 65)
    for name in IMAGE_NAMES:
        img = load_image_rgb(name)
        t0 = time.perf_counter()
        cipher, hb = encrypt_dl_2axis(img, D, nkg)
        enc_ms = (time.perf_counter() - t0) * 1000

        dec = decrypt_dl_2axis(cipher, D, hb, nkg)
        ok = bool(np.array_equal(img, dec))
        assert ok, f"Roundtrip FAILED for {name}"

        ent = {c: entropy_channel(cipher[:, :, i]) for i, c in enumerate("RGB")}
        ent["joint"] = joint_entropy_rgb(cipher)
        corr_h = {c: pixel_corr(cipher[:, :, i], "H") for i, c in enumerate("RGB")}
        corr_v = {c: pixel_corr(cipher[:, :, i], "V") for i, c in enumerate("RGB")}
        corr_d = {c: pixel_corr(cipher[:, :, i], "D") for i, c in enumerate("RGB")}

        img2 = img.copy()
        img2[0, 0, 0] = int(img2[0, 0, 0]) ^ 1
        cipher2, _ = encrypt_dl_2axis(img2, D, nkg)
        npcr_dp, uaci_dp = npcr_uaci_diff(cipher, cipher2)

        max_corr = max(max(abs(corr_h[c]) for c in "RGB"),
                        max(abs(corr_v[c]) for c in "RGB"),
                        max(abs(corr_d[c]) for c in "RGB"))

        all_metrics[name] = {
            "entropy": ent, "corr_h": corr_h, "corr_v": corr_v, "corr_d": corr_d,
            "npcr_diff": npcr_dp, "uaci_diff": uaci_dp,
            "max_corr": max_corr, "enc_ms": enc_ms, "roundtrip": ok,
        }
        print(f"{name:<10} {ent['joint']:>7.4f} {npcr_dp:>10.5f} {uaci_dp:>10.5f} "
              f"{max_corr:>8.5f} {enc_ms:>8.1f}")

    json.dump(all_metrics, open(os.path.join(_RES, "metrics_dl_v2_2axis.json"), "w"), indent=2)

    def avg(key):
        return float(np.mean([all_metrics[n][key] for n in IMAGE_NAMES]))

    all_corr = [abs(all_metrics[n][d][c]) for n in IMAGE_NAMES
                for d in ("corr_h", "corr_v", "corr_d") for c in "RGB"]
    print("-" * 65)
    print(f"Average joint entropy : {np.mean([all_metrics[n]['entropy']['joint'] for n in IMAGE_NAMES]):.5f}")
    print(f"Average NPCR_diff     : {avg('npcr_diff'):.5f}")
    print(f"Average UACI_diff     : {avg('uaci_diff'):.5f}")
    print(f"Aggregate mean|r|     : {np.mean(all_corr):.5f}  (n={len(all_corr)})")
    print(f"Aggregate std|r|      : {np.std(all_corr):.5f}")
    print(f"Average enc time (ms) : {avg('enc_ms'):.1f}")
    print(f"All roundtrips        : {'PASS' if all(all_metrics[n]['roundtrip'] for n in IMAGE_NAMES) else 'FAIL'}")
    print(f"Saved -> {_RES}/metrics_dl_v2_2axis.json")
    return all_metrics


if __name__ == "__main__":
    main()
