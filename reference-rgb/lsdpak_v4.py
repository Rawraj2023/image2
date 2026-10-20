"""
LS-DPAK v4 -- Two-Axis (Row + Column) Bidirectional Diffusion
==============================================================

Diagnosis of the v3 correlation shortfall (see initiatives/.../BRIEF.md
target: |corr_h|,|corr_v|,|corr_d| < 0.002 on all images):

  v3's bidir_encrypt_joint chains pixels ONLY along the flattened
  row-major order. Horizontal neighbours (i, i+1) are direct chain
  neighbours and get strong mixing; vertical neighbours (i, i+256) and
  diagonal neighbours are ~256 chain-steps apart and are decorrelated
  only by the earlier Latin-square scramble + block shuffle, not by the
  diffusion step itself. Empirically this shows up as corr_h running
  consistently higher than corr_v across every image in
  lsdpak-v3-rgb/results/metrics_v3_rgb.json, and both occasionally
  spiking to 0.006-0.010 on some images/channels -- worse than the
  paper's reported 0.0008-00.0015 range.

Fix (contribution 10): add a SECOND bidirectional diffusion pass along
the orthogonal (column) axis. This makes vertical neighbours direct
chain-neighbours in the second pass, symmetrizing the mixing across
both axes and (since diagonal neighbours are adjacent in neither raw
order but touched by both passes) also tightening diagonal correlation.

IMPORTANT: the first attempt at this reused the SAME keystream
(transposed) for pass 2 to avoid a second 90k-step chaotic
integration. Empirically this made correlation *worse*, not better
(max|corr| rose from ~0.01 to ~0.011) -- reusing a deterministic
function of the same keystream reintroduces a structural relationship
between "plaintext" (the pass-1 cipher) and key that a truly
independent key would not have. The fix below instead derives a
SECOND, independent set of chaotic initial conditions via hash-chaining
(h2 = SHA-256(hb)) and runs a full second Lorenz+Rossler integration
for the column pass. This roughly doubles keystream-generation cost
(~2x encryption time) but the second pass is now genuinely independent
randomness, not a transform of the first pass's key.

Pipeline (per channel, then joint across R/G/B exactly as v3):
  1. scramble3 (3-pass Latin square)               [unchanged from v3]
  2. shuffle_blocks_full (full 256-block shuffle)   [unchanged from v3]
  3. interchannel_shuffle (D5 column shuffle)       [unchanged from v3]
  4. bidir_encrypt_joint, row-major flatten, key K1      (row pass)
  5. reshape -> (H,W,3), transpose spatial axes (H<->W)
  6. bidir_encrypt_joint, row-major flatten of transposed array, an
     INDEPENDENT key K2 (from hash-chained ICs), transposed to match
     the transposed data layout                          (column pass)
  7. transpose back -> final cipher

hb (21 bytes) is still the only metadata transmitted -- K2's ICs are
re-derived from SHA-256(hb) on both sides, so no extra key material
needs to travel with the ciphertext.

Decrypt reverses exactly: undo column pass (transpose, bidir_decrypt,
transpose back), then undo row pass (bidir_decrypt), then the
unchanged v3 unshuffle/unscramble chain.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "lsdpak-v3-rgb"))
import lsdpak_v3_rgb as v3  # noqa: E402  (reuse scramble/shuffle/keystream/metrics code)

IMG_SIZE = v3.IMG_SIZE
IMAGE_NAMES = v3.IMAGE_NAMES
CH_NAMES = v3.CH_NAMES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)


def second_round_ics(hb):
    """
    Independent ICs for the column-axis diffusion pass, hash-chained
    from the 21 transmitted metadata bytes: h2 = SHA-256(hb).
    Both encrypt and decrypt recompute this identically from hb, so no
    extra key material needs to travel with the ciphertext, but the
    resulting keystream is statistically independent of pass 1's
    (different hash input -> unrelated Lorenz/Rossler trajectories).
    """
    h2 = v3.hashlib.sha256(bytes(hb)).digest()
    return v3.ics_from_hash_bytes(h2[:21])


def encrypt_v4(img_rgb, D, ics_override=None, hb_override=None):
    """
    LS-DPAK v4: v3 pipeline + second (column-axis) diffusion pass.

    ics_override/hb_override let the key-sensitivity test reuse this
    single implementation with a deliberately perturbed key instead of
    duplicating the whole pipeline.
    """
    H, W = img_rgb.shape[:2]
    n = D.shape[0]
    band = H // n
    D5 = v3.build_D5(n)

    ics_list = ics_override if ics_override is not None else v3.adaptive_ics_rgb(img_rgb)
    hb = hb_override if hb_override is not None else v3.image_hash_bytes_rgb(img_rgb)

    shuffled, keys = [], []
    for ch in range(3):
        ch_arr = img_rgb[:, :, ch]
        key_ch = v3.build_keystream(ics_list[ch], H)
        sc = v3.scramble3(ch_arr, D)
        sh = v3.shuffle_blocks_full(sc)
        shuffled.append(sh)
        keys.append(key_ch)

    R_sh, G_sh, B_sh = v3.interchannel_shuffle(shuffled[0], shuffled[1], shuffled[2], D5, band)

    # --- Pass 1: row-major joint diffusion (identical to v3) ---
    flat_arr = np.concatenate([R_sh.ravel(), G_sh.ravel(), B_sh.ravel()])
    flat_key = np.concatenate([keys[0].ravel(), keys[1].ravel(), keys[2].ravel()])
    row_cipher_flat = v3.bidir_encrypt_joint(flat_arr, flat_key)

    N = H * W
    row_cipher = np.stack([row_cipher_flat[0:N].reshape(H, W),
                            row_cipher_flat[N:2 * N].reshape(H, W),
                            row_cipher_flat[2 * N:3 * N].reshape(H, W)], axis=2)

    # --- Pass 2: column-major joint diffusion (new in v4) ---
    # Transpose spatial axes so columns become the fast-varying axis;
    # use an INDEPENDENT keystream (hash-chained ICs), transposed to
    # match the transposed data layout.
    row_cipher_T = row_cipher.transpose(1, 0, 2)          # (W,H,3)
    ics2_list = second_round_ics(hb)
    keys2 = [v3.build_keystream(ics2_list[ch], H) for ch in range(3)]
    keys2_T = [k.T for k in keys2]                        # (W,H) each

    flat_arr2 = np.concatenate([row_cipher_T[:, :, 0].ravel(),
                                 row_cipher_T[:, :, 1].ravel(),
                                 row_cipher_T[:, :, 2].ravel()])
    flat_key2 = np.concatenate([keys2_T[0].ravel(), keys2_T[1].ravel(), keys2_T[2].ravel()])
    col_cipher_flat = v3.bidir_encrypt_joint(flat_arr2, flat_key2)

    col_cipher_T = np.stack([col_cipher_flat[0:N].reshape(W, H),
                              col_cipher_flat[N:2 * N].reshape(W, H),
                              col_cipher_flat[2 * N:3 * N].reshape(W, H)], axis=2)
    cipher = col_cipher_T.transpose(1, 0, 2).astype(np.uint8)  # back to (H,W,3)

    return cipher, hb


def decrypt_v4(cipher, D, hb):
    H, W = cipher.shape[:2]
    n = D.shape[0]
    band = H // n
    D5 = v3.build_D5(n)
    N = H * W

    ics_list = v3.ics_from_hash_bytes(hb)
    keys = [v3.build_keystream(ics_list[ch], H) for ch in range(3)]

    ics2_list = second_round_ics(hb)
    keys2 = [v3.build_keystream(ics2_list[ch], H) for ch in range(3)]
    keys2_T = [k.T for k in keys2]

    # --- Undo pass 2 (column-major) ---
    cipher_T = cipher.transpose(1, 0, 2)
    flat_cipher2 = np.concatenate([cipher_T[:, :, 0].ravel(),
                                    cipher_T[:, :, 1].ravel(),
                                    cipher_T[:, :, 2].ravel()])
    flat_key2 = np.concatenate([keys2_T[0].ravel(), keys2_T[1].ravel(), keys2_T[2].ravel()])
    row_cipher_flat_T = v3.bidir_decrypt_joint(flat_cipher2, flat_key2)
    row_cipher_T = np.stack([row_cipher_flat_T[0:N].reshape(W, H),
                              row_cipher_flat_T[N:2 * N].reshape(W, H),
                              row_cipher_flat_T[2 * N:3 * N].reshape(W, H)], axis=2)
    row_cipher = row_cipher_T.transpose(1, 0, 2)

    # --- Undo pass 1 (row-major) ---
    flat_cipher1 = np.concatenate([row_cipher[:, :, 0].ravel(),
                                    row_cipher[:, :, 1].ravel(),
                                    row_cipher[:, :, 2].ravel()])
    flat_key1 = np.concatenate([keys[0].ravel(), keys[1].ravel(), keys[2].ravel()])
    plain_flat = v3.bidir_decrypt_joint(flat_cipher1, flat_key1)

    R_sh = plain_flat[0:N].reshape(H, W).astype(np.uint8)
    G_sh = plain_flat[N:2 * N].reshape(H, W).astype(np.uint8)
    B_sh = plain_flat[2 * N:3 * N].reshape(H, W).astype(np.uint8)

    R_sh, G_sh, B_sh = v3.interchannel_unshuffle(R_sh, G_sh, B_sh, D5, band)

    result = np.empty((H, W, 3), np.uint8)
    for ch_i, sh_ch in enumerate([R_sh, G_sh, B_sh]):
        ush = v3.unshuffle_blocks_full(sh_ch)
        result[:, :, ch_i] = v3.unscramble3(ush, D)

    return result


def main():
    D = v3.build_D(32)

    print("LS-DPAK v4  |  Phase 4: Two-Axis (Row+Column) Joint Diffusion")
    print("=" * 120)
    print(f"{'Image':<10} {'E_jnt':>7} {'NPCR_diff%':>11} {'UACI_diff%':>11} "
          f"{'NPCR_key%':>10} {'UACI_key%':>10} {'maxCorr':>8} {'EncMs':>8}")
    print("-" * 120)

    all_metrics = {}
    for name in IMAGE_NAMES:
        img_rgb = v3.load_image_rgb(name)
        t0 = time.perf_counter()
        cipher, hb = encrypt_v4(img_rgb, D)
        enc_ms = (time.perf_counter() - t0) * 1000

        dec = decrypt_v4(cipher, D, hb)
        ok = np.array_equal(img_rgb, dec)
        assert ok, f"Roundtrip FAILED for {name}!"

        ent = v3.info_entropy_rgb(cipher)
        ch_corr_h = v3.corr_rgb(cipher, "h")
        ch_corr_v = v3.corr_rgb(cipher, "v")
        ch_corr_d = v3.corr_rgb(cipher, "d")
        ic = v3.interchannel_corr(cipher)

        # Differential-plaintext NPCR/UACI using v4 encrypt (mirrors v3.npcr_uaci_rgb)
        img2 = img_rgb.copy()
        cy, cx = IMG_SIZE // 2, IMG_SIZE // 2
        img2[cy, cx, 0] = (int(img2[cy, cx, 0]) + 1) % 256
        c2, _ = encrypt_v4(img2, D)
        diff = (cipher != c2)
        nd = 100.0 * diff.sum() / cipher.size
        ud = 100.0 * np.abs(cipher.astype(int) - c2.astype(int)).sum() / (cipher.size * 255)

        # Key-sensitivity NPCR/UACI: shift all 3 channels' pass-1 Lorenz
        # x0 by 1e-14, keep hb (and hence the pass-2 key) unchanged.
        ics_perturbed = v3.ics_from_hash_bytes(hb)
        for _ch in range(3):
            ics_perturbed[_ch]["lx"] += 1e-14
        c2_key, _ = encrypt_v4(img_rgb, D, ics_override=ics_perturbed, hb_override=hb)
        nk = 100.0 * (cipher != c2_key).sum() / cipher.size
        uk = 100.0 * np.abs(cipher.astype(int) - c2_key.astype(int)).sum() / (cipher.size * 255)

        max_corr = max(abs(ch_corr_h[c]) for c in CH_NAMES)
        max_corr = max(max_corr, max(abs(ch_corr_v[c]) for c in CH_NAMES))
        max_corr = max(max_corr, max(abs(ch_corr_d[c]) for c in CH_NAMES))

        all_metrics[name] = {
            "entropy": ent,
            "corr_h": ch_corr_h, "corr_v": ch_corr_v, "corr_d": ch_corr_d,
            "interchannel_corr": ic,
            "npcr_diff": nd, "uaci_diff": ud,
            "npcr_key": nk, "uaci_key": uk,
            "max_corr": max_corr,
            "enc_ms": enc_ms, "roundtrip": ok,
        }

        print(f"{name:<10} {ent['joint']:>7.4f} {nd:>11.5f} {ud:>11.5f} "
              f"{nk:>10.5f} {uk:>10.5f} {max_corr:>8.5f} {enc_ms:>8.1f}")

    json.dump(all_metrics, open(os.path.join(OUT_DIR, "metrics_v4.json"), "w"), indent=2)

    def avg(key):
        return float(np.mean([all_metrics[n][key] for n in IMAGE_NAMES]))

    def maxabs_corr(direction):
        return max(max(abs(all_metrics[n][direction][c]) for c in CH_NAMES) for n in IMAGE_NAMES)

    print("-" * 120)
    print(f"Average joint entropy : {np.mean([all_metrics[n]['entropy']['joint'] for n in IMAGE_NAMES]):.5f}")
    print(f"Average NPCR_diff     : {avg('npcr_diff'):.5f}")
    print(f"Average UACI_diff     : {avg('uaci_diff'):.5f}")
    print(f"Average NPCR_key      : {avg('npcr_key'):.5f}")
    print(f"Average UACI_key      : {avg('uaci_key'):.5f}")
    print(f"Max |corr_h| overall  : {maxabs_corr('corr_h'):.5f}")
    print(f"Max |corr_v| overall  : {maxabs_corr('corr_v'):.5f}")
    print(f"Max |corr_d| overall  : {maxabs_corr('corr_d'):.5f}")
    print(f"Max |corr|  overall   : {max(all_metrics[n]['max_corr'] for n in IMAGE_NAMES):.5f}")
    print(f"All roundtrips        : {'PASS' if all(all_metrics[n]['roundtrip'] for n in IMAGE_NAMES) else 'FAIL'}")
    print(f"Saved -> {OUT_DIR}/metrics_v4.json")
    return all_metrics


if __name__ == "__main__":
    main()
