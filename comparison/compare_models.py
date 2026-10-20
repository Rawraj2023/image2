"""
Head-to-head comparison: LS-DPAK-DL (1-axis), LS-DPAK-DL v2 (2-axis),
LS-DPAK v4 (2-axis chaotic).

Why this script exists
----------------------
The three per-model evaluation scripts are not comparable as written:

  * different image sets -- evaluate_dl.py lists 10 names (6 of which are
    not in lsdpak_v3_rgb.TIFF_SRC at all), evaluate_dl_v2.py lists 6,
    lsdpak_v4.py uses v3.IMAGE_NAMES. Only 5 images exist on disk.
  * different differential protocol -- lsdpak_v4.py perturbs the centre
    pixel (+1 mod 256); both DL scripts flip the LSB of pixel (0,0,R).
  * incomparable key-sensitivity -- lsdpak_v4.py perturbs the Lorenz x0 by
    1e-14 through the real pipeline; evaluate_dl.py bypasses the pipeline
    entirely (it XOR-diffuses the *raw plaintext* with two np.roll-derived
    keystreams the cipher never uses); evaluate_dl_v2.py omits the test.

This script runs all three ciphers on one image set with one protocol, so
the numbers belong in the same table. It modifies no existing module; the
three ciphers are imported and driven as-is.

The only new cipher code is encrypt_dl_hb(), which reproduces encrypt_dl()
with an injectable hb so the key-sensitivity test can run through the full
pipeline. It is asserted bit-identical to encrypt_dl() on every image
before any metric is computed.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import chi2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_V3 = os.path.join(_ROOT, "lsdpak-v3-rgb")
_DL = os.path.join(_ROOT, "lsdpak-dl")
_V4 = os.path.join(_ROOT, "lsdpak-v4-2d-diffusion")
_RES = os.path.join(_HERE, "results")
os.makedirs(_RES, exist_ok=True)

for p in (_V3, _DL, _V4):
    sys.path.insert(0, p)

import lsdpak_v3_rgb as v3
from lsdpak_dl import NKG, DEVICE, encrypt_dl, decrypt_dl
import evaluate_dl_v2 as dlv2
import lsdpak_v4 as v4

# Baboon.tiff is absent from test_images/, so v4's recorded 6-image set
# cannot be reproduced. These 5 are every image that exists on disk.
IMAGES = ["aerial", "lena", "couple", "clock", "man"]
CH = ["R", "G", "B"]

# Jha et al. (2025), Security and Privacy 8:e70128 -- grayscale, reported averages.
JHA = {"entropy": 7.9973, "npcr": 99.6016, "uaci": 33.4635,
       "corr_h": 0.0012, "corr_v": 0.0008, "corr_d": 0.0015}


# ---------------------------------------------------------------- metrics ----
def entropy_rgb(c):
    out = {}
    for i, n in enumerate(CH):
        h, _ = np.histogram(c[:, :, i].ravel(), 256, (0, 256))
        p = h / h.sum()
        p = p[p > 0]
        out[n] = float(-np.sum(p * np.log2(p)))
    h, _ = np.histogram(c.ravel(), 256, (0, 256))
    p = h / h.sum()
    p = p[p > 0]
    # Pooling 3 channels into one 196608-sample histogram lowers the
    # finite-sample bias of the estimator; it is not a security gain.
    out["joint_pooled"] = float(-np.sum(p * np.log2(p)))
    return out


def corr(arr, d):
    if d == "h":
        x, y = arr[:, :-1].ravel(), arr[:, 1:].ravel()
    elif d == "v":
        x, y = arr[:-1, :].ravel(), arr[1:, :].ravel()
    else:
        x, y = arr[:-1, :-1].ravel(), arr[1:, 1:].ravel()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])


def npcr_uaci(a, b):
    d = a.astype(int) - b.astype(int)
    return float((d != 0).mean() * 100.0), float(np.abs(d).mean() / 255.0 * 100.0)


def chi2_uniformity(c):
    """Per-channel chi-square against a uniform byte histogram, df=255."""
    out = {}
    for i, n in enumerate(CH):
        h, _ = np.histogram(c[:, :, i].ravel(), 256, (0, 256))
        e = h.sum() / 256.0
        stat = float(np.sum((h - e) ** 2 / e))
        pv = float(1.0 - chi2.cdf(stat, df=255))
        out[n] = {"stat": stat, "p": pv, "uniform_at_0.01": bool(pv > 0.01)}
    return out


# ------------------------------------------------- cipher driver wrappers ----
def encrypt_dl_hb(img, D, nkg, hb):
    """encrypt_dl() with an injectable hb (needed for key sensitivity)."""
    H, W = img.shape[:2]
    n = D.shape[0]
    band = H // n
    D5 = v3.build_D5(n)
    keys = nkg.generate_channel_keystreams(hb, H)
    sh = [v3.shuffle_blocks_full(v3.scramble3(img[:, :, c], D)) for c in range(3)]
    R, G, B = v3.interchannel_shuffle(sh[0], sh[1], sh[2], D5, band)
    flat = np.concatenate([R.ravel(), G.ravel(), B.ravel()])
    key = np.concatenate([keys[c].ravel() for c in range(3)])
    cf = v3.bidir_encrypt_joint(flat, key)
    N = H * W
    return np.stack([cf[:N].reshape(H, W), cf[N:2 * N].reshape(H, W),
                     cf[2 * N:].reshape(H, W)], axis=2).astype(np.uint8)


def flip_hb_bit(hb, bit=0):
    b = bytearray(hb)
    b[0] ^= (1 << bit)
    return bytes(b)


class Model:
    """Uniform interface: encrypt, decrypt, re-encrypt under a wrong hb."""

    def __init__(self, name, enc, dec, enc_wrong_hb, keystream, note):
        self.name, self.enc, self.dec = name, enc, dec
        self.enc_wrong_hb, self.keystream, self.note = enc_wrong_hb, keystream, note


def build_models(D, nkg):
    def dl1_enc(img):
        return encrypt_dl(img, D, nkg, use_nkg=True)

    def dl1_dec(c, hb):
        return decrypt_dl(c, D, hb, nkg, use_nkg=True)

    def dl1_wrong(img, hb_bad):
        return encrypt_dl_hb(img, D, nkg, hb_bad)

    def dl2_enc(img):
        return dlv2.encrypt_dl_2axis(img, D, nkg)

    def dl2_dec(c, hb):
        return dlv2.decrypt_dl_2axis(c, D, hb, nkg)

    def dl2_wrong(img, hb_bad):
        return dlv2.encrypt_dl_2axis(img, D, nkg, ics_hb_override=hb_bad)[0]

    def v4_enc(img):
        return v4.encrypt_v4(img, D)

    def v4_dec(c, hb):
        return v4.decrypt_v4(c, D, hb)

    def v4_wrong(img, hb_bad):
        # A wrong hb changes BOTH the pass-1 ICs and the hash-chained pass-2
        # ICs -- what a decryptor holding corrupted metadata would see.
        return v4.encrypt_v4(img, D,
                             ics_override=v3.ics_from_hash_bytes(hb_bad),
                             hb_override=hb_bad)[0]

    def dl_ks(hb):
        return np.concatenate([k.ravel() for k in
                               nkg.generate_channel_keystreams(hb, v3.IMG_SIZE)])

    def v4_ks(hb):
        ics = v3.ics_from_hash_bytes(hb)
        return np.concatenate([v3.build_keystream(ics[c], v3.IMG_SIZE).ravel()
                               for c in range(3)])

    return [
        Model("LS-DPAK-DL (1-axis)", dl1_enc, dl1_dec, dl1_wrong, dl_ks,
              "neural keystream, row-major diffusion only"),
        Model("LS-DPAK-DL v2 (2-axis)", dl2_enc, dl2_dec, dl2_wrong, dl_ks,
              "neural keystream, row + column diffusion"),
        Model("LS-DPAK v4 (2-axis)", v4_enc, v4_dec, v4_wrong, v4_ks,
              "Lorenz/Rossler keystream, row + column diffusion"),
    ]


# ------------------------------------------------------------------- main ----
def eval_model(m, img):
    t0 = time.perf_counter()
    cipher, hb = m.enc(img)
    enc_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    dec = m.dec(cipher, hb)
    dec_ms = (time.perf_counter() - t0) * 1e3
    roundtrip = bool(np.array_equal(img, dec))

    # Differential: centre pixel, R channel, +1 mod 256 (v4's protocol,
    # applied uniformly to all three models here).
    img2 = img.copy()
    cy = cx = v3.IMG_SIZE // 2
    img2[cy, cx, 0] = (int(img2[cy, cx, 0]) + 1) % 256
    c_diff, _ = m.enc(img2)
    npcr_d, uaci_d = npcr_uaci(cipher, c_diff)

    # Key sensitivity: flip 1 bit of the transmitted metadata, same plaintext.
    c_key = m.enc_wrong_hb(img, flip_hb_bit(hb))
    npcr_k, uaci_k = npcr_uaci(cipher, c_key)

    ch = {d: {n: corr(cipher[:, :, i], d) for i, n in enumerate(CH)}
          for d in ("h", "v", "d")}
    allc = [abs(ch[d][n]) for d in ch for n in CH]

    return {
        "entropy": entropy_rgb(cipher),
        "corr_h": ch["h"], "corr_v": ch["v"], "corr_d": ch["d"],
        "max_abs_corr": float(max(allc)), "mean_abs_corr": float(np.mean(allc)),
        "interchannel_corr": v3.interchannel_corr(cipher),
        "chi2": chi2_uniformity(cipher),
        "npcr_diff": npcr_d, "uaci_diff": uaci_d,
        "npcr_key_hbflip": npcr_k, "uaci_key_hbflip": uaci_k,
        "enc_ms": enc_ms, "dec_ms": dec_ms, "roundtrip": roundtrip,
    }


def keystream_quality(m, hb):
    """Is the raw keystream itself uniform? (fp32 all-zeros canary included.)"""
    ks = m.keystream(hb)
    h, _ = np.histogram(ks, 256, (0, 256))
    e = h.sum() / 256.0
    stat = float(np.sum((h - e) ** 2 / e))
    hp = h / h.sum()
    hp = hp[hp > 0]
    return {
        "n_bytes": int(ks.size),
        "mean": float(ks.mean()), "std": float(ks.std()),
        "min": int(ks.min()), "max": int(ks.max()),
        "distinct_values": int(np.unique(ks).size),
        "entropy": float(-np.sum(hp * np.log2(hp))),
        "chi2_stat": stat, "chi2_p": float(1.0 - chi2.cdf(stat, df=255)),
        "all_zero": bool(ks.max() == 0),
    }


def main():
    D = v3.build_D(32)
    ckpt = os.path.join(_DL, "checkpoints", "nkg_gan.pt")
    nkg = NKG().to(DEVICE)
    nkg.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    nkg.eval()
    print(f"NKG checkpoint : {os.path.basename(ckpt)}")
    print(f"torch          : {torch.__version__}  (cuda={torch.cuda.is_available()})")
    print(f"images         : {', '.join(IMAGES)}   [Baboon.tiff absent]")

    models = build_models(D, nkg)

    # Guard: encrypt_dl_hb must equal encrypt_dl on the genuine hb.
    for name in IMAGES:
        img = v3.load_image_rgb(name)
        ref, hb = encrypt_dl(img, D, nkg, use_nkg=True)
        assert np.array_equal(ref, encrypt_dl_hb(img, D, nkg, hb)), \
            f"encrypt_dl_hb diverges from encrypt_dl on {name}"
    print("guard          : encrypt_dl_hb == encrypt_dl on all images  [OK]")

    out = {"meta": {"images": IMAGES, "torch": torch.__version__,
                    "nkg_ckpt": os.path.basename(ckpt),
                    "differential": "centre pixel R +1 mod 256",
                    "key_sensitivity": "1-bit flip of transmitted hb[0]"},
           "models": {}, "keystream": {}, "jha_2025": JHA}

    hb_ref = v3.image_hash_bytes_rgb(v3.load_image_rgb("lena"))
    for m in models:
        print(f"\n=== {m.name}  ({m.note})")
        print(f"{'image':8s} {'E_R':>9s} {'E_pool':>8s} {'maxCorr':>8s} "
              f"{'NPCRd':>9s} {'UACId':>8s} {'NPCRk':>9s} {'UACIk':>8s} "
              f"{'enc_ms':>8s} {'RT':>4s}")
        per = {}
        for name in IMAGES:
            img = v3.load_image_rgb(name)
            r = eval_model(m, img)
            per[name] = r
            print(f"{name:8s} {r['entropy']['R']:9.5f} "
                  f"{r['entropy']['joint_pooled']:8.5f} {r['max_abs_corr']:8.5f} "
                  f"{r['npcr_diff']:9.5f} {r['uaci_diff']:8.5f} "
                  f"{r['npcr_key_hbflip']:9.5f} {r['uaci_key_hbflip']:8.5f} "
                  f"{r['enc_ms']:8.1f} {'ok' if r['roundtrip'] else 'FAIL':>4s}")
        out["models"][m.name] = per
        out["keystream"][m.name] = keystream_quality(m, hb_ref)

    # v4-only: the 1e-14 initial-condition perturbation its own script reports.
    print("\n=== LS-DPAK v4 only: 1e-14 Lorenz x0 perturbation (hb unchanged)")
    v4_ic = {}
    for name in IMAGES:
        img = v3.load_image_rgb(name)
        cipher, hb = v4.encrypt_v4(img, D)
        ics = v3.ics_from_hash_bytes(hb)
        for c in range(3):
            ics[c]["lx"] += 1e-14
        c2, _ = v4.encrypt_v4(img, D, ics_override=ics, hb_override=hb)
        n, u = npcr_uaci(cipher, c2)
        v4_ic[name] = {"npcr_key_ic1e14": n, "uaci_key_ic1e14": u}
        print(f"{name:8s} NPCR={n:9.5f}  UACI={u:8.5f}")
    out["v4_ic_perturbation"] = v4_ic

    json.dump(out, open(os.path.join(_RES, "comparison.json"), "w"), indent=2)
    print(f"\nSaved -> comparison/results/comparison.json")
    return out


if __name__ == "__main__":
    main()
