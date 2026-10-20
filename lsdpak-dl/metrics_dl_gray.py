"""
Metrics for LS-DPAK-DL grayscale -- the NKG keystream (lsdpak_dl.py's
nkg_gan.pt, unmodified) dropped into v5's single-channel pipeline.

Produces the fourth cell the article's comparison table was missing: a
learned keystream measured on the SAME colour space, SAME images, and SAME
two protocols as Contribution B (v5 chaotic grayscale), so "learned vs
chaotic" can be read off one row difference instead of inferred across two
different pipelines (RGB DL vs grayscale chaotic).

Two protocols, kept separate (same discipline as metrics_gray.py and for
the same reason -- conflating them reproduces the exact criticism this
project makes of Jha et al.):

  Protocol A (differential plaintext): img[128,128] += 1 mod 256. hb
    changes, so NKG regenerates K1 AND (for the 2-axis variant) K2.

  Protocol A' (hb-bit-flip "key sensitivity"): same plaintext, hb's first
    byte's LSB flipped. NOT the same test as v5's protocol B -- v5 perturbs
    a continuous Lorenz x0 by 1e-14 while hb (and therefore K2) stays
    completely fixed; NKG has no continuous key, only the 7 hash bytes, and
    flipping any of them changes BOTH K1 and K2 simultaneously since K2 is
    hash-chained from hb. Labelled A' rather than B to make that structural
    difference impossible to miss when reading the table.

Also runs the NIST SP 800-22 battery (via nist_tests.run_all_tests) and a
chi-square uniformity check directly on THIS keystream -- the padded-to-21
NKG output on 7-byte grayscale hb -- rather than assuming the RGB-channel
result transfers. The 14 zero-padding bytes are out of the training
distribution, so this is a measurement, not an inference.
"""

import json
import os
import sys
import time

import numpy as np
from scipy.stats import chi2

_HERE = os.path.dirname(os.path.abspath(__file__))
_V5 = os.path.join(_HERE, "..", "lsdpak-v5-grayscale")
_RES = os.path.join(_HERE, "results")
os.makedirs(_RES, exist_ok=True)
sys.path.insert(0, _HERE)
sys.path.insert(0, _V5)

import lsdpak_gray_cpu as gray
import lsdpak_dl_gray as dlg
import nist_tests as nist


def flip_hb_bit(hb: bytes, bit: int = 0) -> bytes:
    b = bytearray(hb)
    b[0] ^= (1 << bit)
    return bytes(b)


def entropy(img):
    h, _ = np.histogram(img.ravel(), 256, (0, 256))
    p = h / h.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def corr(img, d):
    if d == "h":
        x, y = img[:, :-1].ravel(), img[:, 1:].ravel()
    elif d == "v":
        x, y = img[:-1, :].ravel(), img[1:, :].ravel()
    else:
        x, y = img[:-1, :-1].ravel(), img[1:, 1:].ravel()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])


def npcr_uaci(a, b):
    d = a.astype(int) - b.astype(int)
    return float((d != 0).mean() * 100), float(np.abs(d).mean() / 255 * 100)


def chi2_p(img):
    h, _ = np.histogram(img.ravel(), 256, (0, 256))
    e = h.sum() / 256.0
    stat = float(np.sum((h - e) ** 2 / e))
    return stat, float(1.0 - chi2.cdf(stat, df=255))


def evaluate(name, D, nkg, variant):
    enc = dlg.encrypt_dl_gray_2axis if variant == "2axis" else dlg.encrypt_dl_gray
    dec = dlg.decrypt_dl_gray_2axis if variant == "2axis" else dlg.decrypt_dl_gray

    img = gray.load_image_gray(name)

    t0 = time.perf_counter()
    cipher, hb = enc(img, D, nkg)
    enc_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    dec_img = dec(cipher, hb, D, nkg)
    dec_ms = (time.perf_counter() - t0) * 1e3
    roundtrip = bool(np.array_equal(img, dec_img))

    img_a = img.copy()
    img_a[128, 128] = (int(img_a[128, 128]) + 1) % 256
    c_a, _ = enc(img_a, D, nkg)
    npcr_a, uaci_a = npcr_uaci(cipher, c_a)

    c_ap, _ = enc(img, D, nkg, hb_override=flip_hb_bit(hb))
    npcr_ap, uaci_ap = npcr_uaci(cipher, c_ap)

    ch, cv, cd = corr(cipher, "h"), corr(cipher, "v"), corr(cipher, "d")
    stat, p = chi2_p(cipher)

    return {
        "entropy": entropy(cipher),
        "corr_h": ch, "corr_v": cv, "corr_d": cd,
        "max_corr": float(max(abs(ch), abs(cv), abs(cd))),
        "npcr_diff": npcr_a, "uaci_diff": uaci_a,
        "npcr_key_hbflip": npcr_ap, "uaci_key_hbflip": uaci_ap,
        "chi2_stat": stat, "chi2_p": p, "chi2_uniform": bool(p > 0.01),
        "hb": hb.hex(),
        "enc_ms": enc_ms, "dec_ms": dec_ms,
        "roundtrip": roundtrip,
    }


def keystream_battery(nkg):
    """NIST + chi2 + avalanche on THIS keystream, not the RGB one."""
    hb = gray.image_hash_bytes(gray.load_image_gray("lena"))
    ks = dlg.nkg_keystream_gray(nkg, hb, gray.IMG_SIZE)

    h, _ = np.histogram(ks.ravel(), 256, (0, 256))
    e = h.sum() / 256.0
    stat = float(np.sum((h - e) ** 2 / e))
    p = float(1.0 - chi2.cdf(stat, df=255))

    print(f"\n  chi2 on 65536 keystream bytes: stat={stat:.1f}  p={p:.4f}  "
          f"distinct={np.unique(ks).size}/256  mean={ks.mean():.2f}")

    nist_results = nist.run_all_tests(ks.ravel(), label="DL-gray NKG (padded 7->21)")
    nist_pass = sum(1 for r in nist_results if r["pass"])

    # Avalanche: all 56 single-bit flips of the 7 grayscale hash bytes.
    base = ks.astype(int)
    byte_pct, bit_pct, mean_abs = [], [], []
    for byte_i in range(7):
        for bit in range(8):
            b = bytearray(hb)
            b[byte_i] ^= (1 << bit)
            k2 = dlg.nkg_keystream_gray(nkg, bytes(b), gray.IMG_SIZE).astype(int)
            d = k2 - base
            byte_pct.append((d != 0).mean() * 100)
            bit_pct.append(np.unpackbits(
                (k2.astype(np.uint8) ^ base.astype(np.uint8))).mean() * 100)
            mean_abs.append(np.abs(d).mean())

    print(f"  avalanche over 56 single-bit hb flips: "
          f"bytes_changed={np.mean(byte_pct):.2f}%  "
          f"bits_changed={np.mean(bit_pct):.2f}%  "
          f"mean|delta|={np.mean(mean_abs):.2f}")

    return {
        "chi2_stat": stat, "chi2_p": p,
        "distinct_values": int(np.unique(ks).size),
        "mean_byte": float(ks.mean()),
        "nist_pass": nist_pass, "nist_total": len(nist_results),
        "nist_detail": [{"test": r.get("test", "?"), "pass": r["pass"]}
                        for r in nist_results],
        "avalanche_bytes_pct": float(np.mean(byte_pct)),
        "avalanche_bits_pct": float(np.mean(bit_pct)),
        "avalanche_mean_abs_delta": float(np.mean(mean_abs)),
    }


def main():
    D = gray.build_D()
    nkg, ckpt_name = dlg._load_nkg()
    print(f"NKG checkpoint: {ckpt_name}  (RGB-trained, reused unmodified on grayscale)")
    print(f"images: {', '.join(gray.IMAGE_NAMES)}\n")

    out = {"meta": {"nkg_ckpt": ckpt_name, "images": gray.IMAGE_NAMES,
                    "protocol_a": "differential plaintext: img[128,128] += 1 mod 256",
                    "protocol_ap": "hb-bit-flip; NOT v5's protocol B -- changes "
                                   "both K1 and K2, see module docstring"},
           "variants": {}}

    for variant in ("1axis", "2axis"):
        print(f"=== DL-gray {variant} ===")
        print(f"{'image':8s} {'entropy':>9s} {'maxCorr':>8s} {'NPCR_A':>8s} "
              f"{'UACI_A':>8s} {'NPCR_A2':>8s} {'UACI_A2':>8s} {'enc_ms':>8s} {'RT':>4s}")
        m = {}
        for name in gray.IMAGE_NAMES:
            r = evaluate(name, D, nkg, variant)
            m[name] = r
            print(f"{name:8s} {r['entropy']:9.5f} {r['max_corr']:8.5f} "
                  f"{r['npcr_diff']:8.4f} {r['uaci_diff']:8.4f} "
                  f"{r['npcr_key_hbflip']:8.4f} {r['uaci_key_hbflip']:8.4f} "
                  f"{r['enc_ms']:8.1f} {'ok' if r['roundtrip'] else 'FAIL':>4s}")
        avg = lambda k: float(np.mean([m[n][k] for n in gray.IMAGE_NAMES]))
        print(f"{'mean':8s} {avg('entropy'):9.5f} {avg('max_corr'):8.5f} "
              f"{avg('npcr_diff'):8.4f} {avg('uaci_diff'):8.4f} "
              f"{avg('npcr_key_hbflip'):8.4f} {avg('uaci_key_hbflip'):8.4f}")
        print(f"all round-trips: "
              f"{'PASS' if all(m[n]['roundtrip'] for n in gray.IMAGE_NAMES) else 'FAIL'}\n")
        out["variants"][variant] = {"images": m, "aggregate": {k: avg(k) for k in
                                    ("entropy", "max_corr", "npcr_diff", "uaci_diff",
                                     "npcr_key_hbflip", "uaci_key_hbflip")}}

    print("=== keystream battery (2-axis pass-1 key, lena's hb) ===")
    out["keystream"] = keystream_battery(nkg)

    json.dump(out, open(os.path.join(_RES, "metrics_dl_gray.json"), "w"), indent=2)
    print(f"\nSaved -> lsdpak-dl/results/metrics_dl_gray.json")
    return out


if __name__ == "__main__":
    main()
