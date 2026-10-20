"""
Grayscale metrics for LS-DPAK v5, against Jha et al. (2025).

    python metrics_gray.py

Writes results/metrics_gray.json and prints the comparison tables.

The two NPCR/UACI protocols are reported as separate columns, and that
separation is the point. The critique of Jha et al. is precisely that their
Table 6 is cited in their Sec. 5.6 as a differential-plaintext result and in
Sec. 5.7 as a key-sensitivity result, when only the second protocol could have
produced the reported numbers. Collapsing the two here would reproduce the
exact error being criticised.

  Protocol A -- differential plaintext: same master key, one plaintext pixel
    incremented. Because the ICs are SHA-256-adaptive, this changes hb and
    therefore the entire keystream. Against a single non-feedback XOR the
    ceiling here would be 1/65536 = 0.0015%.

  Protocol B -- key sensitivity: same plaintext, pass-1 Lorenz x0 perturbed by
    1e-14, hb (and hence the pass-2 key) held fixed. This is what Jha et al.'s
    Sec. 5.7 describes, so their Table 6 belongs against this column.

Note the output shape differs from metrics_v4.json: `entropy` is a scalar here
where the RGB file had an object with R/G/B/joint keys, and the RGB-only
`interchannel_corr` is absent. Anything consuming both files has to handle
that rather than reshaping one to look like the other.
"""

import json
import os
import sys
import time

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lsdpak_gray_cpu as cpu

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, "results")
os.makedirs(_RES, exist_ok=True)

# Jha et al. 2025, Security and Privacy 8:e70128 -- their Tables 2, 4 and 6.
JHA = {
    "aerial": dict(entropy=7.9973, npcr=99.6094, uaci=28.5589,
                   corr_h=0.0044187, corr_v=0.0046461, corr_d=0.0058410),
    "baboon": dict(entropy=7.9976, npcr=99.5865, uaci=26.6999,
                   corr_h=-0.0011344, corr_v=-0.0063508, corr_d=0.0011699),
    "clock": dict(entropy=7.9968, npcr=99.6017, uaci=35.5581,
                  corr_h=0.0039243, corr_v=-0.0025781, corr_d=0.0027459),
    "couple": dict(entropy=7.9969, npcr=99.6246, uaci=28.0631,
                   corr_h=-0.0049054, corr_v=0.0014872, corr_d=0.00012246),
    "lena": dict(entropy=7.9974, npcr=99.6277, uaci=28.7091,
                 corr_h=-0.000092173, corr_v=-0.0074797, corr_d=0.0001208),
    "man": dict(entropy=7.9970, npcr=99.5865, uaci=28.8974,
                corr_h=-0.0034279, corr_v=-0.00037986, corr_d=0.00089797),
}

IDEAL_NPCR, IDEAL_UACI = 99.6094, 33.4635
# 2000-trial Monte Carlo on uniform-random 256x256 images -- the best any
# cipher could do. ~40% of trials exceed 0.002, so a sub-0.002 target is noise.
NOISE_FLOOR_MEAN_ABS_R = 0.00316


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


def evaluate(name, D):
    img = cpu.load_image_gray(name)

    # One batched ODE call covers every lane this image needs: pass-1 and
    # pass-2 keys for the original, the pixel-flipped variant, and the
    # key-perturbed variant.
    hb, ics_base = cpu.pass_ics(img)

    img_a = img.copy()
    img_a[128, 128] = (int(img_a[128, 128]) + 1) % 256
    hb_a, ics_a = cpu.pass_ics(img_a)

    ics_pert = cpu.ics_from_hash_bytes(hb)
    ics_pert["lx"] += 1e-14
    _, ics_b = cpu.pass_ics(img, ics_override=ics_pert, hb_override=hb)

    t0 = time.perf_counter()
    keys = cpu.build_keystream_many(ics_base + ics_a + ics_b)
    ks_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    cipher, _ = cpu.encrypt_gray(img, D, keys=keys[0:2])
    enc_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    dec = cpu.decrypt_gray(cipher, hb, D, keys=keys[0:2])
    dec_ms = (time.perf_counter() - t0) * 1e3

    c_a, _ = cpu.encrypt_gray(img_a, D, keys=keys[2:4])
    c_b, _ = cpu.encrypt_gray(img, D, keys=keys[4:6], hb_override=hb)

    npcr_a, uaci_a = npcr_uaci(cipher, c_a)
    npcr_b, uaci_b = npcr_uaci(cipher, c_b)
    ch, cv, cd = corr(cipher, "h"), corr(cipher, "v"), corr(cipher, "d")
    stat, p = chi2_p(cipher)

    return {
        "entropy": entropy(cipher),
        "entropy_plain": entropy(img),
        "corr_h": ch, "corr_v": cv, "corr_d": cd,
        "max_corr": float(max(abs(ch), abs(cv), abs(cd))),
        "mean_abs_corr": float(np.mean([abs(ch), abs(cv), abs(cd)])),
        "npcr_diff": npcr_a, "uaci_diff": uaci_a,
        "npcr_key": npcr_b, "uaci_key": uaci_b,
        "chi2_stat": stat, "chi2_p": p, "chi2_uniform": bool(p > 0.01),
        "hb": hb.hex(),
        "keystream_ms": ks_ms, "enc_ms": enc_ms, "dec_ms": dec_ms,
        "roundtrip": bool(np.array_equal(img, dec)),
    }


def main():
    D = cpu.build_D()
    print("LS-DPAK v5 grayscale - metrics")
    print(f"images: {', '.join(cpu.IMAGE_NAMES)}")
    print("Baboon.tiff and Cameraman are absent; Jha et al. averaged over 7 images.\n")

    print(f"{'image':8s} {'entropy':>9s} {'corr_h':>10s} {'corr_v':>10s} {'corr_d':>10s} "
          f"{'NPCR_A':>9s} {'UACI_A':>8s} {'NPCR_B':>9s} {'UACI_B':>8s} "
          f"{'chi2_p':>7s} {'enc_ms':>8s} {'RT':>4s}")
    print("-" * 118)
    m = {}
    for n in cpu.IMAGE_NAMES:
        r = evaluate(n, D)
        m[n] = r
        print(f"{n:8s} {r['entropy']:9.5f} {r['corr_h']:10.6f} {r['corr_v']:10.6f} "
              f"{r['corr_d']:10.6f} {r['npcr_diff']:9.4f} {r['uaci_diff']:8.4f} "
              f"{r['npcr_key']:9.4f} {r['uaci_key']:8.4f} {r['chi2_p']:7.4f} "
              f"{r['enc_ms']:8.1f} {'ok' if r['roundtrip'] else 'FAIL':>4s}")

    names = cpu.IMAGE_NAMES
    avg = lambda k: float(np.mean([m[n][k] for n in names]))
    all_r = [abs(m[n][d]) for n in names for d in ("corr_h", "corr_v", "corr_d")]

    print("-" * 118)
    print(f"{'mean':8s} {avg('entropy'):9.5f} {'':32s} "
          f"{avg('npcr_diff'):9.4f} {avg('uaci_diff'):8.4f} "
          f"{avg('npcr_key'):9.4f} {avg('uaci_key'):8.4f}")
    print(f"{'ideal':8s} {8.0:9.5f} {'':32s} {IDEAL_NPCR:9.4f} {IDEAL_UACI:8.4f} "
          f"{IDEAL_NPCR:9.4f} {IDEAL_UACI:8.4f}")

    print(f"\naggregate mean |r| over {len(all_r)} draws : {np.mean(all_r):.5f}")
    print(f"true-random noise floor (2000 trials) : {NOISE_FLOOR_MEAN_ABS_R:.5f}")
    print(f"worst single |r|                      : {max(all_r):.5f}")
    print(f"all round-trips                       : "
          f"{'PASS' if all(m[n]['roundtrip'] for n in names) else 'FAIL'}")
    print(f"all histograms uniform (chi2 p>0.01)  : "
          f"{'PASS' if all(m[n]['chi2_uniform'] for n in names) else 'FAIL'}")

    # ---- against Jha et al. -------------------------------------------------
    shared = [n for n in names if n in JHA]
    print(f"\n{'':8s} {'--- entropy ---':>24s}   {'--- NPCR % ---':>24s}   {'--- UACI % ---':>24s}")
    print(f"{'image':8s} {'v5 gray':>11s} {'Jha':>11s}   {'v5 (B)':>11s} {'Jha':>11s}   "
          f"{'v5 (B)':>11s} {'Jha':>11s}")
    print("-" * 90)
    for n in shared:
        j = JHA[n]
        print(f"{n:8s} {m[n]['entropy']:11.5f} {j['entropy']:11.4f}   "
              f"{m[n]['npcr_key']:11.4f} {j['npcr']:11.4f}   "
              f"{m[n]['uaci_key']:11.4f} {j['uaci']:11.4f}")
    print("-" * 90)
    print(f"{'mean':8s} {np.mean([m[n]['entropy'] for n in shared]):11.5f} "
          f"{np.mean([JHA[n]['entropy'] for n in shared]):11.4f}   "
          f"{np.mean([m[n]['npcr_key'] for n in shared]):11.4f} "
          f"{np.mean([JHA[n]['npcr'] for n in shared]):11.4f}   "
          f"{np.mean([m[n]['uaci_key'] for n in shared]):11.4f} "
          f"{np.mean([JHA[n]['uaci'] for n in shared]):11.4f}")

    jha_uaci_err = [abs(JHA[n]["uaci"] - IDEAL_UACI) for n in shared]
    v5_uaci_err = [abs(m[n]["uaci_key"] - IDEAL_UACI) for n in shared]
    print(f"\nmean |UACI - 33.4635| ideal deviation:  v5 {np.mean(v5_uaci_err):.4f}"
          f"   Jha {np.mean(jha_uaci_err):.4f}")
    print("Protocol A (differential plaintext) is the measurement Jha et al.")
    print("never performed; their Table 6 is a protocol-B result cited as both.")

    out = {
        "meta": {
            "images": names,
            "missing_images": ["baboon", "cameraman"],
            "protocol_a": "differential plaintext: img[128,128] += 1 mod 256",
            "protocol_b": "key sensitivity: pass-1 Lorenz x0 += 1e-14, hb fixed",
            "ideal_npcr": IDEAL_NPCR, "ideal_uaci": IDEAL_UACI,
            "noise_floor_mean_abs_r": NOISE_FLOOR_MEAN_ABS_R,
            "entropy_note": "scalar, not the RGB file's R/G/B/joint object",
        },
        "images": m,
        "aggregate": {
            "entropy": avg("entropy"),
            "mean_abs_corr": float(np.mean(all_r)),
            "worst_abs_corr": float(max(all_r)),
            "npcr_diff": avg("npcr_diff"), "uaci_diff": avg("uaci_diff"),
            "npcr_key": avg("npcr_key"), "uaci_key": avg("uaci_key"),
            "keystream_ms": avg("keystream_ms"),
            "enc_ms": avg("enc_ms"), "dec_ms": avg("dec_ms"),
        },
        "jha_2025": JHA,
    }
    json.dump(out, open(os.path.join(_RES, "metrics_gray.json"), "w"), indent=2)
    print("\nSaved -> lsdpak-v5-grayscale/results/metrics_gray.json")
    return out


if __name__ == "__main__":
    main()
