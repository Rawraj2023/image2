"""Aggregate comparison/results/comparison.json into the head-to-head table."""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(_HERE, "results", "comparison.json")))
models = list(d["models"])
imgs = d["meta"]["images"]
CH = ["R", "G", "B"]


def per_image(m, fn):
    return [fn(d["models"][m][i]) for i in imgs]


def row(label, fn, fmt="{:.5f}", ideal=""):
    vals = [np.mean(per_image(m, fn)) for m in models]
    cells = "  ".join(fmt.format(v).rjust(11) for v in vals)
    print(f"{label:<30s} {cells}   {ideal}")


print(f"Images: {', '.join(imgs)}   (Baboon.tiff absent from test_images/)")
print(f"Differential protocol : {d['meta']['differential']}")
print(f"Key-sensitivity       : {d['meta']['key_sensitivity']}")
print(f"torch {d['meta']['torch']}, NKG {d['meta']['nkg_ckpt']}, CPU\n")

hdr = "  ".join(m.replace("LS-DPAK", "").strip()[:11].rjust(11) for m in models)
print(f"{'metric':<30s} {hdr}   ideal / note")
print("-" * 95)

row("entropy, per-channel mean",
    lambda r: np.mean([r["entropy"][c] for c in CH]), "{:.5f}", "8.0 (7.997-7.998 typical)")
row("entropy, pooled RGB",
    lambda r: r["entropy"]["joint_pooled"], "{:.5f}", "pooling artifact, not a gain")
row("mean |corr| (45 draws)",
    lambda r: r["mean_abs_corr"], "{:.5f}", "0.00317 = random floor")
row("mean of per-image max |corr|",
    lambda r: r["max_abs_corr"], "{:.5f}", "0")
# True worst single draw over all 5 images x 3 channels x 3 directions.
worst = [max(per_image(m, lambda r: r["max_abs_corr"])) for m in models]
print(f"{'worst |corr| (1 of 45)':<30s} "
      + "  ".join("{:.5f}".format(v).rjust(11) for v in worst)
      + "   0.01336 = random max/1200")
row("mean of per-image max |corr_ic|",
    lambda r: max(abs(v) for v in r["interchannel_corr"].values()), "{:.5f}", "0")
row("NPCR differential  (%)",
    lambda r: r["npcr_diff"], "{:.4f}", "99.6094")
row("UACI differential  (%)",
    lambda r: r["uaci_diff"], "{:.4f}", "33.4635")
row("NPCR key-sens 1-bit hb (%)",
    lambda r: r["npcr_key_hbflip"], "{:.4f}", "99.6094")
row("UACI key-sens 1-bit hb (%)",
    lambda r: r["uaci_key_hbflip"], "{:.4f}", "33.4635")
row("chi2 min p-value (cipher)",
    lambda r: min(r["chi2"][c]["p"] for c in CH), "{:.4f}", "> 0.01 = uniform")
row("encrypt (ms)", lambda r: r["enc_ms"], "{:.0f}", "lower is better")
row("decrypt (ms)", lambda r: r["dec_ms"], "{:.0f}", "lower is better")
row("round-trip bit-exact",
    lambda r: 1.0 if r["roundtrip"] else 0.0, "{:.0f}", "1 = all 5 images pass")

print("\nraw keystream quality (196608 bytes from lena's hb):")
print(f"{'':<30s} {hdr}")
ks = d["keystream"]
for lbl, k, f in (("  keystream entropy", "entropy", "{:.5f}"),
                  ("  keystream chi2 p-value", "chi2_p", "{:.4f}"),
                  ("  keystream mean byte", "mean", "{:.3f}")):
    print(f"{lbl:<30s} " + "  ".join(f.format(ks[m][k]).rjust(11) for m in models))

v4ic = d["v4_ic_perturbation"]
print(f"\nv4 only, 1e-14 Lorenz x0 perturbation (hb held fixed):"
      f"  NPCR={np.mean([v['npcr_key_ic1e14'] for v in v4ic.values()]):.4f}%"
      f"  UACI={np.mean([v['uaci_key_ic1e14'] for v in v4ic.values()]):.4f}%")

j = d["jha_2025"]
print(f"\nJha et al. 2025 (grayscale): entropy {j['entropy']}, NPCR {j['npcr']}, "
      f"UACI {j['uaci']}, corr H/V/D {j['corr_h']}/{j['corr_v']}/{j['corr_d']}")
print("  -- their corr values are per-image single draws on grayscale; the rows")
print("     above aggregate 45 draws, so neither is comparable to the other.")
print("     True-random floor for 256x256: mean |r| 0.00317, max 0.01336/1200.")
