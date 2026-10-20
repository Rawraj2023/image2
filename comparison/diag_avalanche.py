"""Keystream avalanche: NKG (learned MLP) vs Lorenz/Rossler, over many 1-bit flips."""
import os, sys, json
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for p in ("lsdpak-v3-rgb", "lsdpak-dl"):
    sys.path.insert(0, os.path.join(ROOT, p))
import lsdpak_v3_rgb as v3
from lsdpak_dl import NKG, DEVICE

nkg = NKG().to(DEVICE)
nkg.load_state_dict(torch.load(os.path.join(ROOT, "lsdpak-dl", "checkpoints", "nkg_gan.pt"),
                               map_location=DEVICE))
nkg.eval()

hb = v3.image_hash_bytes_rgb(v3.load_image_rgb("lena"))

def nkg_ch0(h):
    return nkg.generate_channel_keystreams(h, 256)[0].astype(int)

def chaos_ch0(h):
    return v3.build_keystream(v3.ics_from_hash_bytes(h)[0], 256).astype(int)

rows = []
for gen_name, gen in (("NKG (learned MLP)", nkg_ch0), ("Lorenz/Rossler", chaos_ch0)):
    base = gen(hb)
    bytes_changed, mean_abs_delta, bit_changed = [], [], []
    for byte_i in range(7):            # only hb[0:7] feeds channel 0
        for bit in range(8):
            b = bytearray(hb); b[byte_i] ^= (1 << bit)
            k = gen(bytes(b))
            d = k - base
            bytes_changed.append((d != 0).mean() * 100)
            mean_abs_delta.append(np.abs(d).mean())
            bit_changed.append(np.unpackbits(
                (k.astype(np.uint8) ^ base.astype(np.uint8))).mean() * 100)
    rows.append((gen_name, np.mean(bytes_changed), np.mean(bit_changed),
                 np.mean(mean_abs_delta), np.min(bytes_changed), np.max(bytes_changed)))

print(f"{'keystream generator':22s} {'bytes chg %':>11s} {'bits chg %':>10s} "
      f"{'mean|delta|':>11s} {'min%':>7s} {'max%':>7s}")
print("-" * 76)
for r in rows:
    print(f"{r[0]:22s} {r[1]:11.3f} {r[2]:10.3f} {r[3]:11.3f} {r[4]:7.3f} {r[5]:7.3f}")
print("\nideal for a keystream generator: bytes ~99.61 %, bits ~50 %, mean|delta| ~85")
print("(averaged over all 56 single-bit flips of the 7 hash bytes feeding channel 0)")
