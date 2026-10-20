"""Why does 1-axis DL key sensitivity sit at 66.5% instead of 99.6%?"""
import os, sys, hashlib
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for p in ("lsdpak-v3-rgb", "lsdpak-dl", "lsdpak-v4-2d-diffusion", "comparison"):
    sys.path.insert(0, os.path.join(ROOT, p))

import lsdpak_v3_rgb as v3
from lsdpak_dl import NKG, DEVICE, encrypt_dl
import compare_models as cm

D = v3.build_D(32)
nkg = NKG().to(DEVICE)
nkg.load_state_dict(torch.load(os.path.join(ROOT, "lsdpak-dl", "checkpoints", "nkg_gan.pt"),
                               map_location=DEVICE))
nkg.eval()

img = v3.load_image_rgb("lena")
hb = v3.image_hash_bytes_rgb(img)
hb2 = cm.flip_hb_bit(hb)
print("hb  :", hb.hex())
print("hb' :", hb2.hex(), " (1 bit flipped in byte 0)")

print("\n-- NKG per-channel keystream: fraction of bytes that change --")
k1 = nkg.generate_channel_keystreams(hb, 256)
k2 = nkg.generate_channel_keystreams(hb2, 256)
for c, n in enumerate("RGB"):
    frac = (k1[c] != k2[c]).mean() * 100
    print(f"   {n}: {frac:7.3f}%   (channel input slice = hb[{c*7}:{(c+1)*7}])")

print("\n-- v4 pass-1 chaotic keystream (same disjoint 7-byte slicing) --")
i1, i2 = v3.ics_from_hash_bytes(hb), v3.ics_from_hash_bytes(hb2)
for c, n in enumerate("RGB"):
    a = v3.build_keystream(i1[c], 256)
    b = v3.build_keystream(i2[c], 256)
    print(f"   {n}: {(a != b).mean()*100:7.3f}%   ics equal={i1[c] == i2[c]}")

print("\n-- v4 pass-2 keystream (ICs from SHA-256(hb) -> full avalanche) --")
j1 = v3.ics_from_hash_bytes(hashlib.sha256(bytes(hb)).digest()[:21])
j2 = v3.ics_from_hash_bytes(hashlib.sha256(bytes(hb2)).digest()[:21])
for c, n in enumerate("RGB"):
    a = v3.build_keystream(j1[c], 256)
    b = v3.build_keystream(j2[c], 256)
    print(f"   {n}: {(a != b).mean()*100:7.3f}%")

print("\n-- control: v3 (1-axis CHAOTIC) under the SAME 1-bit hb flip --")
# encrypt_v3_rgb has no override hook, so mirror its pipeline with injectable keys.
def enc_1axis(img, keys):
    H, W = img.shape[:2]
    n = D.shape[0]
    D5 = v3.build_D5(n)
    sh = [v3.shuffle_blocks_full(v3.scramble3(img[:, :, c], D)) for c in range(3)]
    R, G, B = v3.interchannel_shuffle(sh[0], sh[1], sh[2], D5, H // n)
    cf = v3.bidir_encrypt_joint(
        np.concatenate([R.ravel(), G.ravel(), B.ravel()]),
        np.concatenate([keys[c].ravel() for c in range(3)]))
    N = H * W
    return np.stack([cf[:N].reshape(H, W), cf[N:2*N].reshape(H, W),
                     cf[2*N:].reshape(H, W)], axis=2).astype(np.uint8)

ck_ref = enc_1axis(img, [v3.build_keystream(i1[c], 256) for c in range(3)])
assert np.array_equal(ck_ref, v3.encrypt_v3_rgb(img, D)[0]), "v3 mirror diverges"
ck_bad = enc_1axis(img, [v3.build_keystream(i2[c], 256) for c in range(3)])
n, u = cm.npcr_uaci(ck_ref, ck_bad)
print(f"   v3 1-axis chaotic : NPCR={n:.5f}%  UACI={u:.5f}%   (mirror verified vs encrypt_v3_rgb)")

print("\n-- per-channel NPCR of the 1-axis DL ciphertext under the flip --")
cref, _ = encrypt_dl(img, D, nkg, use_nkg=True)
cbad = cm.encrypt_dl_hb(img, D, nkg, hb2)
for c, n2 in enumerate("RGB"):
    d = cref[:, :, c].astype(int) - cbad[:, :, c].astype(int)
    print(f"   {n2}: NPCR={(d != 0).mean()*100:7.3f}%  UACI={np.abs(d).mean()/255*100:7.3f}%")
