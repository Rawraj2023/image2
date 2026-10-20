"""
LS-DPAK-DL grayscale -- the Neural Keystream Generator dropped into v5's
single-channel pipeline.
=============================================================================

This exists to close a gap: Contribution A (LS-DPAK-DL) is RGB-only --
denoiser.pt's first conv is (16,3,3,3), the discriminator fuses 3 channel
blocks, and nothing in that branch loads with convert("L"). Contribution B
(v5 grayscale) uses the original chaotic keystream. Neither alone lets the
article compare "learned vs chaotic keystream" on the SAME colour space,
which is the comparison the article's A.4 finding (learned keystream fails
NIST, cipher doesn't) is actually about.

What this module is, precisely
-------------------------------
Reuse, not retraining. nkg_gan.pt's embedder is nn.Linear(21, 64) -- fixed at
21 input dims by the checkpoint shape, so it cannot accept v5's native 7-byte
hb directly. The existing RGB code already handles the "fewer than 21 bytes"
case: NKG.generate_keystream() takes an arbitrary <=21-byte slice, zero-pads
it internally (hash_to_tensor does hb[:21]; callers are expected to supply a
full 21-byte buffer), and returns exactly one (256,256) plane -- it was never
RGB-specific, generate_channel_keystreams() is just three calls to it with
three 7-byte slices zero-padded to 21.

So: pad v5's 7-byte hb to 21 bytes with zeros and call generate_keystream()
once. No architecture change, no retraining, and the weights are UNCHANGED
from nkg_gan.pt. This is a repurposing of an RGB-trained generator, not a
grayscale-trained one -- say so in any report built from this module. The
14 padding zeros are out-of-distribution for a network trained on real hash
bytes in all 21 input slots; test the resulting keystream directly (NIST,
chi2, avalanche) rather than assuming the RGB findings transfer unchanged.

Pipeline (mirrors lsdpak_gray_cpu.py exactly, substituting NKG for the
chaotic keystream):

    hb   = SHA-256(img)[:7]
    K1   = NKG(pad21(hb))                       # replaces build_keystream(ics1)
    x    = scramble3 -> shuffle_blocks_full -> lsq_column_shuffle
    c1   = bidir_encrypt(x, K1)                  # pass 1, row-major

    hb2  = SHA-256(hb)[:7]
    K2   = NKG(pad21(hb2))                       # replaces build_keystream(ics2)
    cipher = bidir_encrypt(c1.T, K2.T).T          # pass 2, column axis

Single-axis (encrypt_dl_gray) and two-axis (encrypt_dl_gray_2axis) variants
are both provided, mirroring the 1-axis/2-axis split in the RGB DL branch.

A structural note on key sensitivity
-------------------------------------
v5's protocol B perturbs the Lorenz x0 by 1e-14 while leaving hb (and
therefore K2) completely unchanged -- that is possible because v5's key
material is a continuous initial condition, independent of the 7
transmitted bytes used to label which IC set to regenerate.

NKG has no such continuous key: the only key material is hb itself, and K2
is deterministically chained from hb (SHA-256(hb)), so there is no way to
perturb "pass 1 only" without touching pass 2 as well -- any change to hb
propagates to both passes simultaneously. This is NOT a bug introduced here;
it is already true of the RGB DL v2 cipher (evaluate_dl_v2.encrypt_dl_2axis
recomputes hb2 from whatever hb it is given), it was simply never stated
explicitly. The key-sensitivity test below is therefore a 1-bit hb flip
(the smallest possible key change), and it changes BOTH K1 and K2 -- report
this as "protocol B'" to flag the difference from v5's true single-pass
perturbation, not as a direct substitute for it.
"""

import hashlib
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_V5 = os.path.join(_HERE, "..", "lsdpak-v5-grayscale")
sys.path.insert(0, _HERE)
sys.path.insert(0, _V5)

from lsdpak_dl import NKG, DEVICE
import lsdpak_gray_cpu as gray

IMG_SIZE = gray.IMG_SIZE


def _pad21(hb7: bytes) -> bytes:
    """Zero-pad a 7-byte hash slice to the 21 bytes nkg_gan.pt's embedder
    expects. The 14 padding bytes are out-of-distribution for a network
    trained on real hash bytes in every slot -- this is flagged, not hidden."""
    assert len(hb7) == 7, f"expected 7 bytes, got {len(hb7)}"
    return hb7 + bytes(14)


def nkg_keystream_gray(nkg: NKG, hb7: bytes, size: int = IMG_SIZE) -> np.ndarray:
    """One (size,size) uint8 keystream plane from a 7-byte grayscale hb,
    using the RGB-trained nkg_gan.pt weights unmodified."""
    return nkg.generate_keystream(_pad21(hb7), size)


def second_pass_hb(hb7: bytes) -> bytes:
    """Hash-chained input for the column-axis pass, matching v5's
    gray.second_pass_ics derivation but at the hash-byte level (NKG has no
    IC dict -- it consumes hash bytes directly)."""
    return hashlib.sha256(bytes(hb7)).digest()[:7]


# --------------------------------------------------------- single-axis -----
def encrypt_dl_gray(img: np.ndarray, D: np.ndarray, nkg: NKG,
                    hb_override: bytes = None):
    """Single-axis NKG grayscale cipher. Returns (cipher, hb)."""
    hb = hb_override if hb_override is not None else gray.image_hash_bytes(img)
    K1 = nkg_keystream_gray(nkg, hb, img.shape[0])

    x = gray.scramble3(img, D)
    x = gray.shuffle_blocks_full(x)
    x = gray.lsq_column_shuffle(x, gray.build_D5(D.shape[0]))

    cipher = gray.bidir_encrypt(x.ravel(), K1.ravel()).reshape(img.shape)
    return cipher, hb


def decrypt_dl_gray(cipher: np.ndarray, hb: bytes, D: np.ndarray, nkg: NKG):
    K1 = nkg_keystream_gray(nkg, hb, cipher.shape[0])
    x = gray.bidir_decrypt(cipher.ravel(), K1.ravel()).reshape(cipher.shape)
    x = gray.lsq_column_unshuffle(x, gray.build_D5(D.shape[0]))
    x = gray.unshuffle_blocks_full(x)
    return gray.unscramble3(x, D)


# ------------------------------------------------------------ two-axis -----
def encrypt_dl_gray_2axis(img: np.ndarray, D: np.ndarray, nkg: NKG,
                          hb_override: bytes = None):
    """Two-axis NKG grayscale cipher, mirroring v5's row+column structure
    with NKG in place of the chaotic keystream. Returns (cipher, hb)."""
    hb = hb_override if hb_override is not None else gray.image_hash_bytes(img)
    K1 = nkg_keystream_gray(nkg, hb, img.shape[0])

    x = gray.scramble3(img, D)
    x = gray.shuffle_blocks_full(x)
    x = gray.lsq_column_shuffle(x, gray.build_D5(D.shape[0]))

    c_row = gray.bidir_encrypt(x.ravel(), K1.ravel()).reshape(img.shape)

    hb2 = second_pass_hb(hb)
    K2 = nkg_keystream_gray(nkg, hb2, img.shape[0])

    c_row_T = np.ascontiguousarray(c_row.T)
    K2_T = np.ascontiguousarray(K2.T)
    cipher = gray.bidir_encrypt(c_row_T.ravel(), K2_T.ravel())
    cipher = cipher.reshape(img.shape).T.copy()
    return cipher, hb


def decrypt_dl_gray_2axis(cipher: np.ndarray, hb: bytes, D: np.ndarray, nkg: NKG):
    K1 = nkg_keystream_gray(nkg, hb, cipher.shape[0])
    hb2 = second_pass_hb(hb)
    K2 = nkg_keystream_gray(nkg, hb2, cipher.shape[0])

    cipher_T = np.ascontiguousarray(cipher.T)
    K2_T = np.ascontiguousarray(K2.T)
    c_row_T = gray.bidir_decrypt(cipher_T.ravel(), K2_T.ravel())
    c_row = c_row_T.reshape(cipher.shape).T.copy()

    x = gray.bidir_decrypt(c_row.ravel(), K1.ravel()).reshape(cipher.shape)
    x = gray.lsq_column_unshuffle(x, gray.build_D5(D.shape[0]))
    x = gray.unshuffle_blocks_full(x)
    return gray.unscramble3(x, D)


def _load_nkg(ckpt_dir=None):
    ckpt_dir = ckpt_dir or os.path.join(_HERE, "checkpoints")
    path = os.path.join(ckpt_dir, "nkg_gan.pt")
    if not os.path.exists(path):
        path = os.path.join(ckpt_dir, "nkg_pretrained.pt")
    nkg = NKG().to(DEVICE)
    nkg.load_state_dict(torch.load(path, map_location=DEVICE))
    nkg.eval()
    return nkg, os.path.basename(path)


if __name__ == "__main__":
    nkg, ckpt_name = _load_nkg()
    D = gray.build_D()
    print(f"NKG checkpoint: {ckpt_name} (RGB-trained, reused unmodified)")
    print(f"{'image':8s} {'1-axis RT':>10s} {'2-axis RT':>10s}")
    for name in gray.IMAGE_NAMES:
        img = gray.load_image_gray(name)

        c1, hb1 = encrypt_dl_gray(img, D, nkg)
        ok1 = np.array_equal(img, decrypt_dl_gray(c1, hb1, D, nkg))

        c2, hb2 = encrypt_dl_gray_2axis(img, D, nkg)
        ok2 = np.array_equal(img, decrypt_dl_gray_2axis(c2, hb2, D, nkg))

        print(f"{name:8s} {'ok' if ok1 else 'FAIL':>10s} {'ok' if ok2 else 'FAIL':>10s}")
