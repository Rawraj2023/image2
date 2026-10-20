"""
LS-DPAK-DL grayscale -- cipher built on the NATIVELY-TRAINED NKGGray model.

Supersedes the earlier version of this file, which reused nkg_gan.pt (trained
on 21-byte RGB hash slices) by zero-padding v5's 7-byte hb to 21 bytes. That
was a repurposing of weights trained on data this repo doesn't actually have:
every one of the five source images is mode "L" (single-channel grayscale) --
verified by checking max|R-G| = max|R-B| = max|G-B| = 0 after .convert("RGB"),
meaning every "RGB" result anywhere in this project's earlier work was
computed on a grayscale image replicated into three identical channels. There
was never a 3-channel input to train an RGB-shaped network on.

This version uses NKGGray (lsdpak_dl_gray_model.py): a 7-byte embedder, no
per-channel split, trained from scratch (train_dl_gray.py, same 3-phase
recipe as train.py: uniformity pre-training, GAN fine-tuning, denoiser) on
the actual grayscale images in test_images/. Checkpoints in
checkpoints_gray/.

Pipeline, identical in structure to lsdpak_gray_cpu.encrypt_gray /
lsdpak-v4-2d-diffusion's two-axis design, with NKGGray in place of the
chaotic keystream:

    hb   = SHA-256(img)[:7]
    K1   = NKGGray(hb)                      # replaces build_keystream(ics1)
    x    = scramble3 -> shuffle_blocks_full -> lsq_column_shuffle
    c1   = bidir_encrypt(x, K1)               # pass 1, row-major

    hb2  = SHA-256(hb)[:7]
    K2   = NKGGray(hb2)                     # replaces build_keystream(ics2)
    cipher = bidir_encrypt(c1.T, K2.T).T      # pass 2, column axis

A structural note on key sensitivity (unchanged from the previous version):
v5's protocol B perturbs a continuous Lorenz x0 by 1e-14 while hb (and
therefore K2) stays fixed. NKGGray has no continuous key -- the only key
material is hb, and K2 is chained from hb -- so there is no way to perturb
"pass 1 only" without touching pass 2 as well. The key-sensitivity test here
is a 1-bit hb flip, changing BOTH K1 and K2, labelled protocol A' rather
than B for that reason.
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

from lsdpak_dl_gray_model import NKGGray, DEVICE
import lsdpak_gray_cpu as gray

IMG_SIZE = gray.IMG_SIZE
CKPT_DIR = os.path.join(_HERE, "checkpoints_gray")


def second_pass_hb(hb7: bytes) -> bytes:
    return hashlib.sha256(bytes(hb7)).digest()[:7]


# --------------------------------------------------------- single-axis -----
def encrypt_dl_gray(img: np.ndarray, D: np.ndarray, nkg: NKGGray,
                    hb_override: bytes = None):
    hb = hb_override if hb_override is not None else gray.image_hash_bytes(img)
    K1 = nkg.generate_keystream(hb, img.shape[0])

    x = gray.scramble3(img, D)
    x = gray.shuffle_blocks_full(x)
    x = gray.lsq_column_shuffle(x, gray.build_D5(D.shape[0]))

    cipher = gray.bidir_encrypt(x.ravel(), K1.ravel()).reshape(img.shape)
    return cipher, hb


def decrypt_dl_gray(cipher: np.ndarray, hb: bytes, D: np.ndarray, nkg: NKGGray):
    K1 = nkg.generate_keystream(hb, cipher.shape[0])
    x = gray.bidir_decrypt(cipher.ravel(), K1.ravel()).reshape(cipher.shape)
    x = gray.lsq_column_unshuffle(x, gray.build_D5(D.shape[0]))
    x = gray.unshuffle_blocks_full(x)
    return gray.unscramble3(x, D)


# ------------------------------------------------------------ two-axis -----
def encrypt_dl_gray_2axis(img: np.ndarray, D: np.ndarray, nkg: NKGGray,
                          hb_override: bytes = None):
    hb = hb_override if hb_override is not None else gray.image_hash_bytes(img)
    K1 = nkg.generate_keystream(hb, img.shape[0])

    x = gray.scramble3(img, D)
    x = gray.shuffle_blocks_full(x)
    x = gray.lsq_column_shuffle(x, gray.build_D5(D.shape[0]))

    c_row = gray.bidir_encrypt(x.ravel(), K1.ravel()).reshape(img.shape)

    hb2 = second_pass_hb(hb)
    K2 = nkg.generate_keystream(hb2, img.shape[0])

    c_row_T = np.ascontiguousarray(c_row.T)
    K2_T = np.ascontiguousarray(K2.T)
    cipher = gray.bidir_encrypt(c_row_T.ravel(), K2_T.ravel())
    cipher = cipher.reshape(img.shape).T.copy()
    return cipher, hb


def decrypt_dl_gray_2axis(cipher: np.ndarray, hb: bytes, D: np.ndarray, nkg: NKGGray):
    K1 = nkg.generate_keystream(hb, cipher.shape[0])
    hb2 = second_pass_hb(hb)
    K2 = nkg.generate_keystream(hb2, cipher.shape[0])

    cipher_T = np.ascontiguousarray(cipher.T)
    K2_T = np.ascontiguousarray(K2.T)
    c_row_T = gray.bidir_decrypt(cipher_T.ravel(), K2_T.ravel())
    c_row = c_row_T.reshape(cipher.shape).T.copy()

    x = gray.bidir_decrypt(c_row.ravel(), K1.ravel()).reshape(cipher.shape)
    x = gray.lsq_column_unshuffle(x, gray.build_D5(D.shape[0]))
    x = gray.unshuffle_blocks_full(x)
    return gray.unscramble3(x, D)


def _load_nkg(ckpt_dir=None):
    ckpt_dir = ckpt_dir or CKPT_DIR
    path = os.path.join(ckpt_dir, "nkg_gray_gan.pt")
    if not os.path.exists(path):
        path = os.path.join(ckpt_dir, "nkg_gray_pretrained.pt")
    nkg = NKGGray().to(DEVICE)
    nkg.load_state_dict(torch.load(path, map_location=DEVICE))
    nkg.eval()
    return nkg, os.path.basename(path)


if __name__ == "__main__":
    nkg, ckpt_name = _load_nkg()
    D = gray.build_D()
    print(f"NKGGray checkpoint: {ckpt_name} (trained natively on grayscale, 7-byte hb)")
    print(f"{'image':8s} {'1-axis RT':>10s} {'2-axis RT':>10s}")
    for name in gray.IMAGE_NAMES:
        img = gray.load_image_gray(name)

        c1, hb1 = encrypt_dl_gray(img, D, nkg)
        ok1 = np.array_equal(img, decrypt_dl_gray(c1, hb1, D, nkg))

        c2, hb2 = encrypt_dl_gray_2axis(img, D, nkg)
        ok2 = np.array_equal(img, decrypt_dl_gray_2axis(c2, hb2, D, nkg))

        print(f"{name:8s} {'ok' if ok1 else 'FAIL':>10s} {'ok' if ok2 else 'FAIL':>10s}")
