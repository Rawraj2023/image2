"""
LS-DPAK v5 -- grayscale, NumPy CPU reference implementation.
============================================================

This is the *authoritative* grayscale implementation. The GPU port
(`lsdpak_gray_gpu.py`) is validated bit-for-bit against it, and the golden
hashes in `golden/ciphertext_hashes.json` are frozen from it.

Why grayscale
-------------
The paper this work is benchmarked against (Jha et al., Security and Privacy
2025;8:e70128) is grayscale-only. The RGB implementation inflates two metrics
through measurement artifacts:

  * joint entropy ~7.9991 pools three channels into one 196,608-sample
    histogram, which lowers the estimator's finite-sample bias. It is not a
    security gain and grayscale cannot (and should not) reproduce it. The
    honest grayscale target is the RGB *per-channel* value, ~7.997.
  * `max_corr` in the RGB results is a maximum over 3 channels x 3 directions
    = 9 noisy draws per image. Grayscale has 3, so its worst case looks
    slightly better from order statistics alone -- not from better mixing.

Relationship to the RGB pipeline
--------------------------------
Everything structural is unchanged: the 4D-Lorenz + Rossler keystream, the
three-pass Latin-square scramble, the full 256-block shuffle, the D5 column
shuffle, and the two-axis (row then column) bidirectional diffusion. What is
dropped is only what three channels implied: the per-channel loops, the
[R||G||B] concatenation (so the diffusion chain is N = 65,536 rather than 3N),
the 21-byte metadata (now 7 bytes -- one IC set per pass), pooled joint
entropy, and interchannel correlation.

One rename: v3's `interchannel_shuffle` is `lsq_column_shuffle` here. The RGB
name is inaccurate -- despite the name and the surrounding write-up, that
function never moves data between R, G and B. It applies the same D5-driven
column-band permutation independently to each channel, making it structurally a
fourth Latin-square permutation pass. (Genuine cross-channel coupling in the
RGB version comes from the diffusion chain running over the concatenated
[R||G||B] stream, not from this function.) It is kept here, applied to the one
grayscale plane, because dropping it would remove a permutation stage and
change the correlation results for reasons unrelated to grayscale.

float64 is mandatory
--------------------
Keystream byte extraction is `floor(|v| * 1e14) % 256`, which reads the bottom
of the mantissa. In float32 every byte comes out 0: at magnitude ~2e15 a
float32 ULP is 2**27, every product is a multiple of 2**27, and
2**27 % 256 == 0. The cipher then degenerates to the permuted plaintext XORed
with zeros while still round-tripping cleanly. Worse, `(-3.0 + 1e-14) == -3.0`
is True in float32, so the key-sensitivity test silently measures nothing and
reports NPCR 0.000%. Run `verify_parity.py` -- it asserts fp64 end to end.
"""

import hashlib
import os

import numpy as np
from PIL import Image

IMG_SIZE = 256
N_ORDER = 32                      # Latin-square order
BAND = IMG_SIZE // N_ORDER        # 8
BLOCK = 16                        # block-shuffle block size
BLOCK_SEED = 2025
N_STEPS = 90_000
STEP_H = 0.01

# Master key: 4D Lorenz + Rossler initial conditions.
MASTER = dict(lx=-3.0, ly=2.0, lz=20.0, lw=0.5, rx=0.1, ry=0.0, rz=0.0)
SCALE = 2560.0                    # byte in [-128,127] -> IC shift in +/-0.05
IC_FIELDS = ("lx", "ly", "lz", "lw", "rx", "ry", "rz")

_HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(_HERE, "..", "test_images")

# Baboon.tiff and Cameraman are absent from test_images/. Jha et al. used seven
# images (these five plus Baboon and Cameraman); see README for the limitation.
TIFF_SRC = {
    "aerial": "Aerial.tiff",
    "lena": "Lena.jpg",
    "couple": "Couple.tiff",
    "clock": "Clock.tiff",
    "man": "Man.tiff",
}
IMAGE_NAMES = list(TIFF_SRC)


# ------------------------------------------------------------------ loading --
def load_image_gray(name, size=IMG_SIZE):
    """Load as single-channel uint8. PIL's "L" uses ITU-R 601-2 luma
    (0.299R + 0.587G + 0.114B), matching MATLAB rgb2gray, so this is
    consistent with how Jha et al. produced their grayscale inputs."""
    img = Image.open(os.path.join(IMAGE_DIR, TIFF_SRC[name])).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


# ------------------------------------------------------------ key schedule --
def image_hash_bytes(img):
    """7 metadata bytes transmitted with the ciphertext (21 in the RGB version,
    which needed 7 per channel)."""
    return hashlib.sha256(img.tobytes()).digest()[:7]


def ics_from_hash_bytes(hb):
    """One IC dict from 7 hash bytes: ic_i = MASTER_i + (hb[i] - 128) / 2560."""
    return {f: MASTER[f] + (hb[i] - 128) / SCALE for i, f in enumerate(IC_FIELDS)}


def second_pass_ics(hb):
    """Independent ICs for the column-axis pass, hash-chained from the
    transmitted metadata: h2 = SHA-256(hb). Both sides recompute this from hb,
    so no extra key material travels with the ciphertext, but the resulting
    keystream is statistically independent of pass 1's."""
    return ics_from_hash_bytes(hashlib.sha256(bytes(hb)).digest()[:7])


def lorenz4d_rk4(x0, y0, z0, w0, sigma=10.0, rho=28.0, beta=8 / 3, k=0.1,
                 h=STEP_H, n_steps=N_STEPS):
    x, y, z, w = float(x0), float(y0), float(z0), float(w0)
    xs = np.empty(n_steps, np.float64)
    ys = np.empty(n_steps, np.float64)
    zs = np.empty(n_steps, np.float64)
    ws = np.empty(n_steps, np.float64)
    for i in range(n_steps):
        f1x = sigma * (y - x) + w; f1y = rho * x - y - x * z
        f1z = x * y - beta * z;    f1w = -k * x
        ax, ay, az, aw = x + .5 * h * f1x, y + .5 * h * f1y, z + .5 * h * f1z, w + .5 * h * f1w
        f2x = sigma * (ay - ax) + aw; f2y = rho * ax - ay - ax * az
        f2z = ax * ay - beta * az;    f2w = -k * ax
        ax, ay, az, aw = x + .5 * h * f2x, y + .5 * h * f2y, z + .5 * h * f2z, w + .5 * h * f2w
        f3x = sigma * (ay - ax) + aw; f3y = rho * ax - ay - ax * az
        f3z = ax * ay - beta * az;    f3w = -k * ax
        ax, ay, az, aw = x + h * f3x, y + h * f3y, z + h * f3z, w + h * f3w
        f4x = sigma * (ay - ax) + aw; f4y = rho * ax - ay - ax * az
        f4z = ax * ay - beta * az;    f4w = -k * ax
        x += h / 6 * (f1x + 2 * f2x + 2 * f3x + f4x)
        y += h / 6 * (f1y + 2 * f2y + 2 * f3y + f4y)
        z += h / 6 * (f1z + 2 * f2z + 2 * f3z + f4z)
        w += h / 6 * (f1w + 2 * f2w + 2 * f3w + f4w)
        xs[i] = x; ys[i] = y; zs[i] = z; ws[i] = w
    return xs, ys, zs, ws


def rossler_rk4(x0, y0, z0, a=0.2, b=0.2, c=5.7, h=STEP_H, n_steps=N_STEPS):
    x, y, z = float(x0), float(y0), float(z0)
    xs = np.empty(n_steps, np.float64)
    ys = np.empty(n_steps, np.float64)
    zs = np.empty(n_steps, np.float64)
    for i in range(n_steps):
        k1x = -y - z;      k1y = x + a * y;    k1z = b + z * (x - c)
        ax, ay, az = x + .5 * h * k1x, y + .5 * h * k1y, z + .5 * h * k1z
        k2x = -ay - az;    k2y = ax + a * ay;  k2z = b + az * (ax - c)
        ax, ay, az = x + .5 * h * k2x, y + .5 * h * k2y, z + .5 * h * k2z
        k3x = -ay - az;    k3y = ax + a * ay;  k3z = b + az * (ax - c)
        ax, ay, az = x + h * k3x, y + h * k3y, z + h * k3z
        k4x = -ay - az;    k4y = ax + a * ay;  k4z = b + az * (ax - c)
        x += h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        y += h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        z += h / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)
        xs[i] = x; ys[i] = y; zs[i] = z
    return xs, ys, zs


def _to_bytes(arr):
    """floor(|v| * 1e14) % 256 -- must happen in fp64 (see module docstring)."""
    assert arr.dtype == np.float64, f"keystream extraction needs fp64, got {arr.dtype}"
    return (np.floor(np.abs(arr) * 1e14).astype(np.uint64) % 256).astype(np.uint8)


def _fold(raw, size):
    """XOR-fold the 630,000 raw bytes down to one size*size keystream.

    7 trajectories x 90,000 steps = 630,000 bytes; with need = 65,536 that is
    9 full rows. Rows are slices of the *concatenated* stream, so they cross
    trajectory boundaries -- the concatenation must happen before the fold.
    """
    need = size * size
    n_full = raw.shape[-1] // need
    k = raw[..., :n_full * need].reshape(*raw.shape[:-1], n_full, need)
    out = k[..., 0, :].copy()
    for row in range(1, n_full):
        out ^= k[..., row, :]
    return out.reshape(*raw.shape[:-1], size, size)


def build_keystream(ics, size=IMG_SIZE):
    """(size, size) uint8 keystream from one IC set -- scalar reference path."""
    xs, ys, zs, ws = lorenz4d_rk4(ics["lx"], ics["ly"], ics["lz"], ics["lw"])
    rx, ry, rz = rossler_rk4(ics["rx"], ics["ry"], ics["rz"])
    raw = np.concatenate([_to_bytes(xs), _to_bytes(ys), _to_bytes(zs), _to_bytes(ws),
                          _to_bytes(rx), _to_bytes(ry), _to_bytes(rz)])
    return _fold(raw, size)


# -- lane-vectorised keystream -------------------------------------------------
# RK4 is sequential in time, so the only parallel axis is across trajectories:
# one lane per integration, with the 90,000-step loop inside. The loop then
# runs once no matter how many keystreams are wanted.
#
# MEASURED RESULT, and it is not the expected one: on CPU this is *slower* per
# lane than the scalar path until roughly B = 36. The ODE state is only 4-7
# doubles, so a NumPy call on a (B,) array of that size costs far more in
# dispatch overhead (~1.5 us) than Python float arithmetic does outright
# (~50 ns). The batched cost is almost flat in B -- ~8 s of pure overhead for
# the Lorenz loop whether B is 2 or 30 -- so it only wins once B is large
# enough for 0.236 s/lane of scalar work to exceed it:
#
#     B=2   batched 7.61 s   scalar 0.47 s   -> scalar
#     B=30  batched 8.34 s   scalar 7.08 s   -> scalar
#     B=60  batched 9.58 s   scalar 14.17 s  -> batched
#
# A five-image metric run needs B = 30 (2 keys x 3 encryptions x 5 images), so
# the scalar path is the right CPU default; build_keystream_many() dispatches
# on that. The batched form remains the correct shape for a CUDA device, where
# per-op overhead is amortised across far more arithmetic, and it is what
# lsdpak_gray_gpu.build_keystream_batch implements.
#
# Bit-identity with the scalar path is asserted in verify_parity.py gate 3:
# IEEE 754 fp64 is deterministic, and the expression order below matches
# lorenz4d_rk4/rossler_rk4 exactly.

BATCH_CROSSOVER = 36        # measured; see the note above

def _lorenz4d_batch_np(ic, n_steps=N_STEPS, h=STEP_H,
                       sigma=10.0, rho=28.0, beta=8 / 3, k=0.1):
    """ic: (B,4) float64. Returns four (B, n_steps) float64 trajectories."""
    assert ic.dtype == np.float64, f"ODE needs fp64, got {ic.dtype}"
    B = ic.shape[0]
    x, y, z, w = (ic[:, i].copy() for i in range(4))
    xs = np.empty((B, n_steps), np.float64); ys = np.empty((B, n_steps), np.float64)
    zs = np.empty((B, n_steps), np.float64); ws = np.empty((B, n_steps), np.float64)
    for i in range(n_steps):
        f1x = sigma * (y - x) + w; f1y = rho * x - y - x * z
        f1z = x * y - beta * z;    f1w = -k * x
        ax, ay, az, aw = x + .5 * h * f1x, y + .5 * h * f1y, z + .5 * h * f1z, w + .5 * h * f1w
        f2x = sigma * (ay - ax) + aw; f2y = rho * ax - ay - ax * az
        f2z = ax * ay - beta * az;    f2w = -k * ax
        ax, ay, az, aw = x + .5 * h * f2x, y + .5 * h * f2y, z + .5 * h * f2z, w + .5 * h * f2w
        f3x = sigma * (ay - ax) + aw; f3y = rho * ax - ay - ax * az
        f3z = ax * ay - beta * az;    f3w = -k * ax
        ax, ay, az, aw = x + h * f3x, y + h * f3y, z + h * f3z, w + h * f3w
        f4x = sigma * (ay - ax) + aw; f4y = rho * ax - ay - ax * az
        f4z = ax * ay - beta * az;    f4w = -k * ax
        x = x + h / 6 * (f1x + 2 * f2x + 2 * f3x + f4x)
        y = y + h / 6 * (f1y + 2 * f2y + 2 * f3y + f4y)
        z = z + h / 6 * (f1z + 2 * f2z + 2 * f3z + f4z)
        w = w + h / 6 * (f1w + 2 * f2w + 2 * f3w + f4w)
        xs[:, i] = x; ys[:, i] = y; zs[:, i] = z; ws[:, i] = w
    return xs, ys, zs, ws


def _rossler_batch_np(ic, n_steps=N_STEPS, h=STEP_H, a=0.2, b=0.2, c=5.7):
    """ic: (B,3) float64. Returns three (B, n_steps) float64 trajectories."""
    assert ic.dtype == np.float64, f"ODE needs fp64, got {ic.dtype}"
    B = ic.shape[0]
    x, y, z = (ic[:, i].copy() for i in range(3))
    xs = np.empty((B, n_steps), np.float64); ys = np.empty((B, n_steps), np.float64)
    zs = np.empty((B, n_steps), np.float64)
    for i in range(n_steps):
        k1x = -y - z;      k1y = x + a * y;    k1z = b + z * (x - c)
        ax, ay, az = x + .5 * h * k1x, y + .5 * h * k1y, z + .5 * h * k1z
        k2x = -ay - az;    k2y = ax + a * ay;  k2z = b + az * (ax - c)
        ax, ay, az = x + .5 * h * k2x, y + .5 * h * k2y, z + .5 * h * k2z
        k3x = -ay - az;    k3y = ax + a * ay;  k3z = b + az * (ax - c)
        ax, ay, az = x + h * k3x, y + h * k3y, z + h * k3z
        k4x = -ay - az;    k4y = ax + a * ay;  k4z = b + az * (ax - c)
        x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        y = y + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        z = z + h / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)
        xs[:, i] = x; ys[:, i] = y; zs[:, i] = z
    return xs, ys, zs


def build_keystream_batch(ics_list, size=IMG_SIZE):
    """(B, size, size) uint8 keystreams, one per IC dict, in one time loop.

    Slower than the scalar path below B = BATCH_CROSSOVER on CPU; prefer
    build_keystream_many() unless you are deliberately testing this path.
    """
    lor = np.array([[d["lx"], d["ly"], d["lz"], d["lw"]] for d in ics_list], np.float64)
    ros = np.array([[d["rx"], d["ry"], d["rz"]] for d in ics_list], np.float64)
    xs, ys, zs, ws = _lorenz4d_batch_np(lor)
    rx, ry, rz = _rossler_batch_np(ros)
    raw = np.concatenate([_to_bytes(xs), _to_bytes(ys), _to_bytes(zs), _to_bytes(ws),
                          _to_bytes(rx), _to_bytes(ry), _to_bytes(rz)], axis=1)
    return _fold(raw, size)


def build_keystream_many(ics_list, size=IMG_SIZE, backend="auto"):
    """(B, size, size) keystreams by whichever CPU path is actually faster.

    backend: "auto" (default) | "scalar" | "batched". Both produce bitwise
    identical output; only the cost differs. See BATCH_CROSSOVER.
    """
    if backend == "auto":
        backend = "batched" if len(ics_list) >= BATCH_CROSSOVER else "scalar"
    if backend == "batched":
        return build_keystream_batch(ics_list, size)
    if backend != "scalar":
        raise ValueError(f"unknown backend {backend!r}")
    return np.stack([build_keystream(d, size) for d in ics_list])


# ------------------------------------------------------- Latin-square stage --
def build_D(n=N_ORDER):
    i = np.arange(n).reshape(n, 1); j = np.arange(n).reshape(1, n)
    return ((i + j) % n).astype(np.int32)


def build_D3(n=N_ORDER):
    i = np.arange(n).reshape(n, 1); j = np.arange(n).reshape(1, n)
    return ((i + 3 * j) % n).astype(np.int32)


def build_D5(n=N_ORDER):
    i = np.arange(n).reshape(n, 1); j = np.arange(n).reshape(1, n)
    return ((i + 5 * j) % n).astype(np.int32)


def _cp(D, rb, band=BAND):
    """Length-256 column index vector for row-band rb."""
    return np.concatenate([np.arange(D[rb, cb] * band, (D[rb, cb] + 1) * band)
                           for cb in range(D.shape[0])])


def _inv_cp(D, rb, size=IMG_SIZE, band=BAND):
    cp = _cp(D, rb, band)
    iv = np.empty(size, np.int32)
    iv[cp] = np.arange(size)
    return iv


def scramble3(img, D):
    """Three-pass Latin-square scramble. Pass 2 operates on the transpose,
    which makes it a row permutation; pass 3 uses D3."""
    n = D.shape[0]
    D3 = build_D3(n)
    sc = np.empty_like(img)
    for rb in range(n):
        r = slice(rb * BAND, (rb + 1) * BAND)
        sc[r, :] = img[r, :][:, _cp(D, rb)]
    sc2 = np.empty_like(sc); scT = sc.T
    for rb in range(n):
        r = slice(rb * BAND, (rb + 1) * BAND)
        sc2.T[r, :] = scT[r, :][:, _cp(D, rb)]
    sc3 = np.empty_like(sc2)
    for rb in range(n):
        r = slice(rb * BAND, (rb + 1) * BAND)
        sc3[r, :] = sc2[r, :][:, _cp(D3, rb)]
    return sc3


def unscramble3(img, D):
    n = D.shape[0]
    D3 = build_D3(n)
    s3 = np.empty_like(img)
    for rb in range(n):
        r = slice(rb * BAND, (rb + 1) * BAND)
        s3[r, :] = img[r, :][:, _inv_cp(D3, rb)]
    s2 = np.empty_like(s3); s3T = s3.T
    for rb in range(n):
        r = slice(rb * BAND, (rb + 1) * BAND)
        s2.T[r, :] = s3T[r, :][:, _inv_cp(D, rb)]
    s1 = np.empty_like(s2)
    for rb in range(n):
        r = slice(rb * BAND, (rb + 1) * BAND)
        s1[r, :] = s2[r, :][:, _inv_cp(D, rb)]
    return s1


# -------------------------------------------------------------- block stage --
_FBC = {}


def block_perms(bs=BLOCK, nb=(IMG_SIZE // BLOCK) ** 2, seed=BLOCK_SEED):
    """256 per-block permutations of 16*16, from np.random.default_rng(2025+b).

    These MUST come from NumPy's PCG64. torch.randperm and
    cupy.random.permutation use different generators, so the same seed yields
    different permutations and the ciphertext changes.
    """
    key = (bs, nb, seed)
    if key not in _FBC:
        fwd, inv = [], []
        for b in range(nb):
            p = np.random.default_rng(seed + b).permutation(bs * bs)
            iv = np.empty_like(p); iv[p] = np.arange(bs * bs)
            fwd.append(p); inv.append(iv)
        _FBC[key] = (np.array(fwd), np.array(inv))
    return _FBC[key]


def shuffle_blocks_full(img, bs=BLOCK, seed=BLOCK_SEED):
    tpr = img.shape[0] // bs
    fwd, _ = block_perms(bs, tpr * tpr, seed)
    out = img.copy()
    for b in range(tpr * tpr):
        r0, c0 = (b // tpr) * bs, (b % tpr) * bs
        blk = out[r0:r0 + bs, c0:c0 + bs].ravel()
        out[r0:r0 + bs, c0:c0 + bs] = blk[fwd[b]].reshape(bs, bs)
    return out


def unshuffle_blocks_full(img, bs=BLOCK, seed=BLOCK_SEED):
    tpr = img.shape[0] // bs
    _, inv = block_perms(bs, tpr * tpr, seed)
    out = img.copy()
    for b in range(tpr * tpr):
        r0, c0 = (b // tpr) * bs, (b % tpr) * bs
        blk = out[r0:r0 + bs, c0:c0 + bs].ravel()
        out[r0:r0 + bs, c0:c0 + bs] = blk[inv[b]].reshape(bs, bs)
    return out


# ----------------------------------------------------- D5 column-band stage --
def lsq_column_shuffle(x, D5, band=BAND):
    """D5-driven column-band permutation, per row-band.

    This is v3's `interchannel_shuffle` applied to one plane. See the module
    docstring for why the RGB name was inaccurate.
    """
    n = D5.shape[0]
    out = np.empty_like(x)
    for rb in range(n):
        rows = slice(rb * band, (rb + 1) * band)
        perm = D5[rb]
        for cb in range(n):
            src = perm[cb]
            out[rows, cb * band:(cb + 1) * band] = x[rows, src * band:(src + 1) * band]
    return out


def lsq_column_unshuffle(x, D5, band=BAND):
    n = D5.shape[0]
    out = np.empty_like(x)
    for rb in range(n):
        rows = slice(rb * band, (rb + 1) * band)
        perm = D5[rb]
        inv = np.empty(n, np.int32); inv[perm] = np.arange(n)
        for cb in range(n):
            src = inv[cb]
            out[rows, cb * band:(cb + 1) * band] = x[rows, src * band:(src + 1) * band]
    return out


# ---------------------------------------------------------- diffusion stage --
_BITS = np.arange(8, dtype=np.uint8)


def cumxor(a):
    """Inclusive prefix-XOR. XOR on a bit-plane is addition mod 2, so
    prefix-XOR is the parity of the prefix-sum -- and cumsum is a parallel
    scan. int32 accumulator: the running sum maxes at N = 65,536, well inside
    int32, whereas a float accumulator would lose integers past 2**24.
    """
    bits = ((a[:, None] >> _BITS) & 1).astype(np.int32)
    pref = np.cumsum(bits, axis=0) & 1
    return (pref << _BITS).sum(axis=1).astype(np.uint8)


def bidir_encrypt(arr, key):
    """Forward chain c_f[i] = arr[i]^key[i]^c_f[i-1] is an inclusive
    prefix-XOR of (arr^key); the backward chain C[i] = c_f[i]^C[i+1] is a
    reverse prefix-XOR. Both are parallel scans."""
    c_f = cumxor(np.bitwise_xor(arr, key))
    return cumxor(c_f[::-1])[::-1].copy()


def bidir_decrypt(arr, key):
    """Both decryption loops are elementwise with a neighbour, so no scan is
    needed -- a shift and two XORs."""
    nxt = np.empty_like(arr); nxt[:-1] = arr[1:]; nxt[-1] = 0
    c_f = np.bitwise_xor(arr, nxt)
    prev = np.empty_like(c_f); prev[1:] = c_f[:-1]; prev[0] = 0
    return np.bitwise_xor(np.bitwise_xor(c_f, key), prev)


def bidir_encrypt_sequential(arr, key):
    """The original loop form, kept only so the scan can be asserted against
    it in verify_parity.py."""
    N = len(arr)
    c_f = np.empty(N, np.uint8)
    c_f[0] = int(arr[0]) ^ int(key[0])
    for i in range(1, N):
        c_f[i] = int(arr[i]) ^ int(key[i]) ^ int(c_f[i - 1])
    out = np.empty(N, np.uint8)
    out[N - 1] = c_f[N - 1]
    for i in range(N - 2, -1, -1):
        out[i] = int(c_f[i]) ^ int(out[i + 1])
    return out


# ------------------------------------------------------- encrypt / decrypt ---
def pass_ics(img, ics_override=None, hb_override=None):
    """The (hb, [ics1, ics2]) a single encryption needs. Exposed so a metric
    run can collect every lane's ICs up front and generate all the keystreams
    in one build_keystream_batch call."""
    hb = hb_override if hb_override is not None else image_hash_bytes(img)
    ics1 = ics_override if ics_override is not None else ics_from_hash_bytes(hb)
    return hb, [ics1, second_pass_ics(hb)]


def encrypt_gray(img, D=None, ics_override=None, hb_override=None, keys=None):
    """Grayscale LS-DPAK v5.

    ics_override / hb_override exist so the key-sensitivity test can reuse this
    single implementation with a deliberately perturbed pass-1 key instead of
    duplicating the pipeline (the pattern v4's encrypt_v4 uses).

    keys optionally supplies a precomputed (2, 256, 256) [K1, K2] so a batched
    run can skip the per-encryption ODE.

    Returns (cipher (256,256) uint8, hb 7 bytes).
    """
    if D is None:
        D = build_D()
    D5 = build_D5(D.shape[0])

    hb = hb_override if hb_override is not None else image_hash_bytes(img)
    if keys is not None:
        K1, K2 = keys[0], keys[1]
        return _encrypt_with_keys(img, D, D5, K1, K2), hb
    ics1 = ics_override if ics_override is not None else ics_from_hash_bytes(hb)
    K1 = build_keystream(ics1, img.shape[0])

    K2 = build_keystream(second_pass_ics(hb), img.shape[0])
    return _encrypt_with_keys(img, D, D5, K1, K2), hb


def _encrypt_with_keys(img, D, D5, K1, K2):
    x = scramble3(img, D)
    x = shuffle_blocks_full(x)
    x = lsq_column_shuffle(x, D5)

    # Pass 1 -- row-major diffusion.
    c_row = bidir_encrypt(x.ravel(), K1.ravel()).reshape(img.shape)

    # Pass 2 -- column axis. Both the data and the key are transposed so the
    # chain runs along columns with a key laid out to match. Transposing only
    # one of them yields a wrong cipher that still round-trips, which is what
    # the golden hashes exist to catch.
    cipher = bidir_encrypt(np.ascontiguousarray(c_row.T).ravel(),
                           np.ascontiguousarray(K2.T).ravel())
    return cipher.reshape(img.shape).T.copy()


def decrypt_gray(cipher, hb, D=None, keys=None):
    """Exact reverse. Pass 2 is undone first because it was applied last."""
    if D is None:
        D = build_D()
    D5 = build_D5(D.shape[0])

    if keys is not None:
        K1, K2 = keys[0], keys[1]
    else:
        K1 = build_keystream(ics_from_hash_bytes(hb), cipher.shape[0])
        K2 = build_keystream(second_pass_ics(hb), cipher.shape[0])

    c_row = bidir_decrypt(np.ascontiguousarray(cipher.T).ravel(),
                          np.ascontiguousarray(K2.T).ravel())
    c_row = c_row.reshape(cipher.shape).T.copy()

    x = bidir_decrypt(c_row.ravel(), K1.ravel()).reshape(cipher.shape)
    x = lsq_column_unshuffle(x, D5)
    x = unshuffle_blocks_full(x)
    return unscramble3(x, D)


if __name__ == "__main__":
    D = build_D()
    print(f"{'image':8s} {'hb':>16s} {'roundtrip':>10s}")
    for name in IMAGE_NAMES:
        img = load_image_gray(name)
        cipher, hb = encrypt_gray(img, D)
        ok = np.array_equal(img, decrypt_gray(cipher, hb, D))
        print(f"{name:8s} {hb.hex():>16s} {'ok' if ok else 'FAIL':>10s}")
