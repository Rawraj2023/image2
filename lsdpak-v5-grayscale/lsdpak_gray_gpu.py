"""
LS-DPAK v5 grayscale -- PyTorch port.
=====================================

Validated bitwise against `lsdpak_gray_cpu.py` (see `verify_parity.py`). Runs
on CUDA when available, otherwise on CPU torch; the device is reported rather
than assumed.

What is actually parallel here
------------------------------
Three stages, with very different payoffs:

1. **Diffusion -- certain win.** The forward chain
   `c_f[i] = s[i]^k[i]^c_f[i-1]` is an inclusive prefix-XOR of `s^k`, and the
   backward chain `C[i] = c_f[i]^C[i+1]` is a reverse prefix-XOR. XOR is
   associative, so both are parallel scans with O(log N) depth. Neither torch
   nor CuPy has a native `cumxor`, but XOR on a single bit is addition mod 2,
   so prefix-XOR of a bit-plane is the parity of its prefix-sum -- and `cumsum`
   is natively parallel. This removes ~262,000 Python-level iterations per
   encryption. Decryption needs no scan at all: both of its loops are
   elementwise with a neighbour, so they reduce to a shift and two XORs.

2. **Permutations -- already cheap.** Pure gathers. The index tensors are built
   once on CPU with NumPy and uploaded. They must come from NumPy: the block
   shuffle uses `np.random.default_rng(2025 + b)` (PCG64), and
   `torch.randperm` uses a different generator, so the same seed would yield
   different permutations, a different ciphertext, and failing golden hashes.

3. **The ODE -- uncertain win, and sequential in time.** RK4's 90,000 steps
   each depend on the last, so time cannot be parallelised; launching a kernel
   per step is slower than NumPy. The only available axis is *across
   trajectories*: one lane per integration, with the step loop inside. A full
   metric run offers 2 lanes per image (pass-1 and pass-2 keys) x 3
   encryptions (original, pixel-flipped, key-perturbed) x however many images
   are batched, so `build_keystream_batch` takes a list of IC dicts.

   On a consumer GeForce fp64 runs at 1/32-1/64 of fp32, so the ODE may not
   beat a good CPU implementation at all. That is a measurement, not a
   failure -- `verify_parity.py --bench` reports it per stage.

float64 is mandatory
--------------------
Every tensor in the ODE is created with an explicit `dtype=torch.float64` and
the trajectory dtype is asserted before byte extraction. In float32 the
extraction `floor(|v| * 1e14) % 256` returns 0 for every byte, and
`(-3.0 + 1e-14) == -3.0` is True, so key sensitivity silently measures nothing.
TF32 is disabled defensively; it affects matmuls rather than the elementwise
ops used here, but the cost of being explicit is zero.
"""

import hashlib

import numpy as np
import torch

import lsdpak_gray_cpu as ref

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F64 = torch.float64


def device_report():
    if DEVICE.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return f"cpu (torch {torch.__version__}; no CUDA device present)"


# ----------------------------------------------------------- ODE, batched ----
def _lorenz4d_batch(ic, n_steps=ref.N_STEPS, h=ref.STEP_H,
                    sigma=10.0, rho=28.0, beta=8 / 3, k=0.1):
    """ic: (B,4) float64. Returns four (B, n_steps) float64 trajectories.

    The arithmetic is written in exactly the same order as the scalar reference
    in lsdpak_gray_cpu.lorenz4d_rk4 -- IEEE 754 fp64 is deterministic, so the
    same operation order gives bit-identical results, which is what lets
    verify_parity.py demand identity rather than closeness.
    """
    assert ic.dtype == F64, f"ODE needs fp64, got {ic.dtype}"
    B = ic.shape[0]
    x, y, z, w = (ic[:, i].clone() for i in range(4))
    xs = torch.empty((B, n_steps), dtype=F64, device=ic.device)
    ys = torch.empty((B, n_steps), dtype=F64, device=ic.device)
    zs = torch.empty((B, n_steps), dtype=F64, device=ic.device)
    ws = torch.empty((B, n_steps), dtype=F64, device=ic.device)
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


def _rossler_batch(ic, n_steps=ref.N_STEPS, h=ref.STEP_H, a=0.2, b=0.2, c=5.7):
    """ic: (B,3) float64. Returns three (B, n_steps) float64 trajectories."""
    assert ic.dtype == F64, f"ODE needs fp64, got {ic.dtype}"
    B = ic.shape[0]
    x, y, z = (ic[:, i].clone() for i in range(3))
    xs = torch.empty((B, n_steps), dtype=F64, device=ic.device)
    ys = torch.empty((B, n_steps), dtype=F64, device=ic.device)
    zs = torch.empty((B, n_steps), dtype=F64, device=ic.device)
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


def _to_bytes(t):
    """floor(|v| * 1e14) % 256, in fp64 then cast. Doing the multiply in fp32
    makes every byte 0 (see module docstring)."""
    assert t.dtype == F64, f"byte extraction needs fp64, got {t.dtype}"
    return (torch.floor(torch.abs(t) * 1e14).to(torch.int64) % 256).to(torch.uint8)


def build_keystream_batch(ics_list, size=ref.IMG_SIZE, device=DEVICE):
    """(B, size, size) uint8 keystreams, one per IC dict.

    Batching across trajectories is the only parallel axis the ODE offers; see
    the module docstring.
    """
    lor = torch.tensor([[d["lx"], d["ly"], d["lz"], d["lw"]] for d in ics_list],
                       dtype=F64, device=device)
    ros = torch.tensor([[d["rx"], d["ry"], d["rz"]] for d in ics_list],
                       dtype=F64, device=device)
    xs, ys, zs, ws = _lorenz4d_batch(lor)
    rx, ry, rz = _rossler_batch(ros)
    raw = torch.cat([_to_bytes(xs), _to_bytes(ys), _to_bytes(zs), _to_bytes(ws),
                     _to_bytes(rx), _to_bytes(ry), _to_bytes(rz)], dim=1)
    need = size * size
    n_full = raw.shape[1] // need
    k = raw[:, :n_full * need].reshape(-1, n_full, need)
    key = k[:, 0, :].clone()
    for row in range(1, n_full):
        key = torch.bitwise_xor(key, k[:, row, :])
    return key.reshape(-1, size, size)


# ---------------------------------------------------- diffusion as a scan ----
def cumxor(a):
    """Inclusive prefix-XOR along dim -1 of a uint8 tensor.

    Prefix-XOR of a bit equals the parity of that bit's prefix-sum, and cumsum
    is a parallel scan. int32 accumulator: the running sum tops out at
    N = 65,536, far inside int32, while a float accumulator would start losing
    integers past 2**24.
    """
    shifts = torch.arange(8, device=a.device, dtype=torch.uint8)
    bits = ((a.unsqueeze(-1) >> shifts) & 1).to(torch.int32)
    pref = torch.cumsum(bits, dim=-2) & 1
    return (pref << shifts.to(torch.int32)).sum(dim=-1).to(torch.uint8)


def bidir_encrypt(arr, key):
    c_f = cumxor(torch.bitwise_xor(arr, key))
    return cumxor(c_f.flip(-1)).flip(-1)


def bidir_decrypt(arr, key):
    """No scan needed -- both loops are elementwise with a neighbour."""
    nxt = torch.cat([arr[..., 1:], torch.zeros_like(arr[..., :1])], dim=-1)
    c_f = torch.bitwise_xor(arr, nxt)
    prev = torch.cat([torch.zeros_like(c_f[..., :1]), c_f[..., :-1]], dim=-1)
    return torch.bitwise_xor(torch.bitwise_xor(c_f, key), prev)


# ------------------------------------------- permutation index tensors -------
_IDX = {}


def _indices(device):
    """Build every permutation index once on CPU with NumPy, then upload.

    NumPy is not optional here: the block permutations come from PCG64 via
    np.random.default_rng, and torch's generator would produce different ones.
    """
    key = str(device)
    if key in _IDX:
        return _IDX[key]
    n, band, size = ref.N_ORDER, ref.BAND, ref.IMG_SIZE
    D, D3, D5 = ref.build_D(), ref.build_D3(), ref.build_D5()

    cp_D = np.stack([ref._cp(D, rb) for rb in range(n)])
    cp_D3 = np.stack([ref._cp(D3, rb) for rb in range(n)])
    iv_D = np.stack([ref._inv_cp(D, rb) for rb in range(n)])
    iv_D3 = np.stack([ref._inv_cp(D3, rb) for rb in range(n)])

    # D5 column-band stage, flattened to a full 256-length gather per row-band.
    col_f, col_i = [], []
    for rb in range(n):
        perm = D5[rb]
        inv = np.empty(n, np.int32); inv[perm] = np.arange(n)
        col_f.append(np.concatenate([np.arange(perm[cb] * band, (perm[cb] + 1) * band)
                                     for cb in range(n)]))
        col_i.append(np.concatenate([np.arange(inv[cb] * band, (inv[cb] + 1) * band)
                                     for cb in range(n)]))
    fwd, inv_b = ref.block_perms()

    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long, device=device)
    out = dict(cp_D=t(cp_D), cp_D3=t(cp_D3), iv_D=t(iv_D), iv_D3=t(iv_D3),
               col_f=t(np.stack(col_f)), col_i=t(np.stack(col_i)),
               blk_f=t(fwd), blk_i=t(inv_b), size=size, band=band, n=n)
    _IDX[key] = out
    return out


def _band_gather(x, idx):
    """Apply idx[rb] as a column gather to row-band rb, for all 32 bands."""
    n, band = idx.shape[0], x.shape[0] // idx.shape[0]
    v = x.reshape(n, band, x.shape[1])
    return torch.gather(v, 2, idx.unsqueeze(1).expand(n, band, x.shape[1])
                        ).reshape(x.shape)


def scramble3(x, I):
    x = _band_gather(x, I["cp_D"])
    x = _band_gather(x.T.contiguous(), I["cp_D"]).T.contiguous()
    return _band_gather(x, I["cp_D3"])


def unscramble3(x, I):
    x = _band_gather(x, I["iv_D3"])
    x = _band_gather(x.T.contiguous(), I["iv_D"]).T.contiguous()
    return _band_gather(x, I["iv_D"])


def shuffle_blocks_full(x, I, inverse=False):
    bs, size = ref.BLOCK, I["size"]
    tpr = size // bs
    perm = I["blk_i"] if inverse else I["blk_f"]
    # (tpr,bs,tpr,bs) -> (nb, bs*bs) so each block is a contiguous row.
    v = x.reshape(tpr, bs, tpr, bs).permute(0, 2, 1, 3).reshape(tpr * tpr, bs * bs)
    v = torch.gather(v, 1, perm)
    return v.reshape(tpr, tpr, bs, bs).permute(0, 2, 1, 3).reshape(size, size)


def lsq_column_shuffle(x, I, inverse=False):
    return _band_gather(x, I["col_i"] if inverse else I["col_f"])


# ------------------------------------------------------ encrypt / decrypt ----
def encrypt_gray(img, ics_override=None, hb_override=None, device=DEVICE,
                 keys=None):
    """Grayscale LS-DPAK v5 on `device`. Returns (cipher ndarray, hb bytes).

    `keys` optionally supplies a precomputed (2, 256, 256) uint8 tensor of
    [K1, K2] so a batched metric run can generate every keystream in one
    `build_keystream_batch` call instead of two per encryption.
    """
    I = _indices(device)
    hb = hb_override if hb_override is not None else ref.image_hash_bytes(img)
    if keys is None:
        ics1 = ics_override if ics_override is not None else ref.ics_from_hash_bytes(hb)
        keys = build_keystream_batch([ics1, ref.second_pass_ics(hb)], device=device)
    K1, K2 = keys[0], keys[1]

    x = torch.as_tensor(img, device=device)
    x = scramble3(x, I)
    x = shuffle_blocks_full(x, I)
    x = lsq_column_shuffle(x, I)

    c_row = bidir_encrypt(x.reshape(-1), K1.reshape(-1)).reshape(img.shape)
    cipher = bidir_encrypt(c_row.T.contiguous().reshape(-1),
                           K2.T.contiguous().reshape(-1))
    cipher = cipher.reshape(img.shape).T.contiguous()
    return cipher.cpu().numpy(), hb


def decrypt_gray(cipher, hb, device=DEVICE, keys=None):
    I = _indices(device)
    if keys is None:
        keys = build_keystream_batch([ref.ics_from_hash_bytes(hb),
                                      ref.second_pass_ics(hb)], device=device)
    K1, K2 = keys[0], keys[1]

    c = torch.as_tensor(cipher, device=device)
    c_row = bidir_decrypt(c.T.contiguous().reshape(-1),
                          K2.T.contiguous().reshape(-1))
    c_row = c_row.reshape(cipher.shape).T.contiguous()
    x = bidir_decrypt(c_row.reshape(-1), K1.reshape(-1)).reshape(cipher.shape)
    x = lsq_column_shuffle(x, I, inverse=True)
    x = shuffle_blocks_full(x, I, inverse=True)
    return unscramble3(x, I).cpu().numpy()


if __name__ == "__main__":
    print(f"device: {device_report()}")
    for name in ref.IMAGE_NAMES:
        img = ref.load_image_gray(name)
        cipher, hb = encrypt_gray(img)
        ok = np.array_equal(img, decrypt_gray(cipher, hb))
        print(f"  {name:8s} roundtrip={'ok' if ok else 'FAIL'}")
