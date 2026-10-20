"""
Acceptance gates for the grayscale port.

    python verify_parity.py            # gates 1-6
    python verify_parity.py --freeze   # (re)write golden/ciphertext_hashes.json
    python verify_parity.py --bench    # gates + per-stage timings

Each gate catches a different class of bug, so they run in order -- a cipher
that fails gate 1 will still produce plausible-looking metrics later on.

Gate 3 demands *identity*, not closeness. "Close enough" is not a meaningful
state for a cipher: one differing byte means the two implementations compute
different functions, and every metric taken from the divergent one is
meaningless.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lsdpak_gray_cpu as cpu
import lsdpak_gray_gpu as gpu

_HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(_HERE, "golden", "ciphertext_hashes.json")

_pass = _fail = 0


def check(label, ok, detail=""):
    global _pass, _fail
    if ok:
        _pass += 1
        print(f"  [PASS] {label}" + (f"  {detail}" if detail else ""))
    else:
        _fail += 1
        print(f"  [FAIL] {label}" + (f"  {detail}" if detail else ""))
    return ok


# ------------------------------------------------------------------- gates ---
def gate0_fp64():
    print("\nGate 0 - float64 end to end")
    ics = cpu.ics_from_hash_bytes(bytes(range(1, 8)))
    xs, *_ = cpu.lorenz4d_rk4(ics["lx"], ics["ly"], ics["lz"], ics["lw"], n_steps=200)
    check("trajectory dtype is float64", xs.dtype == np.float64, str(xs.dtype))
    check("(-3.0 + 1e-14) != -3.0 in fp64",
          (np.float64(-3.0) + np.float64(1e-14)) != np.float64(-3.0),
          "fp32 would be False -> key sensitivity would read 0.000%")
    k = cpu.build_keystream(ics)
    check("keystream is not all-zero", k.max() > 0,
          f"max={k.max()}, distinct={np.unique(k).size}")
    check("keystream uses most of 0-255", np.unique(k).size > 250,
          f"{np.unique(k).size}/256 values present")


def gate1_roundtrip(D):
    print("\nGate 1 - CPU round-trip bit-exact")
    for n in cpu.IMAGE_NAMES:
        img = cpu.load_image_gray(n)
        c, hb = cpu.encrypt_gray(img, D)
        check(f"{n} round-trips", np.array_equal(img, cpu.decrypt_gray(c, hb, D)))


def gate2_primitives(D):
    print("\nGate 2 - primitives match the v3 RGB originals")
    # The v3 RGB module is the oracle for this gate, not part of the v5
    # contribution. It lives in reference-rgb/ on the article branch and in
    # lsdpak-v3-rgb/ on the full-history branch, so try both.
    for cand in ("reference-rgb", "lsdpak-v3-rgb"):
        d = os.path.join(_HERE, "..", cand)
        if os.path.isdir(d):
            sys.path.insert(0, d)
    try:
        import lsdpak_v3_rgb as v3
    except Exception as e:                                   # pragma: no cover
        print(f"  [SKIP] v3 RGB oracle unavailable ({e}) - "
              f"6 faithfulness checks not run")
        return
    ics = cpu.ics_from_hash_bytes(bytes(range(1, 8)))
    check("build_keystream == v3.build_keystream",
          np.array_equal(cpu.build_keystream(ics), v3.build_keystream(ics)))
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    D5 = cpu.build_D5()
    check("scramble3 == v3.scramble3",
          np.array_equal(cpu.scramble3(img, D), v3.scramble3(img, D)))
    check("shuffle_blocks_full == v3",
          np.array_equal(cpu.shuffle_blocks_full(img), v3.shuffle_blocks_full(img)))
    r, _, _ = v3.interchannel_shuffle(img, img, img, D5, cpu.BAND)
    check("lsq_column_shuffle == v3.interchannel_shuffle (one plane)",
          np.array_equal(cpu.lsq_column_shuffle(img, D5), r))
    key = rng.integers(0, 256, 65536, dtype=np.uint8)
    enc = cpu.bidir_encrypt(img.ravel(), key)
    check("scan bidir_encrypt == sequential loop",
          np.array_equal(enc, cpu.bidir_encrypt_sequential(img.ravel(), key)))
    check("scan bidir_encrypt == v3.bidir_encrypt_joint",
          np.array_equal(enc, v3.bidir_encrypt_joint(img.ravel(), key)))


def gate3_batch_keystream():
    print("\nGate 3 - lane-vectorised keystreams == scalar reference")
    ics = [cpu.ics_from_hash_bytes(bytes([i] * 7)) for i in range(1, 5)]
    batch_np = cpu.build_keystream_batch(ics)
    scalar = np.stack([cpu.build_keystream(d) for d in ics])
    check("numpy batched == scalar, bitwise", np.array_equal(batch_np, scalar),
          f"B={len(ics)}")
    batch_t = gpu.build_keystream_batch(ics[:2]).cpu().numpy()
    check("torch batched == scalar, bitwise",
          np.array_equal(batch_t, scalar[:2]),
          f"device={gpu.DEVICE.type}")


def gate4_gpu_parity(D):
    print(f"\nGate 4 - GPU ciphertext bitwise identical to CPU  [{gpu.device_report()}]")
    for n in cpu.IMAGE_NAMES:
        img = cpu.load_image_gray(n)
        hb, ics = cpu.pass_ics(img)
        keys_np = cpu.build_keystream_batch(ics)
        keys_t = torch.as_tensor(keys_np, device=gpu.DEVICE)
        c_cpu, _ = cpu.encrypt_gray(img, D, keys=keys_np)
        c_gpu, _ = gpu.encrypt_gray(img, keys=keys_t)
        ok = check(f"{n} cipher identical", np.array_equal(c_cpu, c_gpu))
        if ok:
            check(f"{n} GPU round-trips",
                  np.array_equal(img, gpu.decrypt_gray(c_gpu, hb, keys=keys_t)))


def gate5_golden(D, freeze=False):
    print("\nGate 5 - golden ciphertext hashes")
    cur = {}
    for n in cpu.IMAGE_NAMES:
        img = cpu.load_image_gray(n)
        c, hb = cpu.encrypt_gray(img, D)
        cur[n] = {"sha256": hashlib.sha256(c.tobytes()).hexdigest(), "hb": hb.hex()}
    if freeze or not os.path.exists(GOLDEN):
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        json.dump({"note": "Frozen from lsdpak_gray_cpu.py. Any change here "
                           "means the algorithm changed.", "images": cur},
                  open(GOLDEN, "w"), indent=2)
        print(f"  [FROZE] {len(cur)} hashes -> golden/ciphertext_hashes.json")
        for n, v in cur.items():
            print(f"          {n:8s} {v['sha256']}")
        return
    want = json.load(open(GOLDEN))["images"]
    for n, v in cur.items():
        if n not in want:
            print(f"  [SKIP] {n} not in golden file")
            continue
        check(f"{n} sha256 matches golden", v["sha256"] == want[n]["sha256"],
              v["sha256"][:16])


def gate6_protocols(D):
    print("\nGate 6 - both NPCR/UACI protocols are wired to the real pipeline")
    img = cpu.load_image_gray("lena")
    c1, hb = cpu.encrypt_gray(img, D)

    img2 = img.copy()
    img2[128, 128] = (int(img2[128, 128]) + 1) % 256
    c2, hb2 = cpu.encrypt_gray(img2, D)
    check("protocol A changes hb (keystream re-derived from new plaintext hash)",
          hb != hb2, f"{hb.hex()} -> {hb2.hex()}")
    npcr_a = float((c1 != c2).mean() * 100)
    check("protocol A NPCR in 99.55-99.65", 99.55 <= npcr_a <= 99.65, f"{npcr_a:.4f}%")

    ics = cpu.ics_from_hash_bytes(hb)
    ics["lx"] += 1e-14
    c3, _ = cpu.encrypt_gray(img, D, ics_override=ics, hb_override=hb)
    npcr_b = float((c1 != c3).mean() * 100)
    check("protocol B NPCR in 99.55-99.65", 99.55 <= npcr_b <= 99.65,
          f"{npcr_b:.4f}%  (0.000% would mean fp32)")


# --------------------------------------------------------------- benchmark ---
def bench(D):
    print("\nPer-stage timings (CPU; no CUDA device present)" if gpu.DEVICE.type == "cpu"
          else f"\nPer-stage timings [{gpu.device_report()}]")
    img = cpu.load_image_gray("lena")
    hb, ics = cpu.pass_ics(img)
    D5 = cpu.build_D5()

    def t(fn, reps=1):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e3

    print("\n  keystream generation (the ODE) -- 2 lanes needed per encryption")
    ms_scalar = t(lambda: cpu.build_keystream(ics[0]))
    print(f"    numpy scalar, per lane            {ms_scalar:9.1f} ms")
    for B in (2, 10, 30):
        lanes = [cpu.ics_from_hash_bytes(bytes([i] * 7)) for i in range(B)]
        ms = t(lambda: cpu.build_keystream_batch(lanes))
        print(f"    numpy batched B={B:<2d}  per lane        {ms / B:9.1f} ms"
              f"   (total {ms / 1000:6.2f} s, speedup {ms_scalar / (ms / B):5.1f}x)")
    # The torch ODE is measured over 300 steps and scaled to 90,000: a full run
    # takes minutes on CPU, and the cost is almost perfectly linear in steps
    # because it is dominated by per-op dispatch, not arithmetic.
    SHORT = 300
    for B in (2, 30):
        ic = torch.tensor([[cpu.MASTER["lx"] + i * 1e-3, cpu.MASTER["ly"],
                            cpu.MASTER["lz"], cpu.MASTER["lw"]] for i in range(B)],
                          dtype=torch.float64, device=gpu.DEVICE)
        ms_short = t(lambda: gpu._lorenz4d_batch(ic, n_steps=SHORT))
        # x2: a keystream lane needs Lorenz + Rossler.
        per_lane = ms_short / SHORT * cpu.N_STEPS * 2 / B
        print(f"    torch batched B={B:<2d}  per lane        {per_lane:9.1f} ms"
              f"   ({per_lane / ms_scalar:5.1f}x the numpy scalar cost, extrapolated)")

    print("\n  diffusion -- 2 passes per encryption, N = 65,536 each")
    flat, key = img.ravel(), cpu.build_keystream(ics[0]).ravel()
    ms_seq = t(lambda: cpu.bidir_encrypt_sequential(flat, key))
    ms_scan = t(lambda: cpu.bidir_encrypt(flat, key), reps=20)
    print(f"    sequential Python loop            {ms_seq:9.1f} ms")
    print(f"    numpy prefix-XOR scan             {ms_scan:9.1f} ms"
          f"   (speedup {ms_seq / ms_scan:5.1f}x)")
    ft = torch.as_tensor(flat, device=gpu.DEVICE)
    kt = torch.as_tensor(key, device=gpu.DEVICE)
    ms_gt = t(lambda: gpu.bidir_encrypt(ft, kt), reps=20)
    print(f"    torch prefix-XOR scan             {ms_gt:9.1f} ms"
          f"   (speedup {ms_seq / ms_gt:5.1f}x)")
    ms_dec = t(lambda: cpu.bidir_decrypt(flat, key), reps=20)
    print(f"    numpy decrypt (no scan needed)    {ms_dec:9.1f} ms")

    print("\n  permutation stages (already cheap)")
    print(f"    scramble3                         {t(lambda: cpu.scramble3(img, D), 5):9.1f} ms")
    print(f"    shuffle_blocks_full               {t(lambda: cpu.shuffle_blocks_full(img), 5):9.1f} ms")
    print(f"    lsq_column_shuffle                {t(lambda: cpu.lsq_column_shuffle(img, D5), 5):9.1f} ms")

    print("\n  end to end, one image")
    keys = cpu.build_keystream_batch(ics)
    kt2 = torch.as_tensor(keys, device=gpu.DEVICE)
    print(f"    CPU encrypt (keystream included)  {t(lambda: cpu.encrypt_gray(img, D)):9.1f} ms")
    print(f"    CPU encrypt (keys precomputed)    {t(lambda: cpu.encrypt_gray(img, D, keys=keys), 5):9.1f} ms")
    print(f"    GPU encrypt (keys precomputed)    {t(lambda: gpu.encrypt_gray(img, keys=kt2), 5):9.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true",
                    help="(re)write the golden hash file instead of checking it")
    ap.add_argument("--bench", action="store_true", help="also print per-stage timings")
    a = ap.parse_args()

    print("LS-DPAK v5 grayscale - acceptance gates")
    print(f"device: {gpu.device_report()}")
    print(f"images: {', '.join(cpu.IMAGE_NAMES)}  "
          f"(Baboon.tiff and Cameraman absent; Jha et al. used 7)")
    D = cpu.build_D()

    gate0_fp64()
    gate1_roundtrip(D)
    gate2_primitives(D)
    gate3_batch_keystream()
    gate4_gpu_parity(D)
    gate5_golden(D, freeze=a.freeze)
    gate6_protocols(D)

    if a.bench:
        bench(D)

    print(f"\n{_pass} passed, {_fail} failed")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
