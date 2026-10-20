# LS-DPAK v5 — Grayscale Two-Axis Chaotic Image Encryption

A grayscale chaotic image cipher: SHA-256-adaptive 4D-hyperchaotic-Lorenz and
Rössler keystreams, a three-pass Latin-square scramble, a full block shuffle, a
Latin-square column-band shuffle, and **two** bidirectional XOR-chain diffusion
passes along orthogonal axes with independently derived keys.

Benchmarked against Jha, Singh, Pradhan, Mishra, Jain & Ansari (2025),
*"An Advanced Image Encryption Algorithm Integrating Orthogonal Latin Square
Transversal for Shuffling With Lorenz Chaotic System for Diffusion"*,
**Security and Privacy 8:e70128**,
[doi:10.1002/spy2.70128](https://doi.org/10.1002/spy2.70128).

> This branch (`selected-model`) presents v5 grayscale only. The `main` branch
> carries the full development history — the RGB v3/v4 predecessors, a
> deep-learning variant with a neural keystream generator, and a cross-model
> comparison harness.

---

## Contribution

**1. A like-for-like comparison.** The reference paper is grayscale-only. Every
RGB measurement against it carries two artifacts that flatter the RGB cipher:

- *Pooled entropy.* Combining three channels into one 196,608-sample histogram
  lowers the entropy estimator's finite-sample bias, producing ~7.9991 against
  a per-channel ~7.997. This is a property of the estimator, not of the cipher,
  and grayscale cannot and should not reproduce it.
- *Inflated worst-case correlation.* An RGB maximum is taken over 9 draws per
  image (3 channels × 3 directions) versus 3 in grayscale, so the RGB worst
  case is worse by order statistics alone.

v5 removes both by operating on one channel, making the comparison honest in
both directions.

**2. Two diffusion axes with independent keys.** A single row-major chain
leaves vertical neighbours ~256 chain-steps apart, so they are decorrelated
only by the earlier permutations. v5 adds a column-axis pass keyed from
`SHA-256(hb)` — hash-chained, so no extra key material is transmitted, but the
second keystream is statistically independent of the first rather than a
transform of it.

**3. The two NPCR/UACI protocols reported separately.** These measure different
things and are routinely conflated. Reported apart, they produce this work's
main quantitative result (below).

**4. A verified-faithful implementation.** Every stage is asserted bitwise
identical to the published RGB original, and the PyTorch port is asserted
bitwise identical to the NumPy reference. 35 acceptance gates, all passing.

---

## Quick start

```bash
pip install -r requirements.txt

cd lsdpak-v5-grayscale
python verify_parity.py            # 35 acceptance gates
python verify_parity.py --bench    # gates + per-stage timings
python metrics_gray.py             # all metrics + the Jha et al. comparison
```

Measured on Python 3.13.7, numpy 2.5.4, torch 2.14.1+cpu, **no CUDA device**.

---

## Algorithm

Input a 256×256 uint8 grayscale image; output the ciphertext plus 7 metadata
bytes transmitted alongside it.

```
hb   = SHA-256(img)[:7]                     # 7 transmitted metadata bytes
ics1 = MASTER + (hb[i] - 128) / 2560        # hash-adaptive initial conditions
K1   = keystream(ics1)                      # (256,256) uint8

x = scramble3(img, D)                       # 3-pass Latin square (D, D, D3)
x = shuffle_blocks_full(x)                  # all 256 blocks of 16x16
x = lsq_column_shuffle(x, D5)               # column-band permutation

c = bidir_encrypt(x.ravel(), K1.ravel())    # pass 1: row-major

ics2 = ics_from_hash(SHA-256(hb)[:7])       # independent, hash-chained
K2   = keystream(ics2)
cipher = bidir_encrypt(c.T.ravel(), K2.T.ravel()).T    # pass 2: column axis
```

Both the data **and** the key are transposed for pass 2, so the chain runs
along columns with a key laid out to match. Transposing only one produces a
cipher that still round-trips but differs from the reference — which is what
the frozen golden hashes exist to catch.

**Bidirectional diffusion.** Forward `c_f[i] = s[i] ^ k[i] ^ c_f[i-1]`, then
backward `C[i] = c_f[i] ^ C[i+1]`. The backward sweep is what makes a change
anywhere propagate everywhere, including to earlier pixels.

**Keystream.** 4D hyperchaotic Lorenz (σ=10, ρ=28, β=8/3, k=0.1) and Rössler
(a=0.2, b=0.2, c=5.7), RK4 at h=0.01 for 90,000 steps each. The 7 trajectories
give 630,000 bytes via `floor(|v| · 1e14) mod 256`; 9 full rows of 65,536 are
XOR-folded into the keystream.

### float64 is mandatory

Byte extraction reads the bottom of the mantissa, which is what makes the
scheme key-sensitive and also what makes it precision-fragile. In float32:

- every keystream byte becomes **0** — at magnitude ~2e15 a float32 ULP is
  2²⁷, every product is a multiple of 2²⁷, and 2²⁷ mod 256 = 0. The cipher
  degenerates to the permuted plaintext XORed with zeros *while still
  round-tripping cleanly*.
- `(-3.0 + 1e-14) == -3.0` is **True**, so the key-sensitivity test compares a
  run against itself and reports NPCR 0.000% — a test silently measuring
  nothing.

Gate 0 asserts both. `float64` is explicit on every tensor, and the dtype is
re-asserted immediately before extraction.

---

## Verification

`verify_parity.py` — **35 gates, 35 passing.** Each catches a different class
of bug, so they run in order: a cipher failing gate 1 still produces
plausible-looking metrics at gate 6.

| gate | check | result |
|---|---|---|
| 0 | fp64 end to end; keystream non-zero and spanning 0–255; `(-3.0 + 1e-14) != -3.0` | pass |
| 1 | CPU round-trip bit-exact | 5/5 |
| 2 | keystream, `scramble3`, `shuffle_blocks_full`, `lsq_column_shuffle` and diffusion each **bitwise equal to the v3 RGB original** | pass |
| 3 | lane-vectorised and torch keystreams bitwise equal to the scalar path | pass |
| 4 | **GPU ciphertext bitwise identical to CPU**, plus GPU round-trip | 5/5 |
| 5 | frozen golden SHA-256 ciphertext hashes | 5/5 |
| 6 | both NPCR/UACI protocols wired to the real pipeline | pass |

Gate 2 is what licenses calling this a *reduction* of a published pipeline
rather than a reimplementation of it. The oracle is in `reference-rgb/`.

Gate 4 demands identity, not closeness. "Close enough" is not a meaningful
state for a cipher: one differing byte means two different functions, and
metrics from the divergent one are worthless. The torch fp64 RK4 reproduces the
scalar Python loop bit for bit, which holds only because the arithmetic is
written in the same order and nothing silently demotes precision.

Gate 5's hashes are frozen in `golden/ciphertext_hashes.json` and re-checked
after every change, converting "silently different cipher" into a loud failure.

---

## Results

Five USC-SIPI images at 256×256, loaded with PIL `"L"` (ITU-R 601-2 luma,
matching MATLAB `rgb2gray`).

**Protocol A — differential plaintext.** `img[128,128] += 1 mod 256`, same
master key. Because the initial conditions are SHA-256-adaptive, this changes
`hb` and therefore the entire keystream.

**Protocol B — key sensitivity.** Same plaintext; pass-1 Lorenz `x0` perturbed
by 1e-14, with `hb` and therefore K2 held fixed.

| image | entropy | corr_h | corr_v | corr_d | NPCR_A | UACI_A | NPCR_B | UACI_B | χ² p |
|---|---|---|---|---|---|---|---|---|---|
| aerial | 7.99716 | −0.005231 | −0.004047 | +0.001612 | 99.6033 | 33.5313 | 99.6017 | 33.5979 | 0.4290 |
| lena | 7.99733 | −0.006769 | −0.001018 | +0.000796 | 99.6201 | 33.4684 | 99.6231 | 33.5543 | 0.7278 |
| couple | 7.99730 | +0.003511 | −0.006805 | −0.004368 | 99.6033 | 33.5016 | 99.5621 | 33.4508 | 0.6664 |
| clock | 7.99704 | −0.003620 | −0.005171 | −0.004211 | 99.6109 | 33.4135 | 99.6201 | 33.2221 | 0.2612 |
| man | 7.99704 | −0.000179 | −0.001216 | +0.000930 | 99.6094 | 33.3430 | 99.6140 | 33.4151 | 0.2764 |
| **mean** | **7.99717** | | | | **99.6094** | **33.4516** | **99.6042** | **33.4481** | |
| **ideal** | 8.0 | 0 | 0 | 0 | 99.6094 | 33.4635 | 99.6094 | 33.4635 | >0.01 |

- **Entropy 7.99717**, inside the 7.99700–7.99763 band of the RGB *per-channel*
  values in `reference-rgb/metrics_v4.json`. A grayscale run reporting ~7.999
  would mean something had been pooled.
- **All five round-trip bit-exactly**; all five pass χ² histogram uniformity.
- Mean NPCR_A lands on 99.6094 — the theoretical ideal to four decimals. Worth
  stating plainly that this is coincidence on five images, not precision;
  genuine runs scatter by a few hundredths, and landing *exactly* on the ideal
  is reassuring only in aggregate.

### Correlation sits at the noise floor

A 256×256 image has 255×256 = 65,280 adjacent pairs, so an adjacent-pixel
correlation has standard error 1/√65280 = 0.00391. A Monte Carlo over uniform
random images — the best any cipher could do — gives mean |r| = **0.00316**.

| statistic | v5 grayscale | true random |
|---|---|---|
| mean \|r\| over 15 draws | 0.00330 | 0.00316 |
| worst single \|r\| | 0.00680 | 0.00789 (95th pct) |

v5 is a statistical tie with a perfect source. This is the ceiling, not a
shortfall: **single-image correlation at this size is dominated by sampling
noise, not cipher quality.** Two consequences for anyone extending this work:

- Do not tune toward a lower single-image correlation — that is fitting noise.
  A target of "|r| < 0.002 on all images" is unreachable: one draw from a
  *perfect* cipher falls below 0.002 only 39.1% of the time, so all nine
  (3 images × 3 directions) doing so has probability 0.02%.
- Do not claim a correlation win from one image, in either direction.

---

## Against Jha et al. (2025)

Their Table 6 is a protocol-B result — their §5.7 describes the 1e-14
perturbation — so it belongs against the protocol-B column.

| image | v5 entropy | Jha | v5 NPCR_B | Jha NPCR | v5 UACI_B | Jha UACI |
|---|---|---|---|---|---|---|
| aerial | 7.99716 | 7.9973 | 99.6017 | 99.6094 | **33.5979** | 28.5589 |
| lena | 7.99733 | 7.9974 | 99.6231 | 99.6277 | **33.5543** | 28.7091 |
| couple | 7.99730 | 7.9969 | 99.5621 | 99.6246 | **33.4508** | 28.0631 |
| clock | 7.99704 | 7.9968 | 99.6201 | 99.6017 | **33.2221** | 35.5581 |
| man | 7.99704 | 7.9970 | 99.6140 | 99.5865 | **33.4151** | 28.8974 |
| **mean** | **7.99717** | **7.9971** | **99.6042** | **99.6100** | **33.4481** | **29.9573** |

**Entropy, NPCR and correlation are ties, and should be.** Both ciphers produce
uniform output; NPCR saturates near 99.61% for anything competent; both
correlations are at the noise floor (v5 mean |r| 0.00330, theirs 0.00287).
Claiming a win on any of these would be claiming a win on noise.

**UACI is not a tie.** Mean absolute deviation from the 33.4635 ideal:

| | mean \|UACI − 33.4635\| |
|---|---|
| v5 grayscale | **0.1056** |
| Jha et al. | **4.3440** |

Five of their six reported values cluster at 28–29%, about 5 points low, with
only Clock (35.56%) above. A one-sided deviation of that magnitude across
almost every image is not sampling scatter, and UACI has no plausible mechanism
to sit 5 points below ideal for a cipher whose NPCR is already at 99.61% — the
two statistics are computed from the same difference image.

**The protocol conflation.** Their Table 6 is cited in §5.6 as a
differential-plaintext result and in §5.7 as a key-sensitivity result. Only the
second protocol could have produced the reported numbers. This matters
structurally rather than cosmetically: with a single non-feedback XOR, the
differential-plaintext ceiling is **1/65536 ≈ 0.0015%**, because flipping one
plaintext pixel can only affect one ciphertext pixel. v5 reaches **99.6094%**
on protocol A because the SHA-256-adaptive initial conditions rebuild the whole
keystream from the new plaintext hash, and the bidirectional chain propagates
it in both directions. Protocol A is the measurement the reference paper never
performed.

---

## Performance

Per 256×256 image on CPU, no CUDA device available.

### The diffusion scan — a real win

The forward chain is an inclusive prefix-XOR of `s ^ k`; the backward chain is
a reverse prefix-XOR. XOR is associative, so both are parallel scans. Neither
NumPy nor torch has `cumxor`, but XOR on one bit is addition mod 2, so
prefix-XOR is the parity of a `cumsum`:

```python
bits = (a[:, None] >> arange(8)) & 1   # bit-planes
pref = bits.cumsum(0) & 1              # parallel scan
out  = (pref << arange(8)).sum(1)      # repack
```

| stage | before | after | speedup |
|---|---|---|---|
| diffusion, sequential Python loop → NumPy scan | 73.9 ms | 33.3 ms | **2.2×** |
| diffusion, → torch scan | 73.9 ms | 12.0 ms | **6.1×** |
| decryption diffusion — needs no scan at all | 73.9 ms | 0.2 ms | **370×** |
| encrypt end to end, keys precomputed (NumPy → torch) | 84.3 ms | 22.8 ms | 3.7× |

Decryption is the cheapest stage because both of its loops are elementwise with
a neighbour — `c_f = C ^ shift(C)`, then `s = c_f ^ k ^ shift(c_f)` — so it is
a shift and two XORs with no scan required.

### The ODE — a measured loss, reported as such

Keystream generation dominates everything else: **2811 ms** of a ~2900 ms
metric step. RK4 is sequential in time, so the only parallel axis is across
trajectories. Neither batched strategy helps on CPU:

| keystream path | per lane | vs scalar |
|---|---|---|
| NumPy scalar loop | 332 ms | 1.00× |
| NumPy lane-vectorised, B=2 | 6944 ms | 0.05× |
| NumPy lane-vectorised, B=30 | 508 ms | 0.65× |
| torch lane-vectorised, B=2 | ~196,900 ms | 0.002× |
| torch lane-vectorised, B=30 | ~11,800 ms | 0.03× |

The ODE state is only 4–7 doubles — far too small for array dispatch to
amortise. A NumPy call on a 2-element array costs ~1.5 µs against ~50 ns for
Python float arithmetic, so the batched cost is nearly flat in B (~8 s for the
Lorenz loop whether B is 2 or 30) and only overtakes the scalar path at
**B ≈ 36**. Torch is a further 13× worse on CPU, purely dispatch-bound at
~200 s per 90,000-step loop regardless of batch size.

`build_keystream_many()` therefore dispatches on that measured crossover:
scalar below B = 36, batched above. A five-image metric run needs B = 30, which
lands on the scalar side.

The batched and torch paths are kept because they are the correct *shape* for a
real CUDA device — but note that fp64 runs at 1/32–1/64 of fp32 throughput on
consumer GeForce cards (1/2 on A100/H100), so even there the ODE is the stage
least likely to pay off. Anyone porting this to a GPU should write the ODE as a
fused kernel — one thread per trajectory, the 90,000-step loop inside, state in
registers, reducing 90,000 launches to one — rather than expecting the batched
form to carry it.

---

## Limitations

- **Five images, not seven.** `Baboon.tiff` is absent from this repository and
  Cameraman was never in it; Jha et al. averaged over seven. The means above
  are over the five shared images, and every per-image value is listed so the
  gap is visible rather than hidden inside an average. Sourcing both (standard
  test images; Cameraman ships with MATLAB as `cameraman.tif`) would close it.
- **No CUDA device was available**, so the torch port is validated for
  correctness and benchmarked on CPU only. Its GPU performance is unmeasured;
  the per-stage reasoning above is explicit about which claims are measured and
  which are expectations.
- **One rename, and a correction to the RGB write-up.** v3's
  `interchannel_shuffle` is `lsq_column_shuffle` here. The RGB name is
  inaccurate: despite the name, and despite the original write-up describing it
  as breaking cross-channel correlation, that function never moves data between
  R, G and B — it applies the same D5 column-band permutation independently to
  each channel, making it structurally a fourth Latin-square permutation pass.
  Genuine cross-channel coupling in the RGB version came from the diffusion
  chain running over the three concatenated planes. The stage is kept here,
  applied to the single plane, because dropping it would remove a permutation
  pass and shift the correlation results for reasons unrelated to grayscale.
- **`results/metrics_gray.json` is not shaped like the RGB files:** `entropy`
  is a scalar, not an `{R,G,B,joint}` object, and `interchannel_corr` is
  absent. Tooling that reads both must handle the difference.
- **The grayscale reduction sidesteps an RGB key-schedule defect rather than
  fixing it.** In the RGB variants each channel's key came from a disjoint
  7-byte slice of `hb`, so a 1-bit metadata change left two of three channel
  keystreams bit-identical and single-axis key sensitivity collapsed to 66.5%.
  With one channel there is nothing to split, so v5 is unaffected — but this
  should not be presented as having solved the problem. The fix for the RGB
  line is to derive each channel's key from `SHA-256(hb ‖ channel_index)`.

---

## Repository layout

```
lsdpak-v5-grayscale/
├── lsdpak_gray_cpu.py          NumPy reference — AUTHORITATIVE
├── lsdpak_gray_gpu.py          PyTorch port, bitwise identical to the above
├── verify_parity.py            35 acceptance gates (--freeze, --bench)
├── metrics_gray.py             both NPCR/UACI protocols, reported separately
├── golden/
│   └── ciphertext_hashes.json  frozen SHA-256 per image
└── results/
    ├── metrics_gray.json       machine-readable results
    └── verify_and_bench.txt    recorded gate + timing output

reference-rgb/                  NOT part of the contribution; see its README
├── lsdpak_v3_rgb.py            verification oracle for gate 2
├── lsdpak_v4.py                the RGB cipher v5 reduces
└── metrics_v4.json             provenance for the RGB per-channel values

test_images/                    Aerial, Clock, Couple, Lena, Man (256×256)
requirements.txt                pinned to the versions used for these results
```

`lsdpak_gray_cpu.py` is authoritative. The golden hashes are frozen from it and
the GPU port is validated against it, so when the two disagree the NumPy
reference is correct by definition.

## Note on test images

`test_images/Lena.jpg` is included for comparability with the reference paper,
which uses it. Its provenance is contested and it is being retired from
image-processing literature, so a venue may ask for it to be replaced. The
pipeline is image-agnostic, so substituting it costs nothing but a re-run of
`verify_parity.py --freeze` and `metrics_gray.py`.
