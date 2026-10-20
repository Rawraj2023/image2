# LS-DPAK: Learned and Chaotic Keystreams for Latin-Square Image Encryption

Three contributions, benchmarked against Jha, Singh, Pradhan, Mishra, Jain &
Ansari (2025), *"An Advanced Image Encryption Algorithm Integrating Orthogonal
Latin Square Transversal for Shuffling With Lorenz Chaotic System for
Diffusion"*, **Security and Privacy 8:e70128**,
[doi:10.1002/spy2.70128](https://doi.org/10.1002/spy2.70128).

**A. LS-DPAK-DL** — replacing the chaotic keystream with a *learned* generator
(a GAN-fine-tuned MLP), plus a U-Net denoiser for noise-robust decryption. This
yields an 8.6× speedup, a measurable robustness gain, **and a negative result
that we consider the more important finding**: the learned keystream is not
cryptographically random, yet the cipher built on it passes every standard
test, because the permutation and diffusion layers mask the defect.

**B. LS-DPAK v5 grayscale** — the two-axis chaotic cipher reduced to one
channel, making the comparison against the (grayscale-only) reference paper
like-for-like and removing two measurement artifacts that flatter RGB
implementations. Verified bitwise against its RGB predecessor at every stage.

**C. LS-DPAK-DL grayscale** — the NKG from Contribution A, unmodified and
unretrained, dropped into Contribution B's single-channel pipeline. This is
the cell the first two contributions leave open: A is learned-but-RGB, B is
chaotic-but-grayscale, and neither alone tests "learned vs. chaotic" on the
same colour space. C closes it, and the finding reproduces: 3/8 NIST
(against the chaotic keystream's 8/8), 12.85% bit-avalanche, yet the same
~9x encrypt speedup as Contribution A and ciphertext metrics indistinguishable
from the chaotic grayscale cipher.

All measurements: 2026-10-20, Python 3.13.7, numpy 2.5.4, torch 2.14.1+cpu,
**no CUDA device**, five 256×256 USC-SIPI images.

---

## Why a single Latin-square/chaos cipher has three contributions in it

All three sit on the same pipeline, so they share primitives and differ only
in the keystream source, the channel count, and the number of diffusion axes:

```
plaintext
  1. SHA-256 -> hb                         (metadata transmitted with cipher)
  2. 3-pass Latin-square scramble          (D, D, D3)
  3. full block shuffle                    (256 blocks of 16x16)
  4. Latin-square column-band shuffle      (D5)
  5. bidirectional XOR-chain diffusion, row-major,  key K1
  6. bidirectional XOR-chain diffusion, column axis, key K2   <- two-axis only
ciphertext
```

| variant | channels | keystream | K2 | axes |
|---|---|---|---|---|
| v3 | RGB | 4D Lorenz + Rössler | — | 1 |
| v4 | RGB | 4D Lorenz + Rössler | `ics(SHA-256(hb))` | 2 |
| **DL** | RGB | **NKG (learned MLP)** | — | 1 |
| **DL v2** | RGB | **NKG (learned MLP)** | `NKG(SHA-256(hb))` | 2 |
| **v5** | **grayscale** | 4D Lorenz + Rössler | `ics(SHA-256(hb))` | 2 |
| **DL-gray** | **grayscale** | **NKG (learned MLP)** | — | 1 |
| **DL-gray v2** | **grayscale** | **NKG (learned MLP)** | `NKG(SHA-256(hb))` | 2 |

That shared substrate is what makes the comparison meaningful: the learned and
chaotic keystreams are dropped into an otherwise *identical* cipher, so any
difference in the measured security is attributable to the generator alone.

---

## Quick start

```bash
pip install -r requirements.txt

# Contribution A -- LS-DPAK-DL
cd lsdpak-dl
python nist_tests.py             # keystream randomness: ODE vs NKG vs np.random
python evaluate_dl.py            # single-axis DL metrics
python evaluate_dl_v2.py         # two-axis DL metrics
python denoiser_eval.py          # robustness across 9 noise scenarios
python ablation_study.py         # does the GAN stage do anything?

# Cross-model comparison, one image set and one protocol
cd ../comparison
python compare_models.py && python summarize.py
python diag_keysens.py           # the key-schedule defect, isolated
python diag_avalanche.py         # keystream avalanche, NKG vs chaotic

# Contribution B -- LS-DPAK v5 grayscale
cd ../lsdpak-v5-grayscale
python verify_parity.py --bench  # 35 acceptance gates + per-stage timings
python metrics_gray.py           # metrics + the Jha et al. comparison

# Contribution C -- LS-DPAK-DL grayscale (reuses A's checkpoint, no retraining)
cd ../lsdpak-dl
python lsdpak_dl_gray.py         # round-trip gate, 1-axis and 2-axis
python metrics_dl_gray.py        # metrics + NIST/avalanche on THIS keystream
```

Note `lsdpak-dl/train.py` retrains the NKG and denoiser and **overwrites the
checkpoints every recorded result depends on**. It is not part of the
reproduction path.

---

# Contribution A — LS-DPAK-DL

## A.1 Architecture

**Neural Keystream Generator (NKG).** A 21-byte SHA-256 slice of the plaintext
is embedded to 128 dimensions, then decoded by a positional MLP into 256 chunks
of 256 bytes — exactly one 256×256 keystream plane per forward pass. Trained in
two phases: a chi-square-style uniformity loss, then adversarial fine-tuning
against a histogram discriminator that learns to distinguish the keystream's
byte histogram from uniform noise.

**U-Net denoiser.** Bidirectional diffusion is deliberately tamper-sensitive:
any ciphertext corruption propagates through the whole decrypted image. A
4-level U-Net (~500 K parameters, L1 + SSIM loss) is applied to the decrypted
output to recover from that propagation.

## A.2 Speed — the positive result

| cipher | encrypt | decrypt |
|---|---|---|
| LS-DPAK-DL (1-axis) | **344 ms** | 369 ms |
| LS-DPAK-DL v2 (2-axis) | **712 ms** | 701 ms |
| LS-DPAK v4 (2-axis chaotic) | 2967 ms | 2754 ms |

The chaotic variants spend almost all their time in a 90,000-step RK4 loop run
12× per RGB image. The NKG replaces that with a single batched forward pass,
giving **8.6×** for the single-axis design and **4.2×** for the two-axis one at
matched security metrics (next section).

## A.3 Security — matched on the standard battery

Five images, one protocol: differential = centre pixel +1 mod 256; key
sensitivity = 1-bit flip of the transmitted `hb`.

| metric | DL (1-axis) | DL v2 (2-axis) | v4 (2-axis) | ideal |
|---|---|---|---|---|
| entropy, per-channel mean | 7.99721 | 7.99697 | 7.99724 | 8.0 |
| mean \|corr\| (45 draws) | 0.00369 | 0.00307 | 0.00304 | 0.00317 = noise floor |
| worst \|corr\| (1 of 45) | 0.01138 | 0.01231 | 0.01067 | 0.01336 = random max/1200 |
| max \|interchannel corr\| | 0.00970 | 0.00459 | 0.00406 | 0 |
| NPCR differential (%) | 99.6082 | 99.6261 | 99.6120 | 99.6094 |
| UACI differential (%) | 33.4338 | 33.4566 | 33.4402 | 33.4635 |
| NPCR key-sensitivity (%) | **66.5120** | 99.6218 | 99.6135 | 99.6094 |
| UACI key-sensitivity (%) | **22.2820** | 33.6385 | 33.4644 | 33.4635 |
| cipher χ² min p | 0.2520 | 0.0583 | 0.2805 | > 0.01 |
| round-trip bit-exact | 5/5 | 5/5 | 5/5 | 5/5 |

**DL v2 matches the chaotic cipher on every entry** at 4.2× less cost. On the
standard battery alone, a learned keystream looks like a free speedup.

## A.4 The negative result: a learned keystream is not a random one

It is not a free speedup. Testing the *keystream itself* rather than the
ciphertext:

| keystream | NIST SP 800-22 (8-test subset) |
|---|---|
| 4D Lorenz + Rössler | **8/8 pass** |
| NKG (`nkg_gan.pt`) | **2/8 pass** — Frequency, Runs, Longest Run, Serial, Approximate Entropy and DFT all at p ≈ 0 |
| `np.random` control | 8/8 pass |

Confirmed independently by χ² on 196,608 keystream bytes: NKG p = 0.0000
(entropy 7.99352, mean byte 123.86) against chaotic p = 0.7067 (entropy
7.99911, mean 127.55). **The NKG was trained explicitly to produce uniform byte
histograms and does not achieve one.**

Avalanche is worse, and is the mechanism. Averaged over all 56 single-bit flips
of the 7 hash bytes feeding one channel:

| generator | bytes changed | bits changed | mean \|Δ\| |
|---|---|---|---|
| NKG (learned MLP) | 45.58 % | **13.32 %** | **1.51** |
| 4D Lorenz + Rössler | 99.61 % | 49.99 % | 85.26 |
| ideal | 99.61 % | 50 % | ~85 |

`mean |Δ| = 1.51` out of 255 is the number that matters: the bytes that change
mostly move by ±1. An MLP is a smooth, Lipschitz function, so nearby inputs
give nearby outputs; the only nonlinearity is the `round(ks · 255)`
quantisation. Knowing the keystream for one `hb` gives it to within a couple of
LSB steps for every neighbouring `hb`.

**Why the cipher still passes everything.** The Latin-square scramble, the
block shuffle and the bidirectional diffusion repair the keystream's defects
downstream. The ciphertext's entropy, correlation, NPCR, UACI and χ² are all
fine *because of the permutation and diffusion layers, not because the
keystream is good*. This is the contribution: on this pipeline the standard
evaluation battery **cannot distinguish a cryptographically sound keystream
from one that fails 6 of 8 NIST tests.** Any proposal substituting a learned
generator into a permutation–diffusion cipher must test the generator directly;
ciphertext-level metrics will not reveal the substitution.

**The GAN stage is inert.** The ablation confirms the adversarial phase changes
nothing measurable, and the GAN-trained checkpoint still fails 6/8:

| config | entropy (pooled) | NPCR_diff | UACI_diff | enc ms |
|---|---|---|---|---|
| v3 ODE baseline | 7.9991 | 99.6162 | 33.4455 | 1626 |
| Phase 1 only (uniformity loss) | 7.9990 | 99.6165 | 33.4361 | 397 |
| Phase 1+2 (GAN fine-tuned) | 7.9991 | 99.6156 | 33.4256 | 416 |

## A.5 A key-schedule defect, and what fixes it

Single-axis key sensitivity is **66.51 % NPCR / 22.28 % UACI** against a
99.61 / 33.46 ideal. Per channel:

| channel | NPCR | UACI |
|---|---|---|
| R | 99.498 % | 31.968 % |
| G | **50.000 %** | 25.096 % |
| B | **50.000 %** | 25.090 % |

Both the NKG and the chaotic `ics_from_hash_bytes` derive each channel's key
from a **disjoint 7-byte slice** of `hb` (`hb[0:7]`, `hb[7:14]`, `hb[14:21]`).
A 1-bit change in `hb[0]` therefore leaves the G and B keystreams
**bit-identical**:

| keystream | R | G | B |
|---|---|---|---|
| NKG | 5.08 % | **0.000 %** | **0.000 %** |
| chaotic, pass 1 | 99.625 % | **0.000 %** | **0.000 %** |
| chaotic, pass 2 — ICs from `SHA-256(hb)` | 99.609 % | 99.596 % | 99.652 % |

The joint bidirectional chain then only half-mixes into the untouched channels,
giving exactly 50 % there and 66.5 % overall.

**This is architectural, not neural.** The v3 single-axis *chaotic* cipher
under the identical test gives 66.529 % / 27.936 % — the same defect. What
repairs it in both two-axis designs is pass 2 keying off `SHA-256(hb)`, which
avalanches across all three channels. So the headline benefit of the two-axis
construction is **key sensitivity**, not the correlation improvement originally
claimed for it; that gain is real but small and mostly interchannel
(0.0097 → 0.0046).

The root fix, applicable to the whole RGB line, is to derive each channel's key
from `SHA-256(hb ‖ channel_index)` rather than a disjoint slice.

## A.6 The U-Net denoiser — the unambiguous positive

| scenario | PSNR w/o | PSNR w/ | gain | SSIM w/o | SSIM w/ | gain |
|---|---|---|---|---|---|---|
| S&P 1 % | 25.60 | 25.78 | +0.18 | 0.958 | 0.966 | +0.009 |
| S&P 5 % | 18.56 | 21.87 | **+3.31** | 0.808 | 0.911 | +0.103 |
| S&P 10 % | 15.65 | 18.83 | +3.17 | 0.665 | 0.815 | +0.149 |
| S&P 15 % | 14.01 | 16.60 | +2.59 | 0.554 | 0.705 | +0.151 |
| Gaussian σ=10 | 15.48 | 17.77 | +2.30 | 0.670 | 0.777 | +0.106 |
| Gaussian σ=25 | 12.32 | 13.25 | +0.93 | 0.446 | 0.497 | +0.051 |
| Gaussian σ=50 | 10.42 | 11.11 | +0.69 | 0.262 | 0.288 | +0.026 |
| Crop 1/16 | 20.49 | 23.31 | +2.81 | 0.873 | 0.937 | +0.064 |
| Crop 1/4 | 14.45 | 17.83 | **+3.38** | 0.592 | 0.768 | +0.176 |

Gains across every scenario, up to +3.4 dB and +0.18 SSIM. Importantly this
contribution is **independent of the keystream question** — the denoiser
operates on the decrypted output and could be attached to the chaotic v4 or to
v5 grayscale unchanged. It is the component of the DL branch that survives
A.4 intact.

---

# Contribution B — LS-DPAK v5 grayscale

## B.1 Why grayscale

The reference paper is grayscale-only, so every RGB comparison against it
carries two artifacts that flatter the RGB cipher:

- **Pooled entropy.** Combining three channels into one 196,608-sample
  histogram lowers the entropy estimator's finite-sample bias, giving ~7.9991
  against a per-channel ~7.997. That is a property of the estimator, not the
  cipher, and grayscale cannot and should not reproduce it.
- **Inflated worst-case correlation.** An RGB maximum is over 9 draws per image
  (3 channels × 3 directions) versus 3 in grayscale, so the RGB worst case is
  worse by order statistics alone.

v5 removes both. Metadata drops from 21 bytes to 7, and the diffusion chain
from 3N = 196,608 to N = 65,536.

## B.2 Verified faithfulness — 35 gates, 35 passing

| gate | check | result |
|---|---|---|
| 0 | fp64 end to end; keystream non-zero and spanning 0–255; `(-3.0 + 1e-14) != -3.0` | pass |
| 1 | CPU round-trip bit-exact | 5/5 |
| 2 | keystream, `scramble3`, `shuffle_blocks_full`, `lsq_column_shuffle` and diffusion each **bitwise equal to the v3 RGB original** | pass |
| 3 | lane-vectorised and torch keystreams bitwise equal to the scalar path | pass |
| 4 | **GPU ciphertext bitwise identical to CPU**, plus GPU round-trip | 5/5 |
| 5 | frozen golden SHA-256 ciphertext hashes | 5/5 |
| 6 | both NPCR/UACI protocols wired to the real pipeline | pass |

Gate 2 licenses calling v5 a *reduction* of a published pipeline rather than a
reimplementation of it. Gate 4 demands identity, not closeness — one differing
byte means two different functions, and metrics from the divergent one are
worthless.

**float64 is mandatory.** Byte extraction is `floor(|v| · 1e14) mod 256`, which
reads the bottom of the mantissa — exactly what makes the scheme key-sensitive
and what makes it precision-fragile. In float32 every keystream byte becomes
**0** (at magnitude ~2e15 a float32 ULP is 2²⁷, and 2²⁷ mod 256 = 0), and the
cipher degenerates to the permuted plaintext XORed with zeros *while still
round-tripping cleanly*. Worse, `(-3.0 + 1e-14) == -3.0` is **True** in
float32, so the key-sensitivity test compares a run against itself and reports
NPCR 0.000 % — a test silently measuring nothing. Gate 0 asserts both.

## B.3 Results

**Protocol A — differential plaintext:** `img[128,128] += 1 mod 256`. Since the
initial conditions are SHA-256-adaptive this changes `hb` and the whole
keystream. **Protocol B — key sensitivity:** same plaintext, pass-1 Lorenz `x0`
perturbed by 1e-14, `hb` and therefore K2 held fixed.

| image | entropy | corr_h | corr_v | corr_d | NPCR_A | UACI_A | NPCR_B | UACI_B | χ² p |
|---|---|---|---|---|---|---|---|---|---|
| aerial | 7.99716 | −0.005231 | −0.004047 | +0.001612 | 99.6033 | 33.5313 | 99.6017 | 33.5979 | 0.4290 |
| lena | 7.99733 | −0.006769 | −0.001018 | +0.000796 | 99.6201 | 33.4684 | 99.6231 | 33.5543 | 0.7278 |
| couple | 7.99730 | +0.003511 | −0.006805 | −0.004368 | 99.6033 | 33.5016 | 99.5621 | 33.4508 | 0.6664 |
| clock | 7.99704 | −0.003620 | −0.005171 | −0.004211 | 99.6109 | 33.4135 | 99.6201 | 33.2221 | 0.2612 |
| man | 7.99704 | −0.000179 | −0.001216 | +0.000930 | 99.6094 | 33.3430 | 99.6140 | 33.4151 | 0.2764 |
| **mean** | **7.99717** | | | | **99.6094** | **33.4516** | **99.6042** | **33.4481** | |
| **ideal** | 8.0 | 0 | 0 | 0 | 99.6094 | 33.4635 | 99.6094 | 33.4635 | >0.01 |

Entropy 7.99717 sits inside the 7.99700–7.99763 band of the RGB *per-channel*
values — not near the pooled 7.9991. A grayscale run reporting ~7.999 would
mean something had been pooled. Mean NPCR_A landing on 99.6094 is coincidence
on five images, not precision.

## B.4 Against Jha et al. — like for like

Their Table 6 is a protocol-B result (their §5.7 describes the 1e-14
perturbation), so it belongs against the protocol-B column.

| image | v5 entropy | Jha | v5 NPCR_B | Jha NPCR | v5 UACI_B | Jha UACI |
|---|---|---|---|---|---|---|
| aerial | 7.99716 | 7.9973 | 99.6017 | 99.6094 | **33.5979** | 28.5589 |
| lena | 7.99733 | 7.9974 | 99.6231 | 99.6277 | **33.5543** | 28.7091 |
| couple | 7.99730 | 7.9969 | 99.5621 | 99.6246 | **33.4508** | 28.0631 |
| clock | 7.99704 | 7.9968 | 99.6201 | 99.6017 | **33.2221** | 35.5581 |
| man | 7.99704 | 7.9970 | 99.6140 | 99.5865 | **33.4151** | 28.8974 |
| **mean** | **7.99717** | **7.9971** | **99.6042** | **99.6100** | **33.4481** | **29.9573** |

**Entropy, NPCR and correlation are ties, and should be.** Both produce uniform
output, NPCR saturates near 99.61 % for anything competent, and both
correlations are at the noise floor (v5 mean |r| 0.00330, theirs 0.00287).
Claiming a win on these would be claiming a win on noise.

**UACI is not a tie.** Mean absolute deviation from the 33.4635 ideal:

| | mean \|UACI − 33.4635\| |
|---|---|
| v5 grayscale | **0.1056** |
| Jha et al. | **4.3440** |

Five of their six values cluster at 28–29 %, about 5 points low, with only
Clock (35.56 %) above. A one-sided deviation of that magnitude across almost
every image is not sampling scatter, and UACI has no plausible mechanism to sit
5 points below ideal for a cipher whose NPCR is already 99.61 % — the two
statistics are computed from the same difference image.

**The protocol conflation.** Their Table 6 is cited in §5.6 as a
differential-plaintext result and in §5.7 as a key-sensitivity result; only the
second protocol could have produced the reported numbers. This matters
structurally: with a single non-feedback XOR the differential-plaintext ceiling
is **1/65536 ≈ 0.0015 %**, because flipping one plaintext pixel can only affect
one ciphertext pixel. v5 reaches **99.6094 %** on protocol A because the
SHA-256-adaptive ICs rebuild the entire keystream from the new plaintext hash
and the bidirectional chain propagates it both ways. Protocol A is the
measurement the reference paper never performed.

## B.5 Correlation sits at the noise floor

A 256×256 image has 65,280 adjacent pairs, so an adjacent-pixel correlation has
standard error 1/√65280 = 0.00391. Monte Carlo over uniform random images — the
best any cipher could do — gives mean |r| = **0.00316**, 95th percentile
0.00789, max 0.01336 over 1200 draws.

| statistic | DL (1-axis) | DL v2 | v4 | v5 gray | true random |
|---|---|---|---|---|---|
| mean \|r\| | 0.00369 | 0.00307 | 0.00304 | 0.00330 | **0.00316** |
| worst single \|r\| | 0.01138 | 0.01231 | 0.01067 | 0.00680 | 0.01336 |

Every two-axis variant is a statistical tie with a perfect source. Consequences
for anyone extending this work:

- Do not tune toward a lower single-image correlation — that is fitting noise.
  A "|r| < 0.002 on all images" target is unreachable: one draw from a
  *perfect* cipher falls below 0.002 only 39.1 % of the time, so nine
  consecutive draws doing so has probability 0.02 %.
- Do not claim a correlation win from one image, in either direction.
- v5's lower worst case is expected from having 3 draws per image instead of 9,
  and is **not** evidence that grayscale mixes better.

## B.6 Performance — one win, one honest loss

The forward diffusion chain is an inclusive prefix-XOR of `s ^ k` and the
backward chain a reverse one. XOR is associative, so both are parallel scans;
and since XOR on a bit is addition mod 2, prefix-XOR is the parity of a
`cumsum`:

| stage | before | after | speedup |
|---|---|---|---|
| diffusion, sequential Python loop → NumPy scan | 73.9 ms | 33.3 ms | **2.2×** |
| diffusion, → torch scan | 73.9 ms | 12.0 ms | **6.1×** |
| decryption diffusion — needs no scan at all | 73.9 ms | 0.2 ms | **370×** |
| encrypt end to end, keys precomputed (NumPy → torch) | 84.3 ms | 22.8 ms | 3.7× |

Decryption is cheapest because both its loops are elementwise with a neighbour.

**The ODE did not go the expected way.** Keystream generation dominates
(2811 ms of a ~2900 ms metric step). RK4 is sequential in time, so the only
axis is across trajectories — and neither batched strategy helps on CPU:

| keystream path | per lane | vs scalar |
|---|---|---|
| NumPy scalar loop | 332 ms | 1.00× |
| NumPy lane-vectorised, B=2 | 6944 ms | 0.05× |
| NumPy lane-vectorised, B=30 | 508 ms | 0.65× |
| torch lane-vectorised, B=2 | ~196,900 ms | 0.002× |
| torch lane-vectorised, B=30 | ~11,800 ms | 0.03× |

The ODE state is only 4–7 doubles, far too small for array dispatch to
amortise: a NumPy call on a 2-element array costs ~1.5 µs against ~50 ns for
Python float arithmetic. Batched cost is nearly flat in B and only overtakes
scalar at **B ≈ 36**. Torch is 13× worse again, purely dispatch-bound at ~200 s
per 90,000-step loop regardless of batch size. `build_keystream_many()`
dispatches on that measured crossover.

The batched and torch paths are retained as the correct *shape* for a real CUDA
device, but fp64 runs at 1/32–1/64 of fp32 on consumer GeForce cards (1/2 on
A100/H100), so even there the ODE is the stage least likely to pay off. A GPU
port should write it as a fused per-trajectory kernel — 90,000 launches reduced
to one — not expect the batched form to carry it.

---

---

# Contribution C — LS-DPAK-DL grayscale

Contributions A and B each leave one axis untested: A measures the learned
keystream only on RGB, B measures the chaotic keystream only on grayscale.
Neither tells you whether "learned vs. chaotic" survives a change of colour
space. This contribution closes that gap: it takes the exact `nkg_gan.pt`
checkpoint from Contribution A — same weights, no retraining — and drops it
into Contribution B's single-channel pipeline in place of the chaotic
keystream.

## C.1 How the reuse works

`nkg_gan.pt`'s embedder is `nn.Linear(21, 64)`, fixed by the checkpoint shape.
v5's grayscale `hb` is 7 bytes. `NKG.generate_keystream()` already handles
inputs shorter than 21 bytes — `generate_channel_keystreams()` in the RGB
branch is just three calls to it with three 7-byte channel slices zero-padded
to 21. `lsdpak_dl_gray.py` does the same thing once: pad the 7-byte grayscale
`hb` with 14 zero bytes and call the same function.

```
hb   = SHA-256(img)[:7]                      # v5's 7-byte metadata
K1   = NKG(pad21(hb))                        # replaces build_keystream(ics1)
x    = scramble3 -> shuffle_blocks_full -> lsq_column_shuffle
c1   = bidir_encrypt(x, K1)                   # pass 1, row-major

hb2  = SHA-256(hb)[:7]
K2   = NKG(pad21(hb2))                        # replaces build_keystream(ics2)
cipher = bidir_encrypt(c1.T, K2.T).T           # pass 2, column axis
```

**This is a repurposing of an RGB-trained generator, not a grayscale-trained
one.** The 14 padding zeros are out of the training distribution — nothing in
`nkg_gan.pt`'s training ever saw an input with those positions fixed at zero.
That is exactly why the keystream is tested directly below rather than
assuming the RGB finding transfers.

Round-trip gate, both single-axis and two-axis, all five images:

```
image     1-axis RT  2-axis RT
aerial           ok         ok
lena             ok         ok
couple           ok         ok
clock            ok         ok
man              ok         ok
```

## C.2 A structural note on key sensitivity

v5's protocol B perturbs a *continuous* Lorenz `x0` by 1e-14 while `hb` (and
therefore K2) stays completely fixed — possible because v5's pass-1 key
material is independent of the 7 transmitted bytes. NKG has no continuous key:
the only key material is `hb` itself, and K2 is deterministically chained from
`hb`, so there is no way to perturb "pass 1 only" without touching pass 2 as
well. (This was already true of the RGB DL v2 cipher — `encrypt_dl_2axis`
recomputes both passes from whatever `hb` it is given — it was simply never
stated.)

So Contribution C's key-sensitivity test is a 1-bit flip of `hb`, changing
**both** K1 and K2, labelled **protocol A′** rather than B to keep this
structural difference visible in the table rather than implying a like-for-like
test with v5's protocol B.

## C.3 Results

The two-axis variant (DL-gray v2 in the architecture table), to compare against
v5's two-axis chaotic design rather than its own weaker single-axis sibling.
Same five images, same protocol A (centre pixel `+1 mod 256`) as Contributions
A and B.

| image | entropy | corr_h | corr_v | corr_d | NPCR_A | UACI_A | NPCR_A′ | UACI_A′ | χ² p | enc ms |
|---|---|---|---|---|---|---|---|---|---|---|
| aerial | 7.99671 | −0.007008 | +0.000744 | −0.000268 | 99.6063 | 33.3689 | 99.6078 | 33.4825 | 0.0382 | 115.4 |
| lena | 7.99725 | +0.004904 | +0.003416 | +0.004100 | 99.6170 | 33.4436 | 99.5819 | 33.7836 | 0.5887 | 104.2 |
| couple | 7.99733 | +0.000286 | −0.004064 | −0.001903 | 99.5941 | 33.4821 | 99.6002 | 33.6010 | 0.7117 | 112.3 |
| clock | 7.99759 | −0.002374 | −0.000119 | +0.002137 | 99.6201 | 33.4246 | 99.5514 | 33.3573 | 0.9547 | 109.2 |
| man | 7.99710 | −0.004285 | +0.001845 | +0.003208 | 99.6216 | 33.4315 | 99.5789 | 33.8740 | 0.3618 | 103.7 |
| **mean** | **7.99720** | | | | **99.6118** | **33.4301** | **99.5840** | **33.6197** | | **109.0** |
| ideal | 8.0 | 0 | 0 | 0 | 99.6094 | 33.4635 | 99.6094 | 33.4635 | > 0.01 | — |

(The single-axis DL-gray variant is in `results/metrics_dl_gray.json`; it
reproduces the Contribution A pattern — NPCR_A′ collapses because the
single-axis design has no hash-chained second pass to repair a one-sided key
change. Omitted from the headline table because v5 and the other grayscale
rows are all two-axis; a single-axis grayscale row would compare the wrong
architecture.)

**Ciphertext metrics are, again, fine.** Entropy 7.99720 sits in the same band
as v5's 7.99717. χ² passes on all 5 images at p > 0.01 (aerial is the lowest,
at p = 0.0382, comfortably above the threshold). NPCR/UACI sit within a few
hundredths of ideal on both protocols. By every ciphertext-level measure, this
cipher is indistinguishable from the chaotic one.

## C.4 The keystream is not — same finding, now in grayscale

Measured directly on the padded 65,536-byte keystream (not inferred from the
RGB result):

| | DL-gray keystream | chaotic keystream (v5) | `np.random` |
|---|---|---|---|
| NIST SP 800-22 (8-test) | **3/8** | 8/8 | 8/8 |
| χ² p-value | **0.0000** | 0.7067 | — |
| bit-avalanche (56 single-bit flips) | **12.85 %** | 49.99 % | — |
| mean \|Δ\| per flip | **1.35** | 85.26 | — |

```
Frequency (Monobit)      p=0.000000  [FAIL]
Block Frequency          p=0.115578  [PASS]
Runs                     p=0.000000  [FAIL]
Longest Run              p=0.063845  [PASS]
Serial                   p=0.000000  [FAIL]
Approx. Entropy          p=0.000000  [FAIL]
Cumulative Sums          p=1.000000  [PASS]
DFT (Spectral)           p=0.000000  [FAIL]
```

This is the RGB finding reproduced on a different colour space, with the same
mechanism: mean |Δ| = 1.35 out of 255 means the bytes that do change mostly
move by ±1, consistent with a smooth MLP whose only nonlinearity is output
quantisation. 3/8 rather than 2/8 and 12.85 % rather than 13.32 % bit-avalanche
— close enough to the RGB numbers that the defect reads as intrinsic to the
architecture, not to the RGB-specific training or channel-splitting.

Note on aerial's p = 0.0382: it is the lowest of the five but still passes
comfortably, and this report does not attribute it to the keystream defect
without the per-image avalanche measurement that would be needed to support
that claim. Treat it as ordinary image-to-image scatter unless that
measurement is added.

## C.5 Speed — the finding also transfers

| cipher | encrypt (one image, keystream included) |
|---|---|
| LS-DPAK-DL grayscale, 2-axis | **109.0 ms** |
| LS-DPAK v5 grayscale (chaotic), 2-axis | 880.4 ms |
| **speedup** | **8.1×** |

Matches Contribution A's RGB speedup (8.6× single-axis, 4.2× two-axis) closely
enough to say the mechanism — replacing a 90,000-step RK4 integration with one
batched MLP forward pass — is colour-space-independent, as expected; nothing
about the speed advantage was RGB-specific to begin with.

## C.6 What Contribution C adds that A and B could not show alone

1. **The NIST failure is not an RGB artifact.** Padding zeros into 14 unused
   input slots, on a different colour space, with a different pipeline
   downstream, reproduces the same keystream defect to within a few percent.
   That is evidence the defect is in the generator's smoothness, not in any
   RGB-specific interaction.
2. **The ciphertext-masks-the-keystream finding is pipeline-general, not
   tied to the 3-channel permutation stages.** v5's single-channel permutation
   stack (3-pass scramble, block shuffle, one column-band shuffle) is simpler
   than the RGB stack (which also has the interchannel shuffle), and it still
   fully masks the defect at the ciphertext level. The claim in A.4 — "this
   evaluation battery cannot distinguish a sound keystream from a NIST-failing
   one" — now holds across two different permutation stacks, not one.
3. **The speed advantage is confirmed independent of RGB-specific
   per-channel batching.** A.2's 8.6× came partly from one NKG call replacing
   three RGB ODE integrations; C.5's 8.1× comes from one NKG call replacing
   one grayscale ODE integration, so the speedup is not an artifact of
   amortising three RGB channels into one forward pass.

---

## Limitations

- **Five images, not seven.** `Baboon.tiff` is absent and Cameraman was never
  present; Jha et al. averaged over seven. Means are over the five shared
  images and every per-image value is listed, so the gap is visible rather than
  hidden in an average. Both are standard test images (Cameraman ships with
  MATLAB as `cameraman.tif`).
- **No CUDA device**, so the torch port is validated for correctness and
  benchmarked on CPU only; its GPU performance is unmeasured.
- **The grayscale reduction sidesteps the A.5 key-schedule defect rather than
  fixing it.** With one channel there is no `hb` to split, so v5 is unaffected —
  but that is not a solution, and the RGB line still needs the
  `SHA-256(hb ‖ channel_index)` fix.
- **`lsdpak-dl/results/metrics_dl_v2_2axis.json` is not reproducible** from the
  current `evaluate_dl_v2.py` (aerial NPCR_diff 99.60225 now vs 99.61700
  recorded). It is not a torch-version effect — the single-axis path uses the
  same NKG forward pass and reproduces bit-exactly — nor the "reuse the
  transposed pass-1 key" variant, which gives 99.61599. It is from a lost
  revision and should be regenerated before being cited. The tables in A.3 use
  freshly computed values from `comparison/`, not this file.
- **`metrics_dl.json`'s `npcr_key`/`uaci_key` fields are meaningless.** That
  test (`evaluate_dl.py:232-252`) bypasses the cipher, XOR-diffusing the raw
  plaintext with `np.roll`-derived keystreams the cipher never uses. A.5 uses
  the corrected measurement from `comparison/compare_models.py`.
- **One rename, and a correction to the RGB write-up.** v3's
  `interchannel_shuffle` is `lsq_column_shuffle` in v5. The RGB name is
  inaccurate: that function never moves data between R, G and B — it applies
  the same D5 column-band permutation independently to each channel, making it
  structurally a fourth Latin-square permutation pass. Genuine cross-channel
  coupling came from the diffusion chain running over the three concatenated
  planes. The stage is kept in v5, applied to the single plane, because
  dropping it would remove a permutation pass.
- **`results/metrics_gray.json` is not shaped like the RGB files:** `entropy`
  is a scalar, not an `{R,G,B,joint}` object, and `interchannel_corr` is
  absent.
- **Unseeded RNG.** `evaluate_dl.py`'s `add_salt_pepper` and
  `generate_extra_images.make_noise_img` use unseeded `np.random`, so denoiser
  figures shift between runs.
- **Dead image names.** Six of the ten names in `evaluate_dl.py:70` and two of
  six in `evaluate_dl_v2.py:53` are not keys of `TIFF_SRC` and can never load;
  the first hides this behind `load_image_safe`, the second raises `KeyError`.
  `generate_extra_images.py:20` writes four levels above the repository.
- **Contribution C has no frozen golden hashes.** v5's `verify_parity.py`
  freezes SHA-256 ciphertext hashes per image and re-checks them on every
  change; `lsdpak_dl_gray.py` has only the round-trip gate shown in C.1. Add a
  golden-hash file for it before relying on C's numbers across future edits to
  the shared grayscale primitives it imports from `lsdpak-v5-grayscale`.
- **The ablation in A (uniformity-loss-only vs. GAN-fine-tuned) was never
  re-run on the grayscale repurposing.** It is plausible the GAN stage is
  equally inert here (A's ablation found it changes nothing measurable on
  RGB), but C.4's NIST/avalanche numbers are for `nkg_gan.pt` only; the
  `nkg_pretrained.pt` checkpoint has not been measured through this pipeline.

---

## Repository layout

```
lsdpak-v3-rgb/lsdpak_v3_rgb.py       shared primitives; also the gate-2 oracle
lsdpak-v4-2d-diffusion/lsdpak_v4.py  two-axis chaotic RGB, the cipher v5 reduces

lsdpak-dl/                           CONTRIBUTION A
├── lsdpak_dl.py                     NKG, histogram discriminator, U-Net denoiser
├── train.py                         retrains -- OVERWRITES the checkpoints
├── evaluate_dl.py                   single-axis metrics
├── evaluate_dl_v2.py                two-axis metrics
├── nist_tests.py                    keystream randomness, ODE vs NKG vs random
├── ablation_study.py                uniformity-only vs GAN-fine-tuned
├── denoiser_eval.py                 9 noise/crop scenarios
├── generate_figures.py              figures
├── checkpoints/                     nkg_gan, nkg_pretrained, discriminator, denoiser
├── lsdpak_dl_gray.py                CONTRIBUTION C -- nkg_gan.pt reused on v5's
│                                     single-channel pipeline, no retraining
├── metrics_dl_gray.py               C's metrics + NIST/avalanche on this keystream
└── results/                         metrics, NIST report, figures, metrics_dl_gray.json

lsdpak-v5-grayscale/                 CONTRIBUTION B (and the pipeline C imports)
├── lsdpak_gray_cpu.py               NumPy reference -- AUTHORITATIVE
├── lsdpak_gray_gpu.py               PyTorch port, bitwise identical
├── verify_parity.py                 35 acceptance gates (--freeze, --bench)
├── metrics_gray.py                  both NPCR/UACI protocols
├── golden/ciphertext_hashes.json    frozen SHA-256 per image
└── results/                         metrics_gray.json, verify_and_bench.txt

comparison/                          cross-model harness, one protocol for all
├── compare_models.py                drives DL, DL v2 and v4 unmodified
├── summarize.py                     the A.3 table
├── diag_keysens.py                  isolates the A.5 defect
├── diag_avalanche.py                the A.4 avalanche measurement
└── RESULTS.md                       narrower findings write-up

test_images/                         Aerial, Clock, Couple, Lena, Man (256×256)
```

The three per-variant evaluation scripts are **not** mutually comparable as
written — different image sets, different differential-pixel choices, and
incomparable key-sensitivity tests. `comparison/compare_models.py` exists
because of that: it drives all three ciphers unmodified over one image set with
one protocol, which is where the A.3 and A.5 numbers come from.

## Note on test images

`test_images/Lena.jpg` is included for comparability with the reference paper,
which uses it. Its provenance is contested and it is being retired from
image-processing literature, so a venue may ask for it to be replaced. The
pipeline is image-agnostic; substituting it costs a re-run of
`verify_parity.py --freeze` and the metric scripts.
