# LS-DPAK: Learned and Chaotic Keystreams for Latin-Square Image Encryption

Three contributions, benchmarked against Jha, Singh, Pradhan, Mishra, Jain &
Ansari (2025), *"An Advanced Image Encryption Algorithm Integrating Orthogonal
Latin Square Transversal for Shuffling With Lorenz Chaotic System for
Diffusion"*, **Security and Privacy 8:e70128**,
[doi:10.1002/spy2.70128](https://doi.org/10.1002/spy2.70128).

**A note on the test images, before anything else.** All five source files in
`test_images/` are PIL mode `"L"` — genuinely single-channel grayscale, not
colour. Verified directly: after `.convert("RGB")`, every pixel has
`max|R−G| = max|R−B| = max|G−B| = 0`. So every "RGB" number ever produced by
v3, v4, or the original RGB-trained Contribution A was computed on a
grayscale image replicated into three identical channels — the per-channel
R/G/B differences in those tables came entirely from each channel being
encrypted with an independently-derived key, never from real colour content.
There was no 3-channel signal to encrypt, ever. Contribution C below is the
one place in this project built to match that fact: a deep-learning cipher
*trained from scratch on the actual grayscale data*, not an RGB network with
padded or triplicated input.

**A. LS-DPAK-DL** — replacing the chaotic keystream with a *learned* generator
(a GAN-fine-tuned MLP), plus a U-Net denoiser for noise-robust decryption, run
on the artificially-triplicated RGB data described above. This yields an 8.6×
speedup, a measurable robustness gain, **and a negative result that we
consider the more important finding**: the learned keystream is not
cryptographically random, yet the cipher built on it passes every standard
test, because the permutation and diffusion layers mask the defect. Kept in
this article because the finding is reproduced independently in Contribution
C on real grayscale data, which is the comparison that actually matters.

**B. LS-DPAK v5 grayscale** — the two-axis chaotic cipher reduced to one
channel, making the comparison against the (grayscale-only) reference paper
like-for-like and removing two measurement artifacts that flatter RGB
implementations. Verified bitwise against its RGB predecessor at every stage.

**C. LS-DPAK-DL grayscale** — a Neural Keystream Generator and histogram
discriminator sized for and trained from scratch on the real grayscale
images (7-byte hash in, one 256×256 plane out, no 21-byte RGB input and no
zero-padding), dropped into Contribution B's single-channel pipeline. This is
the comparison the other two contributions cannot make alone: A tests
learned-vs-chaotic on data that was never really RGB; B tests chaotic-only.
C tests learned-vs-chaotic on the colour space the data actually is, with a
network architecturally sized for that data rather than reused from a
differently-shaped one. The finding reproduces, and more cleanly than an
earlier RGB-weight-reuse attempt did: 2/8 NIST (against the chaotic
keystream's 8/8) but markedly *better* avalanche once the network isn't
carrying 14 unused input dimensions (23.9% bits changed vs. 13.3% for the
RGB-reuse version), and ciphertext metrics still indistinguishable from the
chaotic cipher.

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

# Contribution C -- LS-DPAK-DL grayscale, trained from scratch (not RGB reuse)
cd ../lsdpak-dl
python train_dl_gray.py          # trains NKGGray + discriminator + denoiser
python lsdpak_dl_gray.py         # round-trip gate, 1-axis and 2-axis
python metrics_dl_gray.py        # metrics + NIST/avalanche on THIS keystream
```

Note `lsdpak-dl/train.py` retrains Contribution A's RGB NKG and denoiser and
**overwrites the checkpoints every A result depends on**; it is not part of
the reproduction path for A's recorded numbers. `train_dl_gray.py` is
separate -- its own checkpoints, its own `checkpoints_gray/` directory -- and
re-running it will change Contribution C's numbers (it is a from-scratch
training run, not a deterministic reproduction) unless you fix the RNG seeds
first.

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

Contributions A and B each test one axis: A measures a learned keystream,
but on data that was never actually RGB (see the note at the top of this
document); B measures the chaotic keystream on the real grayscale data. C
tests the learned keystream on the real grayscale data too — not by reusing
A's RGB-shaped weights, but with a new network (`NKGGray`,
`HistogramDiscriminatorGray`) sized for and trained from scratch on the
actual 7-byte grayscale hash and the five real images in `test_images/`.

## C.1 Why a fresh model, not a reuse

An earlier version of this cipher reused `nkg_gan.pt` — the RGB checkpoint
from Contribution A — by zero-padding v5's 7-byte `hb` out to the 21 bytes
that network's embedder expects. That was defensible as a quick cross-check,
but it meant testing a network on 14 input dimensions it was never trained to
see anything in.

`lsdpak_dl_gray_model.py` instead defines `NKGGray` with a
`nn.Linear(7, 64)` embedder — the exact length of v5's `hb`, nothing padded,
nothing wasted — plus a single-channel `HistogramDiscriminatorGray` (one
256-bin histogram, not three fused) and a single-channel `UNetDenoiserGray`
(in/out channels = 1, not 3). `train_dl_gray.py` runs the same three-phase
recipe as `train.py` (uniformity pre-training, GAN fine-tuning, denoiser
training — identical loss functions, optimiser settings, and epoch counts),
on the five real grayscale images instead of the triplicated-RGB ones.
Neither phase involves the slow chaotic ODE — Phase 1/2's data is just
SHA-256 hashes of image patches, Phase 3's is synthetically corrupted
plaintext — so training is fast: Phases 1–2 together take under two minutes
on this CPU.

Checkpoints land in `lsdpak-dl/checkpoints_gray/`, kept separate from
Contribution A's `checkpoints/` so neither training run can overwrite the
other.

```
hb   = SHA-256(img)[:7]                      # v5's 7-byte metadata, used whole
K1   = NKGGray(hb)                           # replaces build_keystream(ics1)
x    = scramble3 -> shuffle_blocks_full -> lsq_column_shuffle
c1   = bidir_encrypt(x, K1)                   # pass 1, row-major

hb2  = SHA-256(hb)[:7]
K2   = NKGGray(hb2)                          # replaces build_keystream(ics2)
cipher = bidir_encrypt(c1.T, K2.T).T           # pass 2, column axis
```

Round-trip gate, both single-axis and two-axis, all five images, with the
trained model:

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
material is independent of the 7 transmitted bytes. `NKGGray` has no
continuous key: the only key material is `hb` itself, and K2 is
deterministically chained from `hb`, so there is no way to perturb "pass 1
only" without touching pass 2 as well.

So Contribution C's key-sensitivity test is a 1-bit flip of `hb`, changing
**both** K1 and K2, labelled **protocol A′** rather than B to keep this
structural difference visible in the table rather than implying a
like-for-like test with v5's protocol B.

## C.3 Results

The two-axis variant, to compare against v5's two-axis chaotic design. Same
five images, same protocol A (centre pixel `+1 mod 256`) as Contributions A
and B.

| image | entropy | corr_h | corr_v | corr_d | χ² p | NPCR_A | UACI_A | NPCR_A′ | UACI_A′ | enc ms |
|---|---|---|---|---|---|---|---|---|---|---|
| aerial | 7.99778 | +0.005490 | -0.004822 | +0.004768 | 0.9945 | 99.5712 | 33.3609 | 99.6445 | 33.5004 | 182.2 |
| lena | 7.99703 | +0.003999 | -0.002015 | -0.003221 | 0.2092 | 99.5895 | 33.5631 | 99.5651 | 33.3394 | 181.3 |
| couple | 7.99658 | -0.000275 | +0.000062 | +0.008223 | 0.0129 | 99.6414 | 33.4701 | 99.6292 | 33.4903 | 177.8 |
| clock | 7.99697 | +0.000611 | -0.004194 | +0.003192 | 0.1838 | 99.6277 | 33.5707 | 99.6246 | 33.5538 | 167.7 |
| man | 7.99698 | +0.002606 | -0.003465 | -0.003462 | 0.1884 | 99.6185 | 33.5506 | 99.6170 | 33.4737 | 179.2 |
| **mean** | **7.99707** | | | | | **99.6097** | **33.5031** | **99.6161** | **33.4715** | **177.6*** |
| ideal | 8.0 | 0 | 0 | 0 | >0.01 | 99.6094 | 33.4635 | 99.6094 | 33.4635 | — |

All 5 images pass chi-square uniformity at p > 0.01; couple is the lowest at
0.0129, comfortably above the threshold. \*The 177.6 ms figure
was measured while the denoiser's Phase 3 training was running concurrently
on the same CPU; it is an upper bound, not a clean measurement, and will be
replaced once that run finishes and the comparison can be re-measured in
isolation.

The single-axis variant (`results/metrics_dl_gray.json`, `variants.1axis`) is
omitted from the headline table for the same reason as before: v5 and the
other grayscale rows are two-axis, so a single-axis row would compare the
wrong architecture. Its NPCR_A′ (99.53% mean) is closer to ideal than the
single-axis RGB result in Contribution A (66.51%) and the earlier RGB-reuse
grayscale attempt would have shown, because there is no disjoint
per-channel hash split here to begin with — grayscale has only one channel,
so the Finding-A.5 defect has nothing to act on. That is a structural
non-issue for grayscale, not a fix.

**Ciphertext metrics are, again, fine.** Entropy 7.99707 sits in the same
band as v5's 7.99717. NPCR/UACI sit within a few hundredths of ideal on both
protocols. By every ciphertext-level measure, this cipher is indistinguishable
from the chaotic one.

## C.4 The keystream is not — and the comparison is now cleaner

Measured directly on the natively-trained 65,536-byte keystream:

| | DL-gray (native) | DL-gray (earlier RGB-weight reuse) | chaotic (v5) | `np.random` |
|---|---|---|---|---|
| NIST SP 800-22 (8-test) | **2/8** | 3/8 | 8/8 | 8/8 |
| χ² p-value | **0.0000** | 0.0000 | 0.7067 | — |
| bit-avalanche (56 single-bit flips) | **23.87 %** | 12.85 % | 49.99 % | — |
| mean \|Δ\| per flip | **6.94** | 1.35 | 85.26 | — |

```
Frequency (Monobit)      p=0.000000  [FAIL]
Block Frequency          p=0.993805  [PASS]
Runs                     p=0.000000  [FAIL]
Longest Run               p=0.000000  [FAIL]
Serial                   p=0.000000  [FAIL]
Approx. Entropy          p=0.000000  [FAIL]
Cumulative Sums          p=1.000000  [PASS]
DFT (Spectral)           p=0.000000  [FAIL]
```

Two things worth separating out:

1. **Avalanche nearly doubles once the network is correctly sized.** 23.87%
   bits changed against the RGB-weight-reuse version's 12.85%, and mean
   |Δ| rises from 1.35 to 6.94 — removing 14 unused, out-of-distribution input
   dimensions measurably helps the network respond to its actual input. This
   is evidence the earlier version's weaker avalanche was partly an artifact
   of the padding hack, not purely the architecture.
2. **NIST pass count does not improve (2/8, if anything one test worse than
   the padding version's 3/8).** Better avalanche does not fix the deeper
   problem: an MLP decoder is still a smooth function end to end, so Frequency,
   Runs, Longest Run, Serial, Approximate Entropy and DFT all still fail. The
   architecture's smoothness, not the input padding, is the root cause — this
   native run isolates that distinction for the first time in this project.

The ciphertext still looks uniform (χ² passes on all 5 images; see
`results/metrics_dl_gray.json`) because the Latin-square scramble, block
shuffle, column shuffle and bidirectional diffusion repair the keystream's
defects downstream — not because the keystream is good. This is the A.4
finding, now demonstrated on data that is genuinely the colour space being
claimed, with a network genuinely sized for that data.

## C.5 Speed

A clean, isolated re-measurement is pending (see C.3's note on concurrent
load during this run). The provisional figure — 177.6 ms end-to-end per
2-axis image — against v5's clean 880.4 ms gives roughly 5×, but that number
was measured under CPU contention and should not be cited as the final
figure. Will be replaced once Contribution C's denoiser training (running at
the time of this commit) completes and a clean benchmark can be taken.

## C.6 What Contribution C adds that A and B could not show alone

1. **"Learned vs. chaotic" is now tested on data that is actually the colour
   space being claimed**, with networks sized for that data rather than
   repurposed from a differently-shaped one. Both earlier framings — RGB
   Contribution A on triplicated grayscale, and a padded-input reuse of its
   weights — were testing something slightly different from what their
   headline claim said. This one is not.
2. **Separates "padding hurt avalanche" from "the architecture is
   fundamentally smooth."** The native model's near-doubled avalanche
   (23.87% vs 12.85%) but unchanged NIST failure count shows those are two
   separate effects, not one — useful for anyone trying to fix this class of
   generator: resizing the input does help the avalanche property, but does
   not touch the smoothness problem that drives the NIST failures.
3. **The ciphertext-masks-the-keystream finding holds a third time**, now on
   a keystream trained with no RGB dependency anywhere in its history.

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
- **The ablation in A (uniformity-loss-only vs. GAN-fine-tuned) has not been
  re-run on the native grayscale model.** C.4's NIST/avalanche numbers are for
  `checkpoints_gray/nkg_gray_gan.pt` (post-GAN-fine-tuning) only; whether
  Phase 2 does anything measurable to the grayscale network specifically
  (A's ablation found it inert on RGB) is not yet checked.
- **Contribution C's denoiser was mid-training at the time of this commit**
  and `checkpoints_gray/denoiser_gray.pt` does not exist yet. The cipher
  itself (encrypt/decrypt, both axes) does not depend on it and is fully
  measured above; the denoiser-robustness analogue of A.6 will follow in a
  separate commit once training completes.
- **Training is not seeded.** `train_dl_gray.py` has no fixed RNG seed, so
  re-running it produces a different `NKGGray` and different C.3/C.4 numbers
  each time — unlike every other result in this document, which is either
  deterministic (the chaotic ciphers) or measured on a frozen checkpoint. Add
  a seed before treating C's exact figures as reproducible rather than
  representative.
- **C.3's enc_ms figure was measured under CPU contention** (concurrent
  denoiser training) and is flagged in-line as an upper bound; treat the
  ~5× provisional speedup with more caution than the other timing numbers in
  this document until it is re-measured in isolation.

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
├── lsdpak_dl_gray_model.py          CONTRIBUTION C -- NKGGray, HistogramDiscriminatorGray,
│                                     UNetDenoiserGray: fresh single-channel architectures
├── train_dl_gray.py                 trains them from scratch on the real grayscale images
├── lsdpak_dl_gray.py                C's cipher (1-axis, 2-axis), built on NKGGray
├── metrics_dl_gray.py               C's metrics + NIST/avalanche on this keystream
├── checkpoints_gray/                nkg_gray_pretrained, nkg_gray_gan, discriminator_gray_gan,
│                                     denoiser_gray -- separate from checkpoints/, A's RGB weights
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
