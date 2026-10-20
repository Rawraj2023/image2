# LS-DPAK-DL vs LS-DPAK-DL v2 vs LS-DPAK v4

Run 2026-10-20. Python 3.13.7, numpy 2.5.4, torch 2.14.1+cpu (no CUDA), CPU only.
Reproduce with `python comparison/compare_models.py && python comparison/summarize.py`.

## What was run

| Script | Result |
|---|---|
| `lsdpak-v4-2d-diffusion/lsdpak_v4.py` | 4/6 images, then `FileNotFoundError: Baboon.tiff` |
| `lsdpak-dl/evaluate_dl.py` | 4/10 images (6 skipped, not in `TIFF_SRC`); completed |
| `lsdpak-dl/evaluate_dl_v2.py` | 4/6 images, then `KeyError: '4_2_03'` |
| `lsdpak-dl/nist_tests.py` | completed |
| `lsdpak-dl/ablation_study.py` | completed |
| `lsdpak-dl/denoiser_eval.py` | completed |
| `lsdpak-dl/generate_figures.py` | completed (needed `matplotlib`) |
| `lsdpak-dl/train.py` | not run — would overwrite the checkpoints the results depend on |

Installed into `.venv` to make the above runnable: `torch` (CPU), `scipy`, `matplotlib`.
`requirements.txt` lists only `numpy` and `Pillow`.

## Head-to-head, one image set and one protocol

5 images (aerial, lena, couple, clock, man), differential = centre pixel R +1 mod 256,
key sensitivity = 1-bit flip of the transmitted `hb`. Averages over the 5 images.

| metric | DL (1-axis) | DL v2 (2-axis) | v4 (2-axis) | ideal |
|---|---|---|---|---|
| entropy, per-channel mean | 7.99721 | 7.99697 | 7.99724 | 8.0 |
| entropy, pooled RGB | 7.99908 | 7.99899 | 7.99908 | artifact, see note |
| mean \|corr\| (45 draws) | 0.00369 | 0.00307 | 0.00304 | 0.00317 = random floor |
| mean of per-image max \|corr\| | 0.00880 | 0.00793 | 0.00775 | 0 |
| worst \|corr\| (1 of 45) | 0.01138 | 0.01231 | 0.01067 | 0.01336 = random max of 1200 |
| max \|interchannel corr\| | 0.00970 | 0.00459 | 0.00406 | 0 |
| NPCR differential (%) | 99.6082 | 99.6261 | 99.6120 | 99.6094 |
| UACI differential (%) | 33.4338 | 33.4566 | 33.4402 | 33.4635 |
| **NPCR key-sens (%)** | **66.5120** | **99.6218** | **99.6135** | 99.6094 |
| **UACI key-sens (%)** | **22.2820** | **33.6385** | **33.4644** | 33.4635 |
| cipher chi2 min p | 0.2520 | 0.0583 | 0.2805 | > 0.01 |
| encrypt (ms) | 344 | 712 | 2967 | lower |
| decrypt (ms) | 369 | 701 | 2754 | lower |
| round-trip bit-exact | 5/5 | 5/5 | 5/5 | 5/5 |

v4 additionally survives the stricter test its own script uses — perturbing the
Lorenz `x0` by 1e-14 with `hb` held fixed: NPCR 99.6135%, UACI 33.4883%.

**Note on pooled entropy.** The ~7.9991 figure pools 3 channels into one
196,608-sample histogram, which lowers the estimator's finite-sample bias. It is
not a security gain over the ~7.997 per-channel value, and a grayscale
implementation cannot reproduce it. Compare per-channel values against Jha et
al.'s 7.9973, not the pooled one. Likewise correlation is sampled 45 times here
(5 images x 3 channels x 3 directions), so a maximum over those draws is an
order statistic, not a property of the cipher; the paper reports single
grayscale draws, so the two are not comparable.

## Finding 1 — the 1-axis DL cipher has a broken key schedule

Flipping one bit of the transmitted `hb` changes only 66.5% of ciphertext bytes,
with UACI 22.3% against an ideal 33.46%. Per channel:

```
R: NPCR 99.498%   G: NPCR 50.000%   B: NPCR 50.000%
```

Cause: both `NKG.generate_channel_keystreams` and v3's `ics_from_hash_bytes`
derive each channel's key from a **disjoint 7-byte slice** of `hb`
(`hb[0:7]`, `hb[7:14]`, `hb[14:21]`). A 1-bit change in `hb[0]` therefore leaves
the G and B keystreams **bit-identical**:

```
NKG keystream change:   R 5.08%    G 0.000%   B 0.000%
v4 pass-1 chaotic:      R 99.625%  G 0.000%   B 0.000%
v4 pass-2 (SHA-256(hb)) R 99.609%  G 99.596%  B 99.652%
```

The joint bidirectional XOR chain then only half-mixes into the untouched
channels, giving exactly 50% there and 66.5% overall.

This is **architectural, not neural**. Running the v3 1-axis chaotic cipher
under the identical test gives NPCR 66.529% / UACI 27.936% — the same defect.
What fixes it in both 2-axis models is the second pass keying off
`SHA-256(hb)`, which avalanches across all three channels. So the headline
benefit of the two-axis design is key sensitivity, not the correlation
improvement its docstrings argue for. The correlation gain is real but small,
and mostly interchannel (max |r_ic| 0.0097 -> 0.0046); on adjacent-pixel
correlation all three variants are at the noise floor (Finding 3).

Note the key-sensitivity numbers recorded in `results/metrics_dl.json`
(npcr_key 99.46–99.83, uaci_key 30.0–37.2) do not measure this. That test in
`evaluate_dl.py:232-252` bypasses the cipher: it XOR-diffuses the **raw
plaintext** with two `np.roll`-derived keystreams the cipher never uses, skipping
the scramble, block shuffle and interchannel shuffle entirely. Its scatter
(30%–37% UACI) is the signature of a statistic computed on something other than
the cipher.

## Finding 2 — the neural keystream is not random, and barely avalanches

Already recorded in `results/nist_report.txt`, and reproduced exactly:

| keystream | NIST SP 800-22 (8-test subset) |
|---|---|
| Lorenz/Rössler (v4) | **8/8 pass** |
| NKG (`nkg_gan.pt`) | **2/8 pass** — Frequency, Runs, Longest Run, Serial, Approx. Entropy and DFT all at p ≈ 0 |
| `np.random` control | 8/8 pass |

Confirmed independently by chi-square on 196,608 keystream bytes: NKG p = 0.0000
(entropy 7.99352, mean byte 123.86) versus chaotic p = 0.7067 (entropy 7.99911,
mean 127.55). The NKG was trained for uniform byte histograms and does not
achieve one.

Avalanche, averaged over all 56 single-bit flips of the 7 hash bytes feeding
channel 0:

| generator | bytes changed | bits changed | mean \|Δ\| |
|---|---|---|---|
| NKG (learned MLP) | 45.58% | 13.32% | 1.51 |
| Lorenz/Rössler | 99.61% | 49.99% | 85.26 |
| ideal | 99.61% | 50% | ~85 |

`mean |Δ| = 1.5` is the important number: the bytes that do change mostly move
by ±1. An MLP is a smooth function, so nearby inputs give nearby outputs, and
only the `round(ks * 255)` quantisation supplies any nonlinearity. Knowing the
keystream for one `hb` gives it to within a couple of least-significant-bit
steps for every neighbouring `hb`.

The ciphertext still looks uniform (chi2 p = 0.25, entropy 7.997) because the
Latin-square scramble, block shuffle and bidirectional diffusion repair the
keystream's defects downstream — not because the keystream is good.

## Finding 3 — the correlation targets are below the noise floor

For a 256×256 image there are 255×256 = 65,280 adjacent pairs, so the standard
error of an adjacent-pixel correlation is 1/√65280 = 0.00391. Measured over
1,200 adjacent-pixel correlations of genuinely uniform random 256×256 images
(`numpy` PCG64, seed 0):

| statistic | true random | DL (1-axis) | DL v2 | v4 |
|---|---|---|---|---|
| mean \|r\| | **0.00317** | 0.00369 | 0.00307 | 0.00304 |
| 95th percentile \|r\| | 0.00789 | — | — | — |
| worst \|r\| | 0.01336 (1200 draws) | 0.01138 (45) | 0.01231 (45) | 0.01067 (45) |

Both 2-axis models sit *at* the mean for true randomness, so their correlation
is statistically indistinguishable from perfect and the 0.00307 vs 0.00304 gap
is meaningless. Only the 1-axis model is measurably above the floor, and only
slightly. Its interchannel correlation is the one place the gap is real
(0.00970 vs 0.00459/0.00406 — roughly halved).

Two consequences worth acting on:

- The target quoted in `lsdpak_v4.py`'s docstring — "|corr_h|, |corr_v|,
  |corr_d| < 0.002 on all images" — is unreachable. A single draw from a
  perfect cipher falls under 0.002 only 39% of the time, so all nine
  (3 channels × 3 directions) doing so has probability 0.02%. Chasing it means
  tuning toward sampling noise.
- Jha et al.'s reported 0.0012 / 0.0008 / 0.0015 are each individually
  plausible (24% / 16% / 30% under the null), but as a consistent across-image
  result they sit well below what a correct cipher produces. That is worth a
  line in `manuscript_comparison.tex` — not as a claim of error on their part,
  but because the right reading is "both are at the noise floor", not "theirs
  is 4× better".

## Finding 4 — reproducibility

| file | reproduces? |
|---|---|
| `lsdpak-v4-2d-diffusion/results/metrics_v4.json` | yes, bit-identical on all 4 loadable images |
| `lsdpak-dl/results/metrics_v3.json` | yes, bit-identical |
| `lsdpak-dl/results/metrics_dl.json` | yes on all cipher metrics (see below for the denoiser block) |
| `lsdpak-dl/results/metrics_dl_v2_2axis.json` | **no** |

`metrics_dl_v2_2axis.json` cannot be regenerated from the current
`evaluate_dl_v2.py` (aerial NPCR_diff 99.60225 now vs 99.61700 recorded;
entropy R 7.996257 vs 7.996949). It is not a torch-version effect — the 1-axis
DL path, which uses the same NKG forward pass, reproduces bit-exactly. It is
also not the "reuse the transposed pass-1 key" variant described in
`lsdpak_v4.py`'s docstring; I implemented that and it gives 99.61599, also not a
match. The file is from some third, now-lost revision and should be regenerated
before it is cited.

Two other stale artifacts:

- The denoiser block in the old `metrics_dl.json` had `psnr_with == psnr_without`
  to all 15 digits on every image, which happens when `denoiser.pt` is missing
  and `decrypt_dl` is called with `denoiser=None`. With the checkpoint present
  the denoiser gives a real +3.3 dB at 5% salt-and-pepper. `denoiser_eval.py`
  confirms gains across the board (+0.18 dB at 1% S&P up to +3.38 dB on a 1/4
  crop), so the denoiser works; the recorded numbers just didn't exercise it.
- `add_salt_pepper` and `generate_extra_images.make_noise_img` both use unseeded
  `np.random`, so every denoiser figure shifts run to run.

## Verdict

**LS-DPAK v4** is the only one of the three with no measured defect: ideal
differential and key-sensitivity behaviour under both the 1-bit `hb` flip and
the 1e-14 initial-condition perturbation, the best correlation figures, and a
keystream that passes 8/8 NIST. It costs 2967 ms/image, 8.6× the 1-axis DL
model and 4.2× DL v2.

**LS-DPAK-DL v2** reaches the same differential and key-sensitivity targets at
712 ms — 4.2× faster than v4 — and its correlation figures essentially match
(0.00307 vs 0.00304 mean, 0.00793 vs 0.00775 max). But it reaches them *despite*
its keystream, not because of it: that keystream fails 6 of 8 NIST tests and has
13% bit-avalanche. The security argument rests entirely on the permutation and
diffusion layers masking a weak generator, which is not a claim a paper should
make without addressing Finding 2.

**LS-DPAK-DL (1-axis)** should not be reported as a security result. Its key
sensitivity is 66.5%/22.3%, and the 99.46–99.83% currently in
`metrics_dl.json` comes from a test that does not run the cipher.

## Suggested next steps

1. Restore `Baboon.tiff`, or drop `baboon` from `v3.IMAGE_NAMES`. Six of the ten
   names in `evaluate_dl.py:70` and two of the six in `evaluate_dl_v2.py:53` are
   not keys of `TIFF_SRC` at all, so they can never load — `evaluate_dl.py`
   hides this behind `load_image_safe`, the other two crash.
   `generate_extra_images.py`'s `OUT_DIR` also points four levels above the
   repo, so it has never been writing to `test_images/`.
2. Replace the key-sensitivity test in `evaluate_dl.py:232-252` with one that
   runs the real pipeline, and add one to `evaluate_dl_v2.py`.
3. Fix the per-channel key derivation to avalanche — e.g. derive each channel's
   key from `SHA-256(hb || channel_index)` rather than a disjoint 7-byte slice.
   This fixes 1-axis key sensitivity at its root and removes the 2-axis models'
   reliance on pass 2 for it.
4. Regenerate `metrics_dl_v2_2axis.json`, and seed the RNG in `add_salt_pepper`.
5. Pin `torch`, `scipy` and `matplotlib` in `requirements.txt`.
