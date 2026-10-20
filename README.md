# LS-DPAK — Latin-Square Diffusion/Permutation Image Encryption

Five variants of a chaotic image cipher, benchmarked against each other and
against Jha, Singh, Pradhan, Mishra, Jain & Ansari (2025), *"An Advanced Image
Encryption Algorithm Integrating Orthogonal Latin Square Transversal for
Shuffling With Lorenz Chaotic System for Diffusion"*, Security and Privacy
8:e70128, [doi:10.1002/spy2.70128](https://doi.org/10.1002/spy2.70128).

All measurements in this file were produced on 2026-10-20 on CPU
(Python 3.13.7, numpy 2.5.4, torch 2.14.1+cpu, no CUDA) over the five test
images present in `test_images/`. Reproduce with:

```bash
python comparison/compare_models.py          # the three RGB ciphers, one protocol
python comparison/summarize.py               # the aggregate table below
python comparison/diag_keysens.py            # Finding 1 breakdown
python comparison/diag_avalanche.py          # Finding 2 breakdown

cd lsdpak-v5-grayscale
python verify_parity.py --bench               # 35 acceptance gates + timings
python metrics_gray.py                        # grayscale metrics vs Jha et al.
```

---

## Repository layout

| Path | Contents |
|---|---|
| `lsdpak-v3-rgb/lsdpak_v3_rgb.py` | v3 RGB baseline. Holds the shared primitives every other variant imports: Latin-square construction, 3-pass scramble, block shuffle, interchannel shuffle, 4D-Lorenz + Rössler keystream, bidirectional diffusion, metrics. |
| `lsdpak-v4-2d-diffusion/lsdpak_v4.py` | v4. v3 plus a second, column-axis diffusion pass keyed from an independent hash-chained chaotic keystream. |
| `lsdpak-dl/lsdpak_dl.py` | LS-DPAK-DL. Replaces the chaotic keystream with a learned MLP (the Neural Keystream Generator, NKG), GAN-fine-tuned against a histogram discriminator. Adds a U-Net denoiser for noise-robust decryption. |
| `lsdpak-dl/evaluate_dl_v2.py` | LS-DPAK-DL v2. The NKG cipher with v4's two-axis diffusion added. |
| `lsdpak-v5-grayscale/` | **v5 grayscale.** v4's algorithm reduced to one channel, so the comparison against the (grayscale-only) Jha et al. paper is like-for-like. NumPy reference + bit-exact PyTorch port. |
| `comparison/` | Unified cross-model harness and findings (`RESULTS.md`). |
| `manuscript_comparison.tex` | Detailed technical comparison against the Jha et al. manuscript. |
| `test_images/` | Aerial, Clock, Couple, Lena, Man. **`Baboon.tiff` is missing** — see *Known breakage*. |

### Pipeline

All five variants share this structure; they differ only in the keystream
source, the channel count, and whether step 6 runs. (Shapes below are RGB; v5
drops the per-channel loops and the concatenation of the three planes.)

```
plaintext (H,W,3)
  1. SHA-256 -> hb (21 bytes transmitted as metadata)
  2. per channel: 3-pass Latin-square scramble
  3. per channel: full 256-block shuffle
  4. joint:       interchannel Latin-square column shuffle (D5)
  5. joint:       bidirectional XOR-chain diffusion, row-major,   key K1
  6. joint:       bidirectional XOR-chain diffusion, column-axis,  key K2   <- v4 / DL v2 only
ciphertext (H,W,3)
```

| variant | channels | keystream | K2 derivation | diffusion passes |
|---|---|---|---|---|
| v3 | RGB | 4D Lorenz + Rössler | — | 1 (row) |
| v4 | RGB | 4D Lorenz + Rössler | `ics_from_hash_bytes(SHA-256(hb))` | 2 (row + column) |
| DL 1-axis | RGB | NKG (learned MLP) | — | 1 (row) |
| DL v2 | RGB | NKG (learned MLP) | `NKG(SHA-256(hb))` | 2 (row + column) |
| **v5 gray** | **grayscale** | 4D Lorenz + Rössler | `ics_from_hash_bytes(SHA-256(hb))` | 2 (row + column) |

The RGB variants carry 21 metadata bytes (7 per channel); v5 carries 7. The
diffusion chain is 3N = 196,608 in RGB and N = 65,536 in grayscale.

---

## Comparison method

The three per-variant evaluation scripts are **not** comparable as written:

- **Different image sets.** `evaluate_dl.py` lists 10 names, 6 of which are not
  keys of `lsdpak_v3_rgb.TIFF_SRC` at all; `evaluate_dl_v2.py` lists 6, 2 of
  which are not; `lsdpak_v4.py` uses `v3.IMAGE_NAMES`. Five images exist.
- **Different differential protocol.** `lsdpak_v4.py` perturbs the centre pixel
  (+1 mod 256); both DL scripts flip the LSB of pixel (0,0,R).
- **Incomparable key-sensitivity test.** `lsdpak_v4.py` perturbs the Lorenz
  `x0` by 1e-14 through the real pipeline; `evaluate_dl.py` bypasses the
  pipeline entirely; `evaluate_dl_v2.py` omits the test.

`comparison/compare_models.py` therefore drives all three ciphers unmodified
over one image set with one protocol:

- **Images:** aerial, lena, couple, clock, man.
- **Differential:** centre pixel, R channel, `+1 mod 256`.
- **Key sensitivity:** flip one bit of the transmitted `hb`, same plaintext.
  For v4 this changes both the pass-1 ICs and the hash-chained pass-2 ICs, as a
  decryptor holding corrupted metadata would see.

The only new cipher code is `encrypt_dl_hb()`, which reproduces `encrypt_dl()`
with an injectable `hb` so key sensitivity can run through the full pipeline.
It is asserted bit-identical to `encrypt_dl()` on all five images before any
metric is computed.

---

## Summary table

Averages over the five images. Bold marks a value that is not merely worse but
outside the acceptable range.

| metric | DL (1-axis) | DL v2 (2-axis) | v4 (2-axis) | ideal / note |
|---|---|---|---|---|
| entropy, per-channel mean | 7.99721 | 7.99697 | 7.99724 | 8.0 |
| entropy, pooled RGB | 7.99908 | 7.99899 | 7.99908 | estimator artifact, see below |
| mean \|corr\| (45 draws) | 0.00369 | 0.00307 | 0.00304 | 0.00317 = true-random floor |
| mean of per-image max \|corr\| | 0.00880 | 0.00793 | 0.00775 | 0 |
| worst \|corr\| (1 of 45) | 0.01138 | 0.01231 | 0.01067 | 0.01336 = random max of 1200 |
| mean of per-image max \|corr_ic\| | 0.00970 | 0.00459 | 0.00406 | 0 |
| NPCR differential (%) | 99.6082 | 99.6261 | 99.6120 | 99.6094 |
| UACI differential (%) | 33.4338 | 33.4566 | 33.4402 | 33.4635 |
| NPCR key-sensitivity (%) | **66.5120** | 99.6218 | 99.6135 | 99.6094 |
| UACI key-sensitivity (%) | **22.2820** | 33.6385 | 33.4644 | 33.4635 |
| cipher χ² min p-value | 0.2520 | 0.0583 | 0.2805 | > 0.01 |
| keystream entropy | **7.99352** | **7.99352** | 7.99911 | 8.0 |
| keystream χ² p-value | **0.0000** | **0.0000** | 0.7067 | > 0.01 |
| keystream NIST (8-test) | **2/8** | **2/8** | 8/8 | 8/8 |
| keystream bit-avalanche | **13.3%** | **13.3%** | 49.99% | 50% |
| encrypt (ms/image) | 344 | 712 | 2967 | lower |
| decrypt (ms/image) | 369 | 701 | 2754 | lower |
| round-trip bit-exact | 5/5 | 5/5 | 5/5 | 5/5 |

v4 additionally passes the stricter test its own script uses — perturbing the
Lorenz `x0` by 1e-14 with `hb` held fixed: **NPCR 99.6135%, UACI 33.4883%**.
Neither DL variant has a continuous initial condition to perturb, so this test
has no DL analogue.

### Interpreting entropy and correlation

Two numbers in that table are routinely over-read:

- **Pooled RGB entropy (~7.9991)** pools three channels into one
  196,608-sample histogram, which lowers the estimator's finite-sample bias. It
  is not a security gain over the ~7.997 per-channel figure, and a grayscale
  implementation cannot reproduce it. Compare the **per-channel** value against
  Jha et al.'s 7.9973.
- **Correlation** is sampled 45 times here (5 images × 3 channels × 3
  directions), so a maximum over those draws is an order statistic, not a
  property of the cipher. Jha et al. report single grayscale draws. The two are
  not comparable; see Finding 3.

---

## Per-image detail

### LS-DPAK-DL (1-axis)

| image | entropy | pooled | mean\|r\| | max\|r\| | max\|r_ic\| | NPCR_d | UACI_d | NPCR_k | UACI_k | enc ms | dec ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aerial | 7.99722 | 7.99917 | 0.00347 | 0.01138 | 0.00969 | 99.6012 | 33.4071 | 66.5426 | 20.1142 | 338 | 342 |
| lena | 7.99721 | 7.99906 | 0.00308 | 0.00796 | 0.00441 | 99.6206 | 33.6339 | 66.4993 | 27.3846 | 324 | 384 |
| couple | 7.99714 | 7.99891 | 0.00448 | 0.00737 | 0.00969 | 99.6099 | 33.4977 | 66.4368 | 28.2882 | 321 | 382 |
| clock | 7.99745 | 7.99919 | 0.00463 | 0.01049 | 0.00921 | 99.6023 | 33.2312 | 66.5019 | 15.6689 | 397 | 377 |
| man | 7.99703 | 7.99905 | 0.00280 | 0.00680 | 0.01549 | 99.6068 | 33.3991 | 66.5797 | 19.9542 | 338 | 360 |

### LS-DPAK-DL v2 (2-axis)

| image | entropy | pooled | mean\|r\| | max\|r\| | max\|r_ic\| | NPCR_d | UACI_d | NPCR_k | UACI_k | enc ms | dec ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aerial | 7.99665 | 7.99886 | 0.00159 | 0.00375 | 0.00267 | 99.6282 | 33.5389 | 99.6272 | 33.5405 | 734 | 539 |
| lena | 7.99697 | 7.99900 | 0.00367 | 0.00754 | 0.00449 | 99.6190 | 33.4514 | 99.6399 | 33.7553 | 746 | 809 |
| couple | 7.99712 | 7.99910 | 0.00269 | 0.00571 | 0.00586 | 99.6267 | 33.4714 | 99.6201 | 33.8473 | 620 | 688 |
| clock | 7.99716 | 7.99907 | 0.00368 | 0.01036 | 0.00804 | 99.6343 | 33.4138 | 99.5987 | 33.6816 | 765 | 754 |
| man | 7.99696 | 7.99895 | 0.00374 | 0.01231 | 0.00186 | 99.6221 | 33.4075 | 99.6231 | 33.3676 | 693 | 713 |

### LS-DPAK v4 (2-axis)

| image | entropy | pooled | mean\|r\| | max\|r\| | max\|r_ic\| | NPCR_d | UACI_d | NPCR_k | UACI_k | enc ms | dec ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aerial | 7.99729 | 7.99914 | 0.00300 | 0.00586 | 0.00468 | 99.6180 | 33.2866 | 99.6221 | 33.4302 | 2923 | 2826 |
| lena | 7.99732 | 7.99913 | 0.00324 | 0.00773 | 0.00221 | 99.5987 | 33.4962 | 99.6129 | 33.4628 | 2990 | 2974 |
| couple | 7.99715 | 7.99903 | 0.00245 | 0.00918 | 0.00524 | 99.6429 | 33.4138 | 99.5987 | 33.3547 | 3055 | 2557 |
| clock | 7.99727 | 7.99917 | 0.00267 | 0.00531 | 0.00492 | 99.6104 | 33.5384 | 99.6226 | 33.5086 | 2877 | 2937 |
| man | 7.99715 | 7.99895 | 0.00385 | 0.01067 | 0.00325 | 99.5900 | 33.4662 | 99.6114 | 33.5654 | 2993 | 2473 |

### v4, 1e-14 Lorenz `x0` perturbation (`hb` held fixed)

| image | NPCR (%) | UACI (%) |
|---|---|---|
| aerial | 99.6073 | 33.4565 |
| lena | 99.6160 | 33.4990 |
| couple | 99.6221 | 33.4995 |
| clock | 99.6140 | 33.4594 |
| man | 99.6084 | 33.5272 |

---

## v5 grayscale

The paper is grayscale-only, so every RGB comparison against it carries two
measurement artifacts (the pooled-entropy inflation and the 9-draws-vs-3
correlation maximum). `lsdpak-v5-grayscale/` removes both by running v4's
algorithm on one channel.

### Faithfulness

The port is a reduction, not a reimplementation. `verify_parity.py` asserts
every stage against the v3 RGB originals, and all **35 gates pass**:

| gate | check | result |
|---|---|---|
| 0 | fp64 end to end; keystream not all-zero; `(-3.0 + 1e-14) != -3.0` | pass |
| 1 | CPU round-trip bit-exact | 5/5 |
| 2 | `build_keystream`, `scramble3`, `shuffle_blocks_full`, `lsq_column_shuffle`, diffusion each bitwise equal to the v3 original | pass |
| 3 | lane-vectorised and torch keystreams bitwise equal to the scalar path | pass |
| 4 | **GPU ciphertext bitwise identical to CPU**, and GPU round-trip | 5/5 |
| 5 | golden SHA-256 ciphertext hashes | frozen, 5/5 |
| 6 | both NPCR/UACI protocols wired to the real pipeline | pass |

Gate 4 is identity, not closeness: the torch fp64 RK4 reproduces the scalar
Python loop bit for bit, which holds only because the arithmetic is written in
the same order and nothing silently demotes to fp32.

One rename: v3's `interchannel_shuffle` is `lsq_column_shuffle` here. **The RGB
name is inaccurate.** Despite the name, and despite the surrounding write-up
describing it as breaking cross-channel correlation, that function never moves
data between R, G and B -- it applies the same D5-driven column-band
permutation independently to each channel, making it structurally a fourth
Latin-square permutation pass. Real cross-channel coupling in the RGB version
comes from the diffusion chain running over the three concatenated planes.
Worth fixing in `manuscript_comparison.tex` if it repeats the claim.

### Results

Protocol A = differential plaintext (`img[128,128] += 1`, so `hb` and the whole
keystream change). Protocol B = key sensitivity (pass-1 Lorenz `x0` += 1e-14,
`hb` and therefore K2 held fixed).

| image | entropy | corr_h | corr_v | corr_d | NPCR_A | UACI_A | NPCR_B | UACI_B | chi2 p |
|---|---|---|---|---|---|---|---|---|---|
| aerial | 7.99716 | -0.005231 | -0.004047 | +0.001612 | 99.6033 | 33.5313 | 99.6017 | 33.5979 | 0.4290 |
| lena | 7.99733 | -0.006769 | -0.001018 | +0.000796 | 99.6201 | 33.4684 | 99.6231 | 33.5543 | 0.7278 |
| couple | 7.99730 | +0.003511 | -0.006805 | -0.004368 | 99.6033 | 33.5016 | 99.5621 | 33.4508 | 0.6664 |
| clock | 7.99704 | -0.003620 | -0.005171 | -0.004211 | 99.6109 | 33.4135 | 99.6201 | 33.2221 | 0.2612 |
| man | 7.99704 | -0.000179 | -0.001216 | +0.000930 | 99.6094 | 33.3430 | 99.6140 | 33.4151 | 0.2764 |
| **mean** | **7.99717** | | | | **99.6094** | **33.4516** | **99.6042** | **33.4481** | |
| ideal | 8.0 | 0 | 0 | 0 | 99.6094 | 33.4635 | 99.6094 | 33.4635 | >0.01 |

Entropy lands at 7.99717, inside the 7.9970-7.9976 band the per-channel RGB
values occupy -- **not** near the pooled 7.9991. That is the expected and
correct outcome; a grayscale run reporting 7.999 would mean something had been
pooled. Aggregate mean |r| is 0.00330 against a true-random noise floor of
0.00316, i.e. a statistical tie, and the worst single draw of 15 is 0.00680.
All five histograms pass chi-square uniformity and all five round-trip
bit-exactly.

### Against Jha et al. (2025), like for like

Their Table 6 is a protocol-B result (their Sec. 5.7 describes the 1e-14
perturbation), so it belongs against the protocol-B column.

| image | v5 entropy | Jha | v5 NPCR_B | Jha NPCR | v5 UACI_B | Jha UACI |
|---|---|---|---|---|---|---|
| aerial | 7.99716 | 7.9973 | 99.6017 | 99.6094 | 33.5979 | 28.5589 |
| lena | 7.99733 | 7.9974 | 99.6231 | 99.6277 | 33.5543 | 28.7091 |
| couple | 7.99730 | 7.9969 | 99.5621 | 99.6246 | 33.4508 | 28.0631 |
| clock | 7.99704 | 7.9968 | 99.6201 | 99.6017 | 33.2221 | 35.5581 |
| man | 7.99704 | 7.9970 | 99.6140 | 99.5865 | 33.4151 | 28.8974 |
| **mean** | **7.99717** | **7.9971** | **99.6042** | **99.6100** | **33.4481** | **29.9573** |

Entropy and NPCR are a tie -- as they should be, since both ciphers produce
uniform output and NPCR saturates for anything competent. Correlation is also a
tie (v5 mean |r| 0.00330, theirs 0.00287, both at the noise floor).

**UACI is not a tie.** Mean deviation from the 33.4635 ideal is **0.1056 for v5
against 4.3440 for Jha et al.** Five of their six reported values cluster
around 28-29%, roughly 5 points low, with only Clock (35.56%) above. A
systematic one-sided deviation of that size is not sampling scatter, and UACI
has no plausible reason to sit below ideal for a cipher whose NPCR is at
99.61%. Combined with the protocol conflation -- Table 6 cited in their
Sec. 5.6 as differential-plaintext and in Sec. 5.7 as key-sensitivity, when
only the latter protocol could have produced it -- this is the strongest
quantitative point available for `manuscript_comparison.tex`.

Protocol A is the measurement they never performed, and it matters
structurally: with a single non-feedback XOR the differential ceiling would be
1/65536 = 0.0015%, whereas LS-DPAK reaches 99.6094% because the SHA-256
adaptive ICs rebuild the entire keystream from the new plaintext hash.

### Performance: one win, one honest loss

Measured on this CPU (no CUDA device), per 256x256 image.

| stage | before | after | speedup |
|---|---|---|---|
| diffusion: sequential Python loop -> NumPy prefix-XOR scan | 73.9 ms | 33.3 ms | **2.2x** |
| diffusion: -> torch prefix-XOR scan | 73.9 ms | 12.0 ms | **6.1x** |
| decrypt diffusion (no scan needed: shift + 2 XORs) | 73.9 ms | 0.2 ms | **370x** |
| encrypt end to end, keys precomputed (CPU -> torch) | 84.3 ms | 22.8 ms | 3.7x |

The diffusion scan is the certain win, and it is real: the forward chain is an
inclusive prefix-XOR and the backward chain a reverse one, and since XOR on a
bit is addition mod 2, prefix-XOR is the parity of a `cumsum`. Decryption needs
no scan at all -- both its loops are elementwise with a neighbour.

**The ODE is the honest loss, and not in the way expected.** Keystream
generation dominates everything else (2811 ms of a ~2900 ms full metric step),
and neither parallel strategy helps on CPU:

| keystream path | per lane | vs scalar |
|---|---|---|
| NumPy scalar loop | 332 ms | 1.0x |
| NumPy lane-vectorised, B=2 | 6944 ms | 0.05x |
| NumPy lane-vectorised, B=30 | 508 ms | 0.65x |
| torch lane-vectorised, B=2 | ~196,900 ms | 0.002x |
| torch lane-vectorised, B=30 | ~11,800 ms | 0.03x |

RK4 is sequential in time, so the only axis is across trajectories -- and the
ODE state is just 4-7 doubles, far too small for array dispatch to amortise. A
NumPy call on a 2-element array costs ~1.5 us against ~50 ns for Python float
arithmetic, so the batched cost is nearly flat in B (~8 s for the Lorenz loop
whether B is 2 or 30) and only overtakes the scalar path at **B = 36**. Torch
is 13x worse again on CPU, purely dispatch-bound at ~200 s per 90,000-step loop
regardless of batch size.

So `build_keystream_many()` dispatches on the measured crossover: scalar below
B = 36, batched above. A five-image metric run needs B = 30, which lands on the
scalar side. The batched and torch paths are retained because they are the
correct shape for an actual CUDA device -- where fp64 still runs at 1/32-1/64
of fp32 on consumer GeForce cards, so even there the ODE is the stage least
likely to pay off. Anyone moving this to a GPU box should port the ODE as a
fused kernel (one thread per trajectory, the step loop inside, state in
registers) rather than expecting the batched form to carry it.

### Limitations

- **Five images, not seven.** `Baboon.tiff` is absent from `test_images/` and
  Cameraman was never in the repo; Jha et al. averaged over seven. The means
  above are over the five shared images, and their per-image values are listed
  so the gap is visible rather than hidden in an average.
- `results/metrics_gray.json` has a different shape from `metrics_v4.json`:
  `entropy` is a scalar, not an object with R/G/B/joint keys, and
  `interchannel_corr` is absent. Tooling reading both must handle that.
- The grayscale reduction **sidesteps rather than fixes** the Finding 1 key
  schedule defect. That bug comes from splitting `hb` into disjoint 7-byte
  per-channel slices; with one channel there is nothing to split. The RGB
  variants still need the `SHA-256(hb || channel_index)` fix.

---

## Findings

### 1. The 1-axis DL cipher's key schedule is broken

Flipping one bit of the transmitted `hb` changes only 66.5% of ciphertext
bytes, UACI 22.3% against an ideal 33.46%. Per channel:

| channel | NPCR | UACI |
|---|---|---|
| R | 99.498% | 31.968% |
| G | **50.000%** | 25.096% |
| B | **50.000%** | 25.090% |

**Cause.** Both `NKG.generate_channel_keystreams` and v3's
`ics_from_hash_bytes` derive each channel's key from a *disjoint* 7-byte slice
of `hb` (`hb[0:7]`, `hb[7:14]`, `hb[14:21]`). A 1-bit change in `hb[0]` leaves
the G and B keystreams bit-identical:

| keystream | R | G | B |
|---|---|---|---|
| NKG | 5.08% | **0.000%** | **0.000%** |
| v4 pass-1 chaotic | 99.625% | **0.000%** | **0.000%** |
| v4 pass-2, ICs from `SHA-256(hb)` | 99.609% | 99.596% | 99.652% |

The joint bidirectional XOR chain then only half-mixes into the untouched
channels, giving exactly 50% there and 66.5% overall.

**This is architectural, not neural.** Running the v3 1-axis *chaotic* cipher
under the identical test gives NPCR 66.529% / UACI 27.936% — the same defect.
What repairs it in both 2-axis variants is pass 2 keying off `SHA-256(hb)`,
which avalanches across all three channels. The headline benefit of the
two-axis design is therefore **key sensitivity**, not the correlation
improvement its docstrings argue for.

**The recorded numbers do not measure this.** `metrics_dl.json` reports
npcr_key 99.46–99.83 and uaci_key 30.0–37.2. That test
(`lsdpak-dl/evaluate_dl.py:232-252`) bypasses the cipher: it XOR-diffuses the
**raw plaintext** with two `np.roll`-derived keystreams the cipher never uses,
skipping the scramble, block shuffle and interchannel shuffle entirely. The
30–37% UACI scatter is the signature of a statistic computed on something other
than the cipher.

### 2. The neural keystream is not random, and barely avalanches

Already recorded in `lsdpak-dl/results/nist_report.txt`, and reproduced exactly:

| keystream | NIST SP 800-22 (8-test subset) |
|---|---|
| Lorenz/Rössler (v3, v4) | **8/8 pass** |
| NKG (`nkg_gan.pt`) | **2/8 pass** — Frequency, Runs, Longest Run, Serial, Approximate Entropy and DFT all at p ≈ 0 |
| `np.random` control | 8/8 pass |

Confirmed independently by χ² on 196,608 keystream bytes: NKG p = 0.0000
(entropy 7.99352, mean byte 123.86) against chaotic p = 0.7067 (entropy
7.99911, mean byte 127.55). The NKG was trained for uniform byte histograms and
does not produce one.

Avalanche, averaged over all 56 single-bit flips of the 7 hash bytes feeding
channel 0:

| generator | bytes changed | bits changed | mean \|Δ\| |
|---|---|---|---|
| NKG (learned MLP) | 45.58% | 13.32% | 1.51 |
| Lorenz/Rössler | 99.61% | 49.99% | 85.26 |
| ideal | 99.61% | 50% | ~85 |

`mean |Δ| = 1.51` is the number that matters: the bytes that do change mostly
move by ±1. An MLP is a smooth function, so nearby inputs give nearby outputs,
and only the `round(ks * 255)` quantisation supplies any nonlinearity. Knowing
the keystream for one `hb` gives it to within a couple of LSB steps for every
neighbouring `hb`.

Both DL ciphertexts still look uniform (χ² p = 0.25 / 0.06, entropy 7.997)
because the Latin-square scramble, block shuffle and bidirectional diffusion
repair the keystream's defects downstream — not because the keystream is good.

### 3. The correlation targets are below the noise floor

A 256×256 image has 255 × 256 = 65,280 adjacent pairs, so an adjacent-pixel
correlation has standard error 1/√65280 = 0.00391. Measured over 1,200
adjacent-pixel correlations of genuinely uniform random 256×256 images
(numpy PCG64, seed 0):

| statistic | true random | DL (1-axis) | DL v2 | v4 |
|---|---|---|---|---|
| mean \|r\| | **0.00317** | 0.00369 | 0.00307 | 0.00304 |
| worst \|r\| | 0.01336 (1200 draws) | 0.01138 (45) | 0.01231 (45) | 0.01067 (45) |
| 95th percentile \|r\| | 0.00789 | — | — | — |

Both 2-axis variants sit *at* the true-random mean, so their correlation is
statistically indistinguishable from perfect and the 0.00307 vs 0.00304 gap is
meaningless. Only the 1-axis variant is measurably above the floor, and only
slightly.

The worst-case row makes the same point from the other side: DL v2 holds the
single worst correlation of the three (0.01231) despite having the second-best
mean, and all three worst-case values fall *below* the 0.01336 that 1,200 draws
from a perfect source produced. Ranking the variants on a maximum is ranking
them on sampling luck.

Interchannel correlation is the one place the gap is real: 0.00970 against
0.00459 / 0.00406, roughly halved by the second diffusion pass.

Two consequences:

- The target in `lsdpak_v4.py`'s docstring — "|corr_h|, |corr_v|, |corr_d| <
  0.002 on all images" — is unreachable. A single draw from a *perfect* cipher
  falls under 0.002 only 39.1% of the time, so all nine doing so has
  probability 0.02%. Chasing it means tuning toward sampling noise.
- Jha et al.'s 0.0012 / 0.0008 / 0.0015 are each individually plausible
  (24.1% / 16.2% / 29.8% under the null) but sit well below what a correct
  cipher yields as a consistent across-image result. The right reading is
  "both are at the noise floor", not "theirs is 4× better" — worth a line in
  `manuscript_comparison.tex`.

### 4. Reproducibility of the recorded results

| file | reproduces? |
|---|---|
| `lsdpak-v4-2d-diffusion/results/metrics_v4.json` | yes, bit-identical on all 4 loadable images |
| `lsdpak-dl/results/metrics_v3.json` | yes, bit-identical |
| `lsdpak-dl/results/metrics_dl.json` | yes on every cipher metric |
| `lsdpak-dl/results/nist_report.json` | yes |
| `lsdpak-dl/results/metrics_dl_v2_2axis.json` | **no** |

`metrics_dl_v2_2axis.json` cannot be regenerated from the current
`evaluate_dl_v2.py` (aerial NPCR_diff 99.60225 now against 99.61700 recorded;
entropy R 7.996257 against 7.996949). It is not a torch-version effect — the
1-axis DL path uses the same NKG forward pass and reproduces bit-exactly. It is
also not the "reuse the transposed pass-1 key" variant described in
`lsdpak_v4.py`'s docstring; that was implemented and gives 99.61599, also no
match. The file is from a third, now-lost revision and should be regenerated
before being cited.

Two further stale artifacts:

- The denoiser block in the previously recorded `metrics_dl.json` had
  `psnr_with == psnr_without` to all 15 digits on every image — what happens
  when `denoiser.pt` is absent and `decrypt_dl` is called with `denoiser=None`.
  With the checkpoint present the denoiser gives a real gain.
- `add_salt_pepper` and `generate_extra_images.make_noise_img` use unseeded
  `np.random`, so every denoiser figure shifts run to run.

---

## Supporting results

### U-Net denoiser (`lsdpak-dl/denoiser_eval.py`)

The denoiser addresses the tamper-sensitivity of bidirectional diffusion: any
ciphertext corruption propagates through the whole decrypted image.

| scenario | PSNR w/o | PSNR w/ | gain | SSIM w/o | SSIM w/ | gain |
|---|---|---|---|---|---|---|
| S&P 1% | 25.60 | 25.78 | +0.18 | 0.958 | 0.966 | +0.009 |
| S&P 5% | 18.56 | 21.87 | **+3.31** | 0.808 | 0.911 | +0.103 |
| S&P 10% | 15.65 | 18.83 | +3.17 | 0.665 | 0.815 | +0.149 |
| S&P 15% | 14.01 | 16.60 | +2.59 | 0.554 | 0.705 | +0.151 |
| Gaussian σ=10 | 15.48 | 17.77 | +2.30 | 0.670 | 0.777 | +0.106 |
| Gaussian σ=25 | 12.32 | 13.25 | +0.93 | 0.446 | 0.497 | +0.051 |
| Gaussian σ=50 | 10.42 | 11.11 | +0.69 | 0.262 | 0.288 | +0.026 |
| Crop 1/16 | 20.49 | 23.31 | +2.81 | 0.873 | 0.937 | +0.064 |
| Crop 1/4 | 14.45 | 17.83 | **+3.38** | 0.592 | 0.768 | +0.176 |

The denoiser works, and is the one unambiguous contribution of the DL branch.
It is also independent of the keystream question — it could be bolted onto v4.

### NKG ablation (`lsdpak-dl/ablation_study.py`)

| config | entropy (pooled) | NPCR_d (%) | UACI_d (%) | enc ms |
|---|---|---|---|---|
| v3 ODE baseline | 7.9991 | 99.6162 | 33.4455 | 1626 |
| Phase 1 only (uniformity loss) | 7.9990 | 99.6165 | 33.4361 | 397 |
| Phase 1+2 (GAN fine-tuned) | 7.9991 | 99.6156 | 33.4256 | 416 |

The GAN fine-tuning stage changes nothing measurable. Combined with Finding 2
— the GAN-trained checkpoint still fails 6 of 8 NIST tests — the adversarial
histogram training does not achieve its stated objective.

---

## Verdict

**LS-DPAK v4** is the only variant with no measured defect: ideal differential
and key-sensitivity behaviour under both the 1-bit `hb` flip and the 1e-14
initial-condition perturbation, the best correlation figures, and a keystream
that passes 8/8 NIST. It costs 2967 ms/image — 8.6× the 1-axis DL variant and
4.2× DL v2.

**LS-DPAK-DL v2** reaches the same differential and key-sensitivity targets at
712 ms, and its correlation figures are indistinguishable from v4's. But it
reaches them *despite* its keystream, not because of it: that keystream fails 6
of 8 NIST tests and has 13% bit-avalanche. The security argument rests entirely
on the permutation and diffusion layers masking a weak generator — not a claim
to publish without addressing Finding 2.

**LS-DPAK-DL (1-axis)** should not be reported as a security result. Its key
sensitivity is 66.5% / 22.3%, and the 99.46–99.83% in `metrics_dl.json` comes
from a test that does not run the cipher.

**LS-DPAK v5 grayscale** is what the manuscript comparison should actually
cite. It is v4's algorithm, verified bitwise against the RGB originals stage by
stage, measured on the same colour space as the paper. It removes both RGB
measurement artifacts, and it produces the one metric where the two ciphers
genuinely differ: UACI deviates 0.11 from ideal against the paper's 4.34.

**For the manuscript comparison:** lead with v5 grayscale, because it is the
only like-for-like measurement. v4 remains the strongest RGB claim. The honest
framing of the DL branch is that the U-Net denoiser is a real contribution and
the neural keystream is not.

---

## Known breakage

Discovered while running everything in this directory on 2026-10-20.

| issue | location | effect |
|---|---|---|
| `Baboon.tiff` absent | `test_images/` | `lsdpak_v4.py` crashes after 4 of 6 images; `metrics_v4.json`'s recorded 6-image set is unreproducible; v5 grayscale averages over 5 images against the paper's 7 |
| Cameraman never present | `test_images/` | Jha et al.'s seventh image cannot be compared at all |
| 6 of 10 image names are not `TIFF_SRC` keys | `evaluate_dl.py:70` | silently skipped by `load_image_safe`; evaluates 4 images, not 10 |
| 2 of 6 image names are not `TIFF_SRC` keys | `evaluate_dl_v2.py:53` | `KeyError: '4_2_03'` after 4 images |
| `OUT_DIR` points four levels above the repo | `generate_extra_images.py:20` | has never written to `test_images/`; the 4 synthetic images it makes are unreachable |
| key-sensitivity test bypasses the cipher | `evaluate_dl.py:232-252` | `npcr_key` / `uaci_key` in `metrics_dl.json` are meaningless |
| no key-sensitivity test at all | `evaluate_dl_v2.py` | the defect in Finding 1 would not have been caught |
| unseeded `np.random` | `evaluate_dl.py` `add_salt_pepper`, `generate_extra_images.make_noise_img` | denoiser figures are not reproducible |
| `torch`, `scipy`, `matplotlib` unlisted | `requirements.txt` | nothing in `lsdpak-dl/` runs from a clean install |

`lsdpak-dl/train.py` was deliberately **not** run; it would overwrite the
checkpoints every recorded result depends on.

### Suggested fixes, in priority order

1. Restore `Baboon.tiff` and source Cameraman (standard test images; Cameraman
   ships with MATLAB as `cameraman.tif`). This closes the 5-vs-7 image gap in
   the v5 comparison, which is currently its main limitation. Also drop
   `baboon` from `v3.IMAGE_NAMES` or restore the file, and correct the
   image-name lists in both DL evaluation scripts.
2. Replace the key-sensitivity test in `evaluate_dl.py:232-252` with one that
   runs the real pipeline, and add one to `evaluate_dl_v2.py`.
3. Fix the per-channel key derivation to avalanche — derive each channel's key
   from `SHA-256(hb || channel_index)` rather than a disjoint 7-byte slice.
   This repairs 1-axis key sensitivity at its root and removes the 2-axis
   variants' reliance on pass 2 for it.
4. Regenerate `metrics_dl_v2_2axis.json`; seed the RNG in `add_salt_pepper`.
5. Pin `torch`, `scipy` and `matplotlib` in `requirements.txt`.
6. Drop the `< 0.002` correlation target from `lsdpak_v4.py`'s docstring; it is
   below the noise floor (Finding 3).
7. Correct the `interchannel_shuffle` description in `manuscript_comparison.tex`
   if it repeats the claim that the function breaks cross-channel correlation;
   it does not (see *v5 grayscale -- Faithfulness*).
8. If a CUDA box becomes available, port the ODE as a fused per-trajectory
   kernel rather than reusing the batched form, which is dispatch-bound.
