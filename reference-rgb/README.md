# reference-rgb — not part of the contribution

These three files are the RGB predecessors of LS-DPAK v5. They are retained on
this branch for two narrow purposes only, and neither is a result being
claimed:

| file | why it is here |
|---|---|
| `lsdpak_v3_rgb.py` | **Verification oracle.** `verify_parity.py` gate 2 asserts that v5's keystream, `scramble3`, `shuffle_blocks_full`, `lsq_column_shuffle` and bidirectional diffusion are each *bitwise equal* to the originals in this file. That is what makes v5 a reduction of a published pipeline rather than a reimplementation of it. Delete this file and those 6 checks degrade to `[SKIP]`; the remaining 29 gates still pass. |
| `lsdpak_v4.py` | The two-axis RGB cipher v5 reduces to one channel. Kept so the reduction can be read side by side with its source. |
| `metrics_v4.json` | **Data provenance.** The RGB *per-channel* entropy values (7.99700–7.99763) that v5's 7.99717 is compared against come from this file. The pooled `entropy.joint` figure of ~7.9991 in it is an estimator artifact and is deliberately *not* used as a target — see the main README. |

Both modules are byte-for-byte unmodified. Do not edit them: `metrics_v4.json`
must stay reproducible from `lsdpak_v4.py`, and gate 2 is only meaningful if
the oracle is untouched.

`lsdpak_v4.py` expects `lsdpak_v3_rgb.py` one directory up in
`../lsdpak-v3-rgb/`, so it will not run from this layout. It is here to be
read, not executed; run it from the `main` branch, which preserves the original
directory structure along with the other variants (`lsdpak-dl`, the
cross-model `comparison/` harness, and their results).
