# Seed 1 — Corruption-robustness evaluation report

Generated: 2026-09-20. Independent reconstruction (VGG-19-BN, CIFAR-10, 30% global
L1 conv-weight pruning + 40-epoch recovery), seed 1. This report covers the
CIFAR-10-C corruption-robustness pass added on top of the already-completed
clean-only run; see [REPORT.md](../../REPORT.md) for the original (stale, blocked)
narrative and [docs/assumptions.md](../../docs/assumptions.md) for unresolved
paper ambiguities (ECE binning scheme, corruption-ECE aggregation).

## Run status

- `manifest.json` → `status: "completed_reconstruction_clean_and_corruption"`,
  `completed_utc: 2026-09-20T13:57:11Z`.
- `cifar_c_sha256` in the manifest records hashes for all 16 CIFAR-10-C files
  (15 corruption arrays + `labels.npy`) actually used.
- Both training phases (160-epoch baseline, 40-epoch pruned recovery) were
  already complete before this run; `--resume` re-loaded the final checkpoints
  with zero additional training epochs, then ran clean + corruption evaluation.
- 510 output files under `results/seed1/` (443M total): per-corruption/severity
  `.npz` predictions and `.json` metrics for each phase, plus `results.json`,
  `manifest.json`, `pruning.json`.

## Clean re-evaluation (sanity check)

Re-derived from the same deterministic checkpoints as the original clean-only
run, to confirm the resume path did not drift:

| Phase | Accuracy | ECE (equal-mass) |
|---|---|---|
| Baseline | 0.9375 | 0.0471 |
| Pruned30 + recovered | 0.9373 | 0.0479 |

Identical to the original clean-only run's numbers.

## Mean-per-corruption accuracy (mPC) by severity

Averaged over all 15 standard corruption types (`results.json`):

| Severity | Baseline mPC | Pruned+recovered mPC | Baseline pooled ECE (equal-mass) | Pruned+recovered pooled ECE (equal-mass) |
|---|---|---|---|---|
| 1 | 0.8696 | 0.8693 | 0.1011 | 0.1020 |
| 2 | 0.8196 | 0.8177 | 0.1415 | 0.1442 |
| 3 | 0.7573 | 0.7557 | 0.1937 | 0.1958 |
| 4 | 0.6851 | 0.6840 | 0.2539 | 0.2559 |
| 5 | 0.5683 | 0.5690 | 0.3564 | 0.3571 |

Accuracy degrades monotonically with severity for both checkpoints; ECE rises
monotonically. The 30%-pruned+recovered model tracks the baseline closely at
every severity (differences on the order of 0.1-0.2pp mPC), i.e. this level of
pruning does not measurably change corruption robustness relative to the
unpruned baseline on this reconstruction.

## Per-corruption-type accuracy

### Baseline

| corruption | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| gaussian_noise | 0.8146 | 0.6157 | 0.4000 | 0.3236 | 0.2650 |
| shot_noise | 0.8730 | 0.7915 | 0.5638 | 0.4732 | 0.3437 |
| impulse_noise | 0.8150 | 0.6957 | 0.5921 | 0.4138 | 0.2686 |
| defocus_blur | 0.9352 | 0.9194 | 0.8735 | 0.7835 | 0.5184 |
| glass_blur | 0.5497 | 0.5657 | 0.6152 | 0.4509 | 0.4856 |
| motion_blur | 0.8912 | 0.8293 | 0.7345 | 0.7355 | 0.6445 |
| zoom_blur | 0.8677 | 0.8413 | 0.7922 | 0.7392 | 0.6510 |
| snow | 0.8920 | 0.8103 | 0.8261 | 0.8053 | 0.7774 |
| frost | 0.9101 | 0.8659 | 0.7827 | 0.7603 | 0.6567 |
| fog | 0.9341 | 0.9185 | 0.8928 | 0.8518 | 0.7084 |
| brightness | 0.9373 | 0.9348 | 0.9295 | 0.9223 | 0.8991 |
| contrast | 0.9308 | 0.8828 | 0.8151 | 0.6512 | 0.1950 |
| elastic_transform | 0.8911 | 0.8908 | 0.8527 | 0.8022 | 0.7677 |
| pixelate | 0.9218 | 0.8915 | 0.8673 | 0.7597 | 0.5733 |
| jpeg_compression | 0.8801 | 0.8410 | 0.8217 | 0.8047 | 0.7696 |

### Pruned30 + recovered

| corruption | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| gaussian_noise | 0.8126 | 0.6102 | 0.3978 | 0.3202 | 0.2635 |
| shot_noise | 0.8691 | 0.7860 | 0.5606 | 0.4714 | 0.3448 |
| impulse_noise | 0.8102 | 0.6844 | 0.5800 | 0.4084 | 0.2603 |
| defocus_blur | 0.9360 | 0.9203 | 0.8765 | 0.7882 | 0.5258 |
| glass_blur | 0.5521 | 0.5668 | 0.6126 | 0.4533 | 0.4898 |
| motion_blur | 0.8944 | 0.8287 | 0.7349 | 0.7355 | 0.6455 |
| zoom_blur | 0.8709 | 0.8456 | 0.7974 | 0.7461 | 0.6529 |
| snow | 0.8919 | 0.8088 | 0.8264 | 0.8077 | 0.7773 |
| frost | 0.9082 | 0.8626 | 0.7806 | 0.7586 | 0.6588 |
| fog | 0.9335 | 0.9186 | 0.8915 | 0.8483 | 0.7102 |
| brightness | 0.9372 | 0.9342 | 0.9276 | 0.9210 | 0.8994 |
| contrast | 0.9311 | 0.8785 | 0.8088 | 0.6408 | 0.1986 |
| elastic_transform | 0.8928 | 0.8931 | 0.8563 | 0.8108 | 0.7735 |
| pixelate | 0.9201 | 0.8864 | 0.8605 | 0.7479 | 0.5660 |
| jpeg_compression | 0.8790 | 0.8411 | 0.8244 | 0.8024 | 0.7689 |

## Notable observations

- **`contrast` at severity 5** collapses to near chance level (0.195 baseline /
  0.199 pruned; chance = 0.10 for 10 classes) — by far the hardest single
  corruption at max severity for this model, in both checkpoints.
- **`glass_blur`** is non-monotonic in both phases (accuracy at s3 exceeds s2,
  which exceeds s4) — a known property of that corruption's severity scaling
  in the CIFAR-10-C benchmark, not an artifact of this harness.
- Per-corruption accuracy differences between baseline and pruned+recovered
  are small and inconsistent in sign (sometimes pruned is slightly higher,
  e.g. `defocus_blur`, `glass_blur`; sometimes lower, e.g. `gaussian_noise`,
  `contrast`) — consistent with noise rather than a systematic robustness
  effect from this pruning level.

## Environment / disk

- Free disk space immediately after this eval: 9,301,659,648 bytes ≈ 8.66 GiB
  (36G used / 47G, 81%) — effectively unchanged from the 8.72 GiB baseline
  before the eval (prediction/metrics files added only ~443MB).
- CIFAR-10-C source (`~/measurement_env/data/CIFAR-10-C/`, 2.92GB) is being
  kept on disk rather than deleted: it is a fixed, immutable public benchmark
  archive (re-downloadable, not something computed here), and is still needed
  for the seed-2/seed-3 runs required to match the paper's 3-seed mean±SD
  design, plus any `--recovery-control` extension runs.

## What this does and does not establish

This is still an **independent reconstruction, not the authors' released code**
(see [README.md](../../README.md)). It establishes real, measured corruption
accuracy and calibration numbers for one seed of this specific VGG-19-BN
reconstruction, on real CIFAR-10-C data. It does **not** establish numerical
agreement with the paper (Figures 2-3 numeric points remain undigitized/
unavailable, per [REPORT.md](../../REPORT.md)), and does not yet reflect
seed-to-seed variance (only seed 1 exists so far).
