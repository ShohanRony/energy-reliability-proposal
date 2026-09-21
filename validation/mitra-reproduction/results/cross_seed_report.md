# Cross-seed report — 3 seeds (1, 2, 3)

Generated: 2026-09-21. VGG-19-BN, CIFAR-10, 30% global L1 conv-weight pruning +
40-epoch recovery. Seeds 1, 2, 3 each independently trained from scratch (fresh
weight init, data order, and augmentation randomness per seed) — same
architecture, hyperparameters, and CIFAR-10 split throughout. Full clean +
CIFAR-10-C corruption eval run for all three. Source: `results/seed{1,2,3}/results.json`.

**This is the first point at which any claim about the direction of the
baseline-vs-pruned effect is warranted — seed 1 alone (see
[seed1/corruption_eval_report.md](seed1/corruption_eval_report.md)) was
explicitly insufficient.**

## Run status

| Seed | Status | Baseline clean acc | Pruned clean acc |
|---|---|---|---|
| 1 | `completed_reconstruction_clean_and_corruption` | 0.9375 | 0.9373 |
| 2 | `completed_reconstruction_clean_and_corruption` | 0.9380 | 0.9386 |
| 3 | `completed_reconstruction_clean_and_corruption` | 0.9361 | 0.9365 |

No `failure.json` in any of the three run directories.

## Clean accuracy / ECE — mean ± SD across 3 seeds

| Metric | Baseline | Pruned + recovered |
|---|---|---|
| Accuracy | 0.9372 ± 0.0010 | 0.9375 ± 0.0011 |
| ECE (equal-mass) | 0.0472 ± 0.0012 | 0.0480 ± 0.0007 |

Per-seed clean-accuracy delta (pruned − baseline): seed1 −0.0002, seed2
+0.0006, seed3 +0.0004 — **sign not consistent** (2 of 3 positive). Paired
t(2) = 1.11, far below the df=2, α=0.05 two-tailed critical value of 4.303.
No detectable clean-accuracy effect at this pruning level; differences are
within seed-to-seed noise.

Per-seed clean-ECE delta (pruned − baseline): seed1 +0.00078, seed2 +0.00120,
seed3 +0.00035 — **sign consistent (all positive: pruned less well
calibrated)**. Paired t(2) = 3.18 — directionally consistent but does not
reach the strict df=2 critical value of 4.303.

## Mean-per-corruption accuracy (mPC) by severity — mean ± SD across 3 seeds

| Severity | Baseline mPC | Pruned+recovered mPC | Per-seed deltas (pruned−baseline) | Signs | Consistent? | t(2) |
|---|---|---|---|---|---|---|
| 1 | 0.8678 ± 0.0022 | 0.8670 ± 0.0033 | −0.00031, 0.00000, −0.00211 | −, 0/+, − | No | −1.22 |
| 2 | 0.8146 ± 0.0045 | 0.8129 ± 0.0056 | −0.00193, +0.00094, −0.00405 | −, +, − | No | −1.16 |
| 3 | 0.7525 ± 0.0042 | 0.7496 ± 0.0060 | −0.00155, −0.00160, −0.00567 | −, −, − | **Yes** | −2.15 |
| 4 | 0.6812 ± 0.0046 | 0.6770 ± 0.0072 | −0.00111, −0.00500, −0.00651 | −, −, − | **Yes** | −2.61 |
| 5 | 0.5655 ± 0.0061 | 0.5607 ± 0.0090 | +0.00075, −0.00777, −0.00726 | +, −, − | No | −1.72 |

None of the per-severity paired t-statistics reach the df=2 critical value
(4.303). Sign consistency across all 3 seeds holds only at severities 3 and 4;
at severities 1, 2, and 5 at least one seed flips sign (severity 1's seed-2
delta is exactly 0). **No accuracy-degradation effect from this pruning level
survives 3-seed noise.**

## Pooled ECE (equal-mass) by severity — mean ± SD across 3 seeds

| Severity | Baseline pooled ECE | Pruned+recovered pooled ECE | Per-seed deltas (pruned−baseline) | Signs | Consistent? | t(2) |
|---|---|---|---|---|---|---|
| 1 | 0.10237 ± 0.00171 | 0.10442 ± 0.00299 | +0.00087, +0.00185, +0.00345 | +, +, + | **Yes** | 2.74 |
| 2 | 0.14570 ± 0.00391 | 0.14914 ± 0.00535 | +0.00275, +0.00192, +0.00566 | +, +, + | **Yes** | 3.04 |
| 3 | 0.19808 ± 0.00394 | 0.20296 ± 0.00680 | +0.00208, +0.00463, +0.00793 | +, +, + | **Yes** | 2.88 |
| 4 | 0.25832 ± 0.00497 | 0.26498 ± 0.00863 | +0.00200, +0.00862, +0.00936 | +, +, + | **Yes** | 2.85 |
| 5 | 0.35842 ± 0.00786 | 0.36700 ± 0.01100 | +0.00066, +0.01335, +0.01173 | +, +, + | **Yes** | 2.15 |

**Sign is consistent (pruned worse calibrated than baseline) across all 3
seeds at every single severity, and also in the clean condition — 6 out of 6
comparisons agree in direction.** No individual paired t-test crosses the
strict df=2 critical value of 4.303 (values range 2.15–3.04), so none is
"significant" on its own at n=3. But under a simple sign-test framing, 6/6
same-direction outcomes has probability (1/2)^6 ≈ 0.016 under a true null of
no systematic direction — i.e., the *direction* of the ECE effect (pruning
degrades calibration slightly) is the one finding in this study that survives
3-seed replication, even though no single severity's magnitude clears a
conventional per-comparison significance bar. This is suggestive, not proof:
3 seeds is the minimum, not a well-powered sample, and the individual t-tests
should not be treated as passing/failing in isolation from this sign pattern.

## Bottom line

- **Accuracy:** no evidence that 30% global L1 pruning + recovery changes
  either clean or corrupted accuracy at this seed count — deltas are small,
  inconsistent in sign at 3 of 5 corruption severities, and no paired t-test
  approaches significance.
- **Calibration (ECE):** weak but *directionally consistent* evidence that
  pruning degrades calibration — every one of the 6 conditions tested (clean
  + 5 severities) shows pruned ECE higher than baseline ECE, in all 3 seeds.
  Magnitudes are small (0.0004–0.013 absolute ECE) and individual severities
  don't clear a strict per-comparison significance threshold at n=3.
- Neither finding should be treated as final. Three seeds is the paper's
  stated minimum for a mean±SD design, not a well-powered replication; wider
  seed counts or non-parametric tests robust to n=3 would strengthen either claim.
