# Reproduction report

## Outcome

**Blocked execution; no numerical paper reproduction.** The full paper text was
read, a one-configuration experiment harness was implemented, metric arithmetic
was exercised, and the CNN command was invoked. The command failed at importing
PyTorch. The host exposes no NVIDIA GPU, and direct package/data/source access
returned HTTP 403. There is no accuracy, ECE or corruption result for a trained
model, and nothing was measured on the user's RTX 3050.

## Target and comparison

Selected: VGG-19-BN, CIFAR-10, 30% global convolution-weight pruning with recovery.
It is algorithmically simpler than structural pruning, but still requires the
full baseline/recovery training and a verified upstream architecture.

| Quantity | Published reference | This attempt | Agreement |
|---|---|---|---|
| Baseline clean accuracy | Figure 1 VGG weight-pruning panel; numeric point unavailable | Not measured | Not assessable |
| Baseline clean ECE | Figure 1; numeric point unavailable | Not measured | Not assessable |
| 30% recovered clean accuracy | Figure 1; numeric point unavailable | Not measured | Not assessable |
| 30% recovered clean ECE | Figure 1; numeric point unavailable | Not measured | Not assessable |
| Severity 1–5 mPC | Figure 2; numeric points unavailable | Not measured | Not assessable |
| Severity 1–5 ECE | Figure 3; numeric points unavailable | Not measured | Not assessable |

No mismatch has been observed; a comparison has not happened. The immediate cause
is the execution environment, not model convergence or inadequate laptop memory.
Exact numerical references were not guessed from inaccessible figures.

## What was validated

Nine NumPy metric tests passed. For a four-observation, two-bin synthetic fixture,
hand-calculated ECE was 0.275 for equal-mass bins and 0.125 for equal-width bins;
the code returned those values to floating-point precision. This verifies those
calculations on test fixtures, not calibration on CIFAR-10. The PyTorch mask test
was skipped. All raw inputs, bin-level outputs and the actual exception are committed.

## Most consequential reproduction gaps

The paper's ECE equation specifies equal-width intervals while its experimental
section specifies ten equal-mass bins. Recovery is required: an immediately
pruned checkpoint does not reproduce the reported procedure. Corruption robustness
uses mean accuracy, not normalized mCE. Exact author seeds, checkpoint selection,
software revision and corruption-ECE aggregation were not resolved. The harness
records both ECE schemes and both corruption aggregation choices; all other
assumptions are listed in docs/assumptions.md.

## Hardware assessment and next evidence

CIFAR-sized VGG training at batch 64 is a plausible project for the stated 6GB GPU,
but no runtime, peak-memory or completion estimate has been measured on it. Run the
included real-data preflight first; then complete one seed with the full training
schedule, followed by three independent seeds for the comparison. Inspect the
actual upstream implementation before accepting its settings. Obtain numeric
author results or properly digitized curves before applying an agreement bound.

This attempt supports only: a protocol audit, an explicit implementation plan,
tested metric calculations and a recorded environmental blocker. It does **not**
support claiming a validated experimental pipeline in a scholarship proposal.

## Publication status

The delivered folder is a committed local Git repository packaged as a ZIP.
It has not been published to GitHub; GitHub connection is required for that step.

## Resumed attempt

The 20 September 2026 recheck reproduced the same environmental blocker.
See [resumed status](docs/resumed_status.md) and `results/resumed-attempt/`
for the new raw checks and failed command. No trained-model result was produced.
