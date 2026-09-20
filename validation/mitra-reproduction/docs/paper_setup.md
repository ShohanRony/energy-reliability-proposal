# Extracted experimental specification

Source: [Mitra et al., sections 3–5](https://arxiv.org/html/2405.20876v1).
Labels below separate stated facts from reproduction decisions.

| Item | Paper specification |
|---|---|
| Weight pruning | VGG-19; PreResNet-110; global L1 magnitude; convolution weights only |
| Filter pruning | VGG-16, layers 1 and 8–13; ResNet-110, first convolution of each first-stage residual block |
| Channel pruning | VGG-19; ResNet-164; network slimming; global threshold on BN scaling factors |
| Ratios | 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70% |
| Clean data | CIFAR-10; 50,000 training; 10,000 test; 32x32; ten classes |
| Corrupted data | CIFAR-10-C; 15 corruptions; all five severities |
| Baseline | SGD; batch 64; 160 epochs; LR 0.1 / 0.01 / 0.001 at stated boundaries 80/120 |
| Augmentation | Random crop and horizontal flip |
| Recovery | 40 epochs; constant LR 0.001; clean training data |
| ECE | Aggregate top-label; ten equal-mass bins in section 4.4; equation instead defines equal-width bins |
| Robustness | Mean accuracy over corruptions, separately per severity; not normalized mCE |
| Repetitions | Three; mean and standard deviation |

## How this implementation resolves the ECE conflict

The primary implementation sorts confidence values and divides the observations
into ten nearly equally populated groups. Stable sorting determines the treatment
of ties. The secondary implementation uses ten right-closed intervals of width 0.1.
Both calculate the size-weighted absolute confidence–accuracy gap. Neither
conditions on true class, and no per-class calibration reproduction is claimed.
These are our explicit operational definitions; author code is needed to resolve
which estimator actually produced the figures.

## Reference results

Exact per-configuration means/standard deviations are not supplied in the accessible
text tables. The result curves are Figure 1 (clean accuracy/ECE), Figure 2
(corruption accuracy) and Figure 3 (corruption ECE). The target is the VGG weight-
pruning panel at 0% and 30%. Numeric digitization was not possible here: screenshot
calls produced no usable image payload. No guessed points are entered in the CSV.

Textual anchors, not substitutes for configuration-specific numbers: section 5
describes weight-pruned ECE below baseline across the tested ratios; the broad
accuracy discussion contrasts clean accuracy above 90% with severe-corruption
accuracy below 55%; corruption ECE can increase by more than 400%.

Consequently no numerical agreement bound can yet be applied. Obtain author arrays
or digitize the relevant curves, retain figure/panel coordinates and digitization
uncertainty, and label those values approximate. The exact reference is not zero.
