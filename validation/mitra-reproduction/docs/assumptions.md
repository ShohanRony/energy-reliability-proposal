# Unspecified choices and explicit reconstruction decisions

These gaps do not prove the work is irreproducible. Some may be resolved by the
referenced upstream code, but the authors' exact revision/configuration was not
identified and the source files could not be retrieved in this environment.

| Gap or ambiguity | Runner decision | Consequence |
|---|---|---|
| ECE formula conflicts with experimental prose | Equal-mass primary; equal-width secondary | Match cannot identify author estimator by itself |
| Equal-mass confidence ties/remainder handling | Stable sorted chunks; split tied confidences | Can affect ECE; record bins |
| ECE under corruption: pooled predictions or mean of separate ECE values? | Report both separately at each severity | ECE is nonlinear, so these need not agree |
| Authors' code revision and actual checkpoints | Require explicit upstream commit; train anew | Independent reconstruction, not identical rerun |
| Exact CIFAR VGG graph in paper prose | Load referenced upstream vgg19_bn; require 16 conv layers | No ImageNet VGG substitution; graph recorded; compatibility untested |
| Momentum, weight decay, Nesterov | 0.9, 1e-4, false | Declared assumptions, not verified author choices |
| Normalization values | Mean .4914/.4822/.4465; std .2023/.1994/.2010 | Plausible CIFAR convention; must be checked against original configuration |
| Crop padding and flip probability | Padding 4; flip probability .5 | Paper states augmentation types but not parameters |
| Exact LR boundary indexing | One-based epochs 1–79 / 80–119 / 120–160 | Common zero-based implementations may differ by one epoch |
| Last versus best checkpoint, validation split | Last epoch; no test-based selection | May differ from upstream best-checkpoint convention |
| Weight threshold ties and integer rounding | PyTorch global L1 mask; round total*0.3 | Original threshold implementation might differ near cutoff |
| Fresh pruning per ratio or iterative pruning | One-shot 30% from baseline | Only this ratio implemented; no iterative sweep claimed |
| Recovery optimizer state and batch size | Reset optimizer; batch 64 | Not silently inferred from baseline statement |
| Random seeds and meaning of three repetitions | Seeds 1,2,3 with independently trained baselines | Original pairing/seed policy unknown |
| Software/CUDA versions and determinism | Proposed torch 2.5.1/vision .20.1; deterministic FP32 | Neural runtime not installed/tested here |
| Unpruned baseline gets equal extra training? | Main baseline ends at 160; optional extra-40 control | Observed improvement can partly reflect recovery training |
| Exact corruption filenames | Standard 15, not extra four | Dataset version and corruption list should be author-confirmed |
| Channel-pruning regularization strength and training details | Not implemented | Avoids an additional unresolved setting |
| Classwise ECE | Not reproduced; paper reports aggregate | A future per-class extension is a new analysis |

## Interpretation after a real run

Before diagnosing a discrepancy, establish the baseline graph, preprocessing and
accuracy. Then check checkpoint policy and recovery implementation. A calibration
mismatch without a large accuracy mismatch may arise from the ECE ambiguity,
confidence distribution, seed variation or training regularization. Corruption-only
mismatch suggests dataset slice, normalization, corruption list or ECE aggregation.
These are possible explanations, not findings from the blocked run.

Do not declare agreement merely because ECE decreases. That would reproduce a
directional observation, not the numeric curve. Run three seeds; report means and
standard deviations in the same units as the figure. Compare differences with a
prespecified practical tolerance, author variability and digitization uncertainty.
A confidence interval overlapping the paper's interval is not proof of equivalence.
No arbitrary tolerance is applied retroactively to obtain a passing outcome.
