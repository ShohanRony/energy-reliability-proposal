# Mitra 2024 pruning and calibration reproduction attempt

**Status: execution blocked. No trained-model reproduction result is available.**
The CNN command was executed and failed because PyTorch is absent. The execution
environment has no exposed NVIDIA GPU; package, dataset and source downloads
returned HTTP 403. This is not evidence that an RTX 3050 cannot run the experiment.
Nine NumPy metric checks passed; one PyTorch pruning check was skipped.
Synthetic metric outputs are never used as paper comparison results.

Target: Pallavi Mitra, Gesina Schwalbe and Nadja Klein (2024),
*Investigating Calibration and Corruption Robustness of Post-hoc Pruned Perception
CNNs: An Image Classification Benchmark Study*, CVPR Workshops.
[arXiv full text](https://arxiv.org/html/2405.20876v1)
and [PDF](https://arxiv.org/pdf/2405.20876).

Read [REPORT.md](REPORT.md), [paper setup](docs/paper_setup.md),
[explicit assumptions](docs/assumptions.md), and [source audit](docs/sources.md).
This is an independent reconstruction harness, not the authors' released code.
The actual upstream model definition, training and CUDA path could not be tested.
The repository must not be advertised as a successful reproduction.

## Selected configuration

VGG-19-BN from the Liu repository referenced by the paper; CIFAR-10; baseline plus
30% global L1 pruning of convolution weights; masked weights remain zero during
40 recovery epochs. No structured tensor surgery or sparsity acceleration is
claimed. ECE is aggregate top-label ECE with ten bins; equal-mass is primary and
equal-width is also recorded because the paper is internally inconsistent.

The first meaningful experiment needs 160 baseline epochs plus 40 recovery epochs.
Run a single seed first. Three independent seeds are needed to approximate the
paper's mean-and-standard-deviation design; a single seed is a pilot.
This harness intentionally does not substitute a ResNet-18 or ImageNet VGG.

## Setup on the target laptop

Use Python 3.11 or 3.12 and an NVIDIA driver compatible with the selected CUDA
wheel. Commands below are Bash; on Windows use an appropriate Python environment
and the corresponding activation syntax. No GPU runtime was measured here.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
git clone https://github.com/Eric-mingjie/rethinking-network-pruning.git external/rethinking-network-pruning
git -C external/rethinking-network-pruning rev-parse HEAD
```

Inspect the upstream model and record its full commit as `UPSTREAM_COMMIT` in the
commands below. An author-used commit was not identified. The runner rejects a
dirty checkout or a different commit and records source hashes. Modern PyTorch
compatibility of the old upstream package remains unverified; preserve and report
any compatibility failure rather than silently changing architectures.

```bash
UPSTREAM_COMMIT=$(git -C external/rethinking-network-pruning rev-parse HEAD)
python scripts/run_checks.py
python scripts/metric_checks.py
python scripts/check_environment.py --output results/laptop-environment.json
python run_experiment.py --upstream external/rethinking-network-pruning --upstream-revision "$UPSTREAM_COMMIT" --download --device cuda --out results/laptop-preflight --preflight-only
python run_experiment.py --upstream external/rethinking-network-pruning --upstream-revision "$UPSTREAM_COMMIT" --download --device cuda --seed 1 --out results/seed1
```

Preflight performs one real CIFAR batch forward/backward pass and records peak
allocated GPU memory. It does not establish training convergence or paper agreement.
Use separate output directories for preflight and training. An out-of-memory failure
must be recorded; do not silently change the training batch size or use mixed precision.
The fixed batch size is 64; images are 32x32. This is plausibly within 6GB for CIFAR
VGG, but compatibility and memory have not been demonstrated on the user's laptop.

Epoch-boundary restart uses the same training command with `--resume`. The runner
saves optimizer state, RNG states and a last-epoch checkpoint. Mid-epoch work is
repeated after interruption. There is no shortened-training option masquerading as
the paper experiment. Copy finished outputs before changing package versions.

## Corruption evaluation

Obtain the official CIFAR-10-C archive from the
[benchmark repository](https://github.com/hendrycks/robustness) / its linked dataset
record, extract it, and pass the directory containing `labels.npy` and corruption
arrays. Do not generate approximate replacement corruptions.

```bash
python run_experiment.py --upstream external/rethinking-network-pruning --upstream-revision "$UPSTREAM_COMMIT" --device cuda --seed 1 --out results/seed1 --resume --cifar-c data/CIFAR-10-C
```

All 15 standard corruption types and severities 1–5 are evaluated, using the
appropriate 10,000-image slice. The four extra corruption types sometimes shipped
in the archive are excluded. Arrays are memory-mapped, not all loaded into RAM.
The runner records mean accuracy at each severity, mean per-corruption ECE,
and pooled ECE separately. The paper does not resolve the last two ECE choices.
No AlexNet-normalized mCE is used.

Repeat with `--seed 2 --out results/seed2` and seed 3 for independent baselines.
An optional `--recovery-control` also trains an unpruned clone for 40 extra epochs
to probe the extra-training confound. This is an extension, not a reported paper arm.

## Outputs and provenance

Each successful run saves:

- `manifest.json`: settings, upstream revision/source hashes, data hashes, versions,
  device and completion status.
- `epochs/*.json`: one training record per epoch.
- `checkpoints/*_last.pt`: resumable local checkpoints, excluded from Git by default.
- `predictions/<phase>/*.npz`: per-example probabilities, labels and indices.
- `metrics/<phase>/*.json`: both ECE schemes, bin counts, accuracy and prediction hashes.
- `pruning.json`: requested and achieved convolution-weight sparsity.
- `results.json`: clean and optional severity-level corruption results.
- `failure.json`: actual exception if execution fails.

Stored ECE and accuracy values are fractions; multiply by 100 for percentage points.
No energy measurement is included: reproducing calibration does not validate energy
instrumentation. Test data are not used to choose checkpoints; this is a declared
choice, because the author's checkpoint-selection rule was not available.

## Committed results in this delivery

`results/execution-attempt/environment.json` contains actual capability/network
checks. `results/execution-attempt/model-run/failure.json` records the attempted CNN
command. `results/metric-checks/` contains synthetic inputs, bin-level outputs,
test log and test summary. `results/comparison.csv` uses empty fields, not invented
numbers, where reference values or model measurements are unavailable.

The included Git history commits these files. The ZIP contains a local Git
repository; it has **not** been published to GitHub. After connecting GitHub, it can
be published. With an authenticated GitHub CLI, the equivalent command is:

```bash
gh repo create mitra-calibration-reproduction --public --source . --remote origin --push
```

After real runs, commit small summaries and audit raw predictions explicitly:

```bash
git add results/seed1/manifest.json results/seed1/epochs results/seed1/metrics results/seed1/pruning.json results/seed1/results.json
git add -f results/seed1/predictions
git commit -m "Record actual seed 1 reproduction outputs"
```

Review individual file sizes first. Keep large weights and datasets outside ordinary
Git history. Update REPORT.md only from actual outputs and a verified reference.

## License

Original harness code: MIT, see LICENSE. External code/data/paper retain their own
licenses. No third-party model code, weights or dataset images are bundled.

## Resumed attempt

The 20 September 2026 recheck reproduced the same environmental blocker.
See [resumed status](docs/resumed_status.md) and `results/resumed-attempt/`
for the new raw checks and failed command. No trained-model result was produced.
