# Hardware Validation Work

This directory contains real, hardware-executed validation work for the proposal in the
parent directory, run on the target laptop (Intel i5-13450HX, RTX 3050 6GB Laptop GPU,
24GB RAM, Linux Mint 22.3) between 2026-09-20 and the date of the latest commit here.
Everything in this directory is a genuine, measured result — no numbers here are
projected, simulated, or interpolated unless a file explicitly labels them as an
estimate and states the method used to derive them.

This work directly closes two of the "outstanding pre-submission verification items"
listed in the top-level [README.md](../README.md#status): the environment/hardware
preflight, and an initial execution of the minimum-detectable-energy-difference pilot
(WP1-adjacent). It also produces the first real training-time and reliability numbers
for the architectures and reconstruction the proposal's Phase 1 work packages depend on.

## What's here

- **`reports/`** — two narrative reports written during this validation pass:
  - `ENVIRONMENT_REPORT.md`: full hardware/software preflight (OS, CUDA, RAPL, NVML,
    dataset reachability, disk/VRAM headroom) plus an execution-feasibility audit of
    FP32/FP16/INT8/INT4 and structured/unstructured pruning on this specific GPU —
    including which configurations are genuinely executable here and which are not,
    with the tested reason for each exclusion (e.g. INT4 has no Conv2d kernel in any
    installed library; native PyTorch structured/unstructured pruning gives no real
    speedup because zeros are stored and computed densely; a working real-channel-removal
    path was found and verified instead).
  - `STATUS_AND_TRAINING_REPORT.md`: repo setup, dataset download status, the real
    energy-pilot run, and the VGG-19-BN training run to completion.

- **`energy-pilot/`** — a full copy of the energy-measurement pilot harness (RAPL +
  NVML paired A/A idle-window measurement), with its own `tests/` and two real result
  sets under `results/`:
  - `results/smoke-gpu/` — the pilot's **default** polling interval (20ms), which this
    validation pass discovered produces physically impossible energy readings on this
    GPU (implied power up to ~300W against a 60W hardware cap). Kept as evidence of the
    finding, not as a usable measurement.
  - `results/smoke-gpu-v2/` — the corrected run (`--interval 0.5`), giving physically
    plausible idle/active power (17-29W / 71-78W) and a real `summary.csv`.
  See `REPORT.md` inside that directory for the full sweep that isolated the cause
  (NVML counter telescoping artifact at fast polling intervals — reproducible, not a
  one-off).

- **`mitra-reproduction/`** — a full copy of the independent reconstruction harness for
  Mitra, Schwalbe & Klein (2024) (VGG-19-BN, CIFAR-10, 30% global L1 pruning + 40-epoch
  recovery), with:
  - `results/execution-attempt/`, `results/resumed-attempt/`, `results/metric-checks/`,
    `results/comparison.csv` — the original delivered evidence of the harness being
    blocked in its authoring environment (no GPU, no PyTorch, HTTP 403 on data/source).
  - `results/laptop-environment.json`, `results/laptop-preflight/` — real preflight on
    this GPU: an actual forward/backward pass, loss=2.315 (sane for a fresh 10-class
    VGG), peak VRAM 472.5MB, full source/data SHA-256 provenance.
  - `results/seed1/`, `results/seed2/`, `results/seed3/` — three independent,
    **complete, real** training runs on this hardware (fresh weight init, data order,
    and augmentation randomness per seed; same architecture/hyperparameters/CIFAR-10
    split): 160 baseline epochs + 40 pruned-recovery epochs each, every epoch's
    loss/accuracy/wall-time recorded individually under `epochs/`, plus a full
    CIFAR-10-C corruption-robustness evaluation per seed (`results.json`, `metrics/`;
    seed1 additionally has a narrative `corruption_eval_report.md`). Per-example
    prediction arrays (`predictions/`, ~58MB/seed) and model checkpoints (~383MB/seed)
    are excluded from this repo per `.gitignore` but exist locally and can be shared
    on request.
  - `results/cross_seed_report.md` — the 3-seed mean±SD analysis, sign-consistency
    check, and paired comparison across seeds. This is the first point in this
    validation pass where any claim about the *direction* of the pruning effect is
    warranted; see key finding 4 below for the headline result.
  - **One fix applied to the harness itself, disclosed here for transparency**:
    `run_experiment.py` originally imported the empty top-level `models` package from
    the vendored upstream instead of `models.cifar` (upstream's own `cifar.py` does
    `import models.cifar as models` — this has been the upstream repo's only-ever
    structure since its single "Initial" commit, not a version drift). One-line fix,
    applied to this harness's own code, not to the vendored third-party dependency.
  - The vendored upstream dependency (`Eric-mingjie/rethinking-network-pruning`,
    commit `2ac473d70a09810df888e932bb394f225f9ed2d1`) is **not** included in this repo
    (third-party code, already excluded by `.gitignore`); re-clone it at that exact
    commit to reproduce — the commit hash is recorded in every `manifest.json` here.

- **`thesis-architecture-comparison/`** — real, measured per-epoch training time for
  the two architectures the thesis proposal actually uses (ResNet-18, MobileNetV3-Small),
  compared against the VGG-19-BN reconstruction's real numbers, specifically to answer
  "is a 10-seed x N-architecture full grid realistic on this hardware within a normal
  thesis timeline." Includes `thesis_arch_train.py` (the training script used) and raw
  per-epoch JSONL results, plus raw `nvidia-smi` telemetry logs (`gpu_health_*.log`)
  used to verify each measurement was genuinely uncontended (or, in one case, to
  characterize a real GPU-sharing contention effect before it was corrected).

## Key findings worth reading first

1. **NVML energy-counter polling-interval artifact** (this GPU specific): polling
   `nvmlDeviceGetTotalEnergyConsumption` faster than ~300ms and summing consecutive
   deltas overestimates energy by up to ~10-20x. Confirmed via a controlled sweep
   (20ms -> 291W implied, 500ms -> 16W implied, for the *identical* idle GPU state).
   Any energy-measurement work package on this hardware must use a polling interval
   of at least 0.3-0.5s.

2. **GPU-sharing contention is real and large, not just theoretical**: running two
   training jobs concurrently on this single GPU caused a measured ~2.04x slowdown for
   both jobs simultaneously (VGG-19's own epoch time went 68.7s -> 140.4s and back to
   68.7s exactly as a concurrent ResNet-18 job started and stopped), confirmed via
   `nvidia-smi` telemetry (SM clock dropped from ~1972MHz to 1627-1845MHz under
   contention; no thermal throttling — temperature stayed at 68-72°C throughout).
   Relevant to WP1/WP3 scheduling: do not run two GPU workloads concurrently on this
   machine and expect either measurement to be clean.

3. **Real per-epoch training costs on this hardware** (all clean, uncontended,
   telemetry-verified):

   | Architecture | Per-epoch | 1 seed (200 ep) | 10 seeds sequential |
   |---|---|---|---|
   | VGG-19-BN | 68.68s | 3.82h | 38.2h (1.59 days) |
   | ResNet-18 | 47.71s | 2.65h | 26.5h (1.10 days) |
   | MobileNetV3-Small | 7.08s | 0.39h | 3.9h (0.16 days) |

   **Plain conclusion**: a full 10-seed grid across all three architectures totals
   roughly 68.6h (~2.86 days) of unattended sequential GPU time — comfortably realistic
   within a normal thesis timeline. The evidence does not support that a reduced
   5-seed / ResNet-18-primary grid is *necessary* on raw compute-time grounds.

4. **Real reliability result for the pruning literature this proposal engages with,
   now at the paper's own 3-seed mean±SD design** (see
   `mitra-reproduction/results/cross_seed_report.md` for full tables and paired
   t-statistics):
   - **Accuracy**: no detectable effect from 30% global L1 pruning + 40-epoch
     recovery, clean or under CIFAR-10-C corruption. Per-seed deltas flip sign at
     3 of 5 corruption severities, and no paired t-test (df=2) approaches the 0.05
     critical value.
   - **Calibration (ECE)**: weak but seed-consistent evidence that pruning slightly
     *degrades* calibration — pruned ECE is higher than baseline ECE in **all 3
     seeds at every severity tested, plus clean (6/6 same-direction)**. Magnitudes
     are small (0.0004-0.013 absolute ECE) and no single severity's paired t-test
     clears a strict per-comparison threshold at n=3, but 6/6 unanimous direction
     would occur by chance only ~1.6% of the time under a true null — suggestive,
     not proof, and explicitly flagged in the report as not a well-powered result.
   - Neither conclusion should be treated as final: 3 seeds is the paper's stated
     minimum, not a well-powered replication.

## Reproducing this work

Each sub-directory (`energy-pilot/`, `mitra-reproduction/`) has its own `README.md`,
`requirements.txt`, and `tests/`. Both were run in separate Python virtual environments
(different pinned PyTorch/CUDA wheel builds — cu121 and cu124 respectively — installed
from `download.pytorch.org`, not plain PyPI, to get real CUDA-enabled wheels matching
each harness's own documented requirements) rather than forced into one shared
environment, since their pinned dependency versions differ.
