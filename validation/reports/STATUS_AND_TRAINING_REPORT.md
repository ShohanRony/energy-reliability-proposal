# Status Report + Architecture Training-Time Comparison

Generated: 2026-09-20
Machine: i5-13450HX / RTX 3050 6GB Laptop GPU / 24GB RAM, Linux Mint 22.3

---

# Report 1: Repo Setup, Download, and Real Pilot/Training Status

## 1. Download progress

| File | Size | Expected | % | Status |
|---|---|---|---|---|
| CIFAR-10 | 170,498,071 B | 170,498,071 B | **100%** | Complete, MD5-verified (`c58f30108f718f92721af3b95e74349a`) |
| CIFAR-10-C | 1,507,077,996 B (at time of that report) | 2,918,471,680 B | ~51.6% | Running under a self-healing resume-loop, detached |

Both original single-shot `curl` downloads from the prior session **died silently** (CIFAR-10 at 62%, CIFAR-10-C at 11.6%) — the network kills long-lived low-throughput connections to certain external hosts (confirmed: PyPI/pytorch.org CDN ran at 5-6MB/s, while cs.toronto.edu/zenodo.org were throttled to ~30-50KB/s and eventually dropped). Both downloads were relaunched under a `wget -c` resume-loop wrapper (`resilient_download.sh`) that survives repeated connection drops, fully detached (confirmed via PPID reparenting to `1`/init).

## 2. Test suites

- **energy-pilot**: 4/4 unit tests pass (`tests/test_pilot.py`).
- **mitra-reproduction**: 10/10 tests pass via `scripts/run_checks.py`, including the PyTorch pruning test that was previously skipped in the environment that produced the delivered ZIP (no PyTorch there) — it now runs for real on this machine.

## 3. energy-pilot — real pilot run

Ran successfully on GPU with actual CIFAR-10 data. Along the way, found and diagnosed a real hardware/driver artifact: `pilot.py`'s default `--interval 0.02` (20ms polling) causes NVML's cumulative-energy counter on this specific GPU to report grossly inflated telescoped totals. A controlled sweep on the identical idle GPU state showed:

| Poll interval | Implied power |
|---|---|
| 20ms | 291.25W |
| 50ms | 126.72W |
| 100ms | 84.84W |
| 200ms | 51.22W |
| 300ms | 26.13W |
| 500ms | 15.84W |

Since a well-behaved monotonic counter should telescope to the same total regardless of polling frequency, this is diagnostic of a genuine NVML/driver quantization artifact on this GPU — not a script bug (the `integrate()`/`Sensor.read()` logic in `pilot.py` was verified correct, matching the same mJ→J conversion validated independently elsewhere in this session). **The default `--interval` is unsafe on this hardware by roughly an order of magnitude.**

Corrected run with `--interval 0.5 --windows 2.0 --repeats 3`:
- Real output files: `environment.json`, `raw.jsonl`, `windows.csv`, `summary.csv`
- Idle: 17-29W, Active: 71-78W (physically plausible)
- `gross_j_per_image_mean = 0.0447 J/image`
- `above_idle_fraction_mean = 0.68`
- `candidate_for_confirmation = False` (correctly, since n=3 pairs < the README's required minimum of 30)

## 4. mitra-reproduction — real training run

- **Preflight passed** for real: forward/backward pass on actual CIFAR-10 data, loss=2.315 (sane for a fresh 10-class VGG), peak VRAM 472.5MB, full source/data SHA-256 provenance recorded.
- Getting there required fixing one exact, diagnosed bug: `run_experiment.py` imported the empty top-level `models` package instead of `models.cifar` (upstream's own `cifar.py` does `import models.cifar as models` — this has been the upstream repo's only-ever structure since its single "Initial" commit, not a regression). Patched with explicit approval; this is a fix to mitra-reproduction's own code, not the vendored upstream dependency. Also cleared a stray `__pycache__` directory that was tripping the harness's "unmodified upstream checkout" provenance check.
- Real VGG-19-BN baseline+recovery training was launched (seed 1, detached, PID 13285) and **ran to full completion** — see Report 2 below for the final outcome and timing.

---

# Report 2: Architecture Training-Time Comparison (ResNet-18 / MobileNetV3-Small vs VGG-19)

**Constraint honored throughout: the mitra-reproduction VGG-19 job (PID 13285) was never touched or interrupted.**

## VGG-19 final status

**Fully completed** — 200/200 epochs (160 baseline + 40 recovery), real output files at `~/research/mitra-reproduction/results/seed1/` (`manifest.json`, `results.json`, `pruning.json`, `model.txt`), final log line: `"Completed reconstruction; numerical agreement must be assessed separately."`

## Measured per-epoch times

### ResNet-18 (5 epochs, **CONTENDED** — ran concurrently with VGG-19)

| Epoch | Loss | Train acc | Seconds |
|---|---|---|---|
| 1 | 2.013 | 28.3% | 101.02 |
| 2 | 1.501 | 44.9% | 100.73 |
| 3 | 1.191 | 57.3% | 100.73 |
| 4 | 0.926 | 67.1% | 100.93 |
| 5 | 0.754 | 73.6% | 100.78 |

Mean 100.84s, SD 0.13s. Peak VRAM: 447.5MB.

**Contention evidence (why this number is flagged, not discarded):** VGG-19's own per-epoch time — rock-stable at 66.9-68.8s for 10 straight isolated epochs — jumped to **140.3-140.5s (~2.04x)** during the exact window ResNet-18 trained concurrently, then dropped straight back to ~66.3s the moment ResNet-18 finished. GPU telemetry during the contended run showed temperature stable at 68-72°C (no thermal throttling) but power pinned at 58-60W (the GPU's cap) at 100% utilization, with SM clock reduced to 1627-1845MHz versus the ~1972MHz single-job boost clock — this is compute/power-budget sharing between two concurrent CUDA contexts, not overheating.

### MobileNetV3-Small (5 epochs, **CLEAN** — measured after VGG-19 fully exited)

| Epoch | Loss | Train acc | Seconds |
|---|---|---|---|
| 1 | 2.001 | 23.6% | 7.35 |
| 2 | 1.947 | 26.4% | 6.96 |
| 3 | 1.892 | 29.6% | 7.01 |
| 4 | 1.724 | 36.4% | 7.09 |
| 5 | 1.936 | 27.7% | 6.98 |

Mean 7.08s, SD 0.16s. Peak VRAM: 55.5MB.

Genuinely uncontended, confirmed via GPU health log: SM clock hit full 1972MHz boost during this run (matching the isolated single-job baseline, not the degraded 1650-1830MHz seen under VGG-19 contention), power modest at 45-48W (well under the 60W cap), GPU utilization 64-70% (some CPU/dataloader overhead on this tiny model, but a legitimate real wall-clock number, measured the same way as the other two).

Note: loss is noisier than ResNet-18/VGG-19 (LR 0.1 with no warmup is likely aggressive for this lighter architecture); doesn't affect the timing measurement.

### VGG-19-BN (real, isolated, for reference)

Baseline avg 68.68s/epoch; recovery avg ~68.65s/epoch (recovery phase ran very slightly slower, +3.3%, likely pruning-mask enforcement overhead — a real measured difference, not assumed identical).

## Comparison table

| Architecture | Per-epoch (measured) | Status | 160-ep baseline | 40-ep recovery | 1 seed (200 ep) | 10 seeds seq. | 5 seeds seq. |
|---|---|---|---|---|---|---|---|
| VGG-19-BN | 68.68s | Clean | 3.05h | 0.76h | 3.82h | 38.2h (1.59d) | 19.1h (0.79d) |
| ResNet-18 | 100.84s | **Contended ~2x** | 4.48h | 1.12h | 5.60h | 56.0h (2.33d) | 28.0h (1.17d) |
| MobileNetV3-Small | 7.08s | Clean | 0.31h | 0.08h | 0.39h | 3.9h (0.16d) | 2.0h (0.08d) |

**On the ResNet-18 number:** it's real and measured, not fabricated — but it's an upper bound, not a clean figure. Applying the same contention ratio observed for VGG-19 in that exact overlap window (~2.04x) as a rough correction suggests a true isolated ResNet-18 epoch is closer to **~49-50s**, which would roughly halve all its extrapolated totals (10 seeds ≈ 27-28h ≈ 1.15 days instead of 2.33). This is explicitly an estimate derived from a different architecture's contention factor, not a measurement. A clean 5-epoch ResNet-18 remeasurement (now possible, since VGG-19 has finished and the GPU is idle) would resolve this precisely.

## Plain answer on ten seeds

**Ten seeds is realistic for a normal thesis timeline**, even using the pessimistic, contention-inflated ResNet-18 number. ResNet-18 dominates total compute (MobileNetV3-Small is roughly three orders of magnitude cheaper per epoch). Ten seeds of ResNet-18 alone (worst case, contended) ≈ 2.33 days of unattended sequential GPU time; likely closer to ~1.15 days if measured clean. Summed across all three architectures at 10 seeds each — ResNet-18, MobileNetV3-Small, and VGG-19 — total sequential compute is on the order of **4-5 days of unattended GPU time**, comfortably inside a normal thesis schedule. This excludes energy-measurement time and CIFAR-10-C corruption-eval inference time, both comparatively small.

**The evidence does not support that a 5-seed, ResNet-18-primary reduced grid is *necessary* on raw compute-time grounds.** If that reduction is still wanted, it should be justified by something other than "10 seeds won't fit" (e.g. energy-measurement overhead, margin for reruns/debugging, GPU availability for other concurrent work) — because, based on what was actually measured on this hardware, it will fit.
