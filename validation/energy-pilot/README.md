# ResNet-18 energy noise-floor pilot

**Status: NOT measured on the requested laptop. No minimum viable workload has been established.** The execution host was AMD EPYC, with no exposed Intel RAPL or NVIDIA device. `results/session-attempt/` records the actual failed invocation. The numerical unit tests are synthetic and are not experimental observations. CPU/GPU hardware paths have not been validated here.

## What this measures

Full grid: 32, 64, 128, 224 pixels × batches 1, 16, 64, 128, 256 × CPU/GPU separately. Model is torchvision ResNet-18 modified to a CIFAR stem (3×3, stride 1, no max-pool), ten output classes, eval mode, FP32, TF32 disabled. This is an explicit choice, not a claim about your existing model. Change the stem if your intended model differs, and rerun.

Real CIFAR-10 test inputs are preloaded, resized and normalized outside timing. The same first batch of images is repeated: this isolates resident inference, not a full data-loading pipeline. Default weights are seeded random weights, adequate only for an initial dense-kernel timing pilot; supply `--checkpoint` with a matching state_dict for your actual experiment. No accuracy claim is made. Checkpoint-based sparse models require an implementation that really executes sparse kernels; zero weights in dense tensors need not save energy.

Each block consists of a matched-duration idle window, two identical-model active windows, and a second idle window. Models and input tensors remain resident in idle windows. Three-second settling or active warm-up precedes each window; this does not establish thermal equilibrium. The grid is shuffled separately for each repetition. CPU and GPU are not run simultaneously.

A background thread samples every 20 ms in both idle and active windows. RAPL package counters are summed without adding their subdomains. NVML cumulative energy is preferred; only an explicit not-supported error triggers integrated sampled-power fallback. Raw traces retain timestamps. Faster polling does not create higher sensor resolution. Counter wrap is handled between samples; multiple wraps within one sampling gap cannot be recovered.

GPU synchronization after every batch bounds queuing and records completed work. This is synchronous batch inference; asynchronous serving may have different energy behavior. Repeated windows make small native-resolution workloads measurable in principle, but cannot guarantee that model differences exceed noise.

## Run on your laptop

Use native Linux with an NVIDIA driver. Windows/WSL RAPL access is not assumed. Use AC power, a stable power profile, close competing tasks and keep display settings fixed. Inspect counter permissions and arrange read access through your normal system administration process; do not run the entire Python workload as root merely to read counters.

Python 3.10–3.12 is recommended for the proposed dependency set. These versions were not installed or tested here. Install a PyTorch build compatible with your driver using PyTorch's installation instructions.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
# Preliminary full-grid screening; output directories must be new.
python pilot.py --device cpu --data data --download --out results/cpu-screen
python pilot.py --device cuda --data data --out results/gpu-screen
```

CIFAR download requires internet, or place an extracted torchvision-compatible dataset in `data`. Default screening uses six paired repetitions, five-second windows and four CPU inference threads. That is approximately 80 minutes of measured windows across both devices, plus warm-up, settling and other overhead. Screening with six pairs cannot establish the final minimum. Large GPU configurations may fail with OOM; these are recorded, not silently reduced.

Confirm promising cells with **at least 30 independent pairs**, then repeat on another session/day. Example below is a *candidate test*, not a recommendation already supported by data:

```bash
python pilot.py --device cpu --sizes 32 --batches 16 --windows 1 5 10 --repeats 30 --out results/cpu-confirm
python pilot.py --device cuda --sizes 32 --batches 16 --windows 1 5 10 --repeats 30 --out results/gpu-confirm
```

Your planned window duration was not supplied: five seconds is a declared screening assumption, and 1/5/10 seconds are proposed confirmation durations. Pass your actual durations through `--windows`.

## Outputs and calculations

`raw.jsonl`: each actual window, energy, elapsed duration, completed batches, sensor backend and full trace. `windows.csv`: individual readings and, for complete pairs, idle subtraction and energy per image. `summary.csv`: per-cell idle mean/SD, total mean/SD, above-idle mean/fraction, A/A difference noise and provisional detectability. `environment.json`: software/CPU and arguments. `skipped.jsonl`: GPU memory failures. `failure.txt`: any fatal error. Partial runs retain completed raw data.

For each block, baseline power is the mean of before/after idle watts. Above-idle energy = measured active energy − baseline watts × actual active duration. Idle mean/SD in the summary are normalized to the requested duration because actual windows can overshoot. Individual raw idle energies remain available. Fractions are not clipped: negative estimates expose noise/drift. **Above-idle energy is an incremental device-domain estimate, not isolated compute energy.** Idle subtraction cannot disentangle clock changes, memory, fans or shared package activity. These domains also exclude parts of whole-system energy.

Primary comparison is **gross joules per completed image**, not joules per fixed-duration window: a faster model processes more images in a fixed window. Idle-adjusted energy is a secondary diagnostic. CPU RAPL covers CPU package(s), GPU NVML covers its device domain; their absolute joules are not equivalent whole-system boundaries.

## Decision rule and limits

A 5% change in gross J/image is the prespecified meaningful effect for this pilot: a practical precision target, not a universal definition of worthwhile energy savings. Change it before inspecting results if your research needs smaller effects.

Within each identical-model A/A pair, calculate the difference in J/image. For n pairs with sample SD s, approximate minimum detectable effect (two-sided alpha=.05, power=.80) is **2.80 × s / sqrt(n)**. We additionally add the absolute mean A/A difference as a conservative order-drift allowance. `conservative_screen_fraction` divides this sum by mean gross J/image. A candidate is flagged only for n≥30 and a fraction ≤.05. This is a normal approximation and a screening heuristic, not a formal power guarantee. It assumes future A/B variance resembles A/A variance; autocorrelation, drift, sensor smoothing and multiple-cell selection can invalidate that assumption. Low quantization-induced variance can falsely suggest precision.

Before accepting a candidate:

1. Inspect raw cumulative-energy updates/power traces. Stale or strongly quantized telemetry requires longer windows or another meter; constant sampled power can be real or stale and needs inspection. Check maximum sampling gaps.
2. Repeat that cell with 100-ms polling and compare the estimate and throughput with 20-ms polling. Comparable idle sampling alone does not prove overhead is negligible. Use an independently timed, unsampled run if sampler overhead appears material.
3. Confirm in a fresh session; use randomized A/B versus B/A order with the actual two model configurations, identical image counts or J/image normalization, and report a confidence interval for the paired relative difference. The present runner performs A/A screening, not this A/B confirmation.
4. Require the independent A/B confidence interval to exclude zero before claiming a detected difference; distinguish statistical detection from a practically meaningful ≥5% saving.

There is no single smallest workload until batch/latency requirements are specified. Prefer native 32×32 at the intended deployment batch, extending the window first. Among feasible native cells, consider the smallest batch that passes independent confirmation at an acceptable duration. Upscaling alters the task and FLOPs, so use it only if native-resolution repetition cannot meet precision requirements and the larger-input task fits the research question.

Compare CPU and GPU by the confirmed relative noise bound and stability for the same resolution/batch, at a declared CPU thread count. Neither device wins by assumption. A large active-minus-idle fraction alone does not establish that compression effects are detectable.

## References

- [Linux powercap interface](https://www.kernel.org/doc/html/latest/power/powercap/powercap.html): domain hierarchy, energy counter and wrap range.
- [NVIDIA NVML reference](https://docs.nvidia.com/deploy/nvml-api/latest/index.html): device telemetry API; actual feature support must be tested on this laptop.
- [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html): model constructor; CIFAR modifications above are our choice.

This pilot is a local scaffold, not a published benchmark or a validated power meter. Do not cite its unexecuted grid as a result.
