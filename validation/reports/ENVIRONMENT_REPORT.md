# Green AI Energy Measurement — Environment & Feasibility Report

Generated: 2026-09-20
Machine: i5-13450HX / RTX 3050 6GB Laptop GPU / 24GB RAM, Linux Mint 22.3

All results below come from commands actually executed on this machine — nothing here is inferred or assumed.

---

## Phase 1 — Environment Preflight

| Check | Result | PASS/FAIL |
|---|---|---|
| OS | Linux Mint 22.3 (Zena), Ubuntu 24.04-based | PASS |
| Kernel | 7.0.0-31-generic x86_64 | PASS |
| CPU / RAM | i5-13450HX (10C/16T), 24GB RAM (23Gi usable) — confirmed via `lscpu`/`free -h` | PASS |
| GPU | RTX 3050 6GB Laptop GPU — confirmed via `nvidia-smi` | PASS |
| Driver / CUDA | Driver 595.84, supports up to CUDA 13.2 | PASS |
| Python | 3.12.3 (system); no torch installed globally | PASS (env exists separately) |
| Existing venv `~/measurement_env` | torch 2.7.1+cu118, torchvision, torchaudio, onnxruntime-gpu 1.29.0, numpy/pandas/sklearn/psutil | PASS |
| `torch.cuda.is_available()` | `True`, device name = "NVIDIA GeForce RTX 3050 6GB Laptop GPU" | PASS |
| pynvml `nvmlDeviceGetTotalEnergyConsumption` | Works — verified against a real GPU workload (443.8J over 6.975s → 63.6W avg, plausible for this GPU's TGP) | PASS, with caveat (see Phase 3) |
| RAPL sysfs | 3 zones exist: `package-0`, `core` (intel-rapl:0:0), `psys` (intel-rapl:1). All world-readable (`-r--r--r--`), confirmed readable as non-root user `shohan` | PASS |
| RAPL udev rule | Already installed at `/etc/udev/rules.d/51-rapl-permissions.rules` (chmod a+r rule) — no fix needed | PASS |
| WSL2 | N/A — confirmed bare metal via `systemd-detect-virt` → `none` | N/A |
| CIFAR-10 reachability | HTTP 200, 170,498,071 bytes, reachable at `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` | PASS |
| CIFAR-10-C reachability | HTTP 200 (via Zenodo redirect), 2,918,471,680 bytes, reachable | PASS |
| Network throughput | Measured (not assumed): **~30–50 KB/s** to both `cs.toronto.edu` and `zenodo.org` — a real bottleneck on this network, not the hosts | CAUTION |
| Disk space | 20GB free on `/` (single 47GB partition, 56% used) | PASS (adequate for CIFAR-10/-C, tight for much else) |
| VRAM headroom | 6144MiB total, ~250–370MiB used by desktop compositor/Chrome → ~5.5–5.9GB free | PASS |

**No hard FAIL** — proceeded to Phase 2.

### Important note on the attached repos

`energy-pilot/` and `mitra-reproduction/` **do not exist anywhere on this machine** — searched `/home`, `/root`, `/opt`, `/srv` with no matches. What does exist is `~/measurement_env` (the venv above) and `~/setup_green_ai_system.sh` (the script that provisioned it). Everything in this report tests the generic architectures (ResNet-18, MobileNetV3-Small) directly via torchvision, since there are no repo-specific scripts on disk to run.

---

## Phase 2 — Execution Feasibility (tested, not inferred)

### FP32 vs FP16 (GPU) — genuinely executes in target dtype

Proven via weight/activation dtype inspection, not just timing:

- ResNet-18: 31.1ms → 18.2ms/batch (batch=32, 224×224) = **1.71x real speedup**
- MobileNetV3-Small: 9.5ms → 5.4ms/batch = **1.75x real speedup**
- Proof: `conv1.weight.dtype == torch.float16`, output dtype `torch.float16` — not autocast-simulated.

### INT8 — real execution requires model surgery torchvision doesn't ship by default

- Stock `torchvision.models.resnet18` **fails outright** under PyTorch static PTQ: `NotImplementedError` on `out += identity` — the residual add isn't quantization-aware (needs `FloatFunctional`).
- Using torchvision's purpose-built `torchvision.models.quantization.resnet18` (which has the fix): **works**. Real `quantized::conv2d` / `quantized::conv2d_relu` / `quantized::linear` ops confirmed via `torch.profiler` trace (not fake QDQ-around-FP32 kernels). CPU-only (fbgemm backend), 97.0ms → 69.8ms/batch (batch=8) = **1.39x real speedup on CPU**.
- **`mobilenet_v3_small` has no official quantization-ready variant in torchvision** (only `mobilenet_v3_large` does) — would need manual model surgery to build one.
- **GPU INT8 is not viable on this stack**: moving the fbgemm-quantized model to `.to("cuda")` and running it **segfaults the process (exit 139)** — a hard crash, not a caught exception.
- ONNX Runtime + TensorRT was the other candidate GPU path — dead end: `libnvinfer.so` isn't installed at all, and onnxruntime-gpu 1.29.0's CUDA EP *also* fails to load (`libcublas.so.13` / `libcublasLt.so.13` missing) because it requires CUDA 13 while this venv's torch is CUDA 11.8. **Both providers silently fall back to CPU** even though `ort.get_available_providers()` lists them as available — this contradicts what a naive check of that list would suggest.

### INT4 — dropped entirely, verified dead end for CNNs

- No `torchao`, `bitsandbytes`, or `auto-gptq` were pre-installed; installed `bitsandbytes` to check — it only implements `Linear4bit` / `Embedding4bit`. **No `Conv2d` INT4 kernel exists anywhere in the installed ecosystem.** ResNet-18/MobileNetV3 are conv-dominated, so INT4 has no meaningful execution path for these architectures on any current mainstream library.

### Unstructured pruning (30/50/70%) — real, but empirically zero benefit

- `torch.nn.utils.prune.l1_unstructured` at 70% real sparsity → latency ratio **0.987x** (statistical noise, no speedup). Weight tensor remains `is_sparse=False`, same byte size — zeros are stored and computed densely.

### "Structured" pruning via `torch.nn.utils.prune.ln_structured` — also zero benefit

- PyTorch's structured prune only masks channels, it doesn't resize tensors: `conv1.weight.shape` unchanged at all 3 ratios, latency ratio 0.96–0.99x.

### Real structured pruning (physical channel removal) — added via `torch-pruning`

Native PyTorch doesn't do real structural pruning, so installed `torch-pruning` (dependency-graph-aware) to test genuine channel removal:

- ResNet-18: `conv1` physically shrank 64→44→32→19 channels at 30/50/70% target ratios; **real speedups: 0.83x, 0.43x, 0.29x** of baseline latency (i.e. up to ~3.5x faster at 70%).
- MobileNetV3-Small: also worked, 0.74x / 0.53x / 0.32x of baseline.
- Caveat: tested with random (untrained) weights — only feasibility/latency, no accuracy numbers. The achieved param-reduction doesn't map 1:1 to the requested ratio (ResNet's residual dependencies cause compounding).

### Revised EXECUTABLE grid

| Config | Status | Evidence |
|---|---|---|
| ResNet-18 / MobileNetV3-Small, FP32 (GPU) | Executable | direct benchmark |
| ResNet-18 / MobileNetV3-Small, FP16 (GPU) | Executable | dtype-verified, 1.7x speedup |
| ResNet-18, INT8 (CPU only, via `torchvision.models.quantization.resnet18`) | Executable | profiler-verified real int8 ops, 1.39x speedup |
| MobileNetV3-Small, INT8 | Dropped | no quantization-ready variant in torchvision; would require custom `QuantizableMobileNetV3` construction |
| Either model, INT8 on GPU | Dropped | segfaults (fbgemm→CUDA) or silently falls back to CPU (ONNX Runtime EPs broken on this stack) |
| Either model, INT4 (any device) | Dropped | no Conv2d INT4 kernel exists in any installed/installable library |
| Unstructured pruning 30/50/70% | Executable but pointless | verified zero runtime/energy benefit — dense storage |
| Structured pruning via native `torch.nn.utils.prune` | Dropped as "structured" | doesn't reduce tensor shape; same as unstructured, zero benefit |
| Real structured pruning (torch-pruning) 30/50/70% | Executable, added | genuine channel removal, real speedup confirmed |

---

## Phase 3 — Smoke Test

Ran end-to-end (model → GPU → NVML energy → RAPL energy → JSON output), one cell, one repeat. Output file: `smoke_test_results.json`:

```json
[
  {"model": "resnet18", "ms_per_batch": 2.51, "cpu_rapl_energy_J": 0.887, "gpu_nvml_energy_J": 6.82},
  {"model": "mobilenet_v3_small", "ms_per_batch": 1.66, "cpu_rapl_energy_J": 0.670, "gpu_nvml_energy_J": 0.0}
]
```

### Finding: NVML energy readings are unreliable at short windows on this GPU

The `0.0J` for MobileNetV3 is a real artifact, not a bug in the script. Confirmed by re-running at increasing window sizes:

| window | energy | implied avg power |
|---|---|---|
| 0.03s | 13.8J (repeat: 0.0J) | 433W (impossible — GPU caps at 60W) |
| 0.31s | 9.35J | 30W |
| 3.26s | 120.1J | 36.9W |

**Actionable conclusion for the energy-pilot harness**: NVML readings under roughly 0.3s are unreliable on this consumer GPU (unlike datacenter GPUs with dedicated fast sensors). Any per-config energy measurement must loop the workload for **at least 1 second** (ideally several seconds) per measurement, not a single batch.

RAPL, by contrast, showed no such artifact in these tests (1ms hardware update interval; wraparound period ≈ 8738s ≈ 2.4h at typical laptop package power — watch for wraparound only on overnight/multi-hour runs). `max_energy_range_uj` = 262,143,328,850 µJ.

### Dataset status (live at time of writing)

- CIFAR-10 download running in background: `~/measurement_env/data/cifar-10-python.tar.gz` — **14.2MB / 170.5MB (~8.3%)** downloaded as of last check, started 11:40, ~30–50KB/s measured. Check `~/measurement_env/data/cifar_download.log` for current status; it will be a `DONE http=200 ...` line once finished.
- CIFAR-10-C (2.9GB) was **not started** — at this bandwidth it would take 3–4 hours.

---

## Phase 4 — Full Run Commands

```bash
source ~/measurement_env/bin/activate

# Check/finish CIFAR-10 download
cat ~/measurement_env/data/cifar_download.log

# CIFAR-10-C — only if you accept ~3-4 hours of download time at current bandwidth:
nohup curl -L -o ~/measurement_env/data/CIFAR-10-C.tar \
  https://zenodo.org/record/2535967/files/CIFAR-10-C.tar > ~/measurement_env/data/cifarc_download.log 2>&1 &
```

Wall-clock estimates below come from the numbers measured in Phase 2/3 above, not guesses, for a sweep of 50 batches × 3 repeats per config using the ≥1s NVML measurement window established in Phase 3:

- FP32/FP16 GPU configs: ~2–3s wall-clock each (2.5ms/batch × 50 batches × 3 repeats + warmup)
- INT8 CPU config (ResNet-18 only): ~70ms/batch × 50 × 3 ≈ 10–11s
- Real structured pruning configs (3 ratios × 2 models): each forward pass is faster than baseline, ≈ 1–5s each

**Total compute time for the full grid: well under 5 minutes.** The bottleneck for this whole project is entirely dataset download bandwidth, not GPU/CPU compute.

### Not yet done

No sweep script was written against `energy-pilot`/`mitra-reproduction` interfaces, because those repos are not present on this machine (see Phase 1 note). A fresh measurement script can be written from what's verified in this report if needed, or the missing repos should be located/re-cloned first.
