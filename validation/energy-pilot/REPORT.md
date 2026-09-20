# Empirical status

Requested machine: Intel i5-13450HX, RTX 3050 6 GB, 24 GB RAM.
Accessible machine: AMD EPYC host; no exposed Intel RAPL counters or NVIDIA device. Actual CPU invocation failed before inference with `No accessible Intel RAPL package counters; native Linux required.`

| Requested finding | Observed status |
|---|---|
| Idle energy mean and SD | Not measured on target hardware |
| 20 workload cells per device | Not measured |
| Above-idle energy and fractions | Cannot calculate |
| Smallest detectable workload | Undetermined |
| CPU vs GPU noise comparison | Undetermined |

Recommendation before design commitment: retain native 32×32 as the first candidate and vary repeated-window duration and batch size. Do not upscale solely on the unsupported assumption that a small single inference is unmeasurable. Use the provided A/A pilot to estimate the resolution of energy-per-image comparisons, then confirm selected cells with actual randomized model A/B tests. There is currently no evidence to endorse GPU, CPU-only, or any minimum batch/window on this laptop.

Four arithmetic tests passed for counter wrap, power integration, counter reset rejection, and duration-corrected idle subtraction/detection logic. Those synthetic tests do not validate RAPL/NVML hardware paths or establish measurement accuracy.
