# Evaluating Compressed Neural Networks Under Measured Energy and Reliability Constraints

*Toward Deployment-Aware Model Selection: A Controlled, Hardware-Measured Study of Energy
and Calibration Trade-offs in Compressed Vision Models*

**Master's Research Proposal — Two-Year Programme**
Shohinur Pervez Shohan
B.Sc. in Computer Science & Engineering, Rangamati Science and Technology University, Bangladesh
Prospective Applicant, 2027 Intake — Version 4

## Abstract
Making a neural network smaller can reduce the computing power it needs — but it can also
change how confidently the model is wrong. This proposal tests whether compressed
image-classification models that achieve similar accuracy and similar measured energy
consumption remain equally trustworthy once their confidence, not just their correctness, is
examined under realistic image degradation such as blur, noise, and compression artefacts.
This matters directly for anyone deploying vision models on energy- or battery-constrained
devices: an energy saving is only useful if it does not conceal more confident mistakes.
A three-seed pilot study, completed on the applicant's own hardware ahead of this
application, already shows a small but directionally consistent signal in exactly this direction
— compressed models miscalibrated relative to a dense baseline, contradicting the specific
direction reported by the closest prior study. This proposal sets out the confirmatory design
that follows from that pilot: a controlled comparison, at matched measured energy, of what
compression actually buys and what it may quietly cost.
Phase 1 (months 1–12) is the committed thesis deliverable and extends work already under
way on hardware the applicant owns. Phase 2 (months 13–24) is explicitly conditional on
supervision and resources the applicant does not independently possess, and is scoped to
one selected extension rather than a broad programme.

## 1. Introduction

Deep neural networks have become both more capable and more expensive. Deployment
on edge devices, mobile platforms, and energy-constrained infrastructure depends on
compression: pruning removes parameters, quantization reduces numerical precision, and
knowledge distillation transfers behaviour from a large teacher to a compact student.
Evaluation of these methods has matured along two separate tracks. One community
measures efficiency — FLOPs, parameter count, latency, and increasingly energy on real
hardware. The other measures reliability — calibration, robustness under distribution shift,
and class-level degradation. Both communities have produced substantial results. Neither
has produced results that answer the question a practitioner actually faces at deployment
time.
The practical question is this: when two compression configurations achieve comparable
accuracy at comparable measured energy cost, is one of them a worse choice once the
trustworthiness of its confidence is examined? Answering it requires measuring both
properties on the same models, under one protocol, with an energy figure whose limits are
understood rather than assumed. This proposal reports preliminary evidence that the
answer may be yes, and sets out the confirmatory design that follows.

## 2. Preliminary Work and Evidence to Date

This section reports work already completed on the applicant's own hardware, prior to
submission. It exists because an earlier version of this proposal made a claim about
completed pilot work without supplying evidence — a fault identified in review and
corrected here by reporting only what was actually run, with real output values, on real data.

### 2.1 Hardware and environment, as verified

```
 Component                 Verified specification

 CPU                       Intel Core i5-13450HX (10C/16T)

 GPU                       NVIDIA RTX 3050 6GB Laptop GPU, driver 595.84

 RAM                       24GB DDR5

 OS                        Linux Mint 22.3 (Ubuntu 24.04-based), kernel 7.0.0-31-generic, bare metal
                           (confirmed via systemd-detect-virt)

 RAPL access               3 zones (package-0, core, psys), world-readable, confirmed non-root; udev
                           rule installed

 NVML access               nvmlDeviceGetTotalEnergyConsumption confirmed functional against real
                           workloads

 Software                  PyTorch 2.7.1+cu118, torchvision, onnxruntime-gpu 1.29.0
```

Both target datasets (CIFAR-10, 170.5MB; CIFAR-10-C, 2.92GB, all 15 corruption types at 5
severities) are downloaded, checksum-verified, and extracted on this machine.

### 2.2 Execution feasibility: tested, not inferred

Prior to committing to an experimental grid, every proposed compression configuration
was tested directly on this hardware rather than assumed from documentation. This
surfaced three findings that materially changed the design, each verified through dtype
inspection or profiler traces rather than timing alone.

```
 Configuration                  Status                   Evidence

 FP32 / FP16 (GPU, both         Executable               Verified via weight/activation dtype inspection,
 architectures)                                          not autocast simulation. ResNet-18: 1.71× real
                                                         speedup; MobileNetV3-Small: 1.75×

 INT8, ResNet-18 (CPU only,     Executable               Real quantized::conv2d / quantized::linear ops
 fbgemm)                                                 confirmed via torch.profiler trace, not QDQ-
                                                         around-FP32. 1.39× speedup

 INT8, MobileNetV3-Small        Dropped from             No quantization-ready variant exists in torchvision
                                committed scope          (only mobilenet_v3_large has one); would require
                                                         custom model surgery

 INT8, GPU (either              Dropped — confirmed      fbgemm-quantized model segfaults on .to("cuda")
 architecture)                  dead end                 (exit 139). ONNX Runtime + TensorRT:
                                                         CUDA/TensorRT execution providers silently fall
                                                         back to CPU despite being listed as available, due
                                                         to a CUDA 11.8/13 mismatch

 INT4 (any architecture, any    Dropped — confirmed      No Conv2d INT4 kernel exists in any installed or
 device)                        dead end                 installable mainstream library (torchao,
                                                         bitsandbytes, auto-gptq); bitsandbytes only
                                                         implements Linear4bit / Embedding4bit

 Native PyTorch pruning API     Executable but           torch.nn.utils.prune masks weights with zeros
 (structured or unstructured)   confirmed to give zero   that remain densely stored and computed. 70%
                                benefit                  unstructured sparsity: 0.987× latency ratio
                                                         (statistical noise). The “structured” variant only
                                                         masks channels without resizing tensors —
                                                         identical null result

 Real structural (channel)      Executable, adopted      Dependency-graph-aware channel removal,
 pruning, via torch-pruning                              genuine tensor shrinkage confirmed. ResNet-18
                                                         at 30/50/70% target ratios: real speedups of
                                                         0.83×, 0.43×, 0.29× baseline latency (i.e. up to ≈
                                                         3.5× faster at 70%). MobileNetV3-Small: 0.74×,
                                                         0.53×, 0.32×
```

This is treated as a finding, not merely a scoping exercise. The compression method
used in the closest prior calibration study (§2.3) — unstructured, global-magnitude weight
pruning — is exactly the method confirmed above to yield zero measured energy or latency
benefit on consumer hardware, regardless of the nominal sparsity ratio reported. A “30%
pruned” model can therefore mean two operationally unrelated things: a model with 30% of
its weights zeroed but unchanged compute cost, or a model with 30% of its channels
physically removed and genuinely cheaper to run. This distinction, confirmed empirically
rather than assumed, motivates separating the reliability question (§2.3, using the
unstructured method to enable direct comparison with prior work) from the energy-linked
question (§6, WP3, using confirmed real structural pruning) rather than conflating them
under one label.

### 2.3 Reconstruction of the closest prior study

Mitra, Schwalbe & Klein (2024) is the closest existing work to this thesis (§4.2). Their reported
protocol — VGG-19-BN, CIFAR-10, global unstructured L1 conv-weight pruning at 30%, 160
baseline epochs followed by 40 recovery epochs — was independently reconstructed and
run to completion for three seeds, with full clean and CIFAR-10-C (15 corruptions × 5
severities) evaluation. This is not the authors' released code; it is an independent
implementation built from their paper's stated methodology, with full source and data
provenance (SHA-256 hashes, commit references) recorded for every run.

```
 Metric                         Baseline (mean ± SD, 3    Pruned 30% + recovered     Direction across
                                seeds)                    (mean ± SD, 3 seeds)       seeds

 Clean accuracy                 0.9372 ± 0.0010           0.9375 ± 0.0011            Inconsistent (2 of 3
                                                                                     seeds positive) — no
                                                                                     detectable effect

 Clean ECE (equal-mass)         0.0472 ± 0.0012 (across   —                          Consistent: all 3
                                seeds)                                               seeds show pruned
                                                                                     ECE higher

 Corruption mPC (mean over      Ranges 0.8678 – 0.5655    Comparable, within noise   Inconsistent at
 5 severities)                  by severity                                          severities 1, 2, 5;
                                                                                     consistent (worse)
                                                                                     only at 3, 4

 Pooled corruption ECE, all 5   Lower at every severity   Higher at every severity   Consistent across all
 severities                                                                          3 seeds, at every
                                                                                     severity — 6 of 6
                                                                                     conditions
```

The calibration gap was in the same direction across all three seeds and all six conditions
(the clean condition and five corruption severities). This is directionally consistent
descriptive evidence only: the three-seed sample is too small to support a formal statistical
test, which is exactly why the Phase 1 confirmatory design uses 10 seeds (§6).
This directly contradicts the direction Mitra et al. report — that post-hoc pruning can
improve calibration and corruption robustness simultaneously. One plausible, testable
explanation for the discrepancy is protocol-level: in Mitra et al.'s design, the pruned model
receives 40 additional epochs of recovery training that the baseline, as described, does not
receive. If a dense model given the same 40 additional epochs shows comparable
calibration change, the reported improvement may be partly attributable to extra training
rather than to pruning itself — the same class of finding as the applicant's undergraduate
thesis, where a validation-design choice, not the algorithm, determined the reported
conclusion (§10). This is the central confirmatory hypothesis of Phase 1 (§6, H2), not yet
resolved, and is proposed precisely because the preliminary evidence above is directional
but underpowered.

### 2.4 Instrument characterisation

Two measurement-validity findings emerged directly from running the harness,
independent of the substantive compression question, and are reported as preliminary
methods results.
NVML short-window artifact. Measuring the identical idle GPU state at increasing polling
intervals produced implied average power that should be constant but was not:

```
 Polling    20ms        50ms          100ms        200ms        300ms        500ms
```

 interval

```
 Implied    291.25W     126.72W       84.84W       51.22W       26.13W       15.84W
```

 avg.
 power

This GPU has a 60W cap, so the 20ms reading is physically impossible; a correctly behaving
monotonic counter should telescope to the same total regardless of polling frequency. This
is diagnostic of a genuine NVML/driver quantisation artifact on this consumer GPU, distinct
from datacentre-class GPUs with dedicated fast power sensors. RAPL, tested identically,
showed no equivalent artifact (1ms hardware update interval). The practical consequence,
adopted throughout: no NVML-derived energy measurement is trusted below a 1-second
measurement window.
GPU-contention diagnostic. An accidental concurrent-training incident during timing
measurement was diagnosed rather than discarded: SM clock ceiling, not power draw,
distinguishes contended from clean runs on this hardware, because both regimes saturate
the 60W power cap identically. Clean runs reach 1792–1972MHz; a contended run was
capped at 1627–1845MHz. This diagnostic was used to identify and then correct one
contaminated measurement (§2.5).

### 2.5 Real energy pilot and validated compute budget

A corrected pilot run (0.5s polling interval, 2.0s measurement windows, 3 repeats) produced
physically plausible, real values: idle draw 17–29W, active draw 71–78W, gross energy
0.0447 J/image, with 68% of total measured energy above the idle baseline. The harness
correctly reports this pilot as not yet confirmatory (n = 3 paired measurements, below the
pre-specified minimum of 30) — it establishes that the harness produces plausible numbers,
not that any compression effect is yet resolved.
Per-epoch training time was measured directly, with contention explicitly ruled out via the
GPU-clock diagnostic above:

```
 Architecture                              Per-epoch (measured,   1 seed (200   10 seeds,
                                           clean)                 epochs)       sequential

 VGG-19-BN (reconciliation track, §2.3)    68.68s                 3.82h         38.2h (1.59
                                                                                days)

 ResNet-18 (thesis architecture)           47.71s (SD 0.082s)     2.65h         26.5h (1.10
                                                                                days)

 MobileNetV3-Small (thesis architecture)   7.08s (SD 0.16s)       0.39h         3.9h (0.16 days)
```

Total sequential GPU time for the thesis's own confirmatory training grid (ResNet-18 +
MobileNetV3-Small, 10 seeds each): approximately 30.4 hours ≈ 1.3 days , unattended,
measured directly rather than estimated — comfortably within a normal thesis schedule,
with real margin for reruns.

## 3. Research Motivation

### 3.1 The measurement problem

Compression is routinely justified by reductions in FLOPs or parameter count. Direct
hardware measurement has repeatedly shown these proxies unreliable: energy depends on
memory traffic, kernel availability, cache behaviour, and batch size in ways operation counts
do not capture (§4.3). This proposal takes that finding as an established premise, not a
contribution.
The consequence has not been fully followed through. Software estimators such as
CodeCarbon and pyJoules have been validated against physical meters on general AI
workloads, with documented errors up to 40% (Fischer, 2025). They have not been tested on
compressed models, where quantization alters the executed instruction mix. It is also
necessary to be precise about what such a comparison can establish: pyJoules and, in part,
CodeCarbon derive their readings from the same RAPL and NVML interfaces this proposal
also reads directly. Comparing these tools against RAPL/NVML therefore audits accounting
and sampling consistency between layers of the same measurement stack — a legitimate
and, per §4.4, unaddressed question — rather than constituting independent physical
validation. Independent validation against an external reference (a wall-plug or bench
meter, with its measurement boundary explicitly reconciled against the GPU/CPU package
boundary) is treated in this proposal as Phase 2 work (§7, WP5), not claimed in Phase 1.

### 3.2 The reliability problem

A compressed model that retains aggregate accuracy is conventionally treated as
successful. Compression has been shown to cause selective forgetting concentrated in a
small subset of classes (Hooker et al., 2019), to produce disparate accuracy impacts across
groups (Tran, Fioretto & Kim, 2022), and to alter predictive behaviour in ways aggregate
metrics do not surface.
The direction and magnitude of these effects is not settled by isolated comparison. Post-hoc
pruning has been reported to improve calibration and corruption robustness in one study
(Mitra, Schwalbe & Klein, 2024) and in embedded-inference settings (Huber, Göhner &
Trapp, 2025). Other work reports reduced robustness under domain shift and adversarial
perturbation, with quantization outperforming pruning (Shen et al., 2024). Still other work
finds compression preserving or improving corruption robustness under different
conditions (Diffenderfer et al., 2021; da Silva et al., 2025). These are context-dependent
effects — obtained under different training procedures, compression intensities, and forms
of distribution shift — rather than a strict contradiction resolvable by any single additional
study; this proposal's own reconstruction (§2.3) adds one more controlled data point in this
space, under a protocol close enough to Mitra et al.'s to be directly informative about that
specific comparison.

## 4. Related Work

This section states what prior work established and what it leaves open. Characterisations
such as “not identified in our search” reflect systematic searches across Consensus, Scite,
Scholar Gateway, Elicit, ResearchRabbit, and direct publisher retrieval, covering 2019–2026;
the absence of a result in a search is not treated as proof of its absence in the literature.

### 4.1 Neural network compression evaluation

Compression evaluation is mature and extensively benchmarked, typically on accuracy,
parameter count, FLOPs, and increasingly latency; joint pruning–quantization optimisation
via reinforcement learning has achieved 39% average energy reduction for 1.7% average
accuracy loss on embedded accelerators (Balaskas et al., 2023). Evaluation remains
accuracy-centric, and reliability properties are rarely part of the comparison protocol. This
work does not propose a new compression algorithm; it treats the evaluation protocol as
the object of study.

### 4.2 Reliability consequences of compression

Hooker et al. (2019) showed pruning disproportionately degrades a small subset of classes
while aggregate accuracy moves little — 170 ImageNet classes significantly affected at 50%
sparsity, rising to 372 at 70%. Related work characterised bias through compression-
identified exemplars (Hooker et al., 2020), class/cohort-level degradation (Paganini, 2020),
and gradient-norm-mediated group disparity (Tran, Fioretto & Kim, 2022). None measures
calibration or energy.
Mitra, Schwalbe & Klein (2024) is the closest existing work to this thesis, evaluating pruning
of perception CNNs on expected calibration error and corruption robustness — reporting
aggregate top-label ECE, not disaggregated per class, and not measuring energy. Huber,
Göhner & Trapp (2025) measures embedded-system response time alongside ECE in a
different application domain, without a multi-family compression sweep or energy
measurement. Yuan et al. (2023) examined PTQ reliability with attention to calibration-set
distribution and worst-case subgroup performance; Williams & Aletras (2023) established
that calibration-data selection substantially affects compressed-model task performance —
a distinct concept from probability calibration, and a confound this proposal's own
reconstruction (§2.3) is designed to isolate rather than assume away. Kamal & Talbert (2025)
introduced faithfulness metrics beyond accuracy for compressed models on socially
meaningful datasets, without energy measurement.

### 4.3 Energy measurement and hardware-aware evaluation

Tripp et al. (2024, BUTTER-E) provide 63,527 watt-meter measurements across fully
connected networks of varying size and shape, finding energy hardware- and cache-
mediated rather than FLOPs-linear — evaluated on architecture design space, not post-
training compression, with no reliability axis. Pachón, Pedraza & Ballesteros (2025)
evaluated 180 pruned models, confirming strong batch-size dependence in the FLOPs–
energy relationship. Deutel, Woller & Mutschler (2022) and Balaskas et al. (2023) measured
physical power for compressed models on embedded hardware. Each reports accuracy as
the sole quality metric alongside energy.

### 4.4 Software energy estimator validation

Fischer (2025) compared software estimators against external physical meters across
hundreds of AI experiments and found errors up to 40%; this is a general-workload finding
and is not presented here as specific to pyJoules or to compressed models. Aquino-Brítez et
al. (2025) and Rodriguez et al. (2024) provide comparable general validation and review
work. Kocher et al. (2025) found NVIDIA SMI can correlate poorly with an external power
meter under some sampling conditions and proposed corrective calibration — directly
relevant to the NVML behaviour independently observed in §2.4. de Paula, Soni, Upadhyay &
Lagos (2025) is the closest prior work on the efficiency side, comparing pruning,
quantization, and distillation for carbon efficiency via CodeCarbon, without hardware
ground truth, a reliability axis, or a vision-domain application. Rojahn & Grum (2025), in a
103-study systematic review, conclude the field lacks reproducible, calibrated measurement
across hardware tiers and calls for a hybrid estimator-plus-metering architecture —
external corroboration of the gap this proposal addresses.

## 5. Research Gap

The hardware-energy literature and the reliability literature have each advanced
substantially since 2019 without intersecting on a shared protocol. Studies that measure
energy on compressed models report only accuracy; studies that measure calibration or
class-level impact on compressed models report no energy.

```
 Component combination       Representative prior work                             Status

 Compression + hardware      Pachón et al. 2025; Deutel et al. 2022; Balaskas et   Well covered — accuracy
 energy                      al. 2023; Tripp et al. 2024                           only

 Compression + calibration   Mitra et al. 2024; Huber et al. 2025; Yuan et al.     Covered — no energy;
                             2023                                                  independently
                                                                                   reconstructed here
                                                                                   (§2.3)
 Component combination           Representative prior work                              Status

 Compression + robustness        Shen et al. 2024; Diffenderfer et al. 2021; da Silva   Covered — no energy;
                                 et al. 2025                                            context-dependent
                                                                                        directions

 Compression + class-level       Hooker et al. 2019, 2020; Tran et al. 2022;            Mature — no energy
 impact                          Paganini 2020

 Energy + estimator validation   Fischer 2025; Aquino-Brítez et al. 2025; Kocher et     General workloads —
                                 al. 2025                                               not tested on
                                                                                        compressed models

 Compression + measured          —                                                      Not identified in our
 energy + classwise                                                                     search; addressed here
```

 calibration, one protocol

Gap 1. Software estimators have not been tested on compressed models, where
quantization changes the instruction mix on which they were calibrated. Framed here as an
accounting-consistency audit (§3.1), not independent physical validation, which is reserved
for Phase 2.
Gap 2. Reported reliability effects of pruning are context-dependent across studies, and a
specific protocol confound in the closest study — unmatched recovery training — has not
been tested. Preliminary evidence (§2.3) motivates a confirmatory test of this specific
hypothesis.
Gap 3. Compression methods have not, to the extent our searches were able to establish,
been compared at matched measured-energy budgets using statistical equivalence testing;
existing comparisons normalise on FLOPs or nominal compression ratio — proxies whose
validity is itself in question, and whose disconnect from real energy behaviour is
demonstrated directly in §2.2.

## 6. Research Questions and Hypotheses

RQ1 — Instrument agreement. Do software energy estimators agree with the hardware
counters they partly derive from, across the confirmed-executable compression grid (§2.2),
and does that agreement change systematically as quantization precision decreases from
FP32 to INT8 (ResNet-18, CPU)?
RQ2 — Recovery-training confound (Track A). Does the classwise calibration difference
observed between a dense baseline and a globally-pruned, recovery-trained model (§2.3)
persist when the baseline is given the same total training budget, or is the observed
direction attributable to asymmetric training rather than to pruning?
RQ3 — Matched-energy selection risk (Track B). Among real structurally-pruned and
quantized configurations of ResNet-18 and MobileNetV3-Small demonstrated to have
statistically equivalent measured inference energy, does selecting the most accurate
configuration more often yield materially worse classwise calibration under corruption than
an alternative eligible configuration?
RQ4 (Phase 2) — Physical and cross-platform generality. Do the Phase 1 findings on RQ1–
RQ3 hold against external physical power measurement and on one additional hardware
platform?

One primary confirmatory hypothesis is pre-specified for each of Tracks A and B; RQ1 is
treated as a supporting methods result. All other comparisons are exploratory and reported
as such, with Holm correction where multiple exploratory tests are drawn from the same
family.
  • H1 (supporting, methods). Estimator–counter disagreement, where the estimator
    and counter share an underlying interface, increases as quantization precision
    decreases from FP32 to INT8 on the confirmed CPU path. This is reported
    descriptively; no claim of independent instrument validity is made from it.
  • H2 (primary, Track A). A dense VGG-19-BN model trained for the same total epoch
    budget (160 + 40 = 200) as the pruned-and-recovered model shows classwise ECE
    degradation, relative to a 160-epoch-only dense baseline, that is not distinguishable
    (via TOST, pre-specified margin ±0.010 absolute ECE) from the degradation observed
    in the pruned-and-recovered model. Rejection of this equivalence — i.e. the pruned
    model is measurably worse even after the training-time confound is controlled —
    would indicate the effect is attributable to pruning itself, not training time.
  • H3 (primary, Track B). Within pairs of ResNet-18/MobileNetV3-Small configurations
    shown by TOST to be energy-equivalent (measured-energy ratio within 0.95–1.05)
    and accuracy-equivalent (TOST, ±0.010 absolute), the higher-accuracy member of
    each pair shows worst-class ECE degradation exceeding 0.010 absolute, relative to
    the alternative eligible member, more often than the exploratory ranking (ordinary
    accuracy-based selection) would predict by chance.

## 7. Phase 1, Track A — The Recovery-Training Confound (Months 1–6)

This track directly extends the reconstruction already completed (§2.3). It is the fastest-
completing, most directly evidenced component of Phase 1.

WP1 — Matched-training dense control
Architecture, data, and hyperparameters exactly as in §2.3 (VGG-19-BN, CIFAR-10, SGD
momentum 0.9, weight decay 1e-4, batch 64, LR 0.1). Three arms per seed: (a) dense
baseline, 160 epochs; (b) dense model, 160 + 40 = 200 epochs, no pruning — the missing
control in the reconstructed protocol; (c) globally pruned 30% + 40-epoch recovery, as
already run. Ten seeds, extending the three already completed. Full clean and CIFAR-10-C
evaluation for every arm and seed, using the same harness validated in §2.3–2.5.
Estimator, stated deliberately. Equal-width ECE is unstable when conditioned on class,
given typical per-class sample counts; the primary calibration endpoint is classwise one-
versus-rest ECE with ten equal-mass bins and deterministic tie handling, matching the
equal-mass convention already used in §2.3. Brier score is retained as a secondary measure
of probabilistic predictive quality — it is not treated as an isolated calibration measure, since
it reflects discrimination as well as calibration.
Analysis. H2 is tested via TOST with the pre-specified ±0.010 absolute ECE margin (§6). The
worst-minus-mean classwise gap is reported as a secondary, exploratory quantity,
distinguished explicitly from worst-class error alone — the gap can rise because other
classes improve, not because the worst class worsens, and the two are not conflated.
Output: a standalone finding, publishable independent of Track B's outcome — either the
training-time confound explains Mitra et al.'s reported direction, or pruning has an effect
beyond it. Both are informative results.

## 8. Phase 1, Track B — Matched-Energy Selection Risk (Months 4–12)

This track uses only the compression configurations confirmed genuinely executable and
energy-relevant in §2.2: real structural (channel) pruning via torch-pruning, and FP16/INT8
quantization on their confirmed-executable paths. Unstructured weight pruning is excluded
from this track, having been confirmed to produce no measured energy benefit regardless
of nominal sparsity (§2.2).

```
 Architecture                Confirmed-executable arms

 ResNet-18                   FP32 baseline (GPU); FP16 (GPU); INT8 (CPU,
                             torchvision.models.quantization.resnet18); real channel pruning 30/50/70%
                             (GPU, via torch-pruning)

 MobileNetV3-Small           FP32 baseline (GPU); FP16 (GPU); real channel pruning 30/50/70% (GPU, via
                             torch-pruning) — no INT8 arm (§2.2)
```

Eleven confirmed-executable configurations across both architectures. Ten seeds each,
following the validated compute budget of ≈ 1.3 days sequential GPU time for training
(§2.5); energy measurement follows the validated protocol — minimum 1-second NVML
measurement windows (§2.4), 30 paired measurement windows per confirmatory
comparison, RAPL logged concurrently on the CPU-only INT8 arm.

WP2 — Energy-equivalence classification
Configuration pairs are classified as energy-equivalent using TOST against a pre-specified
0.95–1.05 ratio interval on measured joules per image, computed from at least 30 paired
measurement windows per configuration. Configurations are also classified as accuracy-
equivalent via TOST at ±0.010 absolute top-1 accuracy. Failure to reject a difference is never
treated as evidence of equivalence; only a confidence interval falling entirely inside the pre-
specified margin is.
WP3 — Selection-risk test (H3)
Among pairs classified as jointly energy- and accuracy-equivalent, the member with
(marginally) higher accuracy is compared against its pair partner on worst-class ECE under
corruption. H3 is tested via the proportion of such pairs, across seeds and severities, in
which the higher-accuracy member shows worst-class ECE degradation exceeding the pre-
specified 0.010 absolute threshold.
Robustness metric, matching §2.3. Mean-per-corruption accuracy (mPC), averaged per
severity over all 15 standard CIFAR-10-C corruptions — not normalised mCE, for direct
comparability with the Track A reconstruction and with Mitra et al.'s own reported metric.
Timing and exclusion rules. Fixed batch size 64; warm-up passes discarded; a deterministic
idle–active–idle measurement window (10s settling either side); background load logged;
invalid runs (OOM, throttling detected via the clock-ceiling diagnostic of §2.4, failed
convergence) are retained in the raw record and excluded from confirmatory analysis by a
rule fixed in advance, never post hoc.
Multiplicity. H2 and H3 are the two pre-specified primary tests in this proposal. All ranking
or ordering analysis beyond the specific pair-comparison in H3 (e.g. a full cross-
configuration Kendall correlation) is exploratory, reported with Holm correction, and not
used to support the thesis's central claim.

## 9. Timeline

Note on research activity alongside coursework. The plan below sets out the research
programme itself. In taught Master's programmes, coursework typically runs alongside —
particularly in year one — so research activity proceeds at reduced intensity during teaching
periods and intensifies during dedicated thesis time. The preliminary work reported in §2
was deliberately produced ahead of enrolment for this reason: it reduces what remains to
be done once coursework begins.

```
 Months      Activity                                                              Dependency

 1–2         Track A: 7 additional VGG-19-BN seeds (dense-control arm              Already-validated
             added); harness extended to Track B architectures                     harness

 2–4         Track A: full clean + corruption evaluation, all arms, 10 seeds; H2   Track A training
             analysis

 4           Track A complete: first paper drafted (publishable regardless of      —
             H2 outcome)

 4–6         Track B: 10-seed training, ResNet-18 + MobileNetV3-Small, all 11      Validated 1.3-day
             confirmed-executable configurations                                   compute budget

 6–9         Track B: energy measurement (≥30 windows/configuration),              Track B training
             TOST energy- and accuracy-equivalence classification                  complete

 9–11        Track B: H3 selection-risk analysis; RQ1 estimator-agreement          WP2 equivalence
             audit across full grid                                                classification
 Months       Activity                                                       Dependency

 12           Phase 1 complete: two papers (Track A, Track B), public        —
              benchmark repository, arXiv preprint

 13–15        Buffer quarter: onboarding, ethics/procurement for chosen      Placement
              Phase 2 extension

 15–22        Phase 2: the single extension selected with supervisor (§10)   Lab access, supervision

 22–24        Thesis writing; submission                                     —
```

Compulsory deliverables. A completed thesis; the reproducible evaluation pipeline already
partially built (§2); a documented comparison on the two committed architectures under
both tracks. Manuscript submission from Phase 1 is an intended output; acceptance is not
treated as a completion requirement.
Slippage plan. If Track A overruns, seed count for the dense-control arm is reduced to 5
with the reduction reported, not silently absorbed. If Track B overruns, the MobileNetV3-
Small arm is retained in full and the ResNet-18 INT8 arm is reported as incomplete rather
than dropped without disclosure. Track B's core structural-pruning comparison is never cut
— it is the thesis question.

## 10. Phase 2 — One Bounded Extension (Months 13–24)

Phase 1 is presented as a complete, publishable thesis on its own. Phase 2 is not a catalogue
of possible directions; it is one extension, to be selected jointly with a supervisor based on
Phase 1's actual findings and the specific resources and expertise available in their group.
Four candidate extensions are described below with the specific resource each requires; the
applicant proposes to adopt exactly one.

Candidate: physical power ground truth
Requires: bench power meter, current probes, oscilloscope, and expert guidance
reconciling measurement boundaries. RAPL and NVML are themselves modelled estimates;
this extension would establish the floor beneath the accounting-consistency audit in §3.1
and §6 (RQ1), converting it into genuine independent physical validation.

Candidate: cross-platform generality
Requires: access to at least one additional accelerator class (e.g. a server GPU with genuine
sparse-kernel support, or an embedded/edge device). Tests whether Track B's conclusions
are specific to this consumer GPU's lack of sparsity acceleration (§2.2) or generalise.

Candidate: modality generalisation
Requires: compute beyond the 6GB ceiling used throughout Phase 1, and domain
supervision. Tests whether the Phase 1 protocol, applied to a small (≈ 0.5B) language
model, reproduces the same qualitative pattern. UniComp (von Rad, Cao & Geiger, 2026)
occupies adjacent LLM-compression territory; the differentiator remains classwise
probability calibration and measured energy, neither of which that work reports.

Candidate: real subgroup fairness
Requires: dataset access, ethics approval, and institutional affiliation. Phase 1's per-class
analysis is a legitimate but distinct question from fairness across real demographic groups,
which needs data an unaffiliated individual cannot access.

## 11. Expected Contributions

Already produced (reported in §2, prior to enrolment)
  • An empirically verified executable compression grid for consumer vision hardware,
    including two confirmed dead ends (INT4 for CNNs; GPU-side INT8 via the tested
    stack) and one confirmed non-result (native PyTorch pruning APIs give zero energy
    benefit regardless of nominal sparsity).
  • Two independently discovered instrument-characterisation findings: the NVML short-
    window power artifact, and GPU clock-ceiling as the correct diagnostic for CUDA-
    context contention on shared hardware.
  • A 3-seed reconstruction of the closest prior study, showing a directionally consistent
    (6 of 6 conditions, sign-test p ≈ 0.016) but not yet confirmatory disagreement with its
    reported direction on calibration.

Committed (Phase 1, thesis deliverable)
  • A confirmatory 10-seed test of whether the reported calibration effect in the closest
    prior study is attributable to pruning or to an uncontrolled training-time confound
    (Track A, H2) — informative in either direction.
  • The first evaluation, to our knowledge, of compression-method selection risk at
    TOST-established matched measured energy, using confirmed genuinely-executable
    structural compression rather than nominal compression ratios (Track B, H3).
  • A reproducible benchmark with frozen protocol, released per-example outputs, and
    full provenance, covering both tracks.

Aspirational (Phase 2, one selected extension)
  • Physical validation, cross-platform generality, modality generalisation, or real-
    subgroup fairness — one, selected with supervisor input.

Target venues. TMLR or Sustainable Computing: Informatics and Systems for the
measurement and Track A findings; NeurIPS Datasets & Benchmarks or MLSys for the full
framework; SustainML / EfficientML workshops for early-stage results.

## 12. Feasibility and Applicant Preparation

Section 2 of this proposal is itself the primary feasibility evidence: every claim of
executability, every timing figure, and the central directional finding motivating Track A are
backed by output already produced, not projected.
Technical background. Python, PyTorch, scikit-learn, pandas, NumPy, OpenCV; the full
measurement and training harness described in §2 was built, debugged, and validated
independently, including diagnosing and fixing one packaging bug in the reconstruction
scaffold and one measurement-configuration error in the energy harness's default settings.
Methodological background. The applicant's undergraduate thesis reconstructed three
decades of land-cover change from satellite imagery using multiple classifiers under
spatially-blocked rather than random validation. That single design choice reversed the
headline conclusion, while the choice of classifier moved the result by roughly five
percentage points. The full pipeline was released publicly, and the work is under peer review
at ICCIT 2026. A related literature review, surveying the energy-measurement literature this
proposal builds on, is separately under revision following peer review.
The lesson from that thesis — that evaluation design, not algorithm choice, determined the
finding — is the direct origin both of this proposal and of the specific confound this proposal
now proposes to test (§2.3, §6, H2). The subject matter has changed; the disposition has not.
What scholarship support enables. Existing hardware allows preliminary work to continue
independently, as §2 demonstrates. Scholarship support provides sustained research time
free of the applicant's current full-time employment, formal statistical and methodological
training, and supervision to carry the confirmatory design in §6–§8 through to completion
and, in Phase 2, to extend it into regimes (physical instrumentation, larger-scale compute,
real demographic data) that are not reachable independently. The applicant intends to
release the full pipeline and results publicly, usable on comparably modest hardware,
including by researchers and students without access to institutional compute.
Stated limitations. No accepted publications to date; one manuscript under review, one
under revision. Mathematical background is sufficient to apply and extend established
statistical methods (TOST, paired testing, sign-test framing, as used throughout §2 and §6–
§8) with reference to the literature, rather than to derive new theory independently.

## 13. Risks and Mitigation

```
 Risk                            Mitigation

 H2 or H3 is not confirmed       Both are pre-registered as informative in either direction. A well-
                                 powered null result on H2 (training time fully explains the effect) or H3
                                 (no selection risk at matched energy) is itself publishable, because the
                                 equivalence margins are fixed in advance rather than chosen post
                                 hoc.

 Measurement noise exceeds the   The 1-second minimum NVML window and the 30-window-per-
 effect                          configuration requirement are both derived from measured
 Risk                                 Mitigation

                                      instrument behaviour (§2.4–2.5), not assumed.

 Prior work is closer than assessed   Positioning (§4) states specific extension axes rather than an empty
                                      field, and the closest work (Mitra et al.) has been fully reconstructed
                                      and run, not merely cited.

 Compute or timeline overrun          The full Track A + Track B training budget (≈ 2.9 days sequential
                                      across both tracks) is a direct hardware measurement, not an
                                      estimate; the slippage plan (§9) specifies exactly what is cut first.

 Concurrent work occupies the         Track A's first result depends only on already-completed infrastructure
 space                                and is the fastest output (month 4); early preprinting establishes
                                      priority.
```

## 14. Why Supervision Is Sought

This proposal is not a request to begin a project; the preliminary work in §2 demonstrates a
project already under way, with a specific, evidenced, falsifiable confirmatory question.
What that work cannot do alone is precisely what supervision and a funded position provide:
sustained time free of full-time employment, formal training in the statistical methods this
design already uses at an applied level, and — for Phase 2 specifically — access to physical
instrumentation, additional hardware platforms, or demographic data that cannot be
acquired independently.
The specific request: supervision of a Master's thesis testing whether a reported
compression–calibration relationship survives a training-time confound (Track A), and
whether compression-method selection at matched measured energy carries a reliability
risk not visible under standard accuracy-only comparison (Track B) — with the direction of
the final year's single extension (§10) to be agreed jointly once Phase 1's findings are in.

## References
Aquino-Brítez, A., García-Sánchez, P., & Ortíz, A. (2025). Towards an energy consumption index
    for deep learning models: a comparative analysis of architectures, GPUs, and
    measurement tools. Sensors, 25(3), 846. DOI: 10.3390/s25030846
Balaskas, K., Karatzas, A., Sad, C., Siozios, K., & Anagnostopoulos, I. (2023). Hardware-aware
     DNN compression via diverse pruning and mixed-precision quantization. IEEE
     Transactions on Emerging Topics in Computing. DOI: 10.1109/TETC.2023.3346944
Blakeney, C., Huish, N., Yan, Y., & Zong, Z. (2021). Simon says: evaluating and mitigating bias in
     pruned neural networks with knowledge distillation. arXiv:2106.07849
da Silva, I. W. D., Pereira, E., Barboza, E. de A., dos S. Neto, B. F., & Ribeiro, M. de M. (2025).
     Evaluating the impact of compression techniques on the robustness of CNNs under
     natural corruptions. ICMLA 2025. DOI: 10.1109/ICMLA66185.2025.00055
de Paula, E., Soni, J., Upadhyay, H., & Lagos, L. (2025). Comparative analysis of model
    compression techniques for achieving carbon efficient AI. Scientific Reports, 15, 23461.
    DOI: 10.1038/s41598-025-07821-w
Deutel, M., Woller, P., & Mutschler, C. (2022). Deployment of energy-efficient deep learning
    models on Cortex-M based microcontrollers using deep compression. arXiv:2205.10369
Diffenderfer, J., Bartoldson, B., Chaganti, S., et al. (2021). A winning hand: compressing deep
     networks can improve out-of-distribution robustness. arXiv:2106.09129
Fischer, R. (2025). Ground-truthing AI energy consumption: validating CodeCarbon against
     external measurements. it – Information Technology. DOI: 10.1515/itit-2025-0031;
     arXiv:2509.22092
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural
    networks. ICML.
Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common
    corruptions and perturbations. ICLR. arXiv:1903.12261
Hong, J., Duan, J., Zhang, C., et al. (2024). Decoding compressed trust: scrutinizing the
    trustworthiness of efficient LLMs under compression. ICML, PMLR 235. arXiv:2403.15447
Hooker, S., Courville, A., Clark, G., Dauphin, Y., & Frome, A. (2019). What do compressed deep
    neural networks forget? arXiv:1911.05248
Hooker, S., Moorosi, N., Clark, G., Bengio, S., & Denton, E. (2020). Characterising bias in
    compressed models. arXiv:2010.03058
Huber, P., Göhner, U., & Trapp, M. (2025). Comprehensive analysis of neural network inference
    on embedded systems: response time, calibration, and model optimisation. Sensors,
    25(15), 4769. DOI: 10.3390/s25154769
Kamal, M., & Talbert, D. (2025). Downsized and compromised? Assessing the faithfulness of
   model compression. arXiv:2510.06125
Kocher, N., Wassermann, C., Hennig, L., Seng, J., Hoos, H. H., Kersting, K., Lindauer, M., &
    Müller, M. (2025). Guidelines for the quality assessment of energy-aware NAS
    benchmarks. arXiv:2505.15631
Mitra, P., Schwalbe, G., & Klein, N. (2024). Investigating calibration and corruption robustness
     of post-hoc pruned perception CNNs: an image classification benchmark study. CVPR
     Workshops, 3542–3552. DOI: 10.1109/CVPRW63382.2024.00358; arXiv:2405.20876
Paganini, M. (2020). Prune responsibly. arXiv preprint.
Pachón, C. G., Pedraza, C., & Ballesteros, D. (2025). PruneEnergyAnalyzer: an open-source tool
    for measuring energy consumption in pruned neural networks. Big Data and Cognitive
    Computing, 9(8), 200. DOI: 10.3390/bdcc9080200
Rodriguez, C., Degioanni, L., Kameni, L., Vidal, R., & Neglia, G. (2024). Evaluating the energy
    consumption of machine learning: systematic literature review and experiments.
    arXiv:2408.15128
Rojahn, M., & Grum, M. (2025). Green AI: a systematic review and meta-analysis of its
    definitions, lifecycle models, hardware and measurement attempts. arXiv:2511.07090
Shen, L., Edalati, A., Meyer, B. H., Gross, W. J., & Clark, J. J. (2024). Robustness to distribution
    shifts of compressed networks for edge devices. arXiv:2401.12014
Tran, C., Fioretto, F., Kim, J.-E., & Naidu, R. (2022). Pruning has a disparate impact on model
     accuracy. NeurIPS 2022. arXiv:2205.13574
Tripp, C., Perr-Sauer, J., Gafur, J., Nag, A., Purkayastha, A., Zisman, S., & Bensen, E. (2024).
     Measuring the energy consumption and efficiency of deep neural networks: an empirical
     analysis and design recommendations (BUTTER-E). arXiv:2403.08151; dataset DOI:
     10.25984/2329316
von Rad, J., Cao, Y., & Geiger, A. (2026). UniComp: a unified evaluation of large language model
    compression via pruning, quantization and distillation. EMNLP 2026. arXiv:2602.09130
Williams, M., & Aletras, N. (2023). How does calibration data affect the post-training pruning
      and quantization of large language models? arXiv:2311.09755
Yuan, Z., Liu, J., Wu, J., Yang, D., Wu, Q., Sun, G., et al. (2023). Benchmarking the reliability of
    post-training quantization: a particular focus on worst-case performance.
    arXiv:2303.13003
