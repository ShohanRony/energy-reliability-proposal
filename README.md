# Hardware-Validated Energy-Aware Reliability Evaluation of Compressed Neural Networks

Master's research proposal (two-year programme). Adds a reliability/calibration axis
on top of the hardware-energy-measurement argument: compression is judged not just on
whether FLOPs-based savings hold up under hardware measurement, but on whether a
compressed model that keeps aggregate accuracy also keeps the calibration and
robustness properties deployment depends on — and whether any reliability gain is
purchased at a measured energy cost that FLOPs- or latency-based reporting conceals.

## Files

- `Shohan_Masters_Proposal_Final_v2.docx` — canonical version. Edit this one.
- `proposal.md` — plain-text/markdown extract of the docx, auto-generated for
  readability and diffing across revisions. Not authoritative; regenerate from the
  docx rather than hand-editing when the docx changes.

## Structure (from the docx)

Three research questions (instrument validity, reliability under compression, ranking
stability under matched measured energy) plus a Phase 2 generality question. Phase 1
(months 1–12, committed thesis deliverable) runs three work packages — estimator
validation (WP1), disaggregated reliability sweep (WP2), energy-banded ranking
comparison (WP3) — on hardware already in hand. Phase 2 (months 13–24, resource- and
supervision-dependent) adds six further work packages: language-model generalisation,
physical power ground truth, hardware heterogeneity, scale, real-subgroup fairness
data, and methodological formalisation.

## Relationship to other repos/documents

- The literature review "Energy Measurement of Compressed Deep Learning Models"
  (PDF, not in this repo) covers the energy-measurement half of this proposal's
  argument in full — the tool-validation, FLOPs-energy, and cross-platform gaps this
  proposal's WP1 and WP3 draw on.
- [`green-ai-proposal`](https://github.com/ShohanRony/green-ai-proposal) is a separate,
  earlier proposal-variant build system targeting six programme types on the
  energy-only argument (no reliability/calibration axis). This document is a distinct,
  more advanced proposal — check with Shohan before treating one as superseding the
  other if both are being used for active applications.

## Status

Appendix A of the proposal lists outstanding pre-submission verification items
(reading Mitra et al. 2024 in full, completing the minimum-detectable-energy-difference
pilot, confirming a couple of reference DOIs/venues). Check those are closed before
this is sent anywhere.

**[`validation/`](validation/)** contains real, hardware-executed work against these
items, run on the target laptop (i5-13450HX / RTX 3050 6GB / 24GB RAM): a full
environment/execution-feasibility preflight, an initial run of the energy-measurement
pilot (with a real NVML polling-interval artifact found and corrected), a complete
VGG-19-BN reconstruction of Mitra et al. (2024) including CIFAR-10-C corruption
evaluation, and measured per-epoch training costs for ResNet-18 and MobileNetV3-Small
used to check the proposal's 10-seed grid against a normal thesis timeline. See
[`validation/README.md`](validation/README.md) for the full index and key findings —
this closes the minimum-detectable-energy-difference pilot item at pilot scale (not
yet the 30-independent-pair confirmation the harness itself requires before treating
any cell as a validated candidate) and produces the first real numbers behind Mitra
et al. reading item above.
