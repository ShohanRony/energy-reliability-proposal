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
