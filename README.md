# Hardware-Validated Energy-Aware Reliability Evaluation of Compressed Neural Networks

Master's research proposal (two-year programme). Adds a reliability/calibration axis
on top of the hardware-energy-measurement argument: compression is judged not just on
whether FLOPs-based savings hold up under hardware measurement, but on whether a
compressed model that keeps aggregate accuracy also keeps the calibration and
robustness properties deployment depends on — and whether any reliability gain is
purchased at a measured energy cost that FLOPs- or latency-based reporting conceals.

## Files

- `Shohan_Masters_Proposal.pdf` — canonical version. Edit the source document and
  re-export this one. (Supersedes the earlier `Shohan_Masters_Proposal_Final_v2.docx`,
  which this revision replaces outright rather than incrementing — the content is a
  substantial rewrite, not a version bump.)
- `proposal.md` — plain-text/markdown extract of the PDF, auto-generated for
  readability and diffing across revisions. Not authoritative; regenerate from the
  PDF rather than hand-editing when the PDF changes. Tables are kept as fenced code
  blocks (verbatim from the PDF's text layer) rather than reformatted as markdown
  tables, to avoid transcription errors in the numeric results they contain.

## Structure (from the proposal)

Four research questions — RQ1 instrument agreement (software energy estimators vs.
hardware counters), RQ2 the recovery-training confound (Track A), RQ3 matched-energy
selection risk (Track B), and RQ4 (Phase 2) physical/cross-platform generality — with one
supporting hypothesis (H1) and one pre-specified primary hypothesis per Phase-1 track
(H2, H3), tested via TOST equivalence rather than simple significance. Phase 1 (committed
thesis deliverable) runs two tracks on hardware already in hand: Track A (months 1–6) tests
whether the calibration effect found in the §2.3 pilot survives controlling for the pruned
model's extra recovery-training epochs; Track B (months 4–12) tests, among real
structurally-pruned/quantized configurations shown TOST-equivalent in measured energy,
whether picking the most accurate one carries a hidden calibration cost. Phase 2 (months
13–24, resource- and supervision-dependent) is one bounded extension selected with
supervisor input, not a broad programme.

## Relationship to other repos/documents

- The literature review "Energy Measurement of Compressed Deep Learning Models"
  (PDF, not in this repo) covers the energy-measurement half of this proposal's
  argument in full — the tool-validation, FLOPs-energy, and cross-platform gaps this
  proposal's RQ1 (instrument agreement) and Track B / RQ3 (matched-energy selection
  risk) draw on.
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
pilot (with a real NVML polling-interval artifact found and corrected), a 3-seed
VGG-19-BN reconstruction of Mitra et al. (2024) with full CIFAR-10-C corruption
evaluation and a cross-seed mean±SD/significance analysis, and measured per-epoch
training costs for ResNet-18 and MobileNetV3-Small used to check the proposal's
10-seed grid against a normal thesis timeline. See
[`validation/README.md`](validation/README.md) for the full index and key findings —
this closes the minimum-detectable-energy-difference pilot item at pilot scale (not
yet the 30-independent-pair confirmation the harness itself requires before treating
any cell as a validated candidate) and produces the first real, multi-seed numbers
behind the Mitra et al. reading item above (no detectable accuracy effect from 30%
pruning + recovery; weak but 3-seed-consistent evidence of calibration degradation).
