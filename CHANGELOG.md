# Changelog

## 1.0.0 — 2026-08-11

Paper-aligned archival release for the Scientific Reports major revision.

- Adds an explicit BN-clean decoder-plus-head mode that freezes both encoder-like weights and BatchNorm running state.
- Adds paired source/oracle/BN-clean/full execution and revision-result aggregation with provenance validation.
- Adds target-unlabelled entropy and exact class-balanced pseudo-label controls using an input-only target dataset during adaptation.
- Adds deterministic query manifests and a five-event CAS directional-check runner.
- Distinguishes prevalence-preserving fixed-grid sampling from uniformly random cross-fitted support sampling.
- Records the three completed source chains, support-redraw hierarchy, component protocols, and paper-facing summary values.
- Extends tests for frozen-state invariants, sampling aliases, input-only target access, source-free controls, and revision aggregation.

This release contains code and compact paper-facing summaries only. It does not redistribute datasets, checkpoints, private result artifacts, credentials, or machine-specific paths.
