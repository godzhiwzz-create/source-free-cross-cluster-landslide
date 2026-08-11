# Source-Free Cross-Cluster Landslide Segmentation

Paper-aligned code for a Scientific Reports study of cross-cluster landslide
segmentation diagnosis under limited target annotations and unavailable source
imagery during adaptation.

> *Diagnosing Landslide Segmentation Across Geographic Clusters with Limited
> Target Annotations and No Access to Source Imagery*

The repository implements the auditable Sen12Landslides experiment chain used
in the manuscript:

1. train a fixed-budget source model on five geographic clusters;
2. diagnose whether the transfer failure can be repaired by changing only the
   decision threshold;
3. compare full, BN-clean decoder-plus-head, historical decoder,
   segmentation-head, and BatchNorm-only adaptation;
4. sweep target-label and optimization budgets; and
5. evaluate both pixels and patch-level landslide components.

Source imagery is not required during adaptation. Only the source checkpoint
and the labeled target support set are used.

## Paper protocol

- **Data:** six Sen12Landslides clusters in leave-one-cluster-out evaluation.
- **Input:** 15 time steps, 10 Sentinel-2 bands plus DEM (11 channels). SAR and
  SCL channels are excluded.
- **Model:** 3D U-Net, 6.16M parameters, BatchNorm3d, LeakyReLU, temporal
  adaptive pooling.
- **Source training:** AdamW, learning rate `1e-3`, weight decay `1e-2`,
  `BCE(pos_weight=5) + 0.5 * Dice`, batch size 16, 75 epochs.
- **Adaptation default:** full fine-tuning, learning rate `1e-4`, weight decay
  `1e-4`, batch size 8, a small fixed optimization-step budget.
- **Evaluation:** global pixel F1. The historical paper tables use the exact
  AutoDL implementation: SciPy-default 4-connectivity and independent
  target/prediction overlap tests at component IoU greater than `0.3`. A
  stricter 8-connected, greedy one-to-one audit protocol is also implemented.

The revision reports three completed source-training chains separately. On
the same K50-support-excluded query sets, the six-cluster mean source/oracle/
BN-clean-decoder/full pixel F1 values are `0.181/0.221/0.199/0.260` for seed 42
and `0.199/0.256/0.220/0.289` for seed 123, and
`0.180/0.212/0.203/0.270` for seed 777. Each
value averages three support-redraw-specific query results within cluster and
then weights the six clusters equally. The query-label oracle is a
non-deployable diagnostic ceiling. All three reported chains use complete
six-fold source training at the fixed epoch-75 endpoint.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,data]"
```

## Data preparation

Download the harmonized release from the
[official dataset repository](https://huggingface.co/datasets/paulhoehn/Sen12Landslides)
and extract the Sentinel-2 and optional ascending Sentinel-1 archives. Convert
the official NetCDF files to the experiment's NPZ layout:

```bash
python scripts/preprocess_sen12.py \
  --s2-root /path/to/data_harmonized/s2 \
  --s1asc-root /path/to/data_harmonized/s1asc \
  --output-dir /path/to/preprocessed_npz
```

The preprocessing output contains:

```text
X: (N, 15, 14, 128, 128)
Y: (N, 128, 128)
M_mod: (N, 15, 3)    # optional
```

The 14 input channels are ordered as 10 Sentinel-2 bands, 2 Sentinel-1 bands,
DEM, and SCL. Convert the archives to the memory-mapped layout:

```bash
python scripts/convert_to_memmap.py \
  --input-dir /path/to/preprocessed_npz \
  --output-dir /path/to/sen12_memmap
```

See [docs/DATA.md](docs/DATA.md) for the exact file contract. No local path,
dataset copy, model weight, or private training artifact is included here.

## Reproduction

Train the fixed-budget source model for one held-out cluster:

```bash
python scripts/train_source.py \
  --data-root /path/to/sen12_memmap \
  --held Africa \
  --output-dir outputs \
  --seed 42
```

The paper uses `last.pt`, the checkpoint after the fixed 75-epoch budget. It
does not select a source checkpoint using held-out test performance.

Run the threshold diagnosis:

```bash
python scripts/run_threshold_probe.py \
  --data-root /path/to/sen12_memmap \
  --held Africa \
  --checkpoint outputs/source_Africa_seed42/last.pt \
  --source-seed 42 \
  --output results/Africa_threshold_seed42.json
```

Run the query-label-free K50 repair. The threshold is estimated from
out-of-fold support predictions and never uses target test labels:

```bash
python scripts/run_adaptation.py \
  --data-root /path/to/sen12_memmap \
  --held Africa \
  --checkpoint outputs/source_Africa_seed42/last.pt \
  --source-seed 42 \
  --support-size 50 \
  --steps 20 \
  --adapt-mode full \
  --support-sampling random \
  --threshold-mode cross-fit \
  --support-draw 0 \
  --output results/Africa_k50.json
```

Use `--adapt-mode decoder-clean` for the revision's decisive decoder-plus-head
comparison: encoder-like weights and their BatchNorm buffers remain fixed.
The `decoder`, `head`, and `bn` modes reproduce the historical training-state
behaviour and are explicitly labelled as such in JSON metadata. Use
`--support-sampling stratified-prevalence --threshold-mode fixed` for the
primary fixed-threshold budget grid. Replacing `fixed` with `support` generates
the separate support-selected-threshold sensitivity results. The legacy name
`positive-aware` is retained only as an exact command-line alias. Add
`--component-protocol strict-one-to-one-8` for the stricter component audit;
the default `paper-overlap-4` reproduces the historical table implementation.

For an atomic paired source/oracle/BN-clean-decoder/full cell on one support
draw, use `scripts/run_bn_clean_comparison.py`. It evaluates all conditions on
the identical support-excluded query set, verifies the frozen encoder weights
and BatchNorm buffers tensor by tensor, and records checkpoint/code hashes.

Run all six folds with the shell loop in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The JSON outputs include
the support indices, threshold, checkpoint metadata, pixel metrics, and
component metrics required to audit each run.

After all revision cells complete, `scripts/aggregate_revision_results.py`
validates the expected seed, cluster, redraw, epoch, sampler, trainable-scope,
threshold rule, and frozen-state fields before generating the paper-facing
aggregate. Exact commands are provided in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

AutoDL source/result hashes, RNG streams, cluster counts, and protocol
provenance are recorded in
[docs/PROTOCOL_PROVENANCE.md](docs/PROTOCOL_PROVENANCE.md).

The revision also includes two source-free, target-unlabelled transductive
controls. `scripts/make_query_manifest.py` converts a recorded K50 support split
into a label-free query-index manifest, and
`scripts/run_source_free_control.py` runs either one-pass target entropy
minimization or exact class-balanced pseudo-label self-training. The training
dataset opens only the input memmap; target labels are opened after adaptation
solely to report fixed-threshold metrics. See the exact commands and access
classification in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

The separate CAS runner reports the complete finite candidate set used in the
manuscript: source, query-label oracle, threshold-only, historical decoder,
full-fixed, and full-recipe. The query-label oracle is retained only as a
non-deployable diagnostic ceiling.

## Tests

```bash
python -m pytest
python -m compileall src scripts
```

## Scope

This release reproduces the principal Sen12Landslides diagnosis and repair
pipeline. The manuscript's CAS experiment is exposed as a separate
leave-one-event-out runner with an RGB data contract. The Prithvi experiment
depends on external pretrained weights and band projection. These boundaries
are documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md); CAS has a
public runner, whereas the external Prithvi assets are not shipped here.

## License

Code is released under the MIT License. Dataset and pretrained-model assets
retain their original licenses.

## Citation

Release metadata are provided in [`CITATION.cff`](CITATION.cff). Version
`1.0.0` is the paper-aligned archival release; the version-specific Zenodo DOI
is added to the manuscript after the GitHub release has been archived.
