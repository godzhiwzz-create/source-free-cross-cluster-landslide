# Source-Free Cross-Cluster Landslide Segmentation

Paper-aligned code for:

> **Source-Free Few-Shot Cross-Cluster Landslide Segmentation for Emergency
> Response: A Failure-Analysis Diagnosis and Methodological Study**

The repository implements the auditable Sen12Landslides experiment chain used
in the manuscript:

1. train a fixed-budget source model on five geographic clusters;
2. diagnose whether the transfer failure can be repaired by changing only the
   decision threshold;
3. compare full, decoder, segmentation-head, and BatchNorm-only adaptation;
4. sweep target-label and optimization budgets; and
5. evaluate both pixels and 8-connected landslide components.

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
- **Evaluation:** global pixel F1 and one-to-one component matching under
  8-connectivity with component IoU greater than `0.3`.

The manuscript reports a six-fold source-only pixel F1 of about `0.196`.
Peeking at test labels to choose a global oracle threshold adds only about
`0.040`, whereas the deployable K50 repair reaches about `0.306`. Component
recall rises from `0.120` to `0.205`. These values summarize the submitted
manuscript; exact reruns can vary slightly with software and hardware.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Data preparation

Download Sen12Landslides from its official distribution and run its
preprocessing so that each archive contains:

```text
X: (N, 15, 14, 128, 128)
Y: (N, 128, 128)
M_mod: (N, 15, 3)    # optional
```

The 14 input channels must be ordered as 10 Sentinel-2 bands, 2 Sentinel-1
bands, DEM, and SCL. Convert the archives to the memory-mapped layout:

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
  --checkpoint outputs/source_Africa_seed42/last.pt
```

Run the deployable K50 repair. The threshold is estimated from out-of-fold
support predictions and never uses target test labels:

```bash
python scripts/run_adaptation.py \
  --data-root /path/to/sen12_memmap \
  --held Africa \
  --checkpoint outputs/source_Africa_seed42/last.pt \
  --support-size 50 \
  --steps 20 \
  --adapt-mode full \
  --support-sampling random \
  --threshold-mode cross-fit \
  --output results/Africa_k50.json
```

Use `--adapt-mode decoder`, `head`, or `bn` for the parameter-scope probe.
Use `--support-sampling positive-aware --threshold-mode support` to reproduce
the screened-candidate budget-grid convention described in the paper.

Run all six folds with the shell loop in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The JSON outputs include
the support indices, threshold, checkpoint metadata, pixel metrics, and
component metrics required to audit each run.

## Tests

```bash
python -m pytest
python -m compileall src scripts
```

## Scope

This release reproduces the principal Sen12Landslides diagnosis and repair
pipeline. The manuscript's CAS experiment is a cross-dataset boundary check
with a separate RGB data contract. The Prithvi experiment depends on external
pretrained weights and band projection. They are documented in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) but are not presented as
one-command reproductions in this repository.

## License

Code is released under the MIT License. Dataset and pretrained-model assets
retain their original licenses.
