# Cross-cluster landslide segmentation

Code for **Diagnosing landslide segmentation across geographic clusters with
limited target annotations and no access to source imagery**, Scientific Reports (2026).

- [Paper](https://doi.org/10.1038/s41598-026-70729-6)
- [Archived code v1.0.0](https://doi.org/10.5281/zenodo.21899849)

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,data]"
```

## Data

Download [Sen12Landslides](https://huggingface.co/datasets/paulhoehn/Sen12Landslides)
and follow [data preparation](docs/DATA.md). Datasets and model weights are not bundled.

## Run

Train a source model with one geographic cluster held out:

```bash
python scripts/train_source.py --data-root /path/to/sen12_memmap \
  --held Africa --output-dir outputs --seed 42
```

Adapt its fixed epoch-75 checkpoint using K50 target support tiles:

```bash
python scripts/run_adaptation.py --data-root /path/to/sen12_memmap \
  --held Africa --checkpoint outputs/source_Africa_seed42/last.pt \
  --source-seed 42 --support-size 50 --steps 20 --adapt-mode full \
  --support-sampling random --threshold-mode cross-fit --support-draw 0 \
  --output results/Africa_k50.json
```

Source imagery is used for source training, not target adaptation. Support
tiles are excluded from query evaluation. See [running the experiments](docs/REPRODUCIBILITY.md)
for threshold and parameter-scope comparisons, budget grids, controls, CAS, and metrics.

## Test and licence

Run `python -m pytest`. Code is licensed under [MIT](LICENSE); data and external
model assets retain their own licences. Software citation metadata are in [CITATION.cff](CITATION.cff).
