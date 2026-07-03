# Data contract

The repository does not redistribute Sen12Landslides. Obtain the CC BY 4.0
harmonized data from the
[official Hugging Face release](https://huggingface.co/datasets/paulhoehn/Sen12Landslides)
and cite the dataset paper. The
[official code repository](https://github.com/PaulH97/Sen12Landslides)
documents the original NetCDF format.

## Official NetCDF to NPZ

The paper runs used the harmonized release. Generate the intermediate NPZ
files with:

```bash
python scripts/preprocess_sen12.py \
  --s2-root /path/to/data_harmonized/s2 \
  --s1asc-root /path/to/data_harmonized/s1asc \
  --output-dir /path/to/preprocessed_npz
```

The Sentinel-1 argument is optional because the paper model excludes SAR.
Preprocessing follows the AutoDL source: Sentinel-2 is divided by `10000`, DEM
by `1000`, ascending Sentinel-1 is clipped to `[-30, 0]` dB and scaled, and
SCL is converted to a clear-pixel indicator.

The six clusters are a study-defined grouping, not an official dataset split.
The exact mapping and paper patch counts are versioned in
[`configs/region_to_cluster.json`](../configs/region_to_cluster.json).

## Preprocessed NPZ input

`scripts/convert_to_memmap.py` accepts one or more NPZ archives per geographic
cluster. The default filename pattern is `<cluster>__*.npz`.

Each archive must contain:

| Key | Shape | Meaning |
| --- | --- | --- |
| `X` | `(N, T, 14, H, W)` | multi-temporal inputs |
| `Y` | `(N, H, W)` | binary landslide masks |
| `M_mod` | `(N, T, 3)` | optional modality availability |

The paper uses `T=15` and `H=W=128`. The channel order is:

| Index | Data |
| --- | --- |
| `0..9` | ten Sentinel-2 bands |
| `10..11` | Sentinel-1 VV/VH |
| `12` | DEM |
| `13` | SCL |

Only indices `0..9` and `12` enter the 3D U-Net. The conversion script rejects
inputs that do not contain 14 channels so that a silent band-order mismatch
does not contaminate the experiment.

## Memory-mapped output

For each of the six clusters, conversion produces:

```text
Africa.X.dat
Africa.Y.dat
Africa.M.dat
Africa.meta.json
...
```

`X` is stored as `float16`, `Y` and `M` as `uint8`. Metadata records the number
of patches and the per-patch shapes. All training and evaluation scripts take
this directory through `--data-root`; no machine-specific path is embedded in
the code.

## Leakage control

Normalization statistics are recomputed for every leave-one-cluster-out fold
from its five training clusters only. The held-out cluster contributes neither
images nor labels to normalization.

Source training uses the complete five-cluster patch pools. The optional
Landslide Detection benchmark filter, where a labeled patch contains more than
50 landslide pixels, is exposed by `index_cluster(..., labeled_only=True)` for
in-distribution checks but is not applied to the paper's LOCO source models.

## Support/query split

Every adaptation JSON records the exact support indices. Query evaluation
excludes those support patches.

- `random`: uniform target candidates, used by the deployable K50 recipe.
- `positive-aware`: stratified sampling that preserves the target pool's
  positive/negative patch ratio while forcing at least one positive patch,
  matching the AutoDL budget-grid script.

The support labels are the only target labels available to deployable
adaptation and threshold estimation.
