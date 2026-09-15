# Data preparation

Obtain the harmonized data from the [official Sen12Landslides release](https://huggingface.co/datasets/paulhoehn/Sen12Landslides).
The dataset is not redistributed here. Its [source repository](https://github.com/PaulH97/Sen12Landslides)
documents the original NetCDF format.

## Convert NetCDF to NPZ

```bash
python scripts/preprocess_sen12.py \
  --s2-root /path/to/data_harmonized/s2 \
  --s1asc-root /path/to/data_harmonized/s1asc \
  --output-dir /path/to/preprocessed_npz
```

The Sentinel-1 argument is optional; the model excludes SAR. Preprocessing applies
the following scaling: Sentinel-2 is divided by `10000`, DEM by `1000`, and ascending
Sentinel-1 is clipped to `[-30, 0]` dB and scaled. SCL becomes a clear-pixel indicator.
The study-defined geographic grouping is in [region_to_cluster.json](../configs/region_to_cluster.json).

## Convert NPZ to memory maps

```bash
python scripts/convert_to_memmap.py \
  --input-dir /path/to/preprocessed_npz \
  --output-dir /path/to/sen12_memmap
```

The default input pattern is `<cluster>__*.npz`. Each archive contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `X` | `(N, T, 14, H, W)` | Inputs |
| `Y` | `(N, H, W)` | Binary masks |
| `M_mod` | `(N, T, 3)` | Optional modality availability |

Use `T=15`, `H=W=128`. Channels `0..9` are Sentinel-2, `10..11` are Sentinel-1
VV/VH, `12` is DEM, and `13` is SCL. Only `0..9` and `12` enter the model;
the converter requires all 14 channels to preserve the channel ordering.

Each cluster produces `Africa.X.dat`, `Africa.Y.dat`, `Africa.M.dat`, and
`Africa.meta.json` (with the corresponding cluster name). `X` uses `float16`;
`Y` and `M` use `uint8`. Pass this directory as `--data-root`.

## Data selection

- Normalization statistics use only the five source clusters in each fold.
- Source training uses the complete five-cluster patch pools. The optional
  `index_cluster(..., labeled_only=True)` filter is not used for LOCO training.
- Adaptation JSON files record support indices; these tiles are excluded from query evaluation.
- `random` samples target tiles uniformly and is used for the cross-fitted K50 recipe.
- `stratified-prevalence` samples positive and negative tiles in proportion to the
  target pool while forcing at least one positive tile. This is the budget-grid
  sampling rule, not a measured analyst-screening workflow. `positive-aware` is
  an exact CLI alias. K counts tiles, not annotation time.
- Only support labels are used for query-label-free adaptation and threshold estimation.

## CAS input

Install `pip install -e '.[cas]'`. The CAS runner searches each event directory
recursively for paired TIFF files in sibling `img/` and `mask/` directories:

```text
cas_root/
  palu_x/.../img/example.tif
  palu_x/.../mask/example.tif
  Lombok_x/.../img/example.tif
  Lombok_x/.../mask/example.tif
```

RGB images are resized to `128 x 128` and broadcast to 15 temporal frames.
This differs from the 11-channel Sen12 input; absolute scores are not directly comparable.
