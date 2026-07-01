# Data contract

The repository does not redistribute Sen12Landslides. Obtain the data under
the terms of its official release and cite the dataset paper.

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
- `positive-aware`: candidates known to contain at least one positive pixel,
  used by the screened-candidate budget grid.

The support labels are the only target labels available to deployable
adaptation and threshold estimation.
