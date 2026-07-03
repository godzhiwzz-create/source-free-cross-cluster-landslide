# Protocol provenance

This file records the AutoDL artifacts used to align the public implementation.
No dataset copy, checkpoint, credential, or machine address is included.

Audit date: 2026-07-03.

## Source hashes

| Artifact | SHA-256 |
| --- | --- |
| `train_ld_repro.py` | `d8f9db577fa6812e0a5392e0cfbb474afd133151b5914d53c1ebda2a1ba1dbd8` |
| `preprocess_mm.py` | `bbb9de9bae663360c4d9915dd7f0b5b066ee65c01c1d893202f8d582a17dff79` |
| `lazy_dataset_mm.py` | `b3dcea45428df7b25211302400d7b2e311e5ed3a0f71356baea73207f90bc94f` |
| `adapt_kcurve_unbiased.py` | `734a41140bdcf638e4c4da9cd94eae4c6a31a12648e03a79aa09449a3bcee0dd` |
| `diag_revision_grids.py` | `878cdc90bdfc15797a2229727a0842479580e6a71d4fd495e2a0cf22d6d1d00a` |
| `diag_recipe2.py` | `86a0d3ed942bd507c11b5720bc4ae51abe25b1097a0fa5d5bae3f96d55dead25` |
| `diag_component_chain.py` | `922ccb9fa6cf3538777d815b8aedbdb4da53bacdaf400503e7ea07e7821e6b3a` |
| `reval_unbiased.py` | `957adec7e24ce3a5048f412e728eeb34612ff5786825e2e7afd4248c70bc3489` |

## Result hashes

| Artifact | SHA-256 |
| --- | --- |
| `diag_recipe2_metrics.json` | `05b23c65567fa6cdeea797ccb38a599159cbf74536a75d37b5654502ad8272d3` |
| `diag_revision_grids_metrics.json` | `9f7a14bf6fe56b0847b3d8ee99650e37d9d2c0156834a1366b12f2d2e4d3f347` |
| `reval_unbiased.json` | `caa7b21cb323146e58d5acfdcf116d758ec29c480560c4eed4350e5d70d117bd` |
| `diag_oracle_last.json` | `8ba4076b65bedcc3cf72eaf8f8e921abb1141f3b0a48fd93ed6c16598d72d547` |

## Verified implementation details

- Source checkpoints are fixed-budget epoch-75 `last.pt` checkpoints.
- Source seeds are `42`, `123`, and `777`.
- The end-to-end recipe uses random K50 support, 20 full-fine-tuning steps,
  and five-fold out-of-fold threshold estimation.
- Recipe draw `d` uses support RNG seed `2000+d`, cross-fit permutation seed
  `400+d`, auxiliary-model seeds `10d+i`, and final-model seed `d`.
- The budget grid's positive-aware sampler preserves the target pool's
  positive/negative patch ratio and forces at least one positive patch.
- Historical component tables use SciPy-default 4-connectivity and independent
  bidirectional overlap tests at IoU greater than `0.3`.
- The separate strict component-chain diagnostic uses greedy one-to-one
  matching; the public API additionally makes its intended 8-connectivity
  explicit.

## Paper data inventory

| Cluster | Patches | Regions |
| --- | ---: | --- |
| Africa | 1,133 | chimanimani |
| Americas | 1,368 | dominicamaria, usa_alaska, usa_puertorico |
| CentralAsia | 2,146 | china, kyrgyzstan1, kyrgyzstan2, nepal |
| Europe | 5,321 | italy |
| Oceania | 1,163 | newzealand |
| SoutheastAsia | 2,450 | hiroshima, hokkaido, indonesia, itogon, lanaodelnorte, thrissur |

The six-cluster grouping is study-defined and is versioned in
`configs/region_to_cluster.json`.
