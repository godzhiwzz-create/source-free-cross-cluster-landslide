# Running the experiments

Prepare the [data](DATA.md) first. All commands run from the repository root.
Each script supports `--help`; result paths below are output files to be generated.
Use a distinct output filename for every cluster, source seed, support draw, and condition.

## Source training

```bash
for held in Africa Americas CentralAsia Europe Oceania SoutheastAsia; do
  python scripts/train_source.py --data-root /path/to/sen12_memmap \
    --held "$held" --output-dir outputs --seed 42
done
```

Repeat for source seeds `123` and `777`. Use the final epoch-75 `last.pt`,
not a checkpoint selected by held-out performance.

## Threshold and parameter scope

```bash
python scripts/run_threshold_probe.py --data-root /path/to/sen12_memmap \
  --held Africa --checkpoint outputs/source_Africa_seed42/last.pt \
  --source-seed 42 --support-size 50 --support-seed 0 \
  --output results/Africa_threshold.json

python scripts/run_bn_clean_comparison.py --data-root /path/to/sen12_memmap \
  --held Africa --checkpoint outputs/source_Africa_seed42/last.pt \
  --source-seed 42 --support-size 50 --support-draw 0 --steps 20 \
  --output results/bn_clean/Africa_seed42_draw0.json
```

Run the paired comparison for all six clusters, three source seeds, and draws
`0..2` (54 cells). All conditions share the support-excluded query pool.
The query-label oracle is a diagnostic upper bound, not a deployable threshold.

Individual conditions use `scripts/run_adaptation.py --adapt-mode MODE`:

| Mode | Trainable weights | BatchNorm state |
| --- | --- | --- |
| `full` | Entire network (6,161,793 parameters) | Updates throughout |
| `decoder-clean` | `dc4`, `trans3`, `dc3`, `final` (1,826,881 parameters) | Encoder-like state frozen |
| `decoder` | Same weights as `decoder-clean` | Global training mode |
| `head` | `final` | Global training mode |
| `bn` | BatchNorm affine parameters | Global training mode |

The encoder-like modules are `en3`, `en4`, `center_in`, and `center_out`.
Global training mode can update BatchNorm buffers even where weights are frozen.

## Label and step budgets

```bash
for k in 25 50 100; do
  for steps in 10 20 50; do
    for threshold in fixed support; do
      python scripts/run_adaptation.py --data-root /path/to/sen12_memmap \
        --held Africa --checkpoint outputs/source_Africa_seed42/last.pt \
        --source-seed 42 --support-size "$k" --steps "$steps" \
        --adapt-mode full --support-sampling stratified-prevalence \
        --support-draw 0 --threshold-mode "$threshold" \
        --output "results/budget_complete/Africa_seed42_draw0_k${k}_s${steps}_${threshold}.json"
    done
  done
done
```

Repeat across six clusters, seeds `42/123`, and draws `0..2`. The primary grid
uses threshold `0.5`; the `support` mode gives the separate support-selected
threshold comparison (648 cells across both threshold modes). K counts tiles.
Support identities and query pools may differ across K.

The [K50 recipe](../README.md#run) uses random support and five-fold cross-fit
threshold estimation. `--support-draw d` uses the following RNG streams:
support `2000+d`, fold permutation `400+d`, auxiliary-model seeds `10d+i`, and
final-model seed `d`. Auxiliary models predict only on their held-out support
folds. The final model restarts from the source checkpoint and uses all K tiles.

## Target-unlabelled controls

Build the query manifest from an adaptation result with recorded support indices:

```bash
python scripts/make_query_manifest.py --data-root /path/to/sen12_memmap \
  --held Africa --support-result results/Africa_k50.json \
  --output results/Africa_query_draw0.json

for method in target-entropy class-balanced-pseudo; do
  python scripts/run_source_free_control.py --data-root /path/to/sen12_memmap \
    --held Africa --checkpoint outputs/source_Africa_seed42/last.pt \
    --source-seed 42 --method "$method" --query-indices results/Africa_query_draw0.json \
    --seed 0 --learning-rate 1e-4 --weight-decay 1e-4 --batch-size 8 \
    --output "results/source_free_controls/Africa_${method}_seed42.json"
done
```

Each control makes one full pass over the unlabelled query images, updates the
full network, and evaluates at threshold `0.5`. This transductive access differs
from labelled-support adaptation. Repeat for six clusters and three source seeds
(36 cells across the two methods).

Entropy minimization uses every query pixel. Class-balanced pseudo labels are
fixed before adaptation: classes use threshold `0.5`, and the highest-confidence
`ceil(0.2 * n_class_pixels)` pixels are retained within each predicted class.
Labels and selection masks receive the same flips/rotations as the input.
They are not refreshed during optimization; target labels are opened only for evaluation.

## Aggregate outputs

```bash
python scripts/aggregate_revision_results.py \
  --bn-clean-dir results/bn_clean --budget-dir results/budget_complete \
  --source-free-dir results/source_free_controls \
  --output results/aggregate.json
```

The aggregator validates the 54 paired, 648 budget, and 36 control cells,
including seeds, epochs, sampling, parameter scope, thresholds, and frozen states.
It keeps source seeds separate, averages support draws within cluster, then
weights the six clusters equally. The budget grid uses two source seeds;
the paired comparison and unlabelled controls use three.

## Metrics

Pixel metrics use TP, FP, FN, and TN summed over the complete query pool.
The paper component metric uses 4-connectivity and independent overlap tests
at IoU greater than `0.3`: targets and predictions are matched independently
for recall and precision. This is `--component-protocol paper-overlap-4` (default).

`--component-protocol strict-one-to-one-8` instead uses 8-connectivity and greedy
one-to-one matching in descending IoU order. Keep these two metric conventions separate.

## CAS

```bash
python scripts/run_cas_directional_check.py --data-root /path/to/cas_root \
  --source-epochs 30 --support-size 50 --adaptation-steps 20 --seed 0 \
  --checkpoint-dir outputs/cas_source --output results/cas_directional_check.json
```

Use repeated `--event NAME=relative_directory` arguments for other directory names.
The runner uses the [CAS input format](DATA.md#cas-input), trains each leave-one-event-out
source model, excludes support from query, and evaluates the fixed candidate set.
Its decoder condition uses global-training-mode BatchNorm, not `decoder-clean`.
Outputs include scores, sample counts, support identities, checkpoint hashes,
the oracle designation, and finite-candidate regret. The paper uses one seed;
this is a directional cross-dataset comparison.

Prithvi-EO weights and its multispectral band projection are external assets
and are not bundled as an end-to-end runnable experiment here.
