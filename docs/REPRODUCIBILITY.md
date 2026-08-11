# Reproducibility notes

## Six-fold source training

The source checkpoint used by the paper is the final checkpoint after a fixed
75-epoch budget. Do not select a checkpoint by held-out test F1.

```bash
for held in Africa Americas CentralAsia Europe Oceania SoutheastAsia; do
  python scripts/train_source.py \
    --data-root /path/to/sen12_memmap \
    --held "$held" \
    --output-dir outputs \
    --seed 42
done
```

Repeat with seeds `123` and `777`. Every reported chain contains all six folds
at the fixed epoch-75 endpoint under the same completion criteria.

## Threshold diagnosis

```bash
for held in Africa Americas CentralAsia Europe Oceania SoutheastAsia; do
  python scripts/run_threshold_probe.py \
    --data-root /path/to/sen12_memmap \
    --held "$held" \
    --checkpoint "outputs/source_${held}_seed42/last.pt" \
    --source-seed 42 \
    --support-size 50 \
    --support-seed 0 \
    --output "results/${held}_threshold.json"
done
```

The `oracle` field deliberately uses query labels and is only an upper-bound
diagnostic. It is not a deployment result.

## Parameter-scope probe

Run K50 with a fixed threshold while changing only the trainable scope:

```bash
for mode in full decoder-clean decoder bn head; do
  python scripts/run_adaptation.py \
    --data-root /path/to/sen12_memmap \
    --held Africa \
    --checkpoint outputs/source_Africa_seed42/last.pt \
    --source-seed 42 \
    --support-size 50 \
    --steps 20 \
    --adapt-mode "$mode" \
    --support-sampling stratified-prevalence \
    --support-draw 0 \
    --threshold-mode fixed \
    --output "results/Africa_${mode}.json"
done
```

Repeat for every cluster, three support draws, and source seeds 42, 123, and
777. The
full mode updates exactly 6,161,793 parameters. Both decoder modes update
1,826,881 parameters in `dc4`, `trans3`, `dc3`, and `final`; `decoder-clean`
also keeps `en3`, `en4`, `center_in`, and `center_out` in evaluation mode so
their BatchNorm buffers remain fixed. Historical `decoder` retains the global
training-state behaviour. BN and head update approximately 0.003M and 0.001M
weight parameters, respectively, while historical global training mode can
still update BatchNorm buffers outside those weight scopes.

The revision's claim-defining paired cell can be generated directly:

```bash
python scripts/run_bn_clean_comparison.py \
  --data-root /path/to/sen12_memmap \
  --held Africa \
  --checkpoint outputs/source_Africa_seed42/last.pt \
  --source-seed 42 \
  --support-size 50 \
  --support-draw 0 \
  --steps 20 \
  --output results/Africa_seed42_draw0_bn_clean.json
```

Run this for draws 0--2, all six clusters, and all three completed source
seeds. The
query-label oracle in each JSON is a diagnostic ceiling, not a deployable row.

## Label/step budget grid

The manuscript's primary grid uses prevalence-stratified support sampling and
the fixed threshold 0.5. K is a tile count, not measured annotation time:

```bash
for k in 25 50 100; do
  for steps in 10 20 50; do
    python scripts/run_adaptation.py \
      --data-root /path/to/sen12_memmap \
      --held Africa \
      --checkpoint outputs/source_Africa_seed42/last.pt \
      --source-seed 42 \
      --support-size "$k" \
      --steps "$steps" \
      --adapt-mode full \
      --support-sampling stratified-prevalence \
      --support-draw 0 \
      --threshold-mode fixed \
      --output "results/Africa_k${k}_s${steps}_fixed.json"
  done
done
```

Repeat the grid with `--threshold-mode support` to generate the separately
reported support-selected-threshold sensitivity analysis. The cross-fitted
query-label-free recipe instead uses fully random support and five-fold
cross-fit threshold estimation. `--support-draw d` reproduces the AutoDL RNG streams:
support seed `2000+d`, fold seed `400+d`, auxiliary-model seeds `10d+i`, and
final-model seed `d`. Auxiliary cross-fit models never predict on samples used
to update their weights. The final model is initialized again from the source
checkpoint and adapted on all K samples.

## Target-unlabelled source-free controls

These controls are transductive: they adapt on the same unlabelled query images
that are evaluated after adaptation. They use no source images and no target
labels during optimization. To reproduce the revision comparison on the same
support-excluded query identities, first convert a completed K50 result into a
query manifest:

```bash
python scripts/make_query_manifest.py \
  --data-root /path/to/sen12_memmap \
  --held Africa \
  --support-result results/Africa_k50_draw0.json \
  --output results/Africa_query_draw0.json
```

Run one complete target-data pass for each control:

```bash
for method in target-entropy class-balanced-pseudo; do
  python scripts/run_source_free_control.py \
    --data-root /path/to/sen12_memmap \
    --held Africa \
    --checkpoint outputs/source_Africa_seed42/last.pt \
    --source-seed 42 \
    --method "$method" \
    --query-indices results/Africa_query_draw0.json \
    --seed 0 \
    --learning-rate 1e-4 \
    --weight-decay 1e-4 \
    --batch-size 8 \
    --output "results/Africa_${method}_seed42.json"
done
```

Both controls update the full network and evaluate the final state at threshold
0.5. Entropy minimization uses every query pixel. The pseudo-label control
freezes the source predictions before adaptation, predicts classes at 0.5, and
retains exactly `ceil(0.2 * n_class_pixels)` highest-confidence pixels within
each predicted class. Static pseudo labels and selection masks receive the same
flips and rotations as their input tile. No teacher, mask, threshold, or pseudo
label is refreshed after optimization starts.

The controls are unequal-supervision context for the labelled-support K50
experiment and must not be reported as an unqualified head-to-head ranking.

## Revision aggregation

Store the 54 BN-clean cells, the 648 K25/K50/K100 budget cells spanning 10, 20,
and 50 updates under both fixed and support-selected thresholds, and the 36
target-unlabelled control cells in separate directories. The primary manuscript
table uses the fixed-threshold rows; support-selected rows are a sensitivity
analysis. The budget grid remains the frozen two-chain seed-42/123 campaign;
seed 777 was added only to the claim-defining BN-clean comparison and the
source-free controls.
Regenerate
the revision summary with validation of source seed, cluster, support redraw,
epoch, sampler, trainable scope, frozen-state invariant, and expected cell
counts:

```bash
python scripts/aggregate_revision_results.py \
  --bn-clean-dir results/bn_clean \
  --budget-dir results/budget_complete \
  --source-free-dir results/source_free_controls \
  --output results/revision_aggregate.json
```

The aggregator keeps source-training seeds separate, averages the three support
redraws within each cluster, and only then computes an unweighted arithmetic
mean across the six clusters.

## Metrics

Pixel scores aggregate TP, FP, FN, and TN over the complete query pool before
computing binary metrics.

The result-generating AutoDL scripts used SciPy's default 4-connectivity and
independent overlap tests: each ground-truth component is counted as detected
when any predicted component has IoU greater than `0.3`, and predicted
components are assessed independently for precision. This is exposed as
`--component-protocol paper-overlap-4` and is the default so the repository can
regenerate the historical tables.

For a stricter audit, use `--component-protocol strict-one-to-one-8`. It applies
8-connectivity and greedy one-to-one matching in descending IoU order. These
two conventions must not be mixed when comparing component values.

## External boundary checks

The single-run CAS check freezes the historical candidate sequence but changes
the I/O adapter: RGB single-date images are resized to 128 pixels and broadcast
to 15 temporal frames. It trains each leave-one-event-out source model for 30
epochs with seed 0, draws K50 support, excludes support from the query, and runs
20 adaptation updates. The decoder row intentionally reproduces the historical
global-training-mode BatchNorm policy and is not the BN-clean Sen12 condition.

```bash
python scripts/run_cas_directional_check.py \
  --data-root /path/to/cas_root \
  --source-epochs 30 \
  --support-size 50 \
  --adaptation-steps 20 \
  --seed 0 \
  --checkpoint-dir outputs/cas_source \
  --output results/cas_directional_check.json
```

Use repeated `--event NAME=relative_directory` arguments if the event directory
names differ from the historical defaults. The output records absolute scores,
train/held/support/query counts, support identities, checkpoint hashes, run
count, the query-label oracle designation, and finite-candidate regret. The
result is a directional cross-dataset check, not metric comparability with
Sen12Landslides or a universal strategy validation.

The Prithvi-EO check uses external pretrained weights and a multispectral band
projection that is not band-wise equivalent to the 11-channel 3D U-Net input.
Because those assets are not shipped here, the Prithvi-EO check is not
represented as an end-to-end command in this release.
