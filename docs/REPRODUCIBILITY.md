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

For the three-source-seed estimate, repeat with seeds `43` and `44`.

## Threshold diagnosis

```bash
for held in Africa Americas CentralAsia Europe Oceania SoutheastAsia; do
  python scripts/run_threshold_probe.py \
    --data-root /path/to/sen12_memmap \
    --held "$held" \
    --checkpoint "outputs/source_${held}_seed42/last.pt" \
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
for mode in full decoder bn head; do
  python scripts/run_adaptation.py \
    --data-root /path/to/sen12_memmap \
    --held Africa \
    --checkpoint outputs/source_Africa_seed42/last.pt \
    --support-size 50 \
    --steps 20 \
    --adapt-mode "$mode" \
    --support-sampling positive-aware \
    --threshold-mode fixed \
    --output "results/Africa_${mode}.json"
done
```

Repeat for every cluster and support seed. The full mode updates about 6.16M
parameters; decoder, BN, and head modes update about 1.83M, 0.003M, and 0.001M.

## Label/step budget grid

The manuscript's screened-candidate grid uses positive-aware support sampling
and a support-set threshold:

```bash
for k in 25 50 100; do
  for steps in 10 20 50; do
    python scripts/run_adaptation.py \
      --data-root /path/to/sen12_memmap \
      --held Africa \
      --checkpoint outputs/source_Africa_seed42/last.pt \
      --support-size "$k" \
      --steps "$steps" \
      --adapt-mode full \
      --support-sampling positive-aware \
      --threshold-mode support \
      --output "results/Africa_k${k}_s${steps}.json"
  done
done
```

The deployable recipe instead uses fully random support and five-fold cross-fit
threshold estimation. Auxiliary cross-fit models never predict on samples used
to update their weights. The final model is then initialized again from the
source checkpoint and adapted on all K samples.

## Metrics

Pixel scores aggregate TP, FP, FN, and TN over the complete query pool before
computing binary metrics. Connected components are extracted patch by patch
with 8-connectivity. Predicted and ground-truth components are greedily matched
one-to-one in descending IoU order when IoU is strictly greater than `0.3`.

## External boundary checks

The CAS check freezes the diagnostic sequence but changes the I/O adapter: RGB
single-date images are resized to 128 pixels and broadcast to 15 temporal
frames. Its result establishes directional transfer of the diagnosis, not
metric comparability with Sen12Landslides.

The Prithvi-EO check uses external pretrained weights and a multispectral band
projection that is not band-wise equivalent to the 11-channel 3D U-Net input.
Because those assets are not shipped here, neither check is represented as an
end-to-end command in this release.
