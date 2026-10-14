# Experimental 2D BEV Occupancy/Objectness

This experimental camera-only model predicts a binary top-down vehicle mask
from temporal multi-camera images. It uses BEVFormer image features and BEV
encoder output with a small CNN occupancy head.

The original BEVFormer detection path is unchanged. `tools/train.py`,
`tools/test.py`, and the original detection configs remain separate from this
occupancy path.

## Conceptual Flow

```text
multi-camera temporal images
  -> BEVFormer image backbone/FPN
  -> BEVFormer BEV encoder
  -> BEV feature map [B, 256, 50, 50]
  -> small CNN occupancy head
  -> binary occupancy logits [B, 1, 50, 50]
```

The target is built from nuScenes annotations:

```text
nuScenes 3D boxes
  -> select vehicle-like classes
  -> take top-down box footprints
  -> rasterize into binary BEV mask [B, 1, 50, 50]
```

Vehicle-like classes:

- `car`
- `truck`
- `bus`
- `trailer`
- `construction_vehicle`

## Implementation

```text
projects/mmdet3d_plugin/bevformer/occupancy/bev_occupancy.py
projects/mmdet3d_plugin/bevformer/dense_heads/bev_occupancy_head.py
projects/mmdet3d_plugin/bevformer/detectors/bevformer_occupancy.py
projects/configs/bevformer/bevformer_tiny_occ.py
```

## Prepare nuScenes-mini

Download the [nuScenes-mini dataset and CAN bus expansion](https://www.nuscenes.org/download).
Extract mini into `data/nuscenes/` so it contains `v1.0-mini/`, `samples/`,
and `sweeps/`. Extract CAN bus so its pose files are at
`data/can_bus/can_bus/scene-*_pose.json`. From the repository root, run:

```bash
python tools/create_data.py nuscenes \
  --root-path data/nuscenes --out-dir data/nuscenes \
  --extra-tag nuscenes --version v1.0-mini --canbus data
```

This creates `data/nuscenes/nuscenes_infos_temporal_train.pkl` and
`data/nuscenes/nuscenes_infos_temporal_val.pkl`, the paths used by the occupancy
scripts. Download the pretrained Tiny checkpoint as described in the
[README](../README.md#pretrained-checkpoint) before training.

## Run

Train occupancy model:

```bash
python tools/train_occ.py \
  --epochs 10 \
  --work-dir outputs/occ_train
```

Test/visualize occupancy checkpoint:

```bash
python tools/test_occ.py \
  --checkpoint outputs/occ_train/best.pth \
  --split val \
  --output-dir outputs/occ_test
```

One epoch is one pass through the selected training samples. On the full mini
training split, ten epochs take about 3,230 optimizer steps. `--max-iters`
is only for short checks; 100 steps is less than one mini epoch.

The trainer shuffles samples each epoch, gives the pretrained image backbone
one-tenth the learning rate of the occupancy head and BEV encoder, and uses
warmup, cosine decay, and gradient clipping. It evaluates all mini validation
samples after each epoch and saves `best.pth` when validation IoU improves
(breaking ties by lower validation loss).
Validation disables random image distortion and uses a fixed temporal queue.

Training uses the pretrained Tiny checkpoint at
`ckpts/bevformer_tiny_epoch_24.pth` by default. Pass `--no-checkpoint` to
initialize without it. `--num-samples` selects a small cyclic training subset;
its default uses the full mini training split. Inference saves predictions and
box-footprint targets for two samples by default; use `--num-samples` to change
that count.

## Mini Validation Result

In a 10-epoch nuScenes-mini run, the best checkpoint was from epoch 8
(2,584 steps). Across all 81 mini validation frames, it reached **0.4377 IoU**
and **0.6089 Dice** for binary vehicle-footprint masks at threshold 0.5.
`tools/test_occ.py` defaults to visualizing only two frames from this same
validation split; those examples are not an independent test score.

## Tensor Shapes

```text
image batch:        (1, 3, 6, 3, 480, 800)
raw BEV encoder:    (1, 2500, 256), layout [B, H*W, C]
BEV feature map:    (1, 256, 50, 50)
occupancy logits:   (1, 1, 50, 50)
occupancy target:   (1, 1, 50, 50)
```

## Outputs

- `latest.pth`, validation-selected `best.pth`, `loss_history.csv`, and
  `epoch_history.csv` are saved under `--work-dir`. The CSVs include IoU and
  Dice at a 0.5 occupancy threshold.
- Training and validation visualizations under `<work-dir>/vis/` show
  per-sample IoU and Dice in the title.
- Test visualizations and `summary.csv` are saved under `--output-dir` with
  per-sample IoU and Dice at the selected `--threshold`.

## Limitations

- Binary vehicle-objectness from box footprints, not semantic or raw LiDAR occupancy.
- Single-GPU scripts; no distributed training or official occupancy metrics.
- No tracking or motion prediction.
