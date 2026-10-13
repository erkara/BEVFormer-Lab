# Experimental 2D BEV Occupancy/Objectness

This repo includes an experimental camera-only 2D BEV occupancy/objectness path
on top of BEVFormer. It uses BEVFormer as a BEV feature extractor and trains a
small CNN head to predict a binary top-down vehicle occupancy mask from temporal
multi-camera images.

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

## Why This Exists

This is a lightweight custom head for experimenting with BEVFormer as a BEV
feature extractor, without needing the full detection decoder or official
benchmark reproduction.

## Core Files

```text
projects/mmdet3d_plugin/bevformer/occupancy/bev_occupancy.py
projects/mmdet3d_plugin/bevformer/dense_heads/bev_occupancy_head.py
projects/mmdet3d_plugin/bevformer/detectors/bevformer_occupancy.py
projects/configs/bevformer/bevformer_tiny_occ.py
```

## Scripts

```text
tools/debug/visualize_occupancy_target.py   Visualize rasterized GT occupancy masks.
tools/debug/overfit_occupancy_head.py       Fast overfit sanity check.
tools/train_occ.py                          Standalone occupancy training entrypoint.
tools/test_occ.py                           Checkpoint inference and visualization entrypoint.
```

## Commands

Visualize occupancy targets:

```bash
python tools/debug/visualize_occupancy_target.py --num-samples 3 --start-index 0
```

Overfit sanity check:

```bash
python tools/debug/overfit_occupancy_head.py --num-samples 2 --iters 50
```

Train occupancy model:

```bash
python tools/train_occ.py \
  --num-samples 2 \
  --max-iters 5 \
  --work-dir outputs/occ_train_smoke
```

Test/visualize occupancy checkpoint:

```bash
python tools/test_occ.py \
  --checkpoint outputs/occ_train_smoke/latest.pth \
  --num-samples 2 \
  --output-dir outputs/occ_test_smoke
```

## Validated Smoke Shapes

```text
image batch:        (1, 3, 6, 3, 480, 800)
raw BEV encoder:    (1, 2500, 256), layout [B, H*W, C]
BEV feature map:    (1, 256, 50, 50)
occupancy logits:   (1, 1, 50, 50)
occupancy target:   (1, 1, 50, 50)
```

The overfit sanity check reduced loss from about `0.7766` to about `0.0615`
over 50 iterations on two samples. The standalone smoke training reduced loss
from about `0.7768` to about `0.3626` over 5 iterations.

## Outputs

- Training checkpoints and `loss_history.csv` are saved under `--work-dir`.
- Training visualizations are saved under `<work-dir>/vis/`.
- Test visualizations and `summary.csv` are saved under `--output-dir`.

## Limitations

- Binary vehicle-objectness only, not semantic occupancy.
- Supervised by box footprints, not raw LiDAR occupancy.
- Single-GPU/simple training scripts only.
- No distributed training.
- No official occupancy benchmark metrics.
- No tracking or motion prediction yet.
