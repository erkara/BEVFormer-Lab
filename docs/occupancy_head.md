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

## Run

Train occupancy model:

```bash
python tools/train_occ.py \
  --max-iters 100 \
  --work-dir outputs/occ_train
```

Test/visualize occupancy checkpoint:

```bash
python tools/test_occ.py \
  --checkpoint outputs/occ_train/latest.pth \
  --split val \
  --output-dir outputs/occ_test
```

Training uses the pretrained Tiny checkpoint at
`ckpts/bevformer_tiny_epoch_24.pth` by default. Pass `--no-checkpoint` to
initialize without it. `--num-samples` selects a small cyclic training subset;
its default uses the full mini training split. Inference saves predictions and
box-footprint targets for two samples by default; use `--num-samples` to change
that count.

## Tensor Shapes

```text
image batch:        (1, 3, 6, 3, 480, 800)
raw BEV encoder:    (1, 2500, 256), layout [B, H*W, C]
BEV feature map:    (1, 256, 50, 50)
occupancy logits:   (1, 1, 50, 50)
occupancy target:   (1, 1, 50, 50)
```

## Outputs

- Training checkpoints and `loss_history.csv` are saved under `--work-dir`.
- Training visualizations are saved under `<work-dir>/vis/`.
- Test visualizations and `summary.csv` are saved under `--output-dir`.

## Limitations

- Binary vehicle-objectness from box footprints, not semantic or raw LiDAR occupancy.
- Single-GPU scripts; no distributed training or official occupancy metrics.
- No tracking or motion prediction.
