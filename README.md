# Modern BEVFormer Lab

A modernized, single-GPU-friendly BEVFormer fork for understanding and running
BEVFormer on current OpenMMLab versions.

## What this repo provides

- BEVFormer Tiny model construction under modern OpenMMLab
- nuScenes-mini data conversion
- Single-GPU inference with the pretrained tiny checkpoint
- Single-GPU diagnostic training and smoke training
- Single-GPU train/test entrypoints validated for the current port
- Tensor-shape tracing utilities
- Cleaned compatibility layer under `projects/mmdet3d_plugin/modern_compat`
- Debug scripts isolated under `tools/debug`

## Documentation

- [Get started](GET_STARTED.md)
- [Detailed port status report](reports/bevformer_modern_port_status.md)

## Environment

Currently validated stack:

- Python 3.10
- PyTorch 2.12.1+cu130
- MMCV 2.1.0
- MMEngine 0.10.7
- MMDetection 3.3.0
- MMDetection3D 1.4.0

## Data

The repo expects nuScenes-mini under:

```text
data/nuscenes/
```

`tools/create_data.py` has been patched for nuScenes-mini conversion. The
generated info files are consumed by the diagnostic train, test, and debug
entrypoints.

## Quick checks

```bash
python tools/train.py \
  projects/configs/bevformer/bevformer_tiny.py \
  --work-dir outputs/train_entrypoint_smoke \
  --max-iters 5 \
  --no-validate
```

```bash
python tools/test.py \
  projects/configs/bevformer/bevformer_tiny.py \
  ckpts/bevformer_tiny_epoch_24.pth \
  --max-samples 2 \
  --out outputs/test_outputs/test_max2.pkl \
  --show-dir outputs/test_outputs/vis \
  --show-index 0
```

```bash
python tools/debug/trace_one_training_sample_shapes.py
```

## Current validation status

Validated:

- Model build
- Checkpoint loading
- Real nuScenes-mini dataset loading
- Inference on mini validation samples
- Single-GPU diagnostic training
- Loss, backward, and optimizer step
- Shape tracing through image features, BEV encoder, detection head, and loss

## Limitations

- Single-GPU path validated
- Distributed train/test not yet ported
- `tools/dist_train.sh` and `tools/dist_test.sh` are placeholder stubs that
  fail clearly instead of launching unsupported distributed jobs
- `tools/train.py` is diagnostic and smoke-training oriented, not a full
  official training runner
- No full nuScenes trainval training validated
- No official nuScenes metric reproduction claimed
- Generated outputs and checkpoints should not be committed

## Repository organization

```text
tools/train.py                                      Single-GPU diagnostic training entrypoint
tools/test.py                                       Single-GPU inference entrypoint
tools/dist_train.sh                                 Unsupported distributed-training stub
tools/dist_test.sh                                  Unsupported distributed-testing stub
tools/create_data.py                                nuScenes-mini data conversion
tools/debug/                                        Debug and tracing utilities
projects/mmdet3d_plugin/modern_compat/             Reusable modern OpenMMLab compatibility layer
reports/bevformer_modern_port_status.md            Detailed porting notes and validation log
```

## Notes

This repo is intended for learning, debugging, and modernization of BEVFormer.
It does not claim official paper-level reproduction.
