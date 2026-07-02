# Get Started

This is the practical quickstart for the modernized BEVFormer fork. It covers
the validated single-GPU path for nuScenes-mini data conversion, diagnostic
training, inference, and shape tracing.

For detailed engineering notes, see
[`reports/bevformer_modern_port_status.md`](reports/bevformer_modern_port_status.md).

## Validated environment

This repo has been validated with:

```text
Python 3.10
PyTorch 2.12.1+cu130
MMCV 2.1.0
MMEngine 0.10.7
MMDetection 3.3.0
MMDetection3D 1.4.0
```

Install PyTorch for your CUDA/runtime platform, then install compatible
OpenMMLab packages and nuScenes tooling:

```bash
pip install torch torchvision torchaudio
pip install mmcv==2.1.0 mmengine==0.10.7 mmdet==3.3.0 mmdet3d==1.4.0
pip install nuscenes-devkit
```

Depending on your platform, `mmcv` may require a wheel matching your PyTorch and
CUDA versions. Use the OpenMMLab wheel index or your internal package mirror if
a plain PyPI install is not available.

## Repository entrypoints

Public entrypoints:

```text
tools/create_data.py
tools/train.py
tools/test.py
```

Debug tools:

```text
tools/debug/
```

Compatibility layer:

```text
projects/mmdet3d_plugin/modern_compat/
```

Distributed wrappers are intentionally unsupported stubs:

```text
tools/dist_train.sh
tools/dist_test.sh
```

## Prepare nuScenes-mini

Expected layout:

```text
data/
├── can_bus/
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    └── v1.0-mini/
```

Run conversion from the repo root:

```bash
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini \
  --canbus ./data
```

Expected generated files:

```text
data/nuscenes/nuscenes_infos_temporal_train.pkl
data/nuscenes/nuscenes_infos_temporal_val.pkl
data/nuscenes/nuscenes_infos_temporal_train_mono3d.coco.json
data/nuscenes/nuscenes_infos_temporal_val_mono3d.coco.json
```

Validated mini counts:

```text
train: 323 infos
val: 81 infos
```

## Training smoke test

Run the validated single-GPU diagnostic training path:

```bash
python tools/train.py \
  projects/configs/bevformer/bevformer_tiny.py \
  --work-dir outputs/train_entrypoint_smoke \
  --max-iters 5 \
  --no-validate
```

This verifies dataset loading, forward loss, backward, and optimizer step. It is
not a full official training runner.

## Inference smoke test

Place the tiny checkpoint at:

```text
ckpts/bevformer_tiny_epoch_24.pth
```

Then run:

```bash
python tools/test.py \
  projects/configs/bevformer/bevformer_tiny.py \
  ckpts/bevformer_tiny_epoch_24.pth \
  --max-samples 2 \
  --out outputs/test_outputs/test_max2.pkl \
  --show-dir outputs/test_outputs/vis \
  --show-index 0
```

This verifies checkpoint loading and inference on a small nuScenes-mini
validation subset.

## Shape tracing

Run one training sample through the traced debug path:

```bash
python tools/debug/trace_one_training_sample_shapes.py
```

## Limitations

- Single-GPU train/test is validated.
- `tools/train.py` is a diagnostic/smoke-training entrypoint.
- The validation loop in `tools/train.py` is not ported yet; use
  `--no-validate`.
- Distributed train/test is not supported yet.
- `tools/dist_train.sh` and `tools/dist_test.sh` are unsupported stubs that fail
  clearly.
- Full nuScenes trainval training is not validated.
- Official metric reproduction is not claimed.
- Outputs and checkpoints should not be committed.
