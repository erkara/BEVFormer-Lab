# BEVFormer: A Working OpenMMLab Port

[BEVFormer](https://github.com/fundamentalvision/BEVFormer) is an influential
camera-based 3D perception model, but its original code is difficult to run
with current dependencies. This fork brings BEVFormer
Tiny training and inference to a modern OpenMMLab stack, with a validated
single-GPU workflow on nuScenes-mini. It also adds an experimental 2D BEV
occupancy path for exploring the model's BEV features.

Run the commands below from the repository root. The project has been validated
on nuScenes-mini with a single GPU.

## Requirements

Validated versions: Python 3.10, PyTorch 2.12.1+cu130, MMCV 2.1.0,
MMEngine 0.10.7, MMDetection 3.3.0, and MMDetection3D 1.4.0.
Install PyTorch for your CUDA platform, then the OpenMMLab packages and
nuScenes tooling:

```bash
pip install torch torchvision torchaudio
pip install mmcv==2.1.0 mmengine==0.10.7 mmdet==3.3.0 mmdet3d==1.4.0
pip install nuscenes-devkit
```

MMCV may require a wheel compatible with your PyTorch and CUDA versions.

## Prepare nuScenes-mini

Download [nuScenes-mini and the CAN bus expansion](https://www.nuscenes.org/download).
Unpack nuScenes-mini under `data/nuscenes/` (including `samples/`, `sweeps/`,
`maps/`, and `v1.0-mini/`) and the CAN bus expansion under `data/can_bus/`.
Then generate the training and validation info files:

```bash
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini \
  --canbus ./data
```

This command converts the downloaded data; it does not download it. The
scripts expect `data/nuscenes/nuscenes_infos_temporal_train.pkl` and
`data/nuscenes/nuscenes_infos_temporal_val.pkl` afterward.

## Pretrained checkpoint

The [original BEVFormer model zoo](https://github.com/fundamentalvision/BEVFormer#model-zoo)
provides the pretrained Tiny checkpoint. Download it to the path used by the
scripts:

```bash
mkdir -p ckpts
curl -L --fail https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth \
  -o ckpts/bevformer_tiny_epoch_24.pth
```

Detection inference requires a compatible checkpoint; the command below uses
this one. The training scripts use it for initialization by default; pass
`--no-checkpoint` to train without it.

## 3D detection

The Tiny detection path trains with `tools/train.py` and runs inference with
`tools/test.py`:

```bash
python tools/train.py projects/configs/bevformer/bevformer_tiny.py \
  --work-dir outputs/detection_train --max-iters 100 --no-validate

python tools/test.py projects/configs/bevformer/bevformer_tiny.py \
  ckpts/bevformer_tiny_epoch_24.pth --out outputs/detection_predictions.pkl
```

Training cycles over a fixed mini subset for the requested iterations and
saves a checkpoint and loss CSV under `--work-dir`. Validation is not available
in this trainer. Inference saves predictions for the validation split.

## Experimental 2D BEV occupancy

The occupancy path uses BEVFormer features to predict a binary top-down mask
of vehicle footprints from temporal multi-camera images. Its targets come from
rasterized nuScenes 3D boxes. It has its own
[config](projects/configs/bevformer/bevformer_tiny_occ.py); the detection path
is unchanged.

```bash
python tools/train_occ.py --max-iters 100 --work-dir outputs/occ_train

python tools/test_occ.py --checkpoint outputs/occ_train/latest.pth \
  --split val --output-dir outputs/occ_test
```

Training saves checkpoints and `loss_history.csv` under `--work-dir`.
Inference saves prediction/target images and `summary.csv` under
`--output-dir` (two samples by default).

Validated tensor shapes for the Tiny configuration:

```text
image batch:        (1, 3, 6, 3, 480, 800)
raw BEV encoder:    (1, 2500, 256), layout [B, H*W, C]
BEV feature map:    (1, 256, 50, 50)
occupancy logits:   (1, 1, 50, 50)
occupancy target:   (1, 1, 50, 50)
```

See [occupancy head documentation](docs/occupancy_head.md) for the target and
model details. This task predicts box-footprint objectness from camera images.

## Scope

Distributed training, full nuScenes trainval training, and official benchmark
reproduction have not been validated. Generated outputs and checkpoints belong
in the ignored `outputs/` and `ckpts/` directories.
