# BEVFormer: A Working OpenMMLab Port

[BEVFormer](https://github.com/fundamentalvision/BEVFormer) is an influential
camera-based 3D perception model, but its original code is difficult to run
with current dependencies. This fork brings BEVFormer
Tiny training and inference to a modern OpenMMLab stack. With the original
Tiny checkpoint, its full nuScenes validation scores closely match the
original release. It also adds an experimental 2D BEV occupancy path for
exploring the model's BEV features.

Run the commands below from the repository root. Full validation inference and
nuScenes-mini training have been tested on a single GPU.

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
this one.

## Prepare full nuScenes

Download the [nuScenes trainval metadata, all ten trainval keyframe archives,
and CAN bus expansion](https://www.nuscenes.org/download). Extract the metadata
and **all ten keyframe archives** into `data/nuscenes-full/`, so it contains
`v1.0-trainval/` and `samples/`. Extract the CAN bus expansion under
`data/can_bus/`. Generate both the training and validation info files once:

```bash
python tools/create_data.py nuscenes \
  --root-path data/nuscenes-full --out-dir data/nuscenes-full \
  --extra-tag nuscenes --version v1.0-trainval --canbus data
```

This creates `nuscenes_infos_temporal_train.pkl` and
`nuscenes_infos_temporal_val.pkl` in `data/nuscenes-full/`. Both commands below
use this same prepared dataset. The converter reports how many frames have
CAN bus data; check that it is populated.

## Evaluate BEVFormer Tiny

Evaluate the original pretrained checkpoint on the complete nuScenes
validation split:

```bash
python tools/test.py --config-file projects/configs/bevformer/bevformer_tiny.py \
  --checkpoint ckpts/bevformer_tiny_epoch_24.pth \
  --data-root data/nuscenes-full \
  --ann-file data/nuscenes-full/nuscenes_infos_temporal_val.pkl \
  --eval bbox --eval-dir outputs/detection_full_val
```

Results from the original Tiny checkpoint on all 6,019 validation frames:

| Evaluation | mAP | NDS |
| --- | ---: | ---: |
| [Original BEVFormer Tiny log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.log) | 25.24% | 35.40% |
| This port, full validation (6,019 frames) | 25.19% | 35.45% |

The evaluation writes predictions and metrics under `outputs/detection_full_val/`.
For an evaluation-only, camera-only extraction that uses less disk space, see
[`tools/prepare_val_camera_only.py`](tools/prepare_val_camera_only.py).

## Train BEVFormer Tiny

Train from scratch using the same prepared full dataset:

```bash
python tools/train.py --config-file projects/configs/bevformer/bevformer_tiny.py \
  --data-root data/nuscenes-full \
  --train-info data/nuscenes-full/nuscenes_infos_temporal_train.pkl \
  --no-checkpoint --max-epochs 24 --no-validate \
  --work-dir outputs/detection_full_train
```

For multiple GPUs, launch the same trainer with `torchrun` (set the process
count to the number of GPUs):

```bash
torchrun --nproc-per-node=2 tools/train.py \
  --config-file projects/configs/bevformer/bevformer_tiny.py \
  --data-root data/nuscenes-full \
  --train-info data/nuscenes-full/nuscenes_infos_temporal_train.pkl \
  --no-checkpoint --max-epochs 24 --no-validate \
  --work-dir outputs/detection_full_train_multi --launcher pytorch
```

Evaluate the resulting `latest.pth` with the command above, replacing its
`--checkpoint` path. The trainer saves checkpoints and loss history under
`--work-dir`; use `--resume-from` with `latest.pth` to resume. Each GPU process
trains on different samples. Validation is run separately with `tools/test.py`.

For a smaller training run, see the
[nuScenes-mini preparation steps](docs/occupancy_head.md#prepare-nuscenes-mini).
Multi-GPU execution has not yet been tested on more than one GPU.

## Experimental 2D BEV occupancy

The occupancy path uses BEVFormer features to predict a binary top-down mask
of vehicle footprints from temporal multi-camera images. Its targets come from
rasterized nuScenes 3D boxes. It has its own
[config](projects/configs/bevformer/bevformer_tiny_occ.py); the detection path
is unchanged.

Prepare nuScenes-mini at `data/nuscenes/` using the
[occupancy setup steps](docs/occupancy_head.md#prepare-nuscenes-mini) before
running these commands.

```bash
python tools/train_occ.py --epochs 10 --work-dir outputs/occ_train

python tools/test_occ.py --checkpoint outputs/occ_train/best.pth \
  --split val --output-dir outputs/occ_test
```

One epoch is one pass through the mini training samples (about 323 steps).
Ten epochs therefore take about 3,230 steps. `--max-iters` remains available
for short checks; 100 steps is less than one epoch. Training shuffles each
epoch, checks the held-out mini validation split after each epoch, and saves
`best.pth` by validation IoU (breaking ties by lower loss) alongside
`latest.pth`, `loss_history.csv`, and `epoch_history.csv` under `--work-dir`.
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

Distributed training, full nuScenes trainval training, and original benchmark
reproduction have not been validated.
