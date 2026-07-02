# BEVFormer Modern OpenMMLab Port Status

## Environment

- Python: 3.10
- PyTorch: 2.12.1+cu130
- MMCV: 2.1.0
- MMEngine: 0.10.7
- MMDetection: 3.3.0
- MMDetection3D: 1.4.0

## Current Working State

The modernized repo is now validated beyond import/model-build and synthetic
forward probes. The current validated workflow is the single-GPU nuScenes-mini
path documented in `GET_STARTED.md`.

Validated public workflow:

- `tools/create_data.py` converts nuScenes-mini and produces temporal train/val
  info files plus mono3d COCO JSON files.
- Converted nuScenes-mini counts are validated:
  - train: 323 infos
  - val: 81 infos
- `tools/train.py` runs a single-GPU diagnostic/smoke training cycle on
  nuScenes-mini with dataset loading, forward loss, backward pass, and optimizer
  step.
- `tools/test.py` loads `ckpts/bevformer_tiny_epoch_24.pth` and runs
  small-subset inference on nuScenes-mini validation samples.
- `tools/debug/trace_one_training_sample_shapes.py` traces one real training
  sample through image features, BEV encoder, detection head, and loss.
- Debug/developer tools now live under `tools/debug/`.
- Compatibility helpers live under
  `projects/mmdet3d_plugin/modern_compat/`.
- `tools/dist_train.sh` and `tools/dist_test.sh` are unsupported stubs that fail
  clearly instead of launching distributed jobs.

Current validated commands:

```bash
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini \
  --canbus ./data
```

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

Summary status:

| Gate | Status |
| --- | --- |
| Plugin import | Passed |
| Config load | Passed |
| Model build | Passed |
| `init_weights` smoke test | Passed |
| Image backbone/FPN dummy forward | Passed |
| BEV encoder `only_bev` dummy forward | Passed |
| Full `pts_bbox_head` dummy forward | Passed |
| Decoder layout bug | Fixed permanently by setting decoder `CustomMSDeformableAttention.batch_first=False` |
| nuScenes-mini conversion | Passed; 323 train infos, 81 val infos |
| Single-GPU diagnostic training | Passed; dataset load, forward loss, backward, optimizer step |
| Checkpoint loading | Passed with `ckpts/bevformer_tiny_epoch_24.pth` |
| Single-GPU small-subset inference | Passed with `tools/test.py` |
| Shape tracing on real training sample | Passed with `tools/debug/trace_one_training_sample_shapes.py` |
| `tools/train.py` validation loop | Not ported; use `--no-validate` |
| Distributed train/test | Unsupported; wrapper scripts are explicit failing stubs |
| Full nuScenes trainval training | Not validated |
| Official nuScenes metric reproduction | Not claimed |

Current limitations:

- The validated train/test path is single-GPU.
- `tools/train.py` is diagnostic/smoke-training oriented, not an official full
  training reproduction.
- The validation loop in `tools/train.py` is not ported; use `--no-validate`.
- Distributed train/test is unsupported.
- Full nuScenes trainval training is not validated.
- Official metric reproduction is not claimed.
- Generated outputs and checkpoints should not be committed.

Earlier synthetic-forward diagnostics are retained below as historical porting
notes. They no longer define the current validated repo state.

Observed smoke-test metadata:

- Model class: `projects.mmdet3d_plugin.bevformer.detectors.bevformer.BEVFormer`
- Total parameters: `33,521,157`
- Trainable parameters: `33,245,627`
- Image backbone: `mmdet.models.backbones.resnet.ResNet`
- Image neck: `mmdet.models.necks.fpn.FPN`
- 3D bbox head: `projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head.BEVFormerHead`
- BEVFormer transformer: `projects.mmdet3d_plugin.bevformer.modules.transformer.PerceptionTransformer`
- `pts_bbox_head.cls_branches`: present
- `pts_bbox_head.reg_branches`: present

## Dummy Forward Probe

Current script:

- `tools/debug/debug_bevformer_dummy_forward.py`

What it does:

- Builds BEVFormer tiny using the same config fixes.
- Creates synthetic 6-camera image input and synthetic metadata.
- Probes staged forward paths.
- Does not use real nuScenes data.
- Does not load checkpoint.
- Does not train or evaluate.

Results:

- Image backbone + FPN forward worked.
- Output feature shape: `[(1, 6, 256, 4, 4)]`.
- BEVFormer transformer/head was reached.
- BEV encoder with `only_bev=True` worked.
- BEV encoder output shape: `(1, 2500, 256)`.
- CUDA extension path was invoked.
- Full `pts_bbox_head` forward failed in decoder cross-attention.
- Failure location: `projects/mmdet3d_plugin/bevformer/modules/decoder.py:331`.
- Failure reason: assertion that `spatial_shapes` product equals `num_value`.

Historical interpretation:

- At this early synthetic-only stage, the repo had moved beyond model construction.
- Backbone, FPN, transformer entry path, and BEV encoder could execute on synthetic input.
- Full detection head decoding was not yet working in this probe.
- This blocker was later resolved by the decoder layout patch and superseded by real nuScenes-mini inference and diagnostic training.

Historical limitations for this early dummy probe:

- This probe used synthetic inputs only and was superseded by later real-data validations.
- The decoder/head failure was resolved by the decoder layout patch.
- Real nuScenes-mini conversion, checkpoint loading, inference, diagnostic training, and shape tracing are validated in later sections.

## Decoder Shape Diagnostic

Script:

- `tools/debug/debug_bevformer_decoder_shapes.py`

What it does:

- Builds BEVFormer tiny.
- Reuses the synthetic 6-camera dummy input.
- Monkey-patches runtime only for diagnostics.
- Prints decoder/cross-attention tensor shapes.
- Does not modify model code.
- Does not use real nuScenes data.
- Does not load checkpoint.
- Does not train/evaluate.

Results:

- `mlvl_feats`: `[(1, 6, 256, 4, 4)]`.
- BEV encoder output: `(1, 2500, 256)`.
- Decoder cross-attention was reached.
- `CustomMSDeformableAttention.forward` was called.
- Module `batch_first`: `True`.
- Module `num_levels`: `1`.
- `query`: `(900, 1, 256)`.
- `value`: `(2500, 1, 256)`.
- `reference_points`: `(1, 900, 1, 2)`.
- `spatial_shapes`: `[[50, 50]]`.
- `level_start_index`: `[0]`.
- `spatial_shapes` product sum: `2500`.
- `num_value` read by assertion: `1`.
- Assertion comparison: `2500 == 1`.

Interpretation:

- The BEV encoder is producing 2500 BEV tokens, matching the 50 x 50 BEV grid.
- The decoder/cross-attention receives `value` as sequence-first layout: `(N, B, C) = (2500, 1, 256)`.
- But the attention module is configured with `batch_first=True`, so it interprets `value` as `(B, N, C)`.
- Therefore it incorrectly reads `num_value` as `1` instead of `2500`.
- This looks like a model-porting tensor-layout issue, not a basic model construction issue and not necessarily a dummy-input issue.

Historical blocker:

- This was resolved by setting decoder `CustomMSDeformableAttention.batch_first=False` for the sequence-first decoder path.

## Decoder Layout Runtime-Fix Diagnostic

Script:

- `tools/debug/debug_bevformer_decoder_layout_fix.py`

What it does:

- Builds BEVFormer tiny with the same config fixes.
- Reuses the synthetic 6-camera dummy input and metadata.
- Tests runtime-only decoder layout fixes without modifying model code.
- Does not use real nuScenes data.
- Does not load checkpoint.
- Does not train or evaluate.

Experiment A:

- Runtime patch: set decoder `CustomMSDeformableAttention` instances to `batch_first=False`.
- Image backbone/FPN: passed with `mlvl_feats=[(1, 6, 256, 4, 4)]`.
- BEV encoder `only_bev=True`: passed with output `(1, 2500, 256)`.
- Decoder cross-attention assertion: passed across all 6 decoder layers.
- Full `pts_bbox_head`: passed and returned `all_bbox_preds`, `all_cls_scores`, `bev_embed`, `enc_bbox_preds`, and `enc_cls_scores`.
- CUDA extension path: invoked.

Experiment B:

- Runtime patch: keep `batch_first=True`, but transpose `query`, `value`, and related tensors entering the attention call, then transpose output back.
- Image backbone/FPN: passed with `mlvl_feats=[(1, 6, 256, 4, 4)]`.
- BEV encoder `only_bev=True`: passed with output `(1, 2500, 256)`.
- Decoder cross-attention assertion: passed across all 6 decoder layers.
- Full `pts_bbox_head`: passed and returned `all_bbox_preds`, `all_cls_scores`, `bev_embed`, `enc_bbox_preds`, and `enc_cls_scores`.
- CUDA extension path: invoked.

Recommendation:

- Both runtime experiments passed full `pts_bbox_head` dummy forward.
- Experiment A: set decoder `CustomMSDeformableAttention.batch_first=False`.
- Experiment B: transpose tensors around the existing `batch_first=True` attention call.
- Experiment A is the preferred minimal code patch because the surrounding decoder path already uses sequence-first layout `(N, B, C)`: `query=(900, 1, 256)` and `value=(2500, 1, 256)`. Setting decoder `CustomMSDeformableAttention` to `batch_first=False` matches that existing contract with less wrapper logic than transposing tensors around every cross-attention call.

Historical blocker:

- Resolved by the permanent decoder layout patch and subsequent dummy, real inference, and diagnostic training validations.

## Permanent Decoder Layout Patch

Patch applied:

- `projects/mmdet3d_plugin/bevformer/modules/custom_base_transformer_layer.py` now constructs decoder `CustomMSDeformableAttention` cross-attention with `batch_first=False`.
- The patch is intentionally narrow and only affects BEVFormer decoder custom deformable cross-attention.
- Compatibility comment added in code: BEVFormer decoder passes query/value as sequence-first tensors `(N, B, C)`, so the attention module must not interpret `value` as batch-first when validating `spatial_shapes`.

Verification commands and outputs:

```bash
python tools/debug/debug_build_bevformer_tiny.py
```

```text
model built
<class 'projects.mmdet3d_plugin.bevformer.detectors.bevformer.BEVFormer'>
```

```bash
python tools/debug/debug_bevformer_smoke.py
```

```text
model class: <class 'projects.mmdet3d_plugin.bevformer.detectors.bevformer.BEVFormer'>
total parameters: 33521157
trainable parameters: 33245627
img_backbone type: <class 'mmdet.models.backbones.resnet.ResNet'>
img_neck type: <class 'mmdet.models.necks.fpn.FPN'>
pts_bbox_head type: <class 'projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head.BEVFormerHead'>
pts_bbox_head.transformer type: <class 'projects.mmdet3d_plugin.bevformer.modules.transformer.PerceptionTransformer'>
has cls_branches: True
has reg_branches: True
init_weights: ok
smoke test passed
```

```bash
python tools/debug/debug_bevformer_dummy_forward.py
```

Key output:

```text
=== stage: model.extract_img_feat ===
stage ok: model.extract_img_feat
model.extract_img_feat output: [(1, 6, 256, 4, 4)]

=== stage: model.extract_feat ===
stage ok: model.extract_feat
model.extract_feat output: [(1, 6, 256, 4, 4)]

=== stage: model.pts_bbox_head(..., only_bev=True) ===
stage ok: model.pts_bbox_head(..., only_bev=True)
model.pts_bbox_head(..., only_bev=True) output: (1, 2500, 256)

=== stage: model.pts_bbox_head ===
stage ok: model.pts_bbox_head
model.pts_bbox_head output: {'bev_embed': (2500, 1, 256), 'all_cls_scores': (6, 1, 900, 10), 'all_bbox_preds': (6, 1, 900, 10), 'enc_cls_scores': 'NoneType', 'enc_bbox_preds': 'NoneType'}
head output bev_embed: (2500, 1, 256)
head output all_cls_scores: (6, 1, 900, 10)
head output all_bbox_preds: (6, 1, 900, 10)
head output enc_cls_scores: NoneType
head output enc_bbox_preds: NoneType

=== stage: model.forward(return_loss=False) ===
stage failed: model.forward(return_loss=False)
exception type: RuntimeError
exception: shape '[1, 6, 6, 1, 4, 4]' is invalid for input of size 96

=== summary ===
image backbone/FPN forward worked: True
BEVFormer transformer/head forward reached: True
CUDA extension path invoked: True
```

Interpretation:

- The permanent decoder layout patch resolves the previous `spatial_shapes` assertion in `CustomMSDeformableAttention`.
- Full staged `pts_bbox_head` dummy forward now passes without runtime monkey-patching.
- The remaining failure in `tools/debug/debug_bevformer_dummy_forward.py` was limited to the synthetic full-detector `forward_test` packing path. Later real nuScenes-mini inference through `tools/test.py` passed with the verified OpenMMLab test-time packing.

Historical limitations at this point in the port:

- This specific dummy-forward probe used synthetic inputs only.
- Its full-detector `forward_test` failure was a synthetic metadata/input nesting issue.
- Later sections supersede this limitation with real nuScenes-mini dataset loading, checkpoint loading, inference, and diagnostic training.

## Final Decoder Layout Patch Status

Passed gates:

- Plugin import.
- Config load.
- Model build.
- `init_weights` smoke test.
- Image backbone/FPN dummy forward.
- BEV encoder `only_bev` dummy forward.
- Full `pts_bbox_head` dummy forward.
- Decoder layout bug fixed permanently by setting decoder `CustomMSDeformableAttention.batch_first=False`.

Current dummy-forward outputs:

- `model.extract_img_feat`: `[(1, 6, 256, 4, 4)]`.
- `only_bev` output: `(1, 2500, 256)`.
- `bev_embed`: `(2500, 1, 256)`.
- `all_cls_scores`: `(6, 1, 900, 10)`.
- `all_bbox_preds`: `(6, 1, 900, 10)`.
- CUDA extension path invoked: `True`.

Historical synthetic-only failure:

- Full detector `model.forward(return_loss=False)` failed on the synthetic input.
- Failure: `RuntimeError: shape '[1, 6, 6, 1, 4, 4]' is invalid for input of size 96`.
- Interpretation: this was a synthetic metadata/input nesting mismatch in full `forward_test`, not a failure of the BEVFormer head itself.
- Later real nuScenes-mini inference through `tools/test.py` passed.

Superseded by later validation:

- Real nuScenes-mini samples now load successfully.
- Checkpoint loading now passes through `tools/test.py`.
- Small-subset inference now passes through `tools/test.py`.
- Single-GPU diagnostic training now passes through `tools/train.py`.
- Official evaluation remains unclaimed.

## Historical Not-Tested Status Superseded

This section originally recorded the limits of the early synthetic-only porting
stage. It is no longer the current repo status.

Current validated coverage now includes:

- nuScenes-mini data conversion with `tools/create_data.py`.
- Real nuScenes-mini dataset loading.
- Checkpoint loading with `ckpts/bevformer_tiny_epoch_24.pth`.
- Small-subset inference with `tools/test.py`.
- Single-GPU diagnostic training with `tools/train.py`.
- Shape tracing with `tools/debug/trace_one_training_sample_shapes.py`.

Official evaluation, distributed train/test, and full nuScenes trainval training
remain unclaimed.

Warnings still appear from legacy code, including AMP decorator deprecations in `multi_scale_deformable_attn_function.py` and transformer FFN argument deprecations. These warnings do not block the current model-build or smoke-test gates.

## Files Changed And Why

Tracked compatibility changes shown by `git diff`:

- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`
  - Adapts legacy `BEVFormerHead` construction to modern `mmdet` `DETRHead`.
  - Stops passing old-only args such as `transformer` into `DETRHead.__init__`.
  - Builds and stores the BEVFormer transformer manually.
  - Preserves construction-time attributes needed by BEVFormer branches, including query count, transformer decoder layer count, positional encoding, bbox coder, losses, assigner/sampler-related state, and code weights.

- `projects/mmdet3d_plugin/bevformer/modules/transformer.py`
  - Routes legacy transformer layer sequence construction through `mmdet.registry.MODELS` so plugin classes registered there can be found under modern MMEngine/MMCV.

- `projects/mmdet3d_plugin/bevformer/modules/transformerV2.py`
  - Applies the same registry routing compatibility for the V2 transformer module import path.

- `projects/mmdet3d_plugin/bevformer/modules/encoder.py`
  - Routes MMCV transformer-layer construction to the registry where the legacy plugin registers its encoder/layer classes.

- `projects/mmdet3d_plugin/bevformer/modules/decoder.py`
  - Applies the same transformer-layer registry routing for decoder construction.

- `projects/mmdet3d_plugin/bevformer/modules/custom_base_transformer_layer.py`
  - Routes attention and FFN builders through `mmdet.registry.MODELS`.
  - Registers the existing custom transformer layer under the legacy config name `DetrTransformerDecoderLayer`.

- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`
  - Routes deformable attention construction through `mmdet.registry.MODELS` so legacy plugin attention modules resolve correctly.

Untracked debug scripts created for the gates:

- `tools/debug_build_bevformer_tiny.py`
  - Minimal construction check for the tiny BEVFormer model.

- `tools/debug_bevformer_smoke.py`
  - Safe smoke test for model metadata and `init_weights()`.

Current tracked `git diff --stat`:

```text
 .../bevformer/dense_heads/bevformer_head.py        | 62 +++++++++++++++---
 .../modules/custom_base_transformer_layer.py       | 33 ++++++++--
 .../mmdet3d_plugin/bevformer/modules/decoder.py    | 58 ++++++++++++++---
 .../mmdet3d_plugin/bevformer/modules/encoder.py    | 31 ++++++---
 .../bevformer/modules/spatial_cross_attention.py   | 76 +++++++++++++++++++---
 .../bevformer/modules/transformer.py               | 60 ++++++++++++++---
 .../bevformer/modules/transformerV2.py             | 38 +++++++++--
 7 files changed, 304 insertions(+), 54 deletions(-)
```

## Reproduction Commands

Run the model-build gate:

```bash
python tools/debug/debug_build_bevformer_tiny.py
```

Expected tail output:

```text
model built
<class 'projects.mmdet3d_plugin.bevformer.detectors.bevformer.BEVFormer'>
```

Run the smoke-test gate:

```bash
python tools/debug/debug_bevformer_smoke.py
```

Expected tail output:

```text
model class: <class 'projects.mmdet3d_plugin.bevformer.detectors.bevformer.BEVFormer'>
total parameters: 33521157
trainable parameters: 33245627
img_backbone type: <class 'mmdet.models.backbones.resnet.ResNet'>
img_neck type: <class 'mmdet.models.necks.fpn.FPN'>
pts_bbox_head type: <class 'projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head.BEVFormerHead'>
pts_bbox_head.transformer type: <class 'projects.mmdet3d_plugin.bevformer.modules.transformer.PerceptionTransformer'>
has cls_branches: True
has reg_branches: True
init_weights: ok
smoke test passed
```

## Recommended Next-Step Roadmap

Historical next-step plan at that point:

A. Use this state for architecture and code reading. The model object was constructible, module types were inspectable, and initialization succeeded.

B. Checkpoint loading, real nuScenes-mini dataset loading, inference, diagnostic training, and shape tracing were later validated in subsequent sections.

C. Official evaluation, distributed train/test, and full nuScenes trainval training remain outside the validated scope.

## nuScenes Mini Data Conversion

Command used:

```bash
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini \
  --canbus ./data
```

Compatibility patches made:

- `tools/create_data.py`
  - Lazy-loads non-nuScenes converters and `create_groundtruth_database` so `dataset=nuscenes` does not fail on unrelated legacy Kitti/Waymo imports.
- `tools/data_converter/create_gt_database.py`
  - Replaces `from mmcv import track_iter_progress` with `from mmengine.utils import track_iter_progress` so the module can still be imported when needed.
- `tools/data_converter/nuscenes_converter.py`
  - Replaces old MMCV progress helpers with `mmengine.utils.track_iter_progress`.
  - Replaces old MMCV file helpers with `osp.isfile` and `mmengine.utils.check_file_exist`.
  - Replaces old `mmcv.dump/load` calls with `mmengine.fileio.dump/load`.
  - Updates `points_cam2img` import from `mmdet3d.core.bbox.box_np_ops` to `mmdet3d.structures`.
  - Adds a local nuScenes category mapping helper that uses legacy `NuScenesDataset.NameMapping` when present and otherwise falls back to the nuScenes devkit `category_to_detection_name`.

Output files created:

- `data/nuscenes/nuscenes_infos_temporal_train.pkl`
- `data/nuscenes/nuscenes_infos_temporal_val.pkl`
- `data/nuscenes/nuscenes_infos_temporal_train_mono3d.coco.json`
- `data/nuscenes/nuscenes_infos_temporal_val_mono3d.coco.json`

Readable pkl info counts:

- `nuscenes_infos_temporal_train.pkl`: 323 infos, metadata `{'version': 'v1.0-mini'}`
- `nuscenes_infos_temporal_val.pkl`: 81 infos, metadata `{'version': 'v1.0-mini'}`

Remaining limitations:

- This validates the local nuScenes mini data conversion path only.
- It does not validate full `v1.0-trainval`; no full dataset is assumed to exist.
- It does not create or validate a ground-truth database for nuScenes because that call remains commented out in `tools/create_data.py`.
- This conversion command itself does not run model training, inference, or evaluation; those are validated separately in later sections.
- It does not modify model architecture.

## Real nuScenes Mini Inference And Visualization Status

Verified milestones:

- v1.0-mini conversion succeeded.
- Created temporal info files:
  - `data/nuscenes/nuscenes_infos_temporal_train.pkl`
  - `data/nuscenes/nuscenes_infos_temporal_val.pkl`
- `nuscenes_infos_temporal_train.pkl` has 323 infos.
- `nuscenes_infos_temporal_val.pkl` has 81 infos.
- Real validation dataset builds as `CustomNuScenesDataset` with length 81.
- One real dataset item loads successfully.
- Real image tensor shape is `(6, 3, 480, 800)`.
- Full detector forward works with the OpenMMLab test-time packing:
  - `img = [Tensor(1, 6, 3, 480, 800)]`
  - `img_metas = [[metadata_dict]]`
- Matching checkpoint:
  - `ckpts/bevformer_tiny_epoch_24.pth`
- Full forward with checkpoint returns:
  - `bboxes_3d`: `(300, 9)`
  - `scores_3d`: `(300,)`
  - `labels_3d`: `(300,)`
- BEV prediction-vs-GT visualization works.
- Temporal sequence debugging confirms:
  - `video_test_mode=True`
  - `model.prev_frame_info["prev_bev"]` persists across same-scene frames
  - `prev_bev` shape is `(2500, 1, 256)`
  - continuous temporal outputs differ from reset-each-frame outputs
- Single-camera projection works for `CAM_FRONT`.
- Six-camera projection grid was added and succeeds for mini validation indices 0 and 15:
  - `tools/debug/debug_bevformer_project_to_all_cameras.py`
  - `outputs/camera_projection/debug_camera_projection_allcams_index_0_thr_0p2_topk_10.png`
  - `outputs/camera_projection/debug_camera_projection_allcams_index_15_thr_0p2_topk_10.png`

### Current Interpretation

- The implementation pipeline is now verified end-to-end on nuScenes-mini.
- BEVFormer-tiny predictions are plausible but weak/noisy.
- This is expected because tiny uses R50, 50x50 BEV, 3 encoder layers, and single-scale features.
- The current artifacts are for implementation understanding and debugging, not official performance claims.
- Official evaluation and full training reproduction have not been run. Single-GPU diagnostic training is validated in later sections.

### Remaining Not Claimed

- Official nuScenes evaluation.
- Full training reproduction beyond the diagnostic/smoke-training path.
- BEVFormer-small/base checkpoint inference.
- Camera projection quality across many samples.
- Full trainval dataset.

## Official Test Entrypoint Port

Command run:

```bash
python tools/test.py \
  projects/configs/bevformer/bevformer_tiny.py \
  ckpts/bevformer_tiny_epoch_24.pth \
  --eval bbox \
  --max-samples 5
```

Result:

- Single-GPU/non-distributed `tools/test.py` works on nuScenes-mini.
- The script builds the real mini validation dataset from `data/nuscenes/nuscenes_infos_temporal_val.pkl`.
- The script loads `ckpts/bevformer_tiny_epoch_24.pth` with `mmengine.runner.load_checkpoint`.
- Five validation samples ran successfully.
- Outputs are produced for each sample.
- First output summary:
  - `pts_bbox` keys: `bboxes_3d`, `scores_3d`, `labels_3d`
  - `bboxes_3d`: `(300, 9)`
  - `scores_3d`: `(300,)`
  - `labels_3d`: `(300,)`

Compatibility patches made in `tools/test.py`:

- Replaced old `mmcv.Config` usage with `mmengine.config.Config`.
- Replaced old checkpoint loading with `mmengine.runner.load_checkpoint`, with a `torch.load` fallback.
- Imports `projects.mmdet3d_plugin` so local BEVFormer modules register.
- Builds the model through `mmdet.registry.MODELS`.
- Applies the known-good modern config fixes:
  - remove `cfg.model.pretrained`
  - `cfg.model.img_backbone.type = "mmdet.ResNet"`
  - `cfg.model.img_neck.type = "mmdet.FPN"`
  - `cfg.model.pts_bbox_head.type = "mmdet.BEVFormerHead"`
- Registers runtime compatibility aliases for the legacy test pipeline transforms:
  - `LoadMultiViewImageFromFiles`
  - `DefaultFormatBundle3D`
- Builds the real mini validation dataset through `mmdet3d.registry.DATASETS` and attaches the legacy temporal info records.
- Uses the verified BEVFormer test-time packing:
  - `img = [Tensor(1, 6, 3, 480, 800)]`
  - `img_metas = [[metadata_dict]]`
- Adds `--max-samples` for short mini inference checks.

Evaluation status:

- `dataset.evaluate` was not forced for this gate.
- With `--max-samples 5`, evaluation is intentionally skipped because the output list is truncated.
- Official full mini evaluation remains pending.

Distributed wrapper status:

- `tools/dist_train.sh` and `tools/dist_test.sh` have been replaced with
  explicit unsupported stubs.
- They no longer call deprecated `python -m torch.distributed.launch`.
- Distributed train/test remains unsupported.

## Single-GPU Test Script Output Saving And Visualization

Command used:

```bash
python tools/test.py \
  projects/configs/bevformer/bevformer_tiny.py \
  ckpts/bevformer_tiny_epoch_24.pth \
  --max-samples 2 \
  --out outputs/test_outputs/test_max2.pkl \
  --show-dir outputs/test_outputs/vis \
  --show-index 0
```

Output artifacts:

- Output pickle path: `outputs/test_outputs/test_max2.pkl`
- Visualization path: `outputs/test_outputs/vis/test_sample_0_3e8750f331d7.png`

Reload verification:

- The output pickle reloads as a Python `list` with 2 outputs.
- Each output contains `pts_bbox`.
- `pts_bbox` contains:
  - `bboxes_3d`
  - `scores_3d`
  - `labels_3d`

Scope notes:

- This is single-GPU nuScenes-mini inference through `tools/test.py`.
- This is not official nuScenes evaluation.
- `tools/dist_test.sh` is an unsupported stub; distributed mode remains unported.
- No model code was modified for this report update.

Git diff stat at the time of this update:

```text
 projects/configs/bevformer/bevformer_tiny.py       |  16 +-
 .../modules/custom_base_transformer_layer.py       |  10 +-
 reports/bevformer_modern_port_status.md            | 448 ++++++++++++-
 tools/create_data.py                               |  17 +-
 tools/data_converter/create_gt_database.py         |   2 +-
 tools/data_converter/nuscenes_converter.py         |  39 +-
 tools/test.py                                      | 716 ++++++++++++++-------
 7 files changed, 1002 insertions(+), 246 deletions(-)
```

## nuScenes Mini Training Smoke Test

Command run:

```bash
python tools/debug/debug_train_bevformer_tiny_mini.py --max-iters 2 --no-checkpoint
```

Result:

- Dataset builds: yes.
  - Dataset type: `projects.mmdet3d_plugin.datasets.nuscenes_dataset.CustomNuScenesDataset`
  - Dataset length: 323 mini train samples.
  - Attached legacy metadata: `{'version': 'v1.0-mini'}`
- One real mini batch loads: yes.
  - Batch image shape: `(1, 3, 6, 3, 480, 800)`
  - `img_metas` queue keys: `[0, 1, 2]`
  - GT boxes type: `mmdet3d.structures.bbox_3d.lidar_box3d.LiDARInstance3DBoxes`
  - Iteration 1 GT labels shape: `(50,)`
  - Iteration 2 GT labels shape: `(51,)`
- Forward loss works: yes.
- Backward works: yes.
- Optimizer step works: yes.
- Evaluation was not run.
- Distributed training was not used.
- Full nuScenes trainval was not used.

Loss keys and values:

Iteration 1:

```text
loss_cls: 2.170586
loss_bbox: 2.340232
d0.loss_cls: 2.130126
d0.loss_bbox: 1.689647
d1.loss_cls: 2.082804
d1.loss_bbox: 1.693368
d2.loss_cls: 2.152829
d2.loss_bbox: 1.791789
d3.loss_cls: 2.182679
d3.loss_bbox: 1.910809
d4.loss_cls: 2.056765
d4.loss_bbox: 2.106497
total_loss: 24.308130
```

Iteration 2:

```text
loss_cls: 1.899188
loss_bbox: 1.645746
d0.loss_cls: 1.975094
d0.loss_bbox: 1.641891
d1.loss_cls: 1.960442
d1.loss_bbox: 1.599354
d2.loss_cls: 1.896326
d2.loss_bbox: 1.665616
d3.loss_cls: 1.817855
d3.loss_bbox: 1.661279
d4.loss_cls: 1.774689
d4.loss_bbox: 1.659356
total_loss: 21.196838
```

Compatibility notes:

- Added `tools/debug/debug_train_bevformer_tiny_mini.py` as a narrow single-GPU smoke script instead of rewriting `tools/train.py`.
- Applies the same modern model config fixes used by `tools/test.py`:
  - remove `cfg.model.pretrained`
  - `cfg.model.img_backbone.type = "mmdet.ResNet"`
  - `cfg.model.img_neck.type = "mmdet.FPN"`
  - `cfg.model.pts_bbox_head.type = "mmdet.BEVFormerHead"`
- Uses runtime transform compatibility aliases for the legacy BEVFormer train pipeline.
- Attaches legacy nuScenes info records from `data/nuscenes/nuscenes_infos_temporal_train.pkl`.
- Adds script-local dataset compatibility for legacy `get_ann_info` and `pre_pipeline` behavior.
- Disables the legacy dataset empty-GT filter inside the smoke script to avoid old `DataContainer._data` assumptions before returning the sample.
- Adds script-local tensor-based assigner cost and pseudo-sampler shims for MMDetection 3 compatibility. This is not a model architecture change.

Remaining limitations:

- This only proves 1-5 iteration single-GPU smoke training on nuScenes-mini.
- It does not prove full training stability, convergence, checkpoint saving, validation, or official nuScenes evaluation.
- It does not exercise distributed training or `tools/dist_train.sh`.
- It does not modify or validate `tools/train.py` yet.
- The command above used `--no-checkpoint`; checkpoint-initialized smoke training remains optional follow-up.

Git diff stat at the time of this update:

```text
 projects/configs/bevformer/bevformer_tiny.py       |  16 +-
 .../modules/custom_base_transformer_layer.py       |  10 +-
 reports/bevformer_modern_port_status.md            | 496 +++++++++++++-
 tools/create_data.py                               |  17 +-
 tools/data_converter/create_gt_database.py         |   2 +-
 tools/data_converter/nuscenes_converter.py         |  39 +-
 tools/test.py                                      | 716 ++++++++++++++-------
 7 files changed, 1050 insertions(+), 246 deletions(-)
```

Changed/untracked file relevant to this smoke test:

```text
?? tools/debug/debug_train_bevformer_tiny_mini.py
```

## nuScenes Mini Overfit Training Diagnostic

Command run:

```bash
python tools/debug/debug_overfit_bevformer_tiny_mini.py \
  --num-samples 8 \
  --max-iters 30 \
  --no-checkpoint \
  --work-dir outputs/train_smoke
```

Result:

- Subset size: first 8 samples from the nuScenes-mini train split.
- Iterations: 30 optimizer steps, cycling repeatedly over the fixed 8-sample subset.
- Dataset builds: yes.
- Real mini batches load repeatedly: yes.
- Forward loss works repeatedly: yes.
- Backward works repeatedly: yes.
- Optimizer steps work repeatedly: yes.
- Loss decreased: yes.
  - Initial average total loss over first 3 iterations: `22.741053`.
  - Final average total loss over last 3 iterations: `13.918399`.
- No NaNs or non-finite losses were observed.
- This used `--no-checkpoint`, so weights were initialized via `model.init_weights()`.
- Evaluation was not run.
- Distributed training was not used.
- Full nuScenes trainval was not used.

Artifacts:

```text
outputs/train_smoke/overfit_loss_log.csv
outputs/train_smoke/overfit_loss_curve.png
outputs/train_smoke/overfit_last.pth
```

Per-iteration logging includes total loss, `loss_cls`, `loss_bbox`, decoder loss keys, learning rate, gradient norm, and CUDA allocated/reserved memory.

Representative loss trend:

```text
iter=001 total_loss=24.301237 loss_cls=2.164821 loss_bbox=2.345174 grad_norm=30.647108
iter=010 total_loss=16.356071 loss_cls=1.245898 loss_bbox=1.482374 grad_norm=26.941691
iter=020 total_loss=14.427202 loss_cls=1.046930 loss_bbox=1.351265 grad_norm=20.481393
iter=030 total_loss=14.267797 loss_cls=0.964218 loss_bbox=1.427115 grad_norm=20.893341
```

Remaining limitations:

- This is still a diagnostic overfit/loss-decrease gate, not a full training run.
- It does not validate long-run stability, scheduler behavior, checkpoint resume, validation hooks, or official nuScenes metrics.
- It does not exercise `tools/train.py`, distributed training, or `tools/dist_train.sh`.
- Checkpoint-initialized overfit training remains optional because this successful run used `--no-checkpoint`.

Git diff stat at the time of this update:

```text
 projects/configs/bevformer/bevformer_tiny.py       |  16 +-
 .../modules/custom_base_transformer_layer.py       |  10 +-
 reports/bevformer_modern_port_status.md            | 602 ++++++++++++++++-
 tools/create_data.py                               |  17 +-
 tools/data_converter/create_gt_database.py         |   2 +-
 tools/data_converter/nuscenes_converter.py         |  39 +-
 tools/test.py                                      | 716 ++++++++++++++-------
 7 files changed, 1156 insertions(+), 246 deletions(-)
```

Changed/untracked file relevant to this diagnostic:

```text
?? tools/debug/debug_overfit_bevformer_tiny_mini.py
```

## Single-GPU Train Entrypoint Port

Command run:

```bash
python tools/train.py \
  projects/configs/bevformer/bevformer_tiny.py \
  --work-dir outputs/train_entrypoint_smoke \
  --max-iters 5 \
  --no-validate
```

Result:

- `tools/train.py` now runs as a modern single-GPU nuScenes-mini training diagnostic entrypoint.
- Dataset builds: yes.
  - Dataset type: `projects.mmdet3d_plugin.datasets.nuscenes_dataset.CustomNuScenesDataset`.
  - Dataset length: 323 mini train samples.
  - Attached metadata: `{'version': 'v1.0-mini'}`.
- Batch loads: yes.
- Forward loss works: yes.
- Backward works: yes.
- Optimizer step works: yes.
- Default checkpoint loading works with `ckpts/bevformer_tiny_epoch_24.pth`.
- Validation/evaluation was not run.
- Distributed training was not used.
- Full nuScenes trainval was not used.

Compatibility patches made:

- Replaced the old OpenMMLab 1.x `tools/train.py` internals with a narrow diagnostic path that reuses the proven helpers from `tools/debug/debug_train_bevformer_tiny_mini.py`.
- Uses `mmengine.config.Config` and `mmengine.config.DictAction`.
- Applies the known modern config fixes:
  - remove `cfg.model.pretrained`
  - `cfg.model.img_backbone.type = "mmdet.ResNet"`
  - `cfg.model.img_neck.type = "mmdet.FPN"`
  - `cfg.model.pts_bbox_head.type = "mmdet.BEVFormerHead"`
- Reuses the runtime legacy dataset, transform, assigner-cost, and pseudo-sampler compatibility path from the debug training smoke script.
- Unsupported distributed, validation, resume, and autoscale paths now raise clear `NotImplementedError` messages.

Loss behavior from the 5-iteration smoke run:

```text
iter=001 total_loss=8.891555 loss_cls=1.052448 loss_bbox=0.839082
iter=002 total_loss=7.987552 loss_cls=0.636741 loss_bbox=0.794551
iter=003 total_loss=8.485617 loss_cls=0.630582 loss_bbox=0.861024
iter=004 total_loss=7.656710 loss_cls=0.547433 loss_bbox=0.793938
iter=005 total_loss=7.417868 loss_cls=0.516811 loss_bbox=0.744264
```

Saved files:

```text
outputs/train_entrypoint_smoke/bevformer_tiny.py
outputs/train_entrypoint_smoke/train_loss_log.csv
outputs/train_entrypoint_smoke/train_entrypoint_last.pth
```

Remaining limitations:

- This is still diagnostic single-GPU training.
- It is not full 24-epoch training.
- Distributed training remains unported.
- The validation loop remains unported for `tools/train.py`; use `--no-validate`.
- Scheduler, hooks, resume behavior, validation loops, and long-run stability are still pending.

Git diff stat at the time of this update:

```text
 projects/configs/bevformer/bevformer_tiny.py       |  16 +-
 .../modules/custom_base_transformer_layer.py       |  10 +-
 reports/bevformer_modern_port_status.md            | 677 ++++++++++++++++++-
 tools/create_data.py                               |  17 +-
 tools/data_converter/create_gt_database.py         |   2 +-
 tools/data_converter/nuscenes_converter.py         |  39 +-
 tools/test.py                                      | 716 ++++++++++++++-------
 tools/train.py                                     | 503 ++++++++-------
 8 files changed, 1499 insertions(+), 481 deletions(-)
```

## Cleanup and Compatibility Package Refactor

The latest cleanup reorganized the diagnostic scripts and moved reusable
compatibility code out of public entrypoints.

Debug script organization:

- Debug and investigation scripts were moved under `tools/debug/`.
- Package markers were added so public scripts can import shared utilities
  cleanly:
  - `tools/__init__.py`
  - `tools/debug/__init__.py`

Reusable compatibility package:

- Added `projects/mmdet3d_plugin/modern_compat/` as the shared compatibility
  package for the modern OpenMMLab port.
- Current package files:
  - `projects/mmdet3d_plugin/modern_compat/__init__.py`
  - `projects/mmdet3d_plugin/modern_compat/config.py`
  - `projects/mmdet3d_plugin/modern_compat/dataset.py`
  - `projects/mmdet3d_plugin/modern_compat/transforms.py`
  - `projects/mmdet3d_plugin/modern_compat/training.py`
  - `projects/mmdet3d_plugin/modern_compat/checkpoint.py`

Public entrypoints cleaned:

- `tools/train.py`
- `tools/test.py`

`tools/train.py` now imports the higher-level training and compatibility APIs
from `projects.mmdet3d_plugin.modern_compat`:

- `prepare_config`
- `initialize_modern_compat`
- `build_train_dataset`
- `build_train_model`
- `choose_device`
- `prepare_train_batch`
- `parse_losses`

`tools/train.py` no longer directly imports the lower-level compatibility
helpers that implement the details of the port, including:

- `apply_config_fixes`
- `attach_legacy_infos`
- `configure_train_dataset`
- `register_runtime_compat_transforms`
- `patch_legacy_assigner_and_sampler`

Transform compatibility scope:

- `modern_compat/transforms.py` is intentionally narrow. It only provides
  runtime shims for legacy transform names that are missing under the modern
  stack.
- Current shimmed legacy names:
  - `LoadMultiViewImageFromFiles`
  - `DefaultFormatBundle3D`
- The full data pipeline does not come only from
  `modern_compat/transforms.py`. It still comes from the BEVFormer config and
  the original/plugin/MMDetection3D transform registrations. The compatibility
  module only fills the missing legacy names needed by the old BEVFormer
  pipeline.

Latest validation:

```bash
python -m py_compile ...
```

Result:

- Passed for public entrypoints.
- Passed for debug scripts under `tools/debug/`.
- Passed for the reusable compatibility package under
  `projects/mmdet3d_plugin/modern_compat/`.

Command run:

```bash
python tools/train.py \
  projects/configs/bevformer/bevformer_tiny.py \
  --work-dir outputs/train_entrypoint_smoke_clean2 \
  --max-iters 5 \
  --no-validate
```

Result:

- Passed.
- Confirmed the cleaned `tools/train.py` entrypoint can still build the
  nuScenes-mini train dataset, load batches, run forward loss, run backward,
  and take optimizer steps through the higher-level compatibility APIs.

Command run:

```bash
python tools/test.py \
  projects/configs/bevformer/bevformer_tiny.py \
  ckpts/bevformer_tiny_epoch_24.pth \
  --max-samples 2 \
  --out outputs/test_outputs_clean2/test_max2.pkl \
  --show-dir outputs/test_outputs_clean2/vis \
  --show-index 0
```

Result:

- Passed.
- Confirmed the cleaned `tools/test.py` entrypoint can still load the
  pretrained BEVFormer tiny checkpoint, run inference on two nuScenes-mini
  validation samples, save predictions, and generate visualization output for
  the requested sample.

Command run:

```bash
python tools/debug/trace_one_training_sample_shapes.py
```

Result:

- Passed.
- Confirmed the moved debug script still traces one real training sample
  through the image feature path, BEV encoder, detection head, and loss path.

Remaining limitations after cleanup:

- The validated path is still single-GPU.
- The validation loop in `tools/train.py` is not ported; use `--no-validate`.
- Distributed train/test remain unsupported.
- Old distributed wrappers were replaced with explicit unsupported stubs:
  - `tools/dist_train.sh`
  - `tools/dist_test.sh`
- These stubs fail clearly instead of routing to deprecated
  `python -m torch.distributed.launch`, which avoids users accidentally
  thinking distributed training/testing is currently supported.
- `tools/train.py` remains a diagnostic/smoke-training entrypoint, not a full
  official OpenMMLab training runner.
- Full nuScenes trainval training and official nuScenes metric reproduction
  have not been claimed.
- Generated outputs and checkpoints should not be committed.

