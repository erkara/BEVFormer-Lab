import warnings
warnings.filterwarnings("ignore")

import argparse
import copy
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.debug_train_bevformer_tiny_mini import (  # noqa: E402
    CHECKPOINT_PATH,
    CONFIG_PATH,
    apply_config_fixes,
    attach_legacy_infos,
    build_dataset,
    build_model,
    choose_device,
    configure_train_dataset,
    parse_losses,
    patch_legacy_assigner_and_sampler,
    prepare_train_batch,
    print_header,
    register_runtime_compat_transforms,
)


def shape_of(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    if isinstance(value, np.ndarray):
        return tuple(value.shape)
    if hasattr(value, "tensor") and torch.is_tensor(value.tensor):
        return tuple(value.tensor.shape)
    if isinstance(value, dict):
        return {key: shape_of(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [shape_of(item) for item in value]
    return type(value).__name__


def print_shape(label, value):
    print(f"{label}: {shape_of(value)}")


def print_meta_structure(img_metas):
    queue = img_metas[0]
    print(f"img_metas: list len={len(img_metas)}")
    print(f"sample queue: {type(queue).__name__} len={len(queue)}")
    current_meta = queue[len(queue) - 1]
    print(f"current meta keys: {sorted(current_meta.keys())}")
    for key in ("filename", "img_shape", "pad_shape", "lidar2img", "can_bus", "prev_bev_exists"):
        value = current_meta.get(key, "<missing>")
        print_shape(f"current meta {key}", value)


def print_head_output(outputs):
    if not isinstance(outputs, dict):
        print_shape("output", outputs)
        return
    for key in ("bev_embed", "all_cls_scores", "all_bbox_preds", "enc_cls_scores", "enc_bbox_preds"):
        if key in outputs:
            print_shape(key, outputs[key])


def parse_args():
    parser = argparse.ArgumentParser(description="Trace one BEVFormer tiny nuScenes-mini training sample")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="checkpoint path")
    parser.add_argument("--no-checkpoint", action="store_true", help="skip checkpoint loading and initialize weights")
    parser.add_argument("--device", default="auto", help="device, e.g. auto, cuda, cuda:0, cpu")
    parser.add_argument("--start-index", type=int, default=0, help="train dataset index to trace")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.disable(logging.INFO)
    torch.manual_seed(0)

    cfg = Config.fromfile(CONFIG_PATH)
    apply_config_fixes(cfg)

    init_default_scope("mmdet3d")
    register_runtime_compat_transforms()
    train_cfg = configure_train_dataset(cfg)
    dataset = build_dataset(train_cfg)
    attach_legacy_infos(dataset, train_cfg.ann_file, list(cfg.class_names))

    device, device_reason = choose_device(args.device)
    print_header("setup")
    print(f"config: {CONFIG_PATH}")
    print(f"dataset length: {len(dataset)}")
    print(f"selected device: {device} ({device_reason})")

    model = build_model(cfg, args.checkpoint, args.no_checkpoint, device)
    patch_legacy_assigner_and_sampler(model)

    dataset_index = args.start_index % len(dataset)
    batch = prepare_train_batch(dataset, dataset_index, device)
    len_queue = batch["img"].size(1)
    current_img = batch["img"][:, len_queue - 1, ...]
    current_img_metas = [copy.deepcopy(each[len_queue - 1]) for each in batch["img_metas"]]

    print_header("batch")
    print(f"dataset_index: {dataset_index}")
    print_shape("img", batch["img"])
    print_shape("gt_bboxes_3d", batch["gt_bboxes_3d"])
    print_shape("gt_labels_3d", batch["gt_labels_3d"])

    print_header("img_metas structure")
    print_meta_structure(batch["img_metas"])

    with torch.no_grad():
        print_header("extract_img_feat")
        print_shape("input img", current_img)
        img_feats = model.extract_img_feat(current_img.clone(), copy.deepcopy(current_img_metas))
        for index, feat in enumerate(img_feats):
            print_shape(f"output[{index}]", feat)

        print_header("extract_feat")
        extracted_feats = model.extract_feat(img=current_img.clone(), img_metas=copy.deepcopy(current_img_metas))
        for index, feat in enumerate(extracted_feats):
            print_shape(f"output[{index}]", feat)

        print_header("only_bev")
        bev_embed = model.pts_bbox_head(
            extracted_feats,
            copy.deepcopy(current_img_metas),
            prev_bev=None,
            only_bev=True,
        )
        print_shape("bev_embed", bev_embed)

        print_header("pts_bbox_head")
        head_outputs = model.pts_bbox_head(extracted_feats, copy.deepcopy(current_img_metas), prev_bev=None)
        print_head_output(head_outputs)

    print_header("forward_train loss")
    losses = model(return_loss=True, **batch)
    total_loss, scalar_losses = parse_losses(losses)
    for key, value in scalar_losses.items():
        print(f"{key}: {value:.6f}")
    print(f"total_loss: {float(total_loss.detach().cpu()):.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print("\n=== exception ===")
        print(f"exception type: {type(exc).__name__}")
        print(f"exception: {exc}")
        print("\n--- traceback ---")
        traceback.print_exc()
        raise SystemExit(1)
