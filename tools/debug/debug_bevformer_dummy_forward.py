import copy
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmengine.config import Config

import mmdet.models
import mmdet3d.models
import projects.mmdet3d_plugin

from mmdet.registry import MODELS as MMDET_MODELS
from mmdet3d.structures import LiDARInstance3DBoxes


CONFIG_PATH = "projects/configs/bevformer/bevformer_tiny.py"
NUM_CAMS = 6
IMAGE_H = 128
IMAGE_W = 128


def tensor_shape(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return [tensor_shape(item) for item in value]
    if isinstance(value, dict):
        return {key: tensor_shape(item) for key, item in value.items()}
    return type(value).__name__


def print_shapes(label, value):
    print(f"{label}: {tensor_shape(value)}")


def make_lidar2img(num_cams, image_h, image_w):
    scale = min(image_h, image_w) / 160.0
    lidar2img = []
    for _ in range(num_cams):
        mat = np.eye(4, dtype=np.float32)
        mat[0, :] = np.array([scale, 0.0, 0.0, image_w / 2.0], dtype=np.float32)
        mat[1, :] = np.array([0.0, scale, 0.0, image_h / 2.0], dtype=np.float32)
        mat[2, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        mat[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        lidar2img.append(mat)
    return lidar2img


def make_meta(num_cams, image_h, image_w, prev_bev_exists=False):
    can_bus = np.zeros(18, dtype=np.float32)
    return {
        "filename": [f"dummy_cam_{idx}.jpg" for idx in range(num_cams)],
        "ori_shape": [(image_h, image_w, 3) for _ in range(num_cams)],
        "img_shape": [(image_h, image_w, 3) for _ in range(num_cams)],
        "pad_shape": [(image_h, image_w, 3) for _ in range(num_cams)],
        "lidar2img": make_lidar2img(num_cams, image_h, image_w),
        "can_bus": can_bus,
        "prev_bev_exists": prev_bev_exists,
        "scene_token": "dummy_scene",
        "sample_idx": "dummy_sample",
        "box_type_3d": LiDARInstance3DBoxes,
    }


def apply_config_fixes(cfg):
    cfg.model.pop("pretrained", None)
    cfg.model.img_backbone.type = "mmdet.ResNet"
    cfg.model.img_neck.type = "mmdet.FPN"
    cfg.model.pts_bbox_head.type = "mmdet.BEVFormerHead"


def choose_device():
    if not torch.cuda.is_available():
        return torch.device("cpu"), "torch.cuda.is_available() is false"
    try:
        torch.empty(1, device="cuda")
        return torch.device("cuda"), "CUDA is available and a test allocation succeeded"
    except Exception as exc:
        return torch.device("cpu"), f"CUDA test allocation failed: {type(exc).__name__}: {exc}"


def install_cuda_extension_probe():
    state = {
        "hook_installed": False,
        "forward_called": False,
        "install_error": None,
    }
    try:
        from projects.mmdet3d_plugin.bevformer.modules import multi_scale_deformable_attn_function as msda

        original_forward = msda.ext_module.ms_deform_attn_forward

        def wrapped_forward(*args, **kwargs):
            state["forward_called"] = True
            return original_forward(*args, **kwargs)

        msda.ext_module.ms_deform_attn_forward = wrapped_forward
        state["hook_installed"] = True
    except Exception as exc:
        state["install_error"] = f"{type(exc).__name__}: {exc}"
    return state


def run_stage(name, fn, context):
    print(f"\n=== stage: {name} ===")
    try:
        result = fn()
    except Exception as exc:
        print(f"stage failed: {name}")
        print(f"exception type: {type(exc).__name__}")
        print(f"exception: {exc}")
        print("\n--- tensor shapes used ---")
        for key, value in context.items():
            print_shapes(key, value)
        print("\n--- metadata keys used ---")
        metas = context.get("img_metas")
        if metas:
            print(sorted(metas[0].keys()))
        print("\n--- traceback ---")
        traceback.print_exc()
        return None, False
    print(f"stage ok: {name}")
    print_shapes(f"{name} output", result)
    return result, True


def summarize_outputs(outputs):
    if not isinstance(outputs, dict):
        return
    for key, value in outputs.items():
        print_shapes(f"head output {key}", value)


def main():
    logging.disable(logging.INFO)
    torch.manual_seed(0)

    print(f"config: {CONFIG_PATH}")
    cfg = Config.fromfile(CONFIG_PATH)
    apply_config_fixes(cfg)

    queue_length = int(getattr(cfg, "queue_length", 1))
    print(f"queue_length from config: {queue_length}")
    print(f"dummy image shape: B=1, len_queue={queue_length}, num_cams={NUM_CAMS}, C=3, H={IMAGE_H}, W={IMAGE_W}")

    extension_probe = install_cuda_extension_probe()
    print(f"cuda extension hook installed: {extension_probe['hook_installed']}")
    if extension_probe["install_error"]:
        print(f"cuda extension hook install error: {extension_probe['install_error']}")

    device, device_reason = choose_device()
    print(f"selected device: {device} ({device_reason})")

    model = MMDET_MODELS.build(cfg.model)
    model.eval()
    model.to(device)

    img = torch.randn(1, NUM_CAMS, 3, IMAGE_H, IMAGE_W, device=device)
    img_queue = torch.randn(1, queue_length, NUM_CAMS, 3, IMAGE_H, IMAGE_W, device=device)
    img_metas = [make_meta(NUM_CAMS, IMAGE_H, IMAGE_W, prev_bev_exists=False)]
    img_metas_queue = [[make_meta(NUM_CAMS, IMAGE_H, IMAGE_W, prev_bev_exists=(idx > 0))
                        for idx in range(queue_length)]]

    context = {
        "img": img,
        "img_queue": img_queue,
        "img_metas": img_metas,
        "img_metas_queue": img_metas_queue,
    }

    print("\n--- initial tensor shapes ---")
    for key, value in context.items():
        print_shapes(key, value)
    print("\n--- metadata keys ---")
    print(sorted(img_metas[0].keys()))

    image_backbone_ok = False
    head_reached = False

    with torch.no_grad():
        img_feats, ok = run_stage(
            "model.extract_img_feat",
            lambda: model.extract_img_feat(img.clone(), copy.deepcopy(img_metas)),
            context,
        )
        if not ok:
            print("\n=== summary ===")
            print("image backbone/FPN forward worked: no")
            print("BEVFormer transformer/head forward reached: no")
            print(f"CUDA extension path invoked: {extension_probe['forward_called']}")
            return 1
        image_backbone_ok = True
        context["img_feats"] = img_feats

        extracted_feats, ok = run_stage(
            "model.extract_feat",
            lambda: model.extract_feat(img=img.clone(), img_metas=copy.deepcopy(img_metas)),
            context,
        )
        if not ok:
            print("\n=== summary ===")
            print(f"image backbone/FPN forward worked: {image_backbone_ok}")
            print("BEVFormer transformer/head forward reached: no")
            print(f"CUDA extension path invoked: {extension_probe['forward_called']}")
            return 1
        context["extracted_feats"] = extracted_feats

        bev_only, ok = run_stage(
            "model.pts_bbox_head(..., only_bev=True)",
            lambda: model.pts_bbox_head(extracted_feats, copy.deepcopy(img_metas), prev_bev=None, only_bev=True),
            context,
        )
        head_reached = True
        if not ok:
            print("\n=== summary ===")
            print(f"image backbone/FPN forward worked: {image_backbone_ok}")
            print(f"BEVFormer transformer/head forward reached: {head_reached}")
            print(f"CUDA extension path invoked: {extension_probe['forward_called']}")
            return 1
        context["bev_only"] = bev_only

        head_outputs, ok = run_stage(
            "model.pts_bbox_head",
            lambda: model.pts_bbox_head(extracted_feats, copy.deepcopy(img_metas), prev_bev=None),
            context,
        )
        if not ok:
            print("\n=== summary ===")
            print(f"image backbone/FPN forward worked: {image_backbone_ok}")
            print(f"BEVFormer transformer/head forward reached: {head_reached}")
            print(f"CUDA extension path invoked: {extension_probe['forward_called']}")
            return 1
        summarize_outputs(head_outputs)
        context["head_outputs"] = head_outputs

        _, ok = run_stage(
            "model.forward(return_loss=False)",
            lambda: model(return_loss=False, img=img.clone(), img_metas=[copy.deepcopy(img_metas)]),
            context,
        )
        if not ok:
            print("\n=== summary ===")
            print(f"image backbone/FPN forward worked: {image_backbone_ok}")
            print(f"BEVFormer transformer/head forward reached: {head_reached}")
            print(f"CUDA extension path invoked: {extension_probe['forward_called']}")
            return 1

    print("\n=== summary ===")
    print(f"image backbone/FPN forward worked: {image_backbone_ok}")
    print(f"BEVFormer transformer/head forward reached: {head_reached}")
    print("first real blocker: none in staged dummy eval forward")
    print(f"CUDA extension path invoked: {extension_probe['forward_called']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
