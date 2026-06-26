import argparse
import logging
import math
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon, Rectangle
from mmengine.config import Config
from mmengine.registry import init_default_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.debug_bevformer_checkpoint_load import (  # noqa: E402
    load_checkpoint_modern_safe,
    run_full_forward,
)
from tools.debug_nuscenes_real_full_forward import (  # noqa: E402
    CONFIG_PATH,
    choose_device,
    apply_config_fixes,
    configure_val_dataset,
    build_dataset,
    attach_legacy_infos,
    get_primary_item,
    unwrap_datacontainer,
    register_runtime_compat_transforms,
)

import mmdet.models  # noqa: E402,F401
import mmdet3d.models  # noqa: E402,F401
import projects.mmdet3d_plugin  # noqa: E402,F401
from mmdet.registry import MODELS as MMDET_MODELS  # noqa: E402


DEFAULT_CHECKPOINT = "ckpts/bevformer_tiny_epoch_24.pth"
DEFAULT_OUTPUT_DIR = "outputs"


def print_header(name):
    print(f"\n=== {name} ===")


def parse_args():
    parser = argparse.ArgumentParser(description="Run BEVFormer tiny on one mini sample and save a simple BEV PNG")
    parser.add_argument("--index", type=int, default=0, help="Validation dataset index to visualize")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint path")
    parser.add_argument("--score-thr", type=float, default=0.15, help="Prediction score threshold")
    return parser.parse_args()


def prepare_sample(cfg, index):
    print_header("register dataset compatibility")
    init_default_scope("mmdet3d")
    print("default scope set: mmdet3d")
    register_runtime_compat_transforms()

    print_header("build validation dataset")
    val_cfg = configure_val_dataset(cfg)
    dataset = build_dataset(val_cfg)
    metadata = attach_legacy_infos(dataset, val_cfg.ann_file)
    print(f"attached legacy metadata: {metadata}")
    print(f"dataset type: {type(dataset)}")
    print(f"dataset length: {len(dataset)}")
    if index < 0 or index >= len(dataset):
        raise IndexError(f"index {index} is outside dataset length {len(dataset)}")

    print_header("load validation item")
    raw_item = dataset[index]
    item, item_reason = get_primary_item(raw_item)
    print(f"requested index: {index}")
    print(f"raw item type: {type(raw_item)}")
    print(f"effective item selection: {item_reason}")
    print(f"effective item keys: {list(item.keys()) if isinstance(item, dict) else '<not dict>'}")

    img_tensor = unwrap_datacontainer(item["img"])
    metadata_dict = unwrap_datacontainer(item["img_metas"])
    if not torch.is_tensor(img_tensor):
        raise TypeError(f"expected image tensor, got {type(img_tensor)}")
    if not isinstance(metadata_dict, dict):
        raise TypeError(f"expected metadata dict, got {type(metadata_dict)}")
    print(f"unwrapped img shape: {tuple(img_tensor.shape)}")
    print_sample_info(metadata_dict)
    return img_tensor, metadata_dict


def print_sample_info(meta):
    filenames = meta.get("filename") or []
    print(f"sample_idx: {meta.get('sample_idx', '<missing>')}")
    print(f"scene_token: {meta.get('scene_token', '<missing>')}")
    print(f"first camera filename: {filenames[0] if filenames else '<missing>'}")


def build_model_and_load(cfg, checkpoint, device):
    print_header("build model")
    init_default_scope("mmdet")
    print("default scope set: mmdet")
    model = MMDET_MODELS.build(cfg.model)
    model.eval()
    model.to(device)

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint_loaded, load_error = load_checkpoint_modern_safe(model, checkpoint_path)
    print(f"checkpoint loaded: {'yes' if checkpoint_loaded else 'no'}")
    if load_error is not None:
        print(f"checkpoint load error: {type(load_error).__name__}: {load_error}")
    if not checkpoint_loaded:
        raise RuntimeError(f"failed to load checkpoint: {checkpoint_path}")
    return model


def extract_predictions(outputs):
    if not isinstance(outputs, list) or not outputs:
        raise TypeError(f"expected non-empty list output, got {type(outputs)}")
    result = outputs[0]["pts_bbox"]
    boxes = result["bboxes_3d"].tensor.detach().cpu()
    scores = result["scores_3d"].detach().cpu()
    labels = result["labels_3d"].detach().cpu()
    return boxes, scores, labels


def select_predictions(boxes, scores, labels, score_thr):
    raw_count = int(scores.numel())
    keep = scores >= score_thr
    fallback = False
    if int(keep.sum()) == 0:
        fallback = True
        topk = min(20, raw_count)
        top_idx = torch.argsort(scores, descending=True)[:topk]
        selected_idx = top_idx
    else:
        selected_idx = torch.nonzero(keep, as_tuple=False).flatten()
        selected_idx = selected_idx[torch.argsort(scores[selected_idx], descending=True)]
    return boxes[selected_idx], scores[selected_idx], labels[selected_idx], raw_count, fallback


def box_corners_bev(box):
    x, y, _, dx, dy, _, yaw = box[:7].tolist()
    hx = dx / 2.0
    hy = dy / 2.0
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], dtype=np.float32)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def print_prediction_table(boxes, scores, labels, class_names):
    print_header("prediction table")
    print("rank class score x y z dx dy dz yaw")
    for rank, (box, score, label) in enumerate(zip(boxes, scores, labels), start=1):
        label_int = int(label.item())
        class_name = class_names[label_int] if 0 <= label_int < len(class_names) else f"label_{label_int}"
        x, y, z, dx, dy, dz, yaw = box[:7].tolist()
        print(
            f"{rank:03d} {class_name} {float(score):.4f} "
            f"{x:.3f} {y:.3f} {z:.3f} {dx:.3f} {dy:.3f} {dz:.3f} {yaw:.3f}"
        )


def save_bev_plot(boxes, scores, labels, class_names, point_cloud_range, meta, index, score_thr):
    out_dir = Path(DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    thr_token = f"{score_thr:g}".replace(".", "p")
    out_path = out_dir / f"debug_bevformer_bev_index_{index}_thr_{thr_token}.png"

    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor="black", linewidth=1.5))
    ax.scatter([0], [0], c="red", marker="x", s=80, label="ego")

    cmap = plt.get_cmap("tab10")
    for box, score, label in zip(boxes, scores, labels):
        label_int = int(label.item())
        class_name = class_names[label_int] if 0 <= label_int < len(class_names) else f"label_{label_int}"
        corners = box_corners_bev(box)
        color = cmap(label_int % 10)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=1.5))
        x, y = box[0].item(), box[1].item()
        ax.text(x, y, f"{class_name} {float(score):.2f}", fontsize=7, color=color)

    ax.set_xlim(x_min - 2, x_max + 2)
    ax.set_ylim(y_min - 2, y_max + 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="upper right")
    sample_idx = meta.get("sample_idx", "<missing>")
    ax.set_title(f"BEVFormer BEV predictions sample={sample_idx} score_thr={score_thr:g}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    logging.disable(logging.INFO)
    torch.manual_seed(0)
    args = parse_args()

    print_header("load config")
    print(f"config: {CONFIG_PATH}")
    cfg = Config.fromfile(CONFIG_PATH)
    apply_config_fixes(cfg)
    print("modern config fixes applied")

    img_tensor, metadata_dict = prepare_sample(cfg, args.index)

    device, device_reason = choose_device()
    print_header("device")
    print(f"selected device: {device} ({device_reason})")

    model = build_model_and_load(cfg, args.checkpoint, device)

    print_header("full forward")
    outputs = run_full_forward(model, img_tensor, metadata_dict, device)
    boxes, scores, labels = extract_predictions(outputs)
    selected_boxes, selected_scores, selected_labels, raw_count, fallback = select_predictions(
        boxes, scores, labels, args.score_thr)
    if fallback:
        print(f"no predictions met score threshold {args.score_thr:g}; falling back to top {len(selected_scores)} by score")

    class_names = list(cfg.class_names)
    print_prediction_table(selected_boxes, selected_scores, selected_labels, class_names)
    saved_path = save_bev_plot(
        selected_boxes,
        selected_scores,
        selected_labels,
        class_names,
        cfg.point_cloud_range,
        metadata_dict,
        args.index,
        args.score_thr,
    )

    print_header("summary")
    print(f"saved image path: {saved_path}")
    print(f"number of raw predictions: {raw_count}")
    print(f"number of predictions after threshold: {len(selected_scores)}")
    print(f"sample_idx: {metadata_dict.get('sample_idx', '<missing>')}")
    print(f"scene_token: {metadata_dict.get('scene_token', '<missing>')}")
    filenames = metadata_dict.get("filename") or []
    print(f"first camera filename: {filenames[0] if filenames else '<missing>'}")
    print("no camera projection run")
    print("no evaluation run")
    print("no training run")
    print("no model code modified")
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
