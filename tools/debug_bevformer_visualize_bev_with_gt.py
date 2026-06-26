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
from matplotlib.lines import Line2D
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
    parser = argparse.ArgumentParser(description="Run BEVFormer tiny and save BEV predictions with GT overlay")
    parser.add_argument("--index", type=int, default=0, help="Validation dataset index to visualize")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint path")
    parser.add_argument("--score-thr", type=float, default=0.15, help="Prediction score threshold")
    parser.add_argument("--top-k", type=int, default=10,
                        help="After thresholding, keep only the top K predictions by score. Use <=0 to keep all.")
    parser.add_argument("--no-labels", action="store_true", help="Disable text labels on the BEV plot")
    parser.add_argument("--label-top-k", type=int, default=5,
                        help="When labels are enabled, label only the top K predictions by score")
    parser.add_argument("--label-gt", action="store_true", help="Label up to the first 10 GT boxes")
    parser.add_argument("--zoom-preds", action="store_true",
                        help="Zoom plot limits around selected predictions and nearby GT boxes")
    parser.add_argument("--class-filter", default=None,
                        help="Only plot predictions and GT boxes for this class name")
    parser.add_argument("--min-gt-distance-to-pred", type=float, default=8.0,
                        help="When zooming, include GT boxes within this distance of selected predictions")
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
    return dataset, img_tensor, metadata_dict


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


def select_predictions(boxes, scores, labels, score_thr, top_k):
    raw_count = int(scores.numel())
    keep = scores >= score_thr
    fallback = False
    if int(keep.sum()) == 0:
        fallback = True
        topk = min(20, raw_count)
        thresholded_idx = torch.argsort(scores, descending=True)[:topk]
    else:
        thresholded_idx = torch.nonzero(keep, as_tuple=False).flatten()
        thresholded_idx = thresholded_idx[torch.argsort(scores[thresholded_idx], descending=True)]
    threshold_count = int(thresholded_idx.numel())
    if top_k > 0:
        selected_idx = thresholded_idx[:top_k]
    else:
        selected_idx = thresholded_idx
    return boxes[selected_idx], scores[selected_idx], labels[selected_idx], raw_count, threshold_count, fallback


def filter_predictions_by_class(boxes, scores, labels, class_filter, class_names):
    if class_filter is None:
        return boxes, scores, labels, None
    keep = [label_to_class(label, class_names) == class_filter for label in labels]
    if keep:
        keep_tensor = torch.tensor(keep, dtype=torch.bool)
    else:
        keep_tensor = torch.empty((0,), dtype=torch.bool)
    return boxes[keep_tensor], scores[keep_tensor], labels[keep_tensor], int(keep_tensor.sum().item())


def filter_gt_by_class(boxes, names, class_filter):
    if class_filter is None:
        return boxes, names
    keep = [str(name) == class_filter for name in names]
    if keep:
        keep_tensor = torch.tensor(keep, dtype=torch.bool)
    else:
        keep_tensor = torch.empty((0,), dtype=torch.bool)
    filtered_names = [name for name, should_keep in zip(names, keep) if should_keep]
    return boxes[keep_tensor], filtered_names


def filter_gt_near_predictions(gt_boxes, gt_names, pred_boxes, distance):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return gt_boxes[:0], []
    gt_centers = gt_boxes[:, :2]
    pred_centers = pred_boxes[:, :2]
    dists = torch.cdist(gt_centers.float(), pred_centers.float())
    keep = dists.min(dim=1).values <= float(distance)
    filtered_names = [name for name, should_keep in zip(gt_names, keep.tolist()) if should_keep]
    return gt_boxes[keep], filtered_names


def extract_gt_from_dataset(dataset, index, class_names):
    print_header("extract ground truth")
    info = None
    source = None
    if hasattr(dataset, "data_infos") and len(dataset.data_infos) > index:
        candidate = dataset.data_infos[index]
        if isinstance(candidate, dict) and "gt_boxes" in candidate:
            info = candidate
            source = "dataset.data_infos[index]"
    if info is None and hasattr(dataset, "get_ann_info"):
        ann = dataset.get_ann_info(index)
        if isinstance(ann, dict):
            info = ann
            source = "dataset.get_ann_info(index)"
    if info is None:
        print("GT source used: <none>")
        return torch.empty((0, 7), dtype=torch.float32), torch.empty((0,), dtype=torch.long), [], "<none>"

    print(f"GT source used: {source}")
    gt_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32)
    gt_names = info.get("gt_names")
    valid_flag = info.get("valid_flag")
    if gt_boxes.ndim == 1:
        gt_boxes = gt_boxes.reshape(0, 7)
    if valid_flag is not None:
        valid_flag = np.asarray(valid_flag).astype(bool)
        gt_boxes = gt_boxes[valid_flag]
        if gt_names is not None:
            gt_names = np.asarray(gt_names)[valid_flag]
    if gt_names is None:
        names = ["<unknown>" for _ in range(len(gt_boxes))]
        labels = torch.full((len(gt_boxes),), -1, dtype=torch.long)
    else:
        names = [str(name) for name in list(gt_names)]
        label_values = [class_names.index(name) if name in class_names else -1 for name in names]
        labels = torch.tensor(label_values, dtype=torch.long)
    boxes = torch.as_tensor(gt_boxes[:, :7], dtype=torch.float32)
    return boxes, labels, names, source


def box_corners_bev(box):
    x, y, _, dx, dy, _, yaw = box[:7].tolist()
    hx = dx / 2.0
    hy = dy / 2.0
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], dtype=np.float32)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def label_to_class(label, class_names):
    label_int = int(label.item()) if torch.is_tensor(label) else int(label)
    return class_names[label_int] if 0 <= label_int < len(class_names) else f"label_{label_int}"


def print_prediction_table(boxes, scores, labels, class_names):
    print_header("prediction table")
    print("rank class score x y z dx dy dz yaw")
    for rank, (box, score, label) in enumerate(zip(boxes, scores, labels), start=1):
        class_name = label_to_class(label, class_names)
        x, y, z, dx, dy, dz, yaw = box[:7].tolist()
        print(
            f"{rank:03d} {class_name} {float(score):.4f} "
            f"{x:.3f} {y:.3f} {z:.3f} {dx:.3f} {dy:.3f} {dz:.3f} {yaw:.3f}"
        )


def print_gt_table(boxes, names):
    print_header("ground-truth table")
    print("rank class x y z dx dy dz yaw")
    for rank, (box, name) in enumerate(zip(boxes, names), start=1):
        x, y, z, dx, dy, dz, yaw = box[:7].tolist()
        print(f"{rank:03d} {name} {x:.3f} {y:.3f} {z:.3f} {dx:.3f} {dy:.3f} {dz:.3f} {yaw:.3f}")


def output_path_for(index, score_thr, top_k, class_filter, zoom_preds):
    out_dir = Path(DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    thr_token = f"{score_thr:g}".replace(".", "p")
    parts = [
        "debug_bevformer_bev_with_gt",
        f"index_{index}",
        f"thr_{thr_token}",
        f"topk_{top_k}",
    ]
    if class_filter:
        safe_class = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in class_filter)
        parts.append(safe_class)
    parts.append("clear")
    if zoom_preds:
        parts.append("zoom")
    return out_dir / ("_".join(parts) + ".png")


def get_zoom_limits(pred_boxes, gt_boxes, point_cloud_range, padding):
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return x_min - 2, x_max + 2, y_min - 2, y_max + 2
    centers = []
    if len(pred_boxes) > 0:
        centers.append(pred_boxes[:, :2].detach().cpu().numpy())
    if len(gt_boxes) > 0:
        centers.append(gt_boxes[:, :2].detach().cpu().numpy())
    all_centers = np.concatenate(centers, axis=0)
    zx_min = max(x_min - 2, float(all_centers[:, 0].min()) - padding)
    zx_max = min(x_max + 2, float(all_centers[:, 0].max()) + padding)
    zy_min = max(y_min - 2, float(all_centers[:, 1].min()) - padding)
    zy_max = min(y_max + 2, float(all_centers[:, 1].max()) + padding)
    return zx_min, zx_max, zy_min, zy_max


def save_bev_plot(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_names, class_names,
                  point_cloud_range, meta, index, score_thr, top_k, no_labels,
                  label_top_k, label_gt, gt_source, class_filter, zoom_preds):
    out_path = output_path_for(index, score_thr, top_k, class_filter, zoom_preds)

    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=False, edgecolor="0.55", linewidth=1.2, zorder=1))
    ax.scatter([0], [0], c="red", marker="*", s=220, edgecolors="black", linewidths=0.8,
               zorder=10)
    ax.text(1.2, 1.2, "ego", color="red", fontsize=9, fontweight="bold", zorder=10)

    cmap = plt.get_cmap("tab10")
    for rank, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels), start=1):
        class_name = label_to_class(label, class_names)
        color = cmap(int(label.item()) % 10)
        corners = box_corners_bev(box)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color,
                             linewidth=3.0, linestyle="-", zorder=7))
        ax.scatter([box[0].item()], [box[1].item()], s=54, marker="o", color=color,
                   edgecolors="white", linewidths=0.7, zorder=8)
        if not no_labels and rank <= max(label_top_k, 0):
            ax.text(box[0].item(), box[1].item(), f"{class_name} {float(score):.2f}",
                    fontsize=7, color=color, zorder=9)

    for rank, (box, name) in enumerate(zip(gt_boxes, gt_names), start=1):
        corners = box_corners_bev(box)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="0.15",
                             linewidth=1.8, linestyle="--", alpha=0.85, zorder=3))
        ax.scatter([box[0].item()], [box[1].item()], s=40, marker="x", color="0.15",
                   linewidths=1.4, zorder=4)
        if not no_labels and label_gt and rank <= 10:
            ax.text(box[0].item(), box[1].item(), str(name), fontsize=7, color="black", zorder=5)

    handles = [
        Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=3.0, label="prediction box"),
        Line2D([0], [0], color="tab:blue", marker="o", linestyle="None", markersize=7,
               label="prediction center"),
        Line2D([0], [0], color="0.15", linestyle="--", linewidth=1.8, label="ground truth box"),
        Line2D([0], [0], color="0.15", marker="x", linestyle="None", markersize=7,
               label="ground truth center"),
        Line2D([0], [0], color="red", marker="*", linestyle="None", markersize=12, label="ego"),
    ]
    ax.legend(handles=handles, loc="upper right")
    if zoom_preds:
        x_low, x_high, y_low, y_high = get_zoom_limits(pred_boxes, gt_boxes, point_cloud_range, padding=10.0)
    else:
        x_low, x_high, y_low, y_high = x_min - 2, x_max + 2, y_min - 2, y_max + 2
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linewidth=0.3)
    explanation = "solid thick boxes/circles = predictions\n" \
                  "dashed boxes/x markers = ground truth\n" \
                  "red star = ego"
    ax.text(0.02, 0.02, explanation, transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.5", alpha=0.9),
            va="bottom", ha="left")
    sample_idx = meta.get("sample_idx", "<missing>")
    ax.set_title(
        f"sample={sample_idx} thr={score_thr:g} top_k={top_k} preds={len(pred_scores)} gt={len(gt_boxes)}"
    )
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

    dataset, img_tensor, metadata_dict = prepare_sample(cfg, args.index)
    class_names = list(cfg.class_names)
    gt_boxes, gt_labels, gt_names, gt_source = extract_gt_from_dataset(dataset, args.index, class_names)

    device, device_reason = choose_device()
    print_header("device")
    print(f"selected device: {device} ({device_reason})")

    model = build_model_and_load(cfg, args.checkpoint, device)

    print_header("full forward")
    outputs = run_full_forward(model, img_tensor, metadata_dict, device)
    pred_boxes, pred_scores, pred_labels = extract_predictions(outputs)
    selected_boxes, selected_scores, selected_labels, raw_count, threshold_count, fallback = select_predictions(
        pred_boxes, pred_scores, pred_labels, args.score_thr, args.top_k)
    if fallback:
        print(f"no predictions met score threshold {args.score_thr:g}; falling back to top {len(selected_scores)} by score")

    topk_count = len(selected_scores)
    selected_boxes, selected_scores, selected_labels, class_filter_count = filter_predictions_by_class(
        selected_boxes, selected_scores, selected_labels, args.class_filter, class_names)

    plot_gt_boxes, plot_gt_names = filter_gt_by_class(gt_boxes, gt_names, args.class_filter)
    if args.zoom_preds:
        plot_gt_boxes, plot_gt_names = filter_gt_near_predictions(
            plot_gt_boxes, plot_gt_names, selected_boxes, args.min_gt_distance_to_pred)

    print_prediction_table(selected_boxes, selected_scores, selected_labels, class_names)
    print_gt_table(plot_gt_boxes, plot_gt_names)
    saved_path = save_bev_plot(
        selected_boxes,
        selected_scores,
        selected_labels,
        plot_gt_boxes,
        plot_gt_names,
        class_names,
        cfg.point_cloud_range,
        metadata_dict,
        args.index,
        args.score_thr,
        args.top_k,
        args.no_labels,
        args.label_top_k,
        args.label_gt,
        gt_source,
        args.class_filter,
        args.zoom_preds,
    )

    print_header("summary")
    print(f"saved image path: {saved_path}")
    print(f"raw predictions count: {raw_count}")
    print(f"predictions after score threshold: {threshold_count}")
    print(f"predictions after top-k: {topk_count}")
    if args.class_filter:
        print(f"predictions after class filter: {class_filter_count}")
    print(f"full GT box count: {len(gt_boxes)}")
    print(f"plotted GT box count: {len(plot_gt_boxes)}")
    print(f"zoom used: {'yes' if args.zoom_preds else 'no'}")
    print(f"GT source used: {gt_source}")
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
