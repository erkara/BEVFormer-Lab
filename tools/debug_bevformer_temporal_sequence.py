import argparse
import csv
import logging
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from mmengine.config import Config
from mmengine.registry import init_default_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.debug_bevformer_checkpoint_load import load_checkpoint_modern_safe  # noqa: E402
from tools.debug_bevformer_visualize_bev_with_gt import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    box_corners_bev,
    extract_gt_from_dataset,
    extract_predictions,
    label_to_class,
    select_predictions,
)
from tools.debug_nuscenes_real_full_forward import (  # noqa: E402
    CONFIG_PATH,
    apply_config_fixes,
    attach_legacy_infos,
    build_dataset,
    choose_device,
    configure_val_dataset,
    get_primary_item,
    register_runtime_compat_transforms,
    unwrap_datacontainer,
)

import mmdet.models  # noqa: E402,F401
import mmdet3d.models  # noqa: E402,F401
import projects.mmdet3d_plugin  # noqa: E402,F401
from mmdet.registry import MODELS as MMDET_MODELS  # noqa: E402


DEFAULT_SAVE_DIR = "outputs/temporal_debug"


def print_header(name):
    print(f"\n=== {name} ===")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BEVFormer tiny checkpoint inference over consecutive mini validation samples")
    parser.add_argument("--start-index", type=int, default=0, help="First validation dataset index")
    parser.add_argument("--num-frames", type=int, default=10, help="Maximum number of same-scene frames to process")
    parser.add_argument("--score-thr", type=float, default=0.20, help="Prediction score threshold")
    parser.add_argument("--top-k", type=int, default=10,
                        help="After thresholding, keep only the top K predictions by score. Use <=0 to keep all.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint path")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="Directory for frame PNGs and CSV summary")
    return parser.parse_args()


def build_val_dataset(cfg):
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
    return dataset


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


def summarize_value(value):
    if torch.is_tensor(value):
        return f"Tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
    if isinstance(value, dict):
        return {key: summarize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} len={len(value)}"
    return repr(value)


def temporal_attr_names(model):
    names = []
    preferred = ("prev_frame_info", "prev_bev", "video_test_mode")
    for name in preferred:
        if hasattr(model, name):
            names.append(name)
    tokens = ("prev", "bev", "frame", "scene")
    for name in dir(model):
        if name.startswith("_"):
            continue
        lower = name.lower()
        if any(token in lower for token in tokens) and name not in names and hasattr(model, name):
            try:
                value = getattr(model, name)
            except Exception:
                continue
            if callable(value):
                continue
            names.append(name)
    return names


def summarize_temporal_attrs(model):
    summary = {}
    for name in temporal_attr_names(model):
        try:
            summary[name] = summarize_value(getattr(model, name))
        except Exception as exc:
            summary[name] = f"<unreadable {type(exc).__name__}: {exc}>"
    return summary


def print_temporal_summary(label, model):
    print_header(label)
    summary = summarize_temporal_attrs(model)
    if not summary:
        print("no temporal-looking attributes found")
        return
    for name, value in summary.items():
        print(f"{name}: {value}")


def get_prev_frame_info(model):
    info = getattr(model, "prev_frame_info", None)
    return info if isinstance(info, dict) else None


def get_prev_bev_from_model(model):
    info = get_prev_frame_info(model)
    if info is not None and "prev_bev" in info:
        return info.get("prev_bev")
    return getattr(model, "prev_bev", None) if hasattr(model, "prev_bev") else None


def prev_bev_exists_and_shape(model):
    prev_bev = get_prev_bev_from_model(model)
    exists = torch.is_tensor(prev_bev)
    shape = tuple(prev_bev.shape) if exists else None
    return exists, shape


def print_frame_temporal_state(when, model):
    info = get_prev_frame_info(model)
    exists, shape = prev_bev_exists_and_shape(model)
    print(f"model.prev_frame_info {when}: {summarize_value(info) if info is not None else '<missing>'}")
    print(f"prev_bev exists {when}: {exists}")
    print(f"prev_bev shape {when}: {shape if shape is not None else '<none>'}")
    if info is not None:
        print(f"previous sample token stored {when}: {info.get('sample_idx', info.get('sample_token', '<missing>'))}")
        print(f"previous scene token stored {when}: {info.get('scene_token', '<missing>')}")


def reset_temporal_state(model):
    print_temporal_summary("temporal state before reset", model)
    if hasattr(model, "prev_frame_info") and isinstance(model.prev_frame_info, dict):
        model.prev_frame_info["prev_bev"] = None
        model.prev_frame_info["scene_token"] = None
        model.prev_frame_info["prev_pos"] = 0
        model.prev_frame_info["prev_angle"] = 0
    if hasattr(model, "prev_bev"):
        model.prev_bev = None
    print_temporal_summary("temporal state after reset", model)


def scene_token_for_index(dataset, index):
    if hasattr(dataset, "data_infos") and len(dataset.data_infos) > index:
        info = dataset.data_infos[index]
        if isinstance(info, dict):
            return info.get("scene_token")
    raw_item = dataset[index]
    item, _ = get_primary_item(raw_item)
    meta = unwrap_datacontainer(item["img_metas"])
    return meta.get("scene_token")


def consecutive_same_scene_indices(dataset, start_index, num_frames):
    if start_index < 0 or start_index >= len(dataset):
        raise IndexError(f"start index {start_index} is outside dataset length {len(dataset)}")
    if num_frames <= 0:
        return []
    start_scene = scene_token_for_index(dataset, start_index)
    indices = []
    for index in range(start_index, len(dataset)):
        scene = scene_token_for_index(dataset, index)
        if scene != start_scene:
            print(f"stopping at index {index}: scene_token changed from {start_scene} to {scene}")
            break
        indices.append(index)
        if len(indices) >= num_frames:
            break
    return indices


def load_frame(dataset, index):
    raw_item = dataset[index]
    item, item_reason = get_primary_item(raw_item)
    img_tensor = unwrap_datacontainer(item["img"])
    metadata_dict = unwrap_datacontainer(item["img_metas"])
    if not torch.is_tensor(img_tensor):
        raise TypeError(f"expected image tensor, got {type(img_tensor)}")
    if not isinstance(metadata_dict, dict):
        raise TypeError(f"expected metadata dict, got {type(metadata_dict)}")
    return raw_item, item_reason, img_tensor, metadata_dict


def run_model_frame(model, img_tensor, metadata_dict, device):
    img = [img_tensor.unsqueeze(0).to(device)]
    img_metas = [[metadata_dict]]
    with torch.no_grad():
        return model(return_loss=False, img=img, img_metas=img_metas)


def save_frame_plot(out_path, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_names,
                    class_names, point_cloud_range, frame_id, dataset_index, sample_idx,
                    score_thr):
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=False, edgecolor="0.55", linewidth=1.2, zorder=1))
    ax.scatter([0], [0], c="red", marker="*", s=220, edgecolors="black", linewidths=0.8, zorder=10)
    ax.text(1.2, 1.2, "ego", color="red", fontsize=9, fontweight="bold", zorder=10)

    cmap = plt.get_cmap("tab10")
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        color = cmap(int(label.item()) % 10)
        corners = box_corners_bev(box)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color,
                             linewidth=3.0, linestyle="-", zorder=7))
        ax.scatter([box[0].item()], [box[1].item()], s=54, marker="o", color=color,
                   edgecolors="white", linewidths=0.7, zorder=8)

    for box, _name in zip(gt_boxes, gt_names):
        corners = box_corners_bev(box)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="0.15",
                             linewidth=1.8, linestyle="--", alpha=0.85, zorder=3))
        ax.scatter([box[0].item()], [box[1].item()], s=40, marker="x", color="0.15",
                   linewidths=1.4, zorder=4)

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
    ax.set_xlim(x_min - 2, x_max + 2)
    ax.set_ylim(y_min - 2, y_max + 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linewidth=0.3)
    ax.text(0.02, 0.02,
            "solid thick boxes/circles = predictions\ndashed boxes/x markers = ground truth\nred star = ego",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.5", alpha=0.9),
            va="bottom", ha="left")
    ax.set_title(
        f"frame={frame_id:03d} index={dataset_index} sample={sample_idx} "
        f"thr={score_thr:g} preds={len(pred_scores)} gt={len(gt_boxes)}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def top_label_name(labels, class_names):
    if len(labels) == 0:
        return ""
    return label_to_class(labels[0], class_names)



def format_top5_centers(boxes):
    if len(boxes) == 0:
        return ""
    centers = boxes[:5, :2].detach().cpu().tolist()
    return ";".join(f"{x:.4f},{y:.4f}" for x, y in centers)


def run_sequence_mode(model, dataset, indices, cfg, class_names, device, args, mode, reset_each_frame, save_plots):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    saved_images = []

    print_header(f"sequence inference: {mode}")
    if not reset_each_frame:
        reset_temporal_state(model)

    for frame_id, dataset_index in enumerate(indices):
        if reset_each_frame:
            print(f"\n--- reset temporal state before {mode} frame {frame_id:03d} ---")
            reset_temporal_state(model)

        raw_item, item_reason, img_tensor, metadata_dict = load_frame(dataset, dataset_index)
        sample_idx = metadata_dict.get("sample_idx", "<missing>")
        scene_token = metadata_dict.get("scene_token", "<missing>")
        prev_idx = metadata_dict.get("prev_idx", "<missing>")
        next_idx = metadata_dict.get("next_idx", "<missing>")
        has_prev_bev_exists = "prev_bev_exists" in metadata_dict
        prev_bev_exists = metadata_dict.get("prev_bev_exists", "<missing>")

        print(f"\n--- {mode} frame {frame_id:03d} ---")
        print(f"frame_id: {frame_id}")
        print(f"dataset index: {dataset_index}")
        print(f"item selection: {item_reason}")
        print(f"sample_idx: {sample_idx}")
        print(f"scene_token: {scene_token}")
        print(f"prev_idx: {prev_idx}")
        print(f"next_idx: {next_idx}")
        print(f"metadata has prev_bev_exists: {has_prev_bev_exists} value={prev_bev_exists}")
        print_frame_temporal_state("before forward", model)

        outputs = run_model_frame(model, img_tensor, metadata_dict, device)
        pred_boxes, pred_scores, pred_labels = extract_predictions(outputs)
        selected_boxes, selected_scores, selected_labels, raw_count, threshold_count, fallback = select_predictions(
            pred_boxes, pred_scores, pred_labels, args.score_thr, args.top_k)
        if fallback:
            print(f"no predictions met score threshold {args.score_thr:g}; falling back to top {len(selected_scores)} by score")
        gt_boxes, _gt_labels, gt_names, gt_source = extract_gt_from_dataset(dataset, dataset_index, class_names)
        topk_count = len(selected_scores)
        top_score = float(selected_scores[0]) if len(selected_scores) else ""
        top_label = top_label_name(selected_labels, class_names)
        top5_centers = format_top5_centers(selected_boxes)

        print_frame_temporal_state("after forward", model)
        prev_bev_after_exists, prev_bev_after_shape = prev_bev_exists_and_shape(model)
        print(f"raw predictions: {raw_count}")
        print(f"predictions after score threshold: {threshold_count}")
        print(f"predictions after top-k: {topk_count}")
        print(f"GT boxes: {len(gt_boxes)}")
        print(f"GT source used: {gt_source}")
        print(f"top score: {top_score}")
        print(f"top label: {top_label}")
        print(f"top5 centers xy: {top5_centers}")

        if save_plots:
            sample_short = str(sample_idx)[:8]
            out_path = save_dir / f"frame_{frame_id:03d}_index_{dataset_index}_sample_{sample_short}.png"
            save_frame_plot(
                out_path,
                selected_boxes,
                selected_scores,
                selected_labels,
                gt_boxes,
                gt_names,
                class_names,
                cfg.point_cloud_range,
                frame_id,
                dataset_index,
                sample_short,
                args.score_thr,
            )
            saved_images.append(out_path)
            print(f"saved image path: {out_path}")

        rows.append({
            "mode": mode,
            "frame_id": frame_id,
            "dataset_index": dataset_index,
            "sample_idx": sample_idx,
            "scene_token": scene_token,
            "top_score": top_score,
            "top_label": top_label,
            "preds_after_threshold": threshold_count,
            "raw_predictions": raw_count,
            "predictions_after_topk": topk_count,
            "gt_count": len(gt_boxes),
            "prev_idx": prev_idx,
            "next_idx": next_idx,
            "prev_bev_after_exists": prev_bev_after_exists,
            "prev_bev_after_shape": str(prev_bev_after_shape) if prev_bev_after_shape is not None else "",
            "top5_centers_xy": top5_centers,
            "selected_boxes": selected_boxes.detach().cpu(),
            "selected_scores": selected_scores.detach().cpu(),
            "selected_labels": selected_labels.detach().cpu(),
        })
    return rows, saved_images


def write_original_sequence_csv(csv_path, rows):
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id",
            "dataset_index",
            "sample_idx",
            "scene_token",
            "prev_idx",
            "next_idx",
            "raw_predictions",
            "predictions_after_threshold",
            "predictions_after_topk",
            "gt_count",
            "top_score",
            "top_label",
        ])
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_csv(csv_path, rows):
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode",
            "frame_id",
            "dataset_index",
            "sample_idx",
            "scene_token",
            "top_score",
            "top_label",
            "preds_after_threshold",
            "top5_centers_xy",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def rows_to_original_summary(rows):
    summaries = []
    for row in rows:
        summaries.append({
            "frame_id": row["frame_id"],
            "dataset_index": row["dataset_index"],
            "sample_idx": row["sample_idx"],
            "scene_token": row["scene_token"],
            "prev_idx": row["prev_idx"],
            "next_idx": row["next_idx"],
            "raw_predictions": row["raw_predictions"],
            "predictions_after_threshold": row["preds_after_threshold"],
            "predictions_after_topk": row["predictions_after_topk"],
            "gt_count": row["gt_count"],
            "top_score": row["top_score"],
            "top_label": row["top_label"],
        })
    return summaries


def compare_modes(continuous_rows, reset_rows):
    print_header("temporal comparison")
    temporal_persisted = any(
        idx > 0 and row.get("prev_bev_after_exists", False) for idx, row in enumerate(continuous_rows)
    )
    print(f"temporal state appears to persist: {'yes' if temporal_persisted else 'no'}")
    any_changed = False
    for cont, reset in zip(continuous_rows, reset_rows):
        cont_boxes = cont["selected_boxes"][:5, :2]
        reset_boxes = reset["selected_boxes"][:5, :2]
        same_count = cont_boxes.shape == reset_boxes.shape
        if same_count and cont_boxes.numel() > 0:
            max_delta = float((cont_boxes - reset_boxes).abs().max())
        elif same_count:
            max_delta = 0.0
        else:
            max_delta = float("inf")
        score_delta = 0.0
        if cont["top_score"] != "" and reset["top_score"] != "":
            score_delta = abs(float(cont["top_score"]) - float(reset["top_score"]))
        changed = (not same_count) or max_delta > 1e-4 or score_delta > 1e-4 or cont["top_label"] != reset["top_label"]
        any_changed = any_changed or changed
        print(
            f"frame {cont['frame_id']:03d} index {cont['dataset_index']}: "
            f"top boxes changed={changed} max_center_delta={max_delta:.6g} "
            f"top_score_delta={score_delta:.6g} "
            f"continuous_thr={cont['preds_after_threshold']} reset_thr={reset['preds_after_threshold']}"
        )
    print(f"continuous mode outputs differ from reset-each-frame mode: {'yes' if any_changed else 'no'}")


def main():
    logging.disable(logging.INFO)
    torch.manual_seed(0)
    args = parse_args()

    print_header("load config")
    print(f"config: {CONFIG_PATH}")
    cfg = Config.fromfile(CONFIG_PATH)
    apply_config_fixes(cfg)
    print("modern config fixes applied")

    dataset = build_val_dataset(cfg)
    indices = consecutive_same_scene_indices(dataset, args.start_index, args.num_frames)
    print_header("sequence selection")
    print(f"start index: {args.start_index}")
    print(f"requested num frames: {args.num_frames}")
    print(f"selected indices: {indices}")
    if not indices:
        print("no frames selected")
        return 0

    device, device_reason = choose_device()
    print_header("device")
    print(f"selected device: {device} ({device_reason})")
    model = build_model_and_load(cfg, args.checkpoint, device)
    print_temporal_summary("all temporal-looking model attributes before sequence", model)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(cfg.class_names)

    continuous_rows, saved_images = run_sequence_mode(
        model, dataset, indices, cfg, class_names, device, args,
        mode="temporal-continuous", reset_each_frame=False, save_plots=True)
    reset_temporal_state(model)
    reset_rows, _ = run_sequence_mode(
        model, dataset, indices, cfg, class_names, device, args,
        mode="reset-each-frame", reset_each_frame=True, save_plots=False)

    comparison_csv = save_dir / "temporal_vs_reset_summary.csv"
    write_comparison_csv(comparison_csv, continuous_rows + reset_rows)

    sequence_csv = save_dir / "sequence_summary.csv"
    original_rows = rows_to_original_summary(continuous_rows)
    write_original_sequence_csv(sequence_csv, original_rows)

    compare_modes(continuous_rows, reset_rows)

    print_header("summary")
    print(f"frames processed per mode: {len(continuous_rows)}")
    print(f"comparison CSV path: {comparison_csv}")
    print(f"sequence CSV path: {sequence_csv}")
    print("saved image paths:")
    for path in saved_images:
        print(f"  {path}")
    print("no official evaluation run")
    print("no training run")
    print("no camera projection run")
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
