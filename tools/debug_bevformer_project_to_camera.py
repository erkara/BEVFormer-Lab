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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.debug_bevformer_checkpoint_load import run_full_forward  # noqa: E402
from tools.debug_bevformer_visualize_bev_with_gt import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    build_model_and_load,
    box_corners_bev,
    extract_gt_from_dataset,
    extract_predictions,
    label_to_class,
    prepare_sample,
    select_predictions,
)
from tools.debug_nuscenes_real_full_forward import (  # noqa: E402
    CONFIG_PATH,
    apply_config_fixes,
    choose_device,
)

import mmdet.models  # noqa: E402,F401
import mmdet3d.models  # noqa: E402,F401
import projects.mmdet3d_plugin  # noqa: E402,F401


OUTPUT_DIR = Path("outputs/camera_projection")
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def print_header(name):
    print(f"\n=== {name} ===")


def parse_args():
    parser = argparse.ArgumentParser(description="Project BEVFormer predicted and GT 3D boxes onto one camera image")
    parser.add_argument("--index", type=int, default=0, help="Validation dataset index")
    parser.add_argument("--camera", default="CAM_FRONT", help="Camera name to visualize, e.g. CAM_FRONT")
    parser.add_argument("--score-thr", type=float, default=0.20, help="Prediction score threshold")
    parser.add_argument("--top-k", type=int, default=10, help="Keep top K predictions after thresholding")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint path")
    parser.add_argument("--label-gt", action="store_true", help="Write GT class labels on projected boxes")
    return parser.parse_args()


def resolve_repo_path(filename):
    path = Path(filename)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def shape_hw(shape_value):
    if isinstance(shape_value, (list, tuple)) and shape_value and isinstance(shape_value[0], (list, tuple)):
        shape_value = shape_value[0]
    if len(shape_value) < 2:
        raise ValueError(f"invalid shape metadata: {shape_value}")
    return int(shape_value[0]), int(shape_value[1])


def camera_index_from_metadata(meta, camera):
    filenames = meta.get("filename") or []
    for idx, filename in enumerate(filenames):
        normalized = str(filename).replace("\\", "/")
        if f"/{camera}/" in normalized or f"__{camera}__" in normalized or camera in Path(normalized).parts:
            return idx
    raise ValueError(f"camera {camera!r} not found in metadata filenames: {filenames}")


def load_processed_camera_image(filename, meta, camera_idx):
    image_path = resolve_repo_path(filename)
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    img_shapes = meta.get("img_shape")
    pad_shapes = meta.get("pad_shape")
    img_shape = img_shapes[camera_idx] if isinstance(img_shapes, list) else img_shapes
    pad_shape = pad_shapes[camera_idx] if isinstance(pad_shapes, list) else pad_shapes
    img_h, img_w = shape_hw(img_shape)
    pad_h, pad_w = shape_hw(pad_shape)

    resized = image.resize((img_w, img_h), Image.BILINEAR)
    canvas = Image.new("RGB", (pad_w, pad_h), (0, 0, 0))
    canvas.paste(resized, (0, 0))
    print(f"original image size: {original_size}")
    print(f"visualization image size: {(pad_w, pad_h)} from img_shape={(img_h, img_w)} pad_shape={(pad_h, pad_w)}")
    return np.asarray(canvas), image_path, (pad_w, pad_h)


def box_corners_lidar(box):
    vals = box[:7].detach().cpu().numpy().astype(np.float32)
    x, y, z, dx, dy, dz, yaw = vals.tolist()
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    corners = np.array([
        [hx, hy, -hz], [hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz],
        [hx, hy, hz], [hx, -hy, hz], [-hx, -hy, hz], [-hx, hy, hz],
    ], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return corners @ rot.T + np.array([x, y, z], dtype=np.float32)


def project_corners(box, lidar2img):
    corners = box_corners_lidar(box)
    hom = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1)
    proj = hom @ np.asarray(lidar2img, dtype=np.float32).T
    depth = proj[:, 2]
    if not np.all(np.isfinite(proj)):
        return None, depth, "invalid projection"
    if np.all(depth <= 1e-5):
        return None, depth, "depth <= 0"
    pixels = proj[:, :2] / np.maximum(depth[:, None], 1e-5)
    return pixels, depth, None


def projection_visible(pixels, depth, width, height, margin=80):
    valid_depth = depth > 1e-5
    if valid_depth.sum() < 4:
        return False, "fewer than 4 positive-depth corners"
    valid_pixels = pixels[valid_depth]
    inside_x = (valid_pixels[:, 0] >= -margin) & (valid_pixels[:, 0] <= width + margin)
    inside_y = (valid_pixels[:, 1] >= -margin) & (valid_pixels[:, 1] <= height + margin)
    if int((inside_x & inside_y).sum()) < 2:
        return False, "outside image"
    return True, "ok"


def draw_projected_box(ax, pixels, depth, color, linestyle, linewidth, label_text=None, center=None):
    for start, end in BOX_EDGES:
        if depth[start] <= 1e-5 or depth[end] <= 1e-5:
            continue
        ax.plot([pixels[start, 0], pixels[end, 0]], [pixels[start, 1], pixels[end, 1]],
                color=color, linestyle=linestyle, linewidth=linewidth, zorder=6 if linestyle == "-" else 4)
    if center is not None:
        ax.scatter([center[0]], [center[1]], color=color, s=34, marker="o" if linestyle == "-" else "x",
                   edgecolors="white" if linestyle == "-" else None, linewidths=0.7, zorder=7)
    if label_text:
        anchor_idx = int(np.argmin(pixels[:, 1]))
        ax.text(pixels[anchor_idx, 0], pixels[anchor_idx, 1], label_text, color=color, fontsize=8,
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5), zorder=8)


def project_center(box, lidar2img):
    center = np.asarray([box[0].item(), box[1].item(), box[2].item(), 1.0], dtype=np.float32)
    proj = center @ np.asarray(lidar2img, dtype=np.float32).T
    if not np.all(np.isfinite(proj)) or proj[2] <= 1e-5:
        return None
    return proj[:2] / proj[2]


def draw_projection_overlay(image, lidar2img, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_names,
                            class_names, label_gt):
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(image)
    pred_projected = 0
    gt_projected = 0
    skip_reasons = {}
    cmap = plt.get_cmap("tab10")

    for rank, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels), start=1):
        pixels, depth, error = project_corners(box, lidar2img)
        if error is None:
            visible, error = projection_visible(pixels, depth, width, height)
        else:
            visible = False
        if not visible:
            skip_reasons[f"prediction: {error}"] = skip_reasons.get(f"prediction: {error}", 0) + 1
            continue
        color = cmap(int(label.item()) % 10)
        center = project_center(box, lidar2img)
        class_name = label_to_class(label, class_names)
        draw_projected_box(ax, pixels, depth, color, "-", 2.8, f"{class_name} {float(score):.2f}", center)
        pred_projected += 1

    for box, name in zip(gt_boxes, gt_names):
        pixels, depth, error = project_corners(box, lidar2img)
        if error is None:
            visible, error = projection_visible(pixels, depth, width, height)
        else:
            visible = False
        if not visible:
            skip_reasons[f"GT: {error}"] = skip_reasons.get(f"GT: {error}", 0) + 1
            continue
        center = project_center(box, lidar2img)
        draw_projected_box(ax, pixels, depth, "white", "--", 1.8, str(name) if label_gt else None, center)
        gt_projected += 1

    handles = [
        Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=2.8, label="prediction"),
        Line2D([0], [0], color="white", linestyle="--", linewidth=1.8, label="ground truth"),
    ]
    ax.legend(handles=handles, loc="upper right")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig, pred_projected, gt_projected, skip_reasons


def draw_bev_on_axis(ax, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_names, class_names, point_cloud_range):
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=False, edgecolor="0.55", linewidth=1.2, zorder=1))
    ax.scatter([0], [0], c="red", marker="*", s=180, edgecolors="black", linewidths=0.8, zorder=10)
    cmap = plt.get_cmap("tab10")
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        color = cmap(int(label.item()) % 10)
        ax.add_patch(Polygon(box_corners_bev(box), closed=True, fill=False, edgecolor=color,
                             linewidth=3.0, linestyle="-", zorder=7))
        ax.scatter([box[0].item()], [box[1].item()], s=48, marker="o", color=color,
                   edgecolors="white", linewidths=0.7, zorder=8)
    for box, _name in zip(gt_boxes, gt_names):
        ax.add_patch(Polygon(box_corners_bev(box), closed=True, fill=False, edgecolor="0.15",
                             linewidth=1.6, linestyle="--", alpha=0.85, zorder=3))
        ax.scatter([box[0].item()], [box[1].item()], s=36, marker="x", color="0.15", linewidths=1.3, zorder=4)
    ax.set_xlim(x_min - 2, x_max + 2)
    ax.set_ylim(y_min - 2, y_max + 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linewidth=0.3)
    ax.set_title("BEV selected predictions vs GT")


def save_side_by_side(image, lidar2img, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_names,
                      class_names, point_cloud_range, label_gt, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(image)
    height, width = image.shape[:2]
    cmap = plt.get_cmap("tab10")
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        pixels, depth, error = project_corners(box, lidar2img)
        if error is not None or not projection_visible(pixels, depth, width, height)[0]:
            continue
        color = cmap(int(label.item()) % 10)
        draw_projected_box(axes[0], pixels, depth, color, "-", 2.4,
                           f"{label_to_class(label, class_names)} {float(score):.2f}", project_center(box, lidar2img))
    for box, name in zip(gt_boxes, gt_names):
        pixels, depth, error = project_corners(box, lidar2img)
        if error is not None or not projection_visible(pixels, depth, width, height)[0]:
            continue
        draw_projected_box(axes[0], pixels, depth, "white", "--", 1.5, str(name) if label_gt else None,
                           project_center(box, lidar2img))
    axes[0].set_title(title)
    axes[0].axis("off")
    draw_bev_on_axis(axes[1], pred_boxes, pred_scores, pred_labels, gt_boxes, gt_names, class_names, point_cloud_range)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def output_paths(index, camera, score_thr, top_k):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thr_token = f"{score_thr:g}".replace(".", "p")
    base = f"index_{index}_{camera}_thr_{thr_token}_topk_{top_k}"
    return (
        OUTPUT_DIR / f"debug_camera_projection_{base}.png",
        OUTPUT_DIR / f"debug_camera_projection_side_by_side_{base}.png",
    )


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

    device, device_reason = choose_device()
    print_header("device")
    print(f"selected device: {device} ({device_reason})")
    model = build_model_and_load(cfg, args.checkpoint, device)

    outputs = run_full_forward(model, img_tensor, metadata_dict, device)
    pred_boxes, pred_scores, pred_labels = extract_predictions(outputs)
    selected_boxes, selected_scores, selected_labels, raw_count, threshold_count, fallback = select_predictions(
        pred_boxes, pred_scores, pred_labels, args.score_thr, args.top_k)
    actual_threshold_count = int((pred_scores >= args.score_thr).sum().item())
    if fallback:
        print(f"fallback warning: no predictions met score threshold {args.score_thr:g}; kept top {len(selected_scores)}")

    gt_boxes, _gt_labels, gt_names, gt_source = extract_gt_from_dataset(dataset, args.index, class_names)

    cam_idx = camera_index_from_metadata(metadata_dict, args.camera)
    filenames = metadata_dict.get("filename") or []
    lidar2img_list = metadata_dict.get("lidar2img") or []
    lidar2img = np.asarray(lidar2img_list[cam_idx], dtype=np.float32)
    camera_filename = filenames[cam_idx]

    print_header("camera selection")
    print(f"selected camera name: {args.camera}")
    print(f"selected camera index: {cam_idx}")
    print(f"selected camera filename: {camera_filename}")
    print(f"lidar2img matrix shape: {lidar2img.shape}")

    image, image_path, image_size = load_processed_camera_image(camera_filename, metadata_dict, cam_idx)
    overlay_path, side_by_side_path = output_paths(args.index, args.camera, args.score_thr, args.top_k)

    print_header("projection")
    fig, pred_projected, gt_projected, skip_reasons = draw_projection_overlay(
        image, lidar2img, selected_boxes, selected_scores, selected_labels,
        gt_boxes, gt_names, class_names, args.label_gt)
    title = f"{args.camera} index={args.index} thr={args.score_thr:g} top_k={args.top_k}"
    fig.axes[0].set_title(title)
    fig.savefig(overlay_path, dpi=160)
    plt.close(fig)

    save_side_by_side(
        image, lidar2img, selected_boxes, selected_scores, selected_labels,
        gt_boxes, gt_names, class_names, cfg.point_cloud_range, args.label_gt,
        side_by_side_path, title)

    print(f"number of raw predictions: {raw_count}")
    print(f"number of selected predictions: {len(selected_scores)}")
    print(f"predictions after score threshold: {actual_threshold_count}")
    print(f"number of predicted boxes successfully projected: {pred_projected}")
    print(f"number of GT boxes: {len(gt_boxes)}")
    print(f"number of GT boxes successfully projected: {gt_projected}")
    print(f"GT source used: {gt_source}")
    if skip_reasons:
        print("skipped boxes by reason:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"  {reason}: {count}")
    else:
        print("skipped boxes by reason: none")

    print_header("summary")
    print(f"camera overlay path: {overlay_path}")
    print(f"side-by-side path: {side_by_side_path}")
    print(f"sample_idx: {metadata_dict.get('sample_idx', '<missing>')}")
    print(f"scene_token: {metadata_dict.get('scene_token', '<missing>')}")
    print(f"image path resolved: {image_path}")
    print(f"visualization image size used: {image_size}")
    print("no official evaluation run")
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
