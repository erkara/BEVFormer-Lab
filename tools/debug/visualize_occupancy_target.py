import warnings
warnings.filterwarnings("ignore")
import argparse
import logging
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mmdet.models  # noqa: E402,F401
import mmdet3d.models  # noqa: E402,F401
import projects.mmdet3d_plugin  # noqa: E402,F401
from projects.mmdet3d_plugin.bevformer.occupancy import (  # noqa: E402
    VEHICLE_CLASSES,
    bev_grid_from_config,
    boxes_to_bev_footprints,
    rasterize_bev_occupancy,
)
from projects.mmdet3d_plugin.modern_compat import (  # noqa: E402
    CONFIG_PATH,
    apply_config_fixes,
    attach_legacy_infos,
    build_dataset,
    configure_train_dataset,
    print_header,
    register_runtime_compat_transforms,
    unwrap_datacontainer,
)


DEFAULT_OUTPUT_DIR = "outputs/debug_occupancy_targets"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize BEV occupancy targets from nuScenes 3D boxes")
    parser.add_argument("--start-index", type=int, default=0, help="first train dataset index to visualize")
    parser.add_argument("--num-samples", type=int, default=3, help="number of consecutive samples to visualize")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="directory for PNG outputs")
    parser.add_argument("--config", default=CONFIG_PATH, help="BEVFormer config path")
    parser.add_argument("--label-boxes", action="store_true", help="label selected vehicle boxes on overlay plots")
    return parser.parse_args()


def as_tensor_boxes(boxes):
    if hasattr(boxes, "tensor"):
        boxes = boxes.tensor
    if not torch.is_tensor(boxes):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"expected GT boxes with shape [N, >=7], got {tuple(boxes.shape)}")
    return boxes.detach().cpu().float()


def as_tensor_labels(labels):
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels, dtype=torch.long)
    if labels.ndim != 1:
        raise ValueError(f"expected GT labels with shape [N], got {tuple(labels.shape)}")
    return labels.detach().cpu().long()


def current_meta(img_metas):
    img_metas = unwrap_datacontainer(img_metas)
    if isinstance(img_metas, dict) and img_metas and all(isinstance(key, int) for key in img_metas):
        return img_metas[max(img_metas)]
    if isinstance(img_metas, dict):
        return img_metas
    return {}


def label_to_class(label, class_names):
    label = int(label.item()) if torch.is_tensor(label) else int(label)
    return class_names[label] if 0 <= label < len(class_names) else f"label_{label}"


def load_train_sample(dataset, index):
    raw_item = dataset[index]
    if isinstance(raw_item, list) and len(raw_item) == 1:
        raw_item = raw_item[0]
    if not isinstance(raw_item, dict):
        raise TypeError(f"expected dataset item dict, got {type(raw_item)}")
    for key in ("gt_bboxes_3d", "gt_labels_3d", "img_metas"):
        if key not in raw_item:
            raise KeyError(f"dataset item missing {key}")

    boxes = as_tensor_boxes(unwrap_datacontainer(raw_item["gt_bboxes_3d"]))
    labels = as_tensor_labels(unwrap_datacontainer(raw_item["gt_labels_3d"]))
    if len(boxes) != len(labels):
        raise AssertionError(f"box/label count mismatch: {len(boxes)} boxes vs {len(labels)} labels")
    meta = current_meta(raw_item["img_metas"])
    return boxes, labels, meta


def output_stem(output_dir, index, sample_idx):
    safe_sample = str(sample_idx).replace("/", "_") if sample_idx is not None else "missing_sample"
    return Path(output_dir) / f"occupancy_index_{index:04d}_{safe_sample}"


def save_mask_image(mask, grid, out_path, title):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(mask, origin="lower", extent=grid.extent, interpolation="nearest", cmap="gray_r")
    ax.add_patch(Rectangle((grid.x_min, grid.y_min), grid.x_max - grid.x_min, grid.y_max - grid.y_min,
                           fill=False, edgecolor="tab:blue", linewidth=1.0))
    ax.scatter([0.0], [0.0], c="red", marker="*", s=140, edgecolors="black", linewidths=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_overlay_image(mask, boxes, labels, class_names, selected_indices, selected_footprints,
                       grid, meta, out_path, label_boxes):
    all_footprints = boxes_to_bev_footprints(boxes)
    selected_set = set(int(i) for i in selected_indices.tolist())

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(mask, origin="lower", extent=grid.extent, interpolation="nearest",
              cmap="Greens", alpha=0.55, vmin=0, vmax=1)
    ax.add_patch(Rectangle((grid.x_min, grid.y_min), grid.x_max - grid.x_min, grid.y_max - grid.y_min,
                           fill=False, edgecolor="0.4", linewidth=1.0, zorder=2))
    ax.scatter([0.0], [0.0], c="red", marker="*", s=180, edgecolors="black", linewidths=0.7, zorder=10)

    for box_index, polygon in enumerate(all_footprints):
        if box_index in selected_set:
            continue
        ax.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor="0.45",
                             linewidth=0.8, linestyle="--", alpha=0.6, zorder=3))

    for rank, (box_index, polygon) in enumerate(zip(selected_indices.tolist(), selected_footprints), start=1):
        ax.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor="tab:orange",
                             linewidth=2.0, zorder=5))
        center = boxes[int(box_index), :2].numpy()
        ax.scatter([center[0]], [center[1]], c="tab:orange", s=28, edgecolors="black",
                   linewidths=0.4, zorder=6)
        if label_boxes and rank <= 20:
            ax.text(center[0], center[1], label_to_class(labels[int(box_index)], class_names),
                    fontsize=7, color="black", zorder=7)

    handles = [
        Line2D([0], [0], color="tab:orange", linewidth=2.0, label="selected vehicle box"),
        Line2D([0], [0], color="0.45", linestyle="--", linewidth=0.8, label="ignored GT box"),
        Line2D([0], [0], marker="s", color="tab:green", linestyle="None", alpha=0.55,
               markersize=9, label="occupied cell"),
        Line2D([0], [0], marker="*", color="red", linestyle="None", markersize=11, label="ego"),
    ]
    ax.legend(handles=handles, loc="upper right")
    ax.set_xlim(grid.x_min - 2.0, grid.x_max + 2.0)
    ax.set_ylim(grid.y_min - 2.0, grid.y_max + 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title(f"sample={meta.get('sample_idx', '<missing>')} selected={len(selected_indices)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def print_grid_summary(grid):
    print_header("BEV grid")
    print("coordinate frame: ego-centric LiDAR/BEV")
    print(f"x range: [{grid.x_min:.3f}, {grid.x_max:.3f}] m -> columns 0..{grid.width - 1}")
    print(f"y range: [{grid.y_min:.3f}, {grid.y_max:.3f}] m -> rows 0..{grid.height - 1}")
    print(f"mask shape [H, W]: {grid.shape}")
    print(f"cell size: dx={grid.cell_size_x:.3f} m, dy={grid.cell_size_y:.3f} m")
    print("display convention: imshow(origin='lower')")


def main():
    args = parse_args()
    logging.disable(logging.INFO)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    print_header("load config")
    cfg = Config.fromfile(args.config)
    apply_config_fixes(cfg)
    print(f"config: {args.config}")
    print("modern config fixes applied")

    grid = bev_grid_from_config(cfg)
    print_grid_summary(grid)

    print_header("build train dataset")
    init_default_scope("mmdet3d")
    register_runtime_compat_transforms()
    train_cfg = configure_train_dataset(cfg)
    dataset = build_dataset(train_cfg)
    metadata = attach_legacy_infos(dataset, train_cfg.ann_file, list(cfg.class_names))
    print(f"attached legacy metadata: {metadata}")
    print(f"dataset type: {type(dataset)}")
    print(f"dataset length: {len(dataset)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(cfg.class_names)
    selected_classes = set(VEHICLE_CLASSES)
    print_header("selected classes")
    print(", ".join(VEHICLE_CLASSES))

    for offset in range(args.num_samples):
        index = args.start_index + offset
        if index < 0 or index >= len(dataset):
            raise IndexError(f"index {index} is outside dataset length {len(dataset)}")

        boxes, labels, meta = load_train_sample(dataset, index)
        names = [label_to_class(label, class_names) for label in labels]
        selected_name_count = sum(name in selected_classes for name in names)
        mask, footprints, selected_indices = rasterize_bev_occupancy(
            boxes,
            grid,
            labels=labels,
            class_names=class_names,
            selected_classes=VEHICLE_CLASSES,
        )
        if int(selected_indices.size) != int(selected_name_count):
            raise AssertionError("selected count mismatch between names and rasterizer labels")

        occupied_cells = int(mask.sum())
        sample_idx = meta.get("sample_idx", f"index_{index}")
        stem = output_stem(output_dir, index, sample_idx)
        mask_path = stem.with_name(stem.name + "_mask.png")
        overlay_path = stem.with_name(stem.name + "_overlay.png")

        title = f"sample={sample_idx} occupied={occupied_cells} shape={mask.shape}"
        save_mask_image(mask, grid, mask_path, title)
        save_overlay_image(mask, boxes, labels, class_names, selected_indices, footprints,
                           grid, meta, overlay_path, args.label_boxes)

        print_header(f"sample index {index}")
        print(f"sample_idx: {sample_idx}")
        print(f"scene_token: {meta.get('scene_token', '<missing>')}")
        print(f"GT boxes: {len(boxes)}")
        print(f"selected vehicle boxes: {len(selected_indices)}")
        print(f"occupied cells: {occupied_cells}")
        print(f"BEV mask shape: {mask.shape}")
        print(f"saved binary mask: {mask_path}")
        print(f"saved overlay: {overlay_path}")
        if len(selected_indices) > 0 and occupied_cells == 0:
            raise AssertionError("selected vehicle boxes produced an empty occupancy mask")

    print_header("summary")
    print("occupancy target generation: ok")
    print("model/head changes: none")
    print("training script changes: none")
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
