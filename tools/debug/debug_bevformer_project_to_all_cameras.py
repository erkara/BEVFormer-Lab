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
from mmengine.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.debug.debug_bevformer_project_to_camera import (  # noqa: E402
    OUTPUT_DIR,
    draw_projected_box,
    load_processed_camera_image,
    project_center,
    project_corners,
    projection_visible,
)
from tools.debug.debug_bevformer_visualize_bev_with_gt import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    build_model_and_load,
    extract_gt_from_dataset,
    extract_predictions,
    label_to_class,
    prepare_sample,
    select_predictions,
)
from tools.debug.debug_bevformer_checkpoint_load import run_full_forward  # noqa: E402
from tools.debug.debug_nuscenes_real_full_forward import (  # noqa: E402
    CONFIG_PATH,
    apply_config_fixes,
    choose_device,
)

import mmdet.models  # noqa: E402,F401
import mmdet3d.models  # noqa: E402,F401
import projects.mmdet3d_plugin  # noqa: E402,F401


CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def print_header(name):
    print(f"\n=== {name} ===")


def parse_args():
    parser = argparse.ArgumentParser(description="Project BEVFormer predictions and GT onto all six nuScenes cameras")
    parser.add_argument("--index", type=int, default=0, help="Validation dataset index")
    parser.add_argument("--score-thr", type=float, default=0.20, help="Prediction score threshold")
    parser.add_argument("--top-k", type=int, default=10, help="Keep top K predictions after thresholding")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint path")
    parser.add_argument("--label-gt", action="store_true", help="Write GT class labels on camera plots")
    parser.add_argument("--no-labels", action="store_true", help="Do not write prediction class/score labels")
    return parser.parse_args()


def camera_name_from_filename(filename):
    normalized = str(filename).replace("\\", "/")
    for name in CAMERA_ORDER:
        if f"/{name}/" in normalized or f"__{name}__" in normalized:
            return name
    parts = Path(normalized).parts
    for part in parts:
        if part.startswith("CAM_"):
            return part
    return "UNKNOWN_CAMERA"


def output_path(index, score_thr, top_k):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thr_token = f"{score_thr:g}".replace(".", "p")
    return OUTPUT_DIR / f"debug_camera_projection_allcams_index_{index}_thr_{thr_token}_topk_{top_k}.png"


def draw_camera_axis(ax, image, lidar2img, camera_name, pred_boxes, pred_scores, pred_labels,
                     gt_boxes, gt_names, class_names, label_gt, no_labels):
    height, width = image.shape[:2]
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    cmap = plt.get_cmap("tab10")
    pred_projected = 0
    gt_projected = 0
    skipped_pred = 0
    skipped_gt = 0

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        pixels, depth, error = project_corners(box, lidar2img)
        if error is not None or not projection_visible(pixels, depth, width, height)[0]:
            skipped_pred += 1
            continue
        color = cmap(int(label.item()) % 10)
        text = None if no_labels else f"{label_to_class(label, class_names)} {float(score):.2f}"
        draw_projected_box(ax, pixels, depth, color, "-", 2.4, text, project_center(box, lidar2img))
        pred_projected += 1

    for box, name in zip(gt_boxes, gt_names):
        pixels, depth, error = project_corners(box, lidar2img)
        if error is not None or not projection_visible(pixels, depth, width, height)[0]:
            skipped_gt += 1
            continue
        draw_projected_box(ax, pixels, depth, "white", "--", 1.5, str(name) if label_gt else None,
                           project_center(box, lidar2img))
        gt_projected += 1

    ax.set_title(f"{camera_name}\npred={pred_projected} gt={gt_projected}", fontsize=10)
    return {
        "camera": camera_name,
        "projected_pred": pred_projected,
        "projected_gt": gt_projected,
        "skipped_pred": skipped_pred,
        "skipped_gt": skipped_gt,
    }


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
    selected_boxes, selected_scores, selected_labels, raw_count, helper_threshold_count, fallback = select_predictions(
        pred_boxes, pred_scores, pred_labels, args.score_thr, args.top_k)
    threshold_count = int((pred_scores >= args.score_thr).sum().item())
    if fallback:
        print(f"fallback warning: no predictions met score threshold {args.score_thr:g}; kept top {len(selected_scores)} by score")
    topk_count = len(selected_scores)

    gt_boxes, _gt_labels, gt_names, gt_source = extract_gt_from_dataset(dataset, args.index, class_names)
    filenames = metadata_dict.get("filename") or []
    lidar2imgs = metadata_dict.get("lidar2img") or []
    if len(filenames) != 6:
        print(f"warning: expected 6 camera filenames, got {len(filenames)}")

    camera_entries = []
    for idx, filename in enumerate(filenames):
        camera_entries.append((camera_name_from_filename(filename), idx, filename))
    order = {name: i for i, name in enumerate(CAMERA_ORDER)}
    camera_entries.sort(key=lambda item: order.get(item[0], 99))

    print_header("camera projection grid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))
    axes_flat = axes.flatten()
    stats = []
    for axis_idx, (camera_name, cam_idx, filename) in enumerate(camera_entries[:6]):
        image, image_path, image_size = load_processed_camera_image(filename, metadata_dict, cam_idx)
        lidar2img = np.asarray(lidar2imgs[cam_idx], dtype=np.float32)
        stat = draw_camera_axis(
            axes_flat[axis_idx], image, lidar2img, camera_name,
            selected_boxes, selected_scores, selected_labels,
            gt_boxes, gt_names, class_names, args.label_gt, args.no_labels)
        stat.update({"camera_index": cam_idx, "filename": str(filename), "image_size": image_size})
        stats.append(stat)
        print(
            f"{camera_name}: camera_index={cam_idx} projected_pred={stat['projected_pred']} "
            f"projected_gt={stat['projected_gt']} skipped_pred={stat['skipped_pred']} skipped_gt={stat['skipped_gt']}"
        )

    for empty_idx in range(len(camera_entries), 6):
        axes_flat[empty_idx].axis("off")

    handles = [
        Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=2.4, label="solid boxes = predictions"),
        Line2D([0], [0], color="white", linestyle="--", linewidth=1.5, label="dashed boxes = ground truth"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    sample_idx = metadata_dict.get("sample_idx", "<missing>")
    scene_token = metadata_dict.get("scene_token", "<missing>")
    fig.suptitle(
        f"nuScenes mini index={args.index} sample={sample_idx} thr={args.score_thr:g} top_k={args.top_k}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out_path = output_path(args.index, args.score_thr, args.top_k)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print_header("summary")
    print(f"saved image path: {out_path}")
    print(f"sample_idx: {sample_idx}")
    print(f"scene_token: {scene_token}")
    print(f"raw predictions count: {raw_count}")
    print(f"predictions after threshold: {threshold_count}")
    print(f"predictions after top-k: {topk_count}")
    print(f"GT count: {len(gt_boxes)}")
    print(f"GT source used: {gt_source}")
    print("per-camera counts:")
    for stat in stats:
        print(
            f"  {stat['camera']}: projected_pred={stat['projected_pred']} "
            f"projected_gt={stat['projected_gt']} skipped_pred={stat['skipped_pred']} skipped_gt={stat['skipped_gt']}"
        )
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
