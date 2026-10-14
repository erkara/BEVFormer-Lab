"""Visualize occupancy predictions and GT boxes on mini train or validation."""

import argparse
import csv
import logging
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import MODELS as MMDET_MODELS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmdet.models  # noqa: F401,E402
import mmdet3d.models  # noqa: F401,E402
import projects.mmdet3d_plugin  # noqa: F401,E402
from tools.occ_metrics import occupancy_scores  # noqa: E402
from projects.mmdet3d_plugin.modern_compat import (  # noqa: E402
    DATA_ROOT,
    TRAIN_INFO,
    apply_config_fixes,
    attach_legacy_infos,
    build_dataset,
    choose_device,
    configure_occupancy_eval_dataset,
    configure_train_dataset,
    parse_losses,
    prepare_train_batch,
    print_header,
    register_runtime_compat_transforms,
)


DEFAULT_CONFIG = "projects/configs/bevformer/bevformer_tiny_occ.py"
DEFAULT_OUTPUT_DIR = "outputs/occ_test"


def parse_args():
    parser = argparse.ArgumentParser(description="Run occupancy checkpoint inference and save prediction-vs-target visualizations")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="occupancy config path")
    parser.add_argument("--checkpoint", required=True, help="checkpoint saved by tools/train_occ.py or compatible state dict")
    parser.add_argument("--device", default="auto", help="device, e.g. auto, cuda, cuda:0, cpu")
    parser.add_argument("--split", choices=("train", "val"), default="val", help="dataset split; val is deterministic and includes GT targets")
    parser.add_argument("--start-index", type=int, default=0, help="first dataset index")
    parser.add_argument("--num-samples", type=int, default=2, help="number of consecutive/cyclic samples")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="directory for visualizations and CSV summary")
    parser.add_argument("--threshold", type=float, default=0.5, help="probability threshold for positive occupancy")
    return parser.parse_args()


def configure_occ_dataset(cfg, split):
    """Use GT-bearing samples on both splits; make validation deterministic."""
    dataset_cfg = configure_train_dataset(cfg)
    if split == "val":
        return configure_occupancy_eval_dataset(cfg)
    elif split == "train":
        dataset_cfg.ann_file = str(Path(TRAIN_INFO).resolve())
        dataset_cfg.data_root = DATA_ROOT
        dataset_cfg.test_mode = False
    else:
        raise ValueError(f"unsupported split: {split}")
    return dataset_cfg


def selected_indices(dataset_len, start_index, num_samples):
    if dataset_len <= 0:
        raise ValueError("dataset is empty")
    count = max(1, int(num_samples))
    return [(start_index + i) % dataset_len for i in range(count)]


def build_model_and_load(cfg, checkpoint, device):
    init_default_scope("mmdet")
    model = MMDET_MODELS.build(cfg.model)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        iteration = payload.get("iteration", "<missing>")
    elif isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
        iteration = payload.get("iteration", "<missing>")
    else:
        state_dict = payload
        iteration = "<raw-state-dict>"
    incompatible = model.load_state_dict(state_dict, strict=False)
    print(f"checkpoint loaded: {checkpoint_path}")
    print(f"checkpoint iteration: {iteration}")
    print(f"missing keys: {list(incompatible.missing_keys)}")
    print(f"unexpected keys: {list(incompatible.unexpected_keys)}")
    model.to(device)
    model.eval()
    return model


def sample_meta(batch):
    """Read identity metadata from the current (last) temporal frame."""
    queue = batch["img_metas"][0]
    if isinstance(queue, dict) and queue and all(isinstance(k, int) for k in queue):
        return queue[max(queue)]
    if isinstance(queue, dict):
        return queue
    return {}


def save_visualization(debug, out_dir, sample_index, sample_idx, threshold, loss_value, scores):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logits = debug["logits"].detach().cpu()
    targets = debug["targets"].detach().cpu()
    prob = logits.sigmoid()[0, 0].numpy()
    target = targets[0, 0].numpy()
    pred = (prob >= threshold).astype(np.float32)
    error = np.abs(pred - target)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    im0 = axes[0].imshow(prob, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0].set_title("pred prob")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].imshow(target, origin="lower", vmin=0.0, vmax=1.0, cmap="gray_r")
    axes[1].set_title("GT mask")
    axes[2].imshow(pred, origin="lower", vmin=0.0, vmax=1.0, cmap="gray_r")
    axes[2].set_title(f"pred >= {threshold:g}")
    axes[3].imshow(error, origin="lower", vmin=0.0, vmax=1.0, cmap="magma")
    axes[3].set_title("abs error")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"index={sample_index} sample={sample_idx}  IoU={scores['iou']:.3f}  Dice={scores['dice']:.3f}  loss={loss_value:.4f}")
    fig.tight_layout()
    safe_sample = str(sample_idx).replace("/", "_") if sample_idx else "missing_sample"
    out_path = out_dir / f"sample_{sample_index:04d}_{safe_sample}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, prob


def write_summary_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        fieldnames = [
            "sample_index",
            "sample_idx",
            "occupied_target_cells",
            "predicted_positive_cells",
            "mean_predicted_probability",
            "threshold",
            "iou",
            "dice",
            "bce_loss",
            "visualization",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_debug_shapes(batch, debug):
    print_header("tensor shapes")
    print(f"image batch shape: {tuple(batch['img'].shape)}")
    print(f"raw BEV encoder shape: {debug['raw_bev_embed_shape']} layout={debug['raw_bev_embed_layout']}")
    print(f"BEV feature shape: {debug['bev_feature_shape']}")
    print(f"occupancy logits shape: {debug['occupancy_logits_shape']}")
    print(f"occupancy target shape: {debug['occupancy_target_shape']}")


def main():
    args = parse_args()
    logging.disable(logging.INFO)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("load config")
    cfg = Config.fromfile(args.config)
    apply_config_fixes(cfg)
    print(f"config path: {args.config}")

    print_header("build dataset")
    init_default_scope("mmdet3d")
    register_runtime_compat_transforms()
    dataset_cfg = configure_occ_dataset(cfg, args.split)
    dataset = build_dataset(dataset_cfg)
    metadata = attach_legacy_infos(dataset, dataset_cfg.ann_file, list(cfg.class_names))
    indices = selected_indices(len(dataset), args.start_index, args.num_samples)
    print(f"split: {args.split}")
    print(f"attached legacy metadata: {metadata}")
    print(f"dataset type: {type(dataset)}")
    print(f"dataset length: {len(dataset)}")
    print(f"selected sample count: {len(indices)}")

    device, device_reason = choose_device(args.device)
    print_header("build model")
    print(f"device: {device} ({device_reason})")
    model = build_model_and_load(cfg, args.checkpoint, device)
    print(f"model type: {cfg.model.type}")

    rows = []
    print_header("inference")
    with torch.no_grad():
        for rank, dataset_index in enumerate(indices, start=1):
            batch = prepare_train_batch(dataset, dataset_index, device)
            model.eval()
            losses, debug = model(return_loss=True, return_occ_debug=True, **batch)
            total_loss, scalar_losses = parse_losses(losses)
            loss_value = float(total_loss.detach().cpu())
            if rank == 1:
                print_debug_shapes(batch, debug)
            meta = sample_meta(batch)
            sample_idx = meta.get("sample_idx", "")
            scores = occupancy_scores(debug["logits"], debug["targets"], args.threshold)
            vis_path, prob = save_visualization(
                debug, output_dir, dataset_index, sample_idx, args.threshold, loss_value, scores)
            occupied_target_cells = scores["target_cells"]
            predicted_positive_cells = scores["predicted_cells"]
            mean_probability = float(prob.mean())
            rows.append(dict(
                sample_index=dataset_index,
                sample_idx=sample_idx,
                occupied_target_cells=occupied_target_cells,
                predicted_positive_cells=predicted_positive_cells,
                mean_predicted_probability=f"{mean_probability:.8f}",
                threshold=f"{args.threshold:g}",
                iou=f"{scores['iou']:.8f}",
                dice=f"{scores['dice']:.8f}",
                bce_loss=f"{loss_value:.8f}",
                visualization=str(vis_path),
            ))
            print(
                f"sample {rank:03d}/{len(indices):03d} index={dataset_index} "
                f"target_cells={occupied_target_cells} pred_cells={predicted_positive_cells} "
                f"mean_prob={mean_probability:.6f} iou={scores['iou']:.4f} dice={scores['dice']:.4f} "
                f"loss={loss_value:.6f} vis={vis_path}")

    csv_path = output_dir / "summary.csv"
    write_summary_csv(csv_path, rows)

    print_header("summary")
    print(f"summary CSV: {csv_path}")
    print(f"visualization dir: {output_dir}")
    print("model.eval and torch.no_grad: yes")
    print("distributed inference: no")
    print("tools/test.py changes: none")
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
