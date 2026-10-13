import argparse
import copy
import logging
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import MODELS as MMDET_MODELS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mmdet.models  # noqa: F401,E402
import mmdet3d.models  # noqa: F401,E402
import projects.mmdet3d_plugin  # noqa: F401,E402
from projects.mmdet3d_plugin.modern_compat import (  # noqa: E402
    CHECKPOINT_PATH,
    apply_config_fixes,
    attach_legacy_infos,
    build_dataset,
    choose_device,
    configure_train_dataset,
    parse_losses,
    prepare_train_batch,
    print_header,
    register_runtime_compat_transforms,
)


CONFIG_PATH = "projects/configs/bevformer/bevformer_tiny_occ.py"
DEFAULT_OUTPUT_DIR = "outputs/debug_occupancy_overfit"


def parse_args():
    parser = argparse.ArgumentParser(description="Overfit the experimental BEV occupancy head on nuScenes-mini samples")
    parser.add_argument("--config", default=CONFIG_PATH, help="occupancy config path")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="optional BEVFormer checkpoint for backbone/encoder init")
    parser.add_argument("--no-checkpoint", action="store_true", help="skip checkpoint loading and initialize randomly")
    parser.add_argument("--device", default="auto", help="device, e.g. auto, cuda, cuda:0, cpu")
    parser.add_argument("--start-index", type=int, default=0, help="first train dataset index")
    parser.add_argument("--num-samples", type=int, default=2, help="number of samples to cycle over, clamped to 1..4")
    parser.add_argument("--iters", type=int, default=50, help="optimizer iterations")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="directory for prediction/target PNGs")
    parser.add_argument("--vis-interval", type=int, default=10, help="save visualization every N iterations")
    return parser.parse_args()


def build_occ_model(cfg, checkpoint, no_checkpoint, device):
    init_default_scope("mmdet")
    model = MMDET_MODELS.build(cfg.model)
    model.to(device)
    if no_checkpoint:
        print("checkpoint loading skipped by --no-checkpoint")
        if hasattr(model, "init_weights"):
            print("running model.init_weights()")
            model.init_weights()
    else:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        from mmengine.runner import load_checkpoint
        load_checkpoint(model, str(checkpoint_path), map_location="cpu", strict=False, logger=None)
        print(f"checkpoint loaded with strict=False: {checkpoint_path}")
    model.to(device)
    model.train()
    return model


def save_prediction_target(debug, output_dir, iteration, dataset_index, loss_value):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logits = debug["logits"].detach().cpu()
    targets = debug["targets"].detach().cpu()
    probs = logits.sigmoid()
    prob = probs[0, 0].numpy()
    target = targets[0, 0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(prob, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0].set_title("pred prob")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].imshow(target, origin="lower", vmin=0.0, vmax=1.0, cmap="gray_r")
    axes[1].set_title("GT mask")
    axes[2].imshow((prob >= 0.5).astype(target.dtype), origin="lower", vmin=0.0, vmax=1.0, cmap="gray_r")
    axes[2].set_title("pred >= 0.5")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"iter={iteration} dataset_index={dataset_index} loss={loss_value:.6f}")
    fig.tight_layout()
    out_path = output_dir / f"iter_{iteration:04d}_index_{dataset_index:04d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


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
    torch.manual_seed(0)
    args.num_samples = max(1, min(int(args.num_samples), 4))
    if args.iters <= 0:
        raise ValueError("--iters must be positive")

    print_header("load config")
    cfg = Config.fromfile(args.config)
    apply_config_fixes(cfg)
    print(f"config: {args.config}")
    print("modern config fixes applied")

    print_header("build train dataset")
    init_default_scope("mmdet3d")
    register_runtime_compat_transforms()
    train_cfg = configure_train_dataset(cfg)
    dataset = build_dataset(train_cfg)
    metadata = attach_legacy_infos(dataset, train_cfg.ann_file, list(cfg.class_names))
    print(f"attached legacy metadata: {metadata}")
    print(f"dataset type: {type(dataset)}")
    print(f"dataset length: {len(dataset)}")

    device, device_reason = choose_device(args.device)
    print_header("device")
    print(f"selected device: {device} ({device_reason})")

    print_header("build model")
    model = build_occ_model(cfg, args.checkpoint, args.no_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"optimizer: AdamW lr={args.lr:g}")

    sample_indices = [(args.start_index + i) % len(dataset) for i in range(args.num_samples)]
    print_header("overfit samples")
    print(f"sample indices: {sample_indices}")

    initial_loss = None
    final_loss = None
    saved_paths = []
    for iteration in range(1, args.iters + 1):
        dataset_index = sample_indices[(iteration - 1) % len(sample_indices)]
        batch = prepare_train_batch(dataset, dataset_index, device)

        optimizer.zero_grad(set_to_none=True)
        output = model(return_loss=True, return_occ_debug=True, **batch)
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError(f"expected (losses, debug), got {type(output)}")
        losses, debug = output
        total_loss, scalar_losses = parse_losses(losses)
        loss_value = float(total_loss.detach().cpu())
        if initial_loss is None:
            initial_loss = loss_value
            print_debug_shapes(batch, debug)
        final_loss = loss_value
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite loss at iter {iteration}: {loss_value}")

        total_loss.backward()
        optimizer.step()

        print(f"iter {iteration:04d}/{args.iters:04d} dataset_index={dataset_index} loss={loss_value:.6f} components={scalar_losses}")

        should_save = iteration == 1 or iteration == args.iters or (args.vis_interval > 0 and iteration % args.vis_interval == 0)
        if should_save:
            saved_paths.append(save_prediction_target(debug, args.output_dir, iteration, dataset_index, loss_value))

    print_header("summary")
    print(f"initial loss: {initial_loss:.6f}")
    print(f"final loss: {final_loss:.6f}")
    print(f"loss decreased: {'yes' if final_loss < initial_loss else 'no'}")
    print("saved visualizations:")
    for path in saved_paths:
        print(f"  {path}")
    print("distributed training: no")
    print("tools/train.py changes: none")
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
