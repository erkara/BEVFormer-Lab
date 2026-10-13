import argparse
import csv
import logging
import random
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


DEFAULT_CONFIG = "projects/configs/bevformer/bevformer_tiny_occ.py"
DEFAULT_WORK_DIR = "outputs/occ_train"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the experimental BEVFormer occupancy model")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="occupancy config path")
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR, help="directory for checkpoints/logs/visualizations")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="optional initialization checkpoint")
    parser.add_argument("--no-checkpoint", action="store_true", help="skip checkpoint initialization")
    parser.add_argument("--device", default="auto", help="device, e.g. auto, cuda, cuda:0, cpu")
    parser.add_argument("--start-index", type=int, default=0, help="first train dataset index")
    parser.add_argument("--num-samples", type=int, default=0, help="limit training to this many cyclic samples; <=0 uses full train set")
    parser.add_argument("--max-iters", type=int, default=100, help="number of optimizer iterations")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--save-interval", type=int, default=100, help="save iter_N checkpoint every N iterations; <=0 disables interval saves")
    parser.add_argument("--vis-interval", type=int, default=25, help="save prediction-vs-target visualization every N iterations; <=0 disables interval visualizations")
    parser.add_argument("--log-interval", type=int, default=10, help="print training status every N iterations")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_occ_model(cfg, checkpoint, no_checkpoint, device):
    init_default_scope("mmdet")
    model = MMDET_MODELS.build(cfg.model)
    model.to(device)
    loaded = False
    checkpoint_path = None
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
        loaded = True
        print(f"checkpoint loaded with strict=False: {checkpoint_path}")
    model.to(device)
    model.train()
    return model, loaded, checkpoint_path


def selected_indices(dataset_len, start_index, num_samples):
    if dataset_len <= 0:
        raise ValueError("dataset is empty")
    if num_samples is None or num_samples <= 0:
        count = dataset_len
    else:
        count = min(int(num_samples), dataset_len)
    return [(start_index + i) % dataset_len for i in range(count)]


def save_checkpoint(path, model, optimizer, iteration, config_path, loss_history):
    payload = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
        iteration=int(iteration),
        config_path=str(config_path),
        loss_history=list(loss_history),
    )
    torch.save(payload, path)


def append_loss_csv(csv_path, rows):
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "dataset_index", "total_loss", "loss_occ", "running_avg"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_prediction_target(debug, out_dir, iteration, dataset_index, loss_value):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logits = debug["logits"].detach().cpu()
    targets = debug["targets"].detach().cpu()
    prob = logits.sigmoid()[0, 0].numpy()
    target = targets[0, 0].numpy()
    pred = (prob >= 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    im0 = axes[0].imshow(prob, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0].set_title("pred prob")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].imshow(target, origin="lower", vmin=0.0, vmax=1.0, cmap="gray_r")
    axes[1].set_title("GT mask")
    axes[2].imshow(pred, origin="lower", vmin=0.0, vmax=1.0, cmap="gray_r")
    axes[2].set_title("pred >= 0.5")
    axes[3].imshow(np.abs(pred - target), origin="lower", vmin=0.0, vmax=1.0, cmap="magma")
    axes[3].set_title("abs error")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"iter={iteration} dataset_index={dataset_index} loss={loss_value:.6f}")
    fig.tight_layout()
    out_path = out_dir / f"iter_{iteration:06d}_index_{dataset_index:04d}.png"
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
    if args.max_iters <= 0:
        raise ValueError("--max-iters must be positive")
    set_seed(args.seed)

    work_dir = Path(args.work_dir)
    vis_dir = work_dir / "vis"
    work_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = work_dir / "loss_history.csv"
    if csv_path.exists():
        csv_path.unlink()

    print_header("load config")
    cfg = Config.fromfile(args.config)
    apply_config_fixes(cfg)
    print(f"config path: {args.config}")

    print_header("build train dataset")
    init_default_scope("mmdet3d")
    register_runtime_compat_transforms()
    train_cfg = configure_train_dataset(cfg)
    dataset = build_dataset(train_cfg)
    metadata = attach_legacy_infos(dataset, train_cfg.ann_file, list(cfg.class_names))
    indices = selected_indices(len(dataset), args.start_index, args.num_samples)
    print(f"attached legacy metadata: {metadata}")
    print(f"dataset type: {type(dataset)}")
    print(f"dataset length: {len(dataset)}")
    print(f"selected training sample count: {len(indices)}")

    device, device_reason = choose_device(args.device)
    print_header("build model")
    print(f"device: {device} ({device_reason})")
    model, checkpoint_loaded, checkpoint_path = build_occ_model(cfg, args.checkpoint, args.no_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"checkpoint loaded: {'yes' if checkpoint_loaded else 'no'}")
    if checkpoint_path is not None:
        print(f"checkpoint path: {checkpoint_path}")
    print(f"model type: {cfg.model.type}")
    print(f"optimizer: AdamW lr={args.lr:g} weight_decay={args.weight_decay:g}")
    print(f"work-dir: {work_dir}")

    print_header("training")
    loss_history = []
    pending_csv_rows = []
    running_window = []
    for iteration in range(1, args.max_iters + 1):
        dataset_index = indices[(iteration - 1) % len(indices)]
        batch = prepare_train_batch(dataset, dataset_index, device)

        optimizer.zero_grad(set_to_none=True)
        losses, debug = model(return_loss=True, return_occ_debug=True, **batch)
        total_loss, scalar_losses = parse_losses(losses)
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite loss at iter {iteration}: {float(total_loss.detach().cpu())}")
        if iteration == 1:
            print_debug_shapes(batch, debug)

        total_loss.backward()
        optimizer.step()

        total_value = float(total_loss.detach().cpu())
        loss_occ = float(scalar_losses.get("loss_occ", total_value))
        running_window.append(total_value)
        if len(running_window) > 20:
            running_window.pop(0)
        running_avg = sum(running_window) / len(running_window)
        loss_history.append(dict(iteration=iteration, dataset_index=dataset_index, total_loss=total_value, loss_occ=loss_occ))
        pending_csv_rows.append(dict(
            iteration=iteration,
            dataset_index=dataset_index,
            total_loss=f"{total_value:.8f}",
            loss_occ=f"{loss_occ:.8f}",
            running_avg=f"{running_avg:.8f}",
        ))

        should_log = iteration == 1 or iteration == args.max_iters or (args.log_interval > 0 and iteration % args.log_interval == 0)
        if should_log:
            print(f"iter {iteration:06d}/{args.max_iters:06d} dataset_index={dataset_index} total_loss={total_value:.6f} loss_occ={loss_occ:.6f} running_avg={running_avg:.6f}")

        if pending_csv_rows:
            append_loss_csv(csv_path, pending_csv_rows)
            pending_csv_rows = []

        should_vis = iteration == 1 or iteration == args.max_iters or (args.vis_interval > 0 and iteration % args.vis_interval == 0)
        if should_vis:
            vis_path = save_prediction_target(debug, vis_dir, iteration, dataset_index, total_value)
            if should_log:
                print(f"saved visualization: {vis_path}")

        should_save_iter = args.save_interval > 0 and iteration % args.save_interval == 0
        if should_save_iter:
            ckpt_path = work_dir / f"iter_{iteration:06d}.pth"
            save_checkpoint(ckpt_path, model, optimizer, iteration, args.config, loss_history)
            print(f"saved checkpoint: {ckpt_path}")

        save_checkpoint(work_dir / "latest.pth", model, optimizer, iteration, args.config, loss_history)

    print_header("summary")
    print(f"training loss finite: yes")
    print(f"initial loss: {loss_history[0]['total_loss']:.6f}")
    print(f"final loss: {loss_history[-1]['total_loss']:.6f}")
    print(f"latest checkpoint: {work_dir / 'latest.pth'}")
    print(f"loss CSV: {csv_path}")
    print(f"visualization dir: {vis_dir}")
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
