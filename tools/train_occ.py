"""Single-device occupancy trainer using the legacy BEVFormer data pipeline."""

import argparse
import csv
import logging
import math
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
from tools.occ_metrics import occupancy_scores  # noqa: E402
from projects.mmdet3d_plugin.modern_compat import (  # noqa: E402
    CHECKPOINT_PATH,
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
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--epochs", type=int, help="passes over the selected training samples (default: 5)")
    duration.add_argument("--max-iters", type=int, help="optimizer steps for short tests, instead of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate")
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1, help="learning-rate multiplier for pretrained image backbone")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--val-samples", type=int, default=0, help="number of mini-val samples per epoch; 0 uses all")
    parser.add_argument("--no-validate", action="store_true", help="skip held-out validation and best-checkpoint selection")
    parser.add_argument("--save-interval", type=int, default=0, help="also save iter_N checkpoint every N steps; 0 saves latest at epoch ends")
    parser.add_argument("--vis-interval", type=int, default=100, help="save extra visualizations every N steps; 0 saves first and last only")
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


def training_steps(sample_count, epochs=None, max_iters=None):
    if sample_count <= 0:
        raise ValueError("selected training sample count must be positive")
    if epochs is not None and max_iters is not None:
        raise ValueError("use --epochs or --max-iters, not both")
    if max_iters is not None:
        if max_iters <= 0:
            raise ValueError("--max-iters must be positive")
        return max_iters
    epochs = 5 if epochs is None else epochs
    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    return epochs * sample_count


def epoch_order(indices, seed, epoch):
    generator = torch.Generator().manual_seed(seed + epoch)
    return [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]


def make_optimizer(model, lr, weight_decay, backbone_lr_mult):
    """Fine-tune the pretrained image backbone more gently than the BEV/head layers."""
    if lr <= 0 or weight_decay < 0 or backbone_lr_mult <= 0:
        raise ValueError("learning rates must be positive and weight decay nonnegative")
    backbone = []
    other = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if name.startswith("img_backbone.") else other).append(parameter)
    if not backbone or not other:
        raise ValueError("expected both image-backbone and non-backbone trainable parameters")
    optimizer = torch.optim.AdamW(
        [dict(params=backbone, lr=lr * backbone_lr_mult),
         dict(params=other, lr=lr)],
        weight_decay=weight_decay,
    )
    return optimizer


def make_scheduler(optimizer, total_iters, warmup_iters=500, warmup_ratio=1 / 3, min_lr_ratio=1e-3):
    """Apply Tiny-config warmup and cosine decay over this run's update count."""
    warmup = min(warmup_iters, max(total_iters - 1, 0))

    def factor(step):
        if step < warmup:
            return warmup_ratio + (1 - warmup_ratio) * step / warmup
        progress = min(1.0, (step - warmup) / max(total_iters - warmup, 1))
        return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path, model, optimizer, scheduler, iteration, config_path, loss_history,
                    epoch=None, val_loss=None, val_iou=None, include_optimizer=True):
    """Save optimizer/scheduler state in ``latest.pth``; keep ``best.pth`` model-only."""
    payload = dict(
        state_dict=model.state_dict(),
        iteration=int(iteration),
        config_path=str(config_path),
        epoch=epoch,
        val_loss=val_loss,
        val_iou=val_iou,
    )
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
        payload["loss_history"] = list(loss_history)
    torch.save(payload, path)


def append_loss_csv(csv_path, rows):
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "epoch", "dataset_index", "total_loss", "loss_occ", "iou", "dice", "running_avg", "lr", "grad_norm"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_epoch_csv(path, row):
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "iteration", "train_loss", "val_loss", "val_iou", "val_dice", "target_cells", "predicted_cells"])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def evaluate_occupancy(model, dataset, indices, device, vis_dir, epoch, iteration):
    """Score held-out frames; IoU/Dice pool cells across samples before dividing."""
    model.eval()
    losses = []
    intersection = target_cells = predicted_cells = 0
    try:
        with torch.no_grad():
            for rank, index in enumerate(indices):
                batch = prepare_train_batch(dataset, index, device)
                result, debug = model(return_loss=True, return_occ_debug=True, **batch)
                loss, _ = parse_losses(result)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite validation loss at index {index}")
                losses.append(float(loss.detach().cpu()))
                scores = occupancy_scores(debug["logits"], debug["targets"])
                intersection += scores["intersection"]
                target_cells += scores["target_cells"]
                predicted_cells += scores["predicted_cells"]
                if rank == 0:
                    save_prediction_target(debug, vis_dir, iteration, index, losses[-1], scores)
    finally:
        model.train()
    union = target_cells + predicted_cells - intersection
    total = target_cells + predicted_cells
    return dict(loss=sum(losses) / len(losses),
                iou=intersection / union if union else 1.0,
                dice=2 * intersection / total if total else 1.0,
                target_cells=target_cells, predicted_cells=predicted_cells)


def is_better_validation(val, best_iou, best_loss):
    """Prefer mask overlap; use loss only when validation IoU ties."""
    return val["iou"] > best_iou or (val["iou"] == best_iou and val["loss"] < best_loss)


def save_prediction_target(debug, out_dir, iteration, dataset_index, loss_value, scores):
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
    fig.suptitle(f"iter={iteration} dataset_index={dataset_index}  IoU={scores['iou']:.3f}  Dice={scores['dice']:.3f}  loss={loss_value:.4f}")
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
    if args.val_samples < 0:
        raise ValueError("--val-samples must be nonnegative")
    set_seed(args.seed)

    work_dir = Path(args.work_dir)
    vis_dir = work_dir / "vis"
    csv_path = work_dir / "loss_history.csv"
    epoch_csv_path = work_dir / "epoch_history.csv"
    if (work_dir / "latest.pth").exists() or csv_path.exists():
        raise FileExistsError(f"{work_dir} already contains training output; choose a new --work-dir")
    work_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

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
    total_iters = training_steps(len(indices), args.epochs, args.max_iters)
    print(f"attached legacy metadata: {metadata}")
    print(f"dataset type: {type(dataset)}")
    print(f"dataset length: {len(dataset)}")
    print(f"selected training sample count: {len(indices)}")
    print(f"one epoch: {len(indices)} optimizer steps; total: {total_iters} steps")

    val_dataset = None
    val_indices = []
    if not args.no_validate:
        # Detection's test pipeline has no boxes; use the GT-bearing mini-val
        # pipeline configured for repeatable occupancy validation.
        val_cfg = configure_occupancy_eval_dataset(cfg)
        val_dataset = build_dataset(val_cfg)
        attach_legacy_infos(val_dataset, val_cfg.ann_file, list(cfg.class_names))
        val_indices = list(range(min(args.val_samples or len(val_dataset), len(val_dataset))))
        print(f"held-out validation samples: {len(val_indices)}")

    device, device_reason = choose_device(args.device)
    print_header("build model")
    print(f"device: {device} ({device_reason})")
    model, checkpoint_loaded, checkpoint_path = build_occ_model(cfg, args.checkpoint, args.no_checkpoint, device)
    optimizer = make_optimizer(model, args.lr, args.weight_decay, args.backbone_lr_mult)
    scheduler = make_scheduler(
        optimizer, total_iters,
        warmup_iters=int(cfg.lr_config.warmup_iters),
        warmup_ratio=float(cfg.lr_config.warmup_ratio),
        min_lr_ratio=float(cfg.lr_config.min_lr_ratio),
    )
    grad_clip = float(cfg.optimizer_config.grad_clip.max_norm)
    print(f"checkpoint loaded: {'yes' if checkpoint_loaded else 'no'}")
    if checkpoint_path is not None:
        print(f"checkpoint path: {checkpoint_path}")
    print(f"model type: {cfg.model.type}")
    print(f"optimizer: AdamW head/BEV lr={args.lr:g}, backbone lr={args.lr * args.backbone_lr_mult:g}, weight_decay={args.weight_decay:g}")
    print(f"schedule: {cfg.lr_config.warmup_iters} warmup steps then cosine; gradient clip: {grad_clip:g}")
    print(f"work-dir: {work_dir}")

    print_header("training")
    loss_history = []
    pending_csv_rows = []
    running_window = []
    best_val_iou = -1.0
    best_val_loss = float("inf")
    current_epoch = -1
    epoch_losses = []
    for iteration in range(1, total_iters + 1):
        epoch = (iteration - 1) // len(indices)
        position = (iteration - 1) % len(indices)
        if epoch != current_epoch:
            # One epoch visits each selected sample once in a seeded new order.
            order = epoch_order(indices, args.seed, epoch)
            epoch_losses = []
            current_epoch = epoch
        dataset_index = order[position]
        batch = prepare_train_batch(dataset, dataset_index, device)

        optimizer.zero_grad(set_to_none=True)
        losses, debug = model(return_loss=True, return_occ_debug=True, **batch)
        total_loss, scalar_losses = parse_losses(losses)
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite loss at iter {iteration}: {float(total_loss.detach().cpu())}")
        if iteration == 1:
            print_debug_shapes(batch, debug)

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        lr = max(group["lr"] for group in optimizer.param_groups)
        optimizer.step()
        scheduler.step()

        total_value = float(total_loss.detach().cpu())
        loss_occ = float(scalar_losses.get("loss_occ", total_value))
        scores = occupancy_scores(debug["logits"], debug["targets"])
        running_window.append(total_value)
        if len(running_window) > 20:
            running_window.pop(0)
        running_avg = sum(running_window) / len(running_window)
        epoch_losses.append(total_value)
        loss_history.append(dict(iteration=iteration, dataset_index=dataset_index, total_loss=total_value, loss_occ=loss_occ))
        pending_csv_rows.append(dict(
            iteration=iteration,
            epoch=epoch + 1,
            dataset_index=dataset_index,
            total_loss=f"{total_value:.8f}",
            loss_occ=f"{loss_occ:.8f}",
            iou=f"{scores['iou']:.8f}",
            dice=f"{scores['dice']:.8f}",
            running_avg=f"{running_avg:.8f}",
            lr=f"{lr:.8g}",
            grad_norm=f"{float(grad_norm):.8f}",
        ))

        should_log = iteration == 1 or iteration == total_iters or (args.log_interval > 0 and iteration % args.log_interval == 0)
        if should_log:
            print(f"iter {iteration:06d}/{total_iters:06d} epoch={epoch + 1} dataset_index={dataset_index} total_loss={total_value:.6f} loss_occ={loss_occ:.6f} running_avg={running_avg:.6f} lr={lr:.8g}")

        if pending_csv_rows:
            append_loss_csv(csv_path, pending_csv_rows)
            pending_csv_rows = []

        should_vis = iteration == 1 or iteration == total_iters or (args.vis_interval > 0 and iteration % args.vis_interval == 0)
        if should_vis:
            vis_path = save_prediction_target(debug, vis_dir, iteration, dataset_index, total_value, scores)
            if should_log:
                print(f"saved visualization: {vis_path}")

        should_save_iter = args.save_interval > 0 and iteration % args.save_interval == 0
        if should_save_iter:
            ckpt_path = work_dir / f"iter_{iteration:06d}.pth"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, iteration, args.config,
                            loss_history, epoch=epoch + 1)
            print(f"saved checkpoint: {ckpt_path}")

        epoch_end = position == len(indices) - 1 or iteration == total_iters
        if epoch_end:
            val = None
            if val_dataset is not None:
                val = evaluate_occupancy(model, val_dataset, val_indices, device,
                                         vis_dir / "val", epoch + 1, iteration)
                print(f"epoch {epoch + 1}: train_loss={sum(epoch_losses) / len(epoch_losses):.6f} "
                      f"val_loss={val['loss']:.6f} val_iou={val['iou']:.4f} val_dice={val['dice']:.4f} "
                      f"target_cells={val['target_cells']} pred_cells={val['predicted_cells']}")
                if is_better_validation(val, best_val_iou, best_val_loss):
                    best_val_iou = val["iou"]
                    best_val_loss = val["loss"]
                    save_checkpoint(work_dir / "best.pth", model, optimizer, scheduler, iteration,
                                    args.config, loss_history, epoch=epoch + 1,
                                    val_loss=val["loss"], val_iou=val["iou"],
                                    include_optimizer=False)
                    print(f"saved best checkpoint: {work_dir / 'best.pth'}")
            else:
                print(f"epoch {epoch + 1}: train_loss={sum(epoch_losses) / len(epoch_losses):.6f}")
            append_epoch_csv(epoch_csv_path, dict(
                epoch=epoch + 1, iteration=iteration,
                train_loss=f"{sum(epoch_losses) / len(epoch_losses):.8f}",
                val_loss=f"{val['loss']:.8f}" if val else "",
                val_iou=f"{val['iou']:.8f}" if val else "",
                val_dice=f"{val['dice']:.8f}" if val else "",
                target_cells=val["target_cells"] if val else "",
                predicted_cells=val["predicted_cells"] if val else "",
            ))
        if epoch_end or should_save_iter:
            save_checkpoint(work_dir / "latest.pth", model, optimizer, scheduler, iteration,
                            args.config, loss_history, epoch=epoch + 1,
                            val_loss=val["loss"] if epoch_end and val else None,
                            val_iou=val["iou"] if epoch_end and val else None)

    print_header("summary")
    print(f"training loss finite: yes")
    print(f"initial loss: {loss_history[0]['total_loss']:.6f}")
    print(f"final loss: {loss_history[-1]['total_loss']:.6f}")
    print(f"latest checkpoint: {work_dir / 'latest.pth'}")
    if val_dataset is not None:
        print(f"best checkpoint: {work_dir / 'best.pth'} (val_iou={best_val_iou:.4f}, val_loss={best_val_loss:.6f})")
    print(f"loss CSV: {csv_path}")
    print(f"epoch CSV: {epoch_csv_path}")
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
