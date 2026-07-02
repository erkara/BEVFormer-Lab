import argparse
import csv
import logging
import math
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mmdet.models  # noqa: F401,E402
import mmdet3d.models  # noqa: F401,E402
import projects.mmdet3d_plugin  # noqa: F401,E402
from projects.mmdet3d_plugin.modern_compat import (  # noqa: E402
    CHECKPOINT_PATH,
    CONFIG_PATH,
    apply_config_fixes,
    attach_legacy_infos,
    build_dataset,
    build_model,
    choose_device,
    configure_train_dataset,
    parse_losses,
    patch_legacy_assigner_and_sampler,
    prepare_train_batch,
    print_header,
    register_runtime_compat_transforms,
)


def parse_args():
    parser = argparse.ArgumentParser(description="BEVFormer tiny nuScenes-mini overfit diagnostic")
    parser.add_argument("--num-samples", type=int, default=8, help="fixed mini-train subset size")
    parser.add_argument("--max-iters", type=int, default=30, help="optimizer iterations to run")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    parser.add_argument("--no-checkpoint", action="store_true", help="skip checkpoint loading and initialize weights")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="checkpoint path")
    parser.add_argument("--device", default="cuda", help="device, e.g. cuda, cuda:0, cpu, auto")
    parser.add_argument("--work-dir", default="outputs/train_smoke", help="directory for logs and checkpoint")
    return parser.parse_args()


def compute_grad_norm(model):
    total_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.norm(2)
        total_sq += float(param_norm.item() ** 2)
    return total_sq**0.5


def cuda_memory_mb(device):
    if device.type != "cuda":
        return "", ""
    return (
        torch.cuda.memory_allocated(device) / (1024**2),
        torch.cuda.memory_reserved(device) / (1024**2),
    )


def finite_or_raise(iteration, total_loss, scalar_losses):
    total_value = float(total_loss.detach().cpu())
    if not math.isfinite(total_value):
        raise FloatingPointError(f"non-finite total loss at iter {iteration}: {total_value}")
    bad_keys = {key: value for key, value in scalar_losses.items() if not math.isfinite(value)}
    if bad_keys:
        raise FloatingPointError(f"non-finite individual losses at iter {iteration}: {bad_keys}")
    return total_value


def save_csv(path, rows, loss_keys):
    fieldnames = [
        "iter",
        "sample_index",
        "total_loss",
        *loss_keys,
        "lr",
        "grad_norm",
        "cuda_mem_alloc_mb",
        "cuda_mem_reserved_mb",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_curve(path, rows, initial_avg, final_avg, decreased):
    iterations = [row["iter"] for row in rows]
    losses = [row["total_loss"] for row in rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(iterations, losses, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Total loss")
    plt.title(f"BEVFormer tiny mini overfit: {'decreased' if decreased else 'not decreased'}")
    plt.grid(True, alpha=0.3)
    plt.text(
        0.02,
        0.95,
        f"first3 avg={initial_avg:.6f}\nlast3 avg={final_avg:.6f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    logging.disable(logging.INFO)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.max_iters <= 0:
        raise ValueError("--max-iters must be positive")

    torch.manual_seed(0)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print_header("load config")
    cfg = Config.fromfile(CONFIG_PATH)
    apply_config_fixes(cfg)
    print(f"config: {CONFIG_PATH}")
    print("modern config fixes applied")

    print_header("build train dataset")
    init_default_scope("mmdet3d")
    register_runtime_compat_transforms()
    train_cfg = configure_train_dataset(cfg)
    dataset = build_dataset(train_cfg)
    metadata = attach_legacy_infos(dataset, train_cfg.ann_file, list(cfg.class_names))
    subset_indices = list(range(min(args.num_samples, len(dataset))))
    if not subset_indices:
        raise RuntimeError("empty training subset")
    print(f"attached legacy metadata: {metadata}")
    print(f"dataset type: {type(dataset)}")
    print(f"dataset length: {len(dataset)}")
    print(f"fixed subset size: {len(subset_indices)}")

    device, device_reason = choose_device(args.device)
    print_header("device")
    print(f"selected device: {device} ({device_reason})")

    print_header("build model")
    model = build_model(cfg, args.checkpoint, args.no_checkpoint, device)
    patch_legacy_assigner_and_sampler(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    print(f"optimizer: AdamW lr={args.lr:g}")

    print_header("overfit iterations")
    rows = []
    loss_keys = []
    for iteration in range(1, args.max_iters + 1):
        sample_index = subset_indices[(iteration - 1) % len(subset_indices)]
        batch = prepare_train_batch(dataset, sample_index, device)
        optimizer.zero_grad(set_to_none=True)

        losses = model(return_loss=True, **batch)
        total_loss, scalar_losses = parse_losses(losses)
        total_value = finite_or_raise(iteration, total_loss, scalar_losses)
        total_loss.backward()
        grad_norm = compute_grad_norm(model)
        optimizer.step()

        for key in scalar_losses:
            if key not in loss_keys:
                loss_keys.append(key)
        mem_alloc, mem_reserved = cuda_memory_mb(device)
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "iter": iteration,
            "sample_index": sample_index,
            "total_loss": total_value,
            "lr": lr,
            "grad_norm": grad_norm,
            "cuda_mem_alloc_mb": mem_alloc,
            "cuda_mem_reserved_mb": mem_reserved,
        }
        row.update(scalar_losses)
        rows.append(row)

        main_keys = [key for key in ("loss_cls", "loss_bbox") if key in scalar_losses]
        main_text = " ".join(f"{key}={scalar_losses[key]:.6f}" for key in main_keys)
        mem_text = ""
        if device.type == "cuda":
            mem_text = f" mem_alloc={mem_alloc:.1f}MB mem_reserved={mem_reserved:.1f}MB"
        print(
            f"iter={iteration:03d} sample={sample_index} total_loss={total_value:.6f} "
            f"{main_text} grad_norm={grad_norm:.6f}{mem_text}"
        )

    first_count = min(3, len(rows))
    last_count = min(3, len(rows))
    initial_avg = sum(row["total_loss"] for row in rows[:first_count]) / first_count
    final_avg = sum(row["total_loss"] for row in rows[-last_count:]) / last_count
    decreased = final_avg < initial_avg

    csv_path = work_dir / "overfit_loss_log.csv"
    curve_path = work_dir / "overfit_loss_curve.png"
    checkpoint_path = work_dir / "overfit_last.pth"
    save_csv(csv_path, rows, loss_keys)
    save_curve(curve_path, rows, initial_avg, final_avg, decreased)
    torch.save(
        {
            "meta": {
                "config": CONFIG_PATH,
                "checkpoint": None if args.no_checkpoint else args.checkpoint,
                "num_samples": len(subset_indices),
                "max_iters": args.max_iters,
                "lr": args.lr,
            },
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_log": rows,
        },
        checkpoint_path,
    )

    print_header("summary")
    print(f"initial average loss over first {first_count}: {initial_avg:.6f}")
    print(f"final average loss over last {last_count}: {final_avg:.6f}")
    print(f"loss decreased: {decreased}")
    print(f"csv log: {csv_path}")
    print(f"loss curve: {curve_path}")
    print(f"checkpoint: {checkpoint_path}")
    print("no evaluation run")
    print("no distributed training")
    print("no full trainval")
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
