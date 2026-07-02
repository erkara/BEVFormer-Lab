import warnings
warnings.filterwarnings("ignore")
import argparse
import csv
import logging
import math
import sys
import traceback
from pathlib import Path

import torch
from mmengine.config import DictAction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmdet.models  # noqa: F401,E402
import mmdet3d.models  # noqa: F401,E402
import projects.mmdet3d_plugin  # noqa: F401,E402
from projects.mmdet3d_plugin.modern_compat import (  # noqa: E402
    CHECKPOINT_PATH,
    build_train_dataset,
    build_train_model,
    choose_device,
    initialize_modern_compat,
    parse_losses,
    prepare_config,
    prepare_train_batch,
    print_header,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-GPU BEVFormer tiny nuScenes-mini training diagnostic"
    )
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", default="outputs/train_entrypoint_smoke", help="output directory")
    parser.add_argument("--max-iters", type=int, default=5, help="maximum optimizer iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    parser.add_argument("--no-checkpoint", action="store_true", help="skip checkpoint loading and initialize weights")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="checkpoint path")
    parser.add_argument("--device", default="cuda", help="device, e.g. cuda, cuda:0, cpu, auto")
    parser.add_argument(
        "--num-samples",
        "--limit-samples",
        dest="num_samples",
        type=int,
        default=8,
        help="fixed mini-train subset size to cycle through",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--no-validate", action="store_true", help="required; validation is not ported here")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config settings before applying compatibility fixes",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="deprecated alias for --cfg-options",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="distributed launcher; only none is supported in this diagnostic port",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume-from", help="unsupported in this diagnostic port")
    parser.add_argument("--gpus", type=int, help="legacy argument accepted only for single-GPU compatibility")
    parser.add_argument("--gpu-ids", type=int, nargs="+", help="legacy argument accepted only for single-GPU compatibility")
    parser.add_argument("--deterministic", action="store_true", help="accepted for CLI compatibility")
    parser.add_argument("--autoscale-lr", action="store_true", help="unsupported in this diagnostic port")
    args = parser.parse_args()

    if args.options and args.cfg_options:
        raise ValueError("--options and --cfg-options cannot both be specified")
    if args.options:
        args.cfg_options = args.options
    return args


def ensure_supported_args(args):
    if args.launcher != "none":
        raise NotImplementedError("Distributed training is not ported for tools/train.py yet. Use --launcher none.")
    if not args.no_validate:
        raise NotImplementedError("Validation/evaluation is not ported for this diagnostic entrypoint. Pass --no-validate.")
    if args.resume_from:
        raise NotImplementedError("--resume-from is not supported in this diagnostic entrypoint yet.")
    if args.autoscale_lr:
        raise NotImplementedError("--autoscale-lr is not supported in this diagnostic entrypoint.")
    if args.gpus not in (None, 1):
        raise NotImplementedError("Only single-GPU diagnostic training is supported.")
    if args.gpu_ids is not None and len(args.gpu_ids) > 1:
        raise NotImplementedError("Only one GPU id is supported in this diagnostic entrypoint.")
    if args.max_iters <= 0:
        raise ValueError("--max-iters must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")


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


def main():
    args = parse_args()
    ensure_supported_args(args)
    logging.disable(logging.INFO)
    torch.manual_seed(args.seed)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print_header("load config")
    cfg = prepare_config(args.config, args.cfg_options)
    print(f"config: {args.config}")
    print("modern config fixes applied")

    config_copy = work_dir / Path(args.config).name
    cfg.dump(config_copy)

    print_header("build train dataset")
    initialize_modern_compat()
    dataset, metadata = build_train_dataset(cfg)
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
    model = build_train_model(cfg, args.checkpoint, args.no_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    print(f"optimizer: AdamW lr={args.lr:g}")

    print_header("single-gpu train iterations")
    rows = []
    loss_keys = []
    first_batch_loaded = False
    forward_ok = False
    backward_ok = False
    step_ok = False
    for iteration in range(1, args.max_iters + 1):
        sample_index = subset_indices[(iteration - 1) % len(subset_indices)]
        batch = prepare_train_batch(dataset, sample_index, device)
        first_batch_loaded = True
        optimizer.zero_grad(set_to_none=True)

        losses = model(return_loss=True, **batch)
        total_loss, scalar_losses = parse_losses(losses)
        forward_ok = True
        total_value = finite_or_raise(iteration, total_loss, scalar_losses)

        total_loss.backward()
        backward_ok = True
        grad_norm = compute_grad_norm(model)
        optimizer.step()
        step_ok = True

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

    csv_path = work_dir / "train_loss_log.csv"
    checkpoint_path = work_dir / "train_entrypoint_last.pth"
    save_csv(csv_path, rows, loss_keys)
    torch.save(
        {
            "meta": {
                "config": args.config,
                "checkpoint": None if args.no_checkpoint else args.checkpoint,
                "num_samples": len(subset_indices),
                "max_iters": args.max_iters,
                "lr": args.lr,
                "seed": args.seed,
                "diagnostic_entrypoint": "tools/train.py",
            },
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_log": rows,
        },
        checkpoint_path,
    )

    print_header("summary")
    print("dataset builds: yes")
    print(f"one batch loads: {'yes' if first_batch_loaded else 'no'}")
    print(f"forward loss works: {'yes' if forward_ok else 'no'}")
    print(f"backward works: {'yes' if backward_ok else 'no'}")
    print(f"optimizer step works: {'yes' if step_ok else 'no'}")
    print(f"last total loss: {rows[-1]['total_loss']:.6f}")
    print(f"loss keys: {loss_keys}")
    print(f"csv log: {csv_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"config copy: {config_copy}")
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
