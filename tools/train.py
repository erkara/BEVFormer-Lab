import warnings
warnings.filterwarnings("ignore")
import argparse
import csv
import logging
import math
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import DictAction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmdet.models  # noqa: F401,E402
import mmdet3d.models  # noqa: F401,E402
import projects.mmdet3d_plugin  # noqa: F401,E402
from projects.mmdet3d_plugin.modern_compat import (  # noqa: E402
    CHECKPOINT_PATH,
    attach_legacy_infos,
    build_dataset,
    build_train_model,
    choose_device,
    configure_train_dataset,
    initialize_modern_compat,
    parse_losses,
    prepare_config,
    prepare_train_batch,
    print_header,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BEVFormer detection training"
    )
    parser.add_argument("config", nargs="?", help="train config file path (positional form)")
    parser.add_argument("--config-file", help="train config file path")
    parser.add_argument("--work-dir", default="outputs/detection_train", help="output directory")
    parser.add_argument("--data-root", help="override training data root from config")
    parser.add_argument("--train-info", help="override training annotation file from config")
    parser.add_argument("--max-epochs", type=int, help="training epochs; defaults to config runner.max_epochs")
    parser.add_argument("--max-iters", type=int, help="optional optimizer-step limit for short runs")
    parser.add_argument("--lr", type=float, help="override config AdamW learning rate")
    parser.add_argument("--no-checkpoint", action="store_true", help="skip checkpoint loading and initialize weights")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="checkpoint path")
    parser.add_argument("--device", default="cuda", help="device, e.g. cuda, cuda:0, cpu, auto")
    parser.add_argument(
        "--num-samples",
        "--limit-samples",
        dest="num_samples",
        type=int,
        default=0,
        help="limit to the first N training samples; 0 uses the full split",
    )
    parser.add_argument("--save-interval", type=int, default=0, help="also save latest every N steps; 0 saves at epoch ends")
    parser.add_argument("--log-interval", type=int, default=50, help="print status every N steps")
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
        help="distributed launcher; pytorch uses torchrun (slurm and mpi are unsupported)",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--resume-from", help="resume model, optimizer, schedule, and RNG from a training checkpoint")
    parser.add_argument("--gpus", type=int, help="legacy single-GPU argument; torchrun sets distributed process count")
    parser.add_argument("--gpu-ids", type=int, nargs="+", help="legacy single-GPU argument; torchrun sets local device")
    parser.add_argument("--deterministic", action="store_true", help="accepted for CLI compatibility")
    parser.add_argument("--autoscale-lr", action="store_true", help="unsupported in this diagnostic port")
    args = parser.parse_args()

    if args.config and args.config_file:
        parser.error("specify the config either positionally or with --config-file, not both")
    args.config = args.config_file or args.config
    if not args.config:
        parser.error("a config is required (positional or --config-file)")

    if args.options and args.cfg_options:
        raise ValueError("--options and --cfg-options cannot both be specified")
    if args.options:
        args.cfg_options = args.options
    return args


def ensure_supported_args(args):
    if args.launcher not in ("none", "pytorch"):
        raise NotImplementedError("Use --launcher pytorch with torchrun for distributed training.")
    if args.launcher == "none" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("torchrun with multiple processes requires --launcher pytorch")
    if args.launcher == "pytorch" and not all(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise ValueError("--launcher pytorch requires torchrun (RANK, WORLD_SIZE, LOCAL_RANK)")
    if not args.no_validate:
        raise NotImplementedError("Validation/evaluation is not ported for this diagnostic entrypoint. Pass --no-validate.")
    if args.autoscale_lr:
        raise NotImplementedError("--autoscale-lr is not supported in this diagnostic entrypoint.")
    if args.launcher == "pytorch" and (args.gpus is not None or args.gpu_ids is not None):
        raise ValueError("with torchrun, use --nproc-per-node instead of --gpus or --gpu-ids")
    if args.gpus not in (None, 1) or (args.gpu_ids is not None and len(args.gpu_ids) > 1):
        raise NotImplementedError("without torchrun, only one GPU is supported")
    if args.max_iters is not None and args.max_iters <= 0:
        raise ValueError("--max-iters must be positive")
    if args.max_epochs is not None and args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    if args.num_samples < 0:
        raise ValueError("--num-samples must be nonnegative")
    if args.lr is not None and args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.save_interval < 0 or args.log_interval < 0:
        raise ValueError("intervals must be nonnegative")
    if args.resume_from and (args.no_checkpoint or args.checkpoint != CHECKPOINT_PATH):
        raise ValueError("--resume-from cannot be combined with checkpoint initialization options")
    if args.launcher == "pytorch" and args.device not in ("cuda", "auto"):
        raise ValueError("distributed training requires --device cuda or auto")


def setup_distributed(args):
    if args.launcher == "none":
        return False, 0, 1, None
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 1 or not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError("torchrun needs one visible CUDA device per local process")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, world_size, torch.device("cuda", local_rank)


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
        "epoch",
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


def make_optimizer(cfg, model, lr_override):
    optimizer_cfg = cfg.optimizer
    if optimizer_cfg.type != "AdamW":
        raise NotImplementedError(f"optimizer {optimizer_cfg.type} is not supported")
    base_lr = float(lr_override if lr_override is not None else optimizer_cfg.lr)
    weight_decay = float(optimizer_cfg.weight_decay)
    custom_keys = optimizer_cfg.get("paramwise_cfg", {}).get("custom_keys", {})
    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = next((key for key in sorted(custom_keys, key=len, reverse=True) if key in name), None)
        rule = custom_keys[key] if key is not None else {}
        lr = base_lr * float(rule.get("lr_mult", 1.0))
        decay = weight_decay * float(rule.get("decay_mult", 1.0))
        grouped.setdefault((lr, decay), []).append(parameter)
    groups = [dict(params=params, lr=lr, weight_decay=decay) for (lr, decay), params in grouped.items()]
    return torch.optim.AdamW(groups), base_lr


def make_scheduler(cfg, optimizer, total_steps):
    policy = cfg.lr_config
    if policy.policy != "CosineAnnealing" or policy.get("warmup") != "linear":
        raise NotImplementedError("only the Tiny config's cosine schedule with linear warmup is supported")
    warmup_steps = int(policy.warmup_iters)
    warmup_ratio = float(policy.warmup_ratio)
    min_ratio = float(policy.min_lr_ratio)

    def factor(step):
        if step < warmup_steps:
            return warmup_ratio + (1.0 - warmup_ratio) * step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def capture_rng_state():
    numpy_state = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def save_training_state(path, model, optimizer, scheduler, iteration, rows, run_info, rank=0, world_size=1):
    rng_states = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(rng_states, capture_rng_state())
    else:
        rng_states[0] = capture_rng_state()
    if rank != 0:
        return
    payload = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        iteration=iteration,
        loss_log=rows,
        run_info=run_info,
        rng_state=rng_states[0],
    )
    if world_size > 1:
        payload["rng_states"] = rng_states
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_training_state(path, model, optimizer, scheduler, run_info, rank=0, world_size=1):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("run_info") != run_info:
        raise ValueError("resume checkpoint was made with a different config, dataset, subset, seed, or epoch count")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if world_size > 1:
        states = payload.get("rng_states")
        if states is None or len(states) != world_size:
            raise ValueError("distributed resume needs RNG states for every rank")
        state = states[rank]
    else:
        state = payload["rng_state"]
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    return int(payload["iteration"]), list(payload["loss_log"])


def main():
    args = parse_args()
    ensure_supported_args(args)
    distributed, rank, world_size, distributed_device = setup_distributed(args)
    try:
        return run_training(args, distributed, rank, world_size, distributed_device)
    finally:
        if distributed:
            dist.destroy_process_group()


def run_training(args, distributed, rank, world_size, distributed_device):
    logging.disable(logging.INFO)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    work_dir = Path(args.work_dir)
    if (work_dir / "latest.pth").exists() and not args.resume_from:
        raise FileExistsError(f"{work_dir / 'latest.pth'} exists; use --resume-from or a new --work-dir")
    work_dir.mkdir(parents=True, exist_ok=True)

    print_header("load config")
    cfg = prepare_config(args.config, args.cfg_options)
    print(f"config: {args.config}")
    print("modern config fixes applied")

    print_header("build train dataset")
    initialize_modern_compat()
    train_cfg = configure_train_dataset(cfg)
    train_cfg.data_root = args.data_root or cfg.data.train.data_root
    if args.train_info:
        info_path = Path(args.train_info)
    elif args.data_root:
        info_path = Path(args.data_root) / Path(cfg.data.train.ann_file).name
    else:
        info_path = Path(cfg.data.train.ann_file)
    train_cfg.ann_file = str(info_path.expanduser().resolve())
    dataset = build_dataset(train_cfg)
    metadata = attach_legacy_infos(dataset, train_cfg.ann_file, list(cfg.class_names))
    subset_size = min(args.num_samples, len(dataset)) if args.num_samples else len(dataset)
    if subset_size < world_size:
        raise RuntimeError("selected training sample count must be at least the process count")
    steps_per_epoch = subset_size // world_size
    max_epochs = int(args.max_epochs or cfg.runner.max_epochs)
    total_steps = steps_per_epoch * max_epochs
    max_steps = min(args.max_iters or total_steps, total_steps)
    print(f"attached legacy metadata: {metadata}")
    print(f"dataset type: {type(dataset)}")
    print(f"dataset length: {len(dataset)}")
    print(f"data root: {train_cfg.data_root}")
    print(f"train info: {train_cfg.ann_file}")
    print(f"selected sample count: {subset_size}")
    print(f"world size: {world_size}; samples per rank per epoch: {steps_per_epoch}; dropped tail: {subset_size % world_size}")
    print(f"epochs: {max_epochs}; planned steps: {total_steps}; step limit: {max_steps}")

    device, device_reason = (distributed_device, "torchrun LOCAL_RANK") if distributed else choose_device(args.device)
    print_header("device")
    print(f"selected device: {device} ({device_reason})")

    print_header("build model")
    model = build_train_model(cfg, args.checkpoint, args.no_checkpoint or bool(args.resume_from), device)
    optimizer, base_lr = make_optimizer(cfg, model, args.lr)
    scheduler = make_scheduler(cfg, optimizer, total_steps)
    print(f"optimizer: AdamW lr={base_lr:g} weight_decay={float(cfg.optimizer.weight_decay):g}")
    print(f"schedule: {cfg.lr_config.policy}, warmup={cfg.lr_config.warmup_iters} steps")

    run_info = dict(
        config=str(Path(args.config).resolve()),
        train_info=train_cfg.ann_file,
        data_root=str(Path(train_cfg.data_root).resolve()),
        subset_size=subset_size,
        max_epochs=max_epochs,
        seed=args.seed,
        base_lr=base_lr,
    )
    if world_size > 1:
        run_info["world_size"] = world_size
    rows = []
    loss_keys = []
    completed = 0
    if args.resume_from:
        completed, rows = restore_training_state(args.resume_from, model, optimizer, scheduler, run_info, rank, world_size)
        loss_keys = [key for key in rows[0] if "loss" in key and key != "total_loss"] if rows else []
        print(f"resumed from {args.resume_from} at step {completed}")
    if completed >= max_steps:
        raise ValueError(f"checkpoint already completed {completed} steps; requested limit is {max_steps}")

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index,
            find_unused_parameters=bool(cfg.get("find_unused_parameters", True)),
        )

    if rank == 0:
        print_header("distributed training" if world_size > 1 else "single-gpu training")
    checkpoint_path = work_dir / "latest.pth"
    csv_path = work_dir / "train_loss_log.csv"
    current_epoch = None
    permutation = None
    for step in range(completed, max_steps):
        epoch = step // steps_per_epoch
        position = step % steps_per_epoch
        if epoch != current_epoch:
            generator = torch.Generator().manual_seed(args.seed + epoch)
            permutation = torch.randperm(subset_size, generator=generator).tolist()
            current_epoch = epoch
        sample_index = permutation[position * world_size + rank]
        batch = prepare_train_batch(dataset, sample_index, device)
        optimizer.zero_grad(set_to_none=True)

        losses = model(return_loss=True, **batch)
        total_loss, scalar_losses = parse_losses(losses)
        iteration = step + 1
        if distributed:
            is_finite = int(torch.isfinite(total_loss.detach()).item())
            is_finite = min(is_finite, *(int(math.isfinite(value)) for value in scalar_losses.values()))
            finite_flag = torch.tensor(is_finite, device=device)
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if not finite_flag.item():
                raise FloatingPointError(f"non-finite loss on at least one rank at iter {iteration}")
        total_value = finite_or_raise(iteration, total_loss, scalar_losses)

        total_loss.backward()
        grad_norm = compute_grad_norm(model)
        clip_cfg = cfg.optimizer_config.get("grad_clip")
        if clip_cfg:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_cfg.max_norm), norm_type=float(clip_cfg.norm_type))
        lr = max(group["lr"] for group in optimizer.param_groups)
        optimizer.step()
        scheduler.step()

        for key in scalar_losses:
            if key not in loss_keys:
                loss_keys.append(key)
        mem_alloc, mem_reserved = cuda_memory_mb(device)
        row = {
            "iter": iteration,
            "epoch": epoch + 1,
            "sample_index": sample_index,
            "total_loss": total_value,
            "lr": lr,
            "grad_norm": grad_norm,
            "cuda_mem_alloc_mb": mem_alloc,
            "cuda_mem_reserved_mb": mem_reserved,
        }
        row.update(scalar_losses)
        if rank == 0:
            rows.append(row)

        main_keys = [key for key in ("loss_cls", "loss_bbox") if key in scalar_losses]
        main_text = " ".join(f"{key}={scalar_losses[key]:.6f}" for key in main_keys)
        mem_text = ""
        if device.type == "cuda":
            mem_text = f" mem_alloc={mem_alloc:.1f}MB mem_reserved={mem_reserved:.1f}MB"
        if rank == 0 and (iteration == 1 or position == steps_per_epoch - 1 or iteration == max_steps or (args.log_interval and iteration % args.log_interval == 0)):
            print(f"epoch={epoch + 1}/{max_epochs} iter={iteration}/{max_steps} sample={sample_index} "
                  f"total_loss={total_value:.6f} {main_text} lr={lr:.8g} grad_norm={grad_norm:.6f}{mem_text}")
        if position == steps_per_epoch - 1 or iteration == max_steps or (args.save_interval and iteration % args.save_interval == 0):
            save_training_state(checkpoint_path, model.module if distributed else model,
                                optimizer, scheduler, iteration, rows, run_info, rank, world_size)
            if rank == 0:
                save_csv(csv_path, rows, loss_keys)

    if rank == 0:
        print_header("summary")
        print(f"completed steps: {len(rows)}")
        print(f"last total loss: {rows[-1]['total_loss']:.6f}")
        print(f"loss keys: {loss_keys}")
        print(f"csv log: {csv_path}")
        print(f"checkpoint: {checkpoint_path}")
        print("no evaluation run")
        print(f"distributed processes: {world_size}" if distributed else "single-process run")
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
