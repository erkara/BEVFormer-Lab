import argparse
import logging
import sys
import traceback
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Minimal BEVFormer tiny nuScenes-mini training smoke test")
    parser.add_argument("--max-iters", type=int, default=2, help="maximum optimizer iterations to run")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate")
    parser.add_argument("--no-checkpoint", action="store_true", help="skip checkpoint loading and initialize weights")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="checkpoint path")
    parser.add_argument("--device", default="auto", help="device, e.g. auto, cuda, cuda:0, cpu")
    parser.add_argument("--start-index", type=int, default=0, help="first train dataset index to try")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.disable(logging.INFO)
    if args.max_iters <= 0:
        raise ValueError("--max-iters must be positive")

    torch.manual_seed(0)

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
    print(f"attached legacy metadata: {metadata}")
    print(f"dataset type: {type(dataset)}")
    print(f"dataset length: {len(dataset)}")

    device, device_reason = choose_device(args.device)
    print_header("device")
    print(f"selected device: {device} ({device_reason})")

    print_header("build model")
    model = build_model(cfg, args.checkpoint, args.no_checkpoint, device)
    patch_legacy_assigner_and_sampler(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    print(f"optimizer: AdamW lr={args.lr:g}")

    print_header("training smoke iterations")
    last_scalar_losses = None
    for iteration in range(args.max_iters):
        dataset_index = (args.start_index + iteration) % len(dataset)
        print(f"\n--- iteration {iteration + 1}/{args.max_iters} dataset_index={dataset_index} ---")
        batch = prepare_train_batch(dataset, dataset_index, device)
        print(f"img shape: {tuple(batch['img'].shape)}")
        print(f"img_metas queue keys: {list(batch['img_metas'][0].keys())}")
        print(f"gt boxes type: {type(batch['gt_bboxes_3d'][0])}")
        print(f"gt labels shape: {tuple(batch['gt_labels_3d'][0].shape)}")

        losses = model(return_loss=True, **batch)
        total_loss, scalar_losses = parse_losses(losses)
        last_scalar_losses = scalar_losses
        print("losses:")
        for key, value in scalar_losses.items():
            print(f"  {key}: {value:.6f}")
        print(f"  total_loss: {float(total_loss.detach().cpu()):.6f}")

        total_loss.backward()
        print("backward: ok")
        optimizer.step()
        print("optimizer step: ok")
        optimizer.zero_grad(set_to_none=True)

    print_header("summary")
    print("dataset builds: yes")
    print("one batch loads: yes")
    print("forward loss works: yes")
    print("backward works: yes")
    print("optimizer step works: yes")
    if last_scalar_losses is not None:
        print(f"last loss keys: {list(last_scalar_losses.keys())}")
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
