import argparse
import logging
import sys
import traceback
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.debug_nuscenes_real_full_forward import (  # noqa: E402
    CONFIG_PATH,
    choose_device,
    apply_config_fixes,
    prepare_real_inputs,
    summarize_recursive,
)

import mmdet.models  # noqa: E402,F401
import mmdet3d.models  # noqa: E402,F401
import projects.mmdet3d_plugin  # noqa: E402,F401
from mmdet.registry import MODELS as MMDET_MODELS  # noqa: E402


SEARCH_ROOTS = ("work_dirs", "checkpoints", "ckpts", "pretrained", ".")
CHECKPOINT_SUFFIXES = (".pth", ".pt")


def print_header(name):
    print(f"\n=== {name} ===")


def find_checkpoint_candidates():
    repo_root = Path.cwd()
    seen = set()
    candidates = []
    for root_name in SEARCH_ROOTS:
        root = Path(root_name)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in CHECKPOINT_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                display = resolved.relative_to(repo_root)
            except ValueError:
                display = resolved
            candidates.append((str(display), resolved, resolved.stat().st_size))
    return sorted(candidates, key=lambda item: item[0])


def format_size(num_bytes):
    mib = num_bytes / (1024 * 1024)
    return f"{num_bytes} bytes ({mib:.2f} MiB)"


def print_candidates(candidates):
    print_header("checkpoint candidates")
    if not candidates:
        print("no checkpoint found")
        return
    for idx, (display, _, size) in enumerate(candidates):
        print(f"[{idx}] {display} - {format_size(size)}")


def choose_checkpoint(cli_checkpoint, candidates):
    if cli_checkpoint:
        path = Path(cli_checkpoint).expanduser().resolve()
        print(f"checkpoint selected from CLI: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint path does not exist: {path}")
        return path
    display, path, _ = candidates[0]
    print(f"checkpoint selected by default: {display}")
    return path


def load_checkpoint_modern_safe(model, checkpoint_path):
    print_header("load checkpoint")
    print(f"checkpoint path: {checkpoint_path}")
    checkpoint_loaded = False
    load_error = None
    checkpoint_obj = None

    try:
        from mmengine.runner import load_checkpoint
        checkpoint_obj = load_checkpoint(
            model,
            str(checkpoint_path),
            map_location="cpu",
            strict=False,
            logger=None,
        )
        checkpoint_loaded = True
        print("load method: mmengine.runner.load_checkpoint")
        if isinstance(checkpoint_obj, dict):
            print(f"checkpoint top-level keys: {list(checkpoint_obj.keys())}")
            missing = checkpoint_obj.get("missing_keys")
            unexpected = checkpoint_obj.get("unexpected_keys")
            if missing is not None:
                print(f"missing_keys from checkpoint object: {missing}")
            if unexpected is not None:
                print(f"unexpected_keys from checkpoint object: {unexpected}")
        else:
            print(f"checkpoint return type: {type(checkpoint_obj)}")
    except Exception as exc:
        load_error = exc
        print("mmengine.runner.load_checkpoint failed")
        print(f"exception type: {type(exc).__name__}")
        print(f"exception: {exc}")
        print("\n--- traceback ---")
        traceback.print_exc()

    if checkpoint_loaded:
        return True, None

    print("\n--- torch.load fallback ---")
    try:
        raw = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(raw, dict):
            print(f"checkpoint top-level keys: {list(raw.keys())}")
            state_dict = raw.get("state_dict") or raw.get("model") or raw.get("model_state_dict") or raw
        else:
            print(f"checkpoint object type: {type(raw)}")
            state_dict = raw
        incompatible = model.load_state_dict(state_dict, strict=False)
        print("load method: torch.load + model.load_state_dict(strict=False)")
        print(f"missing_keys: {list(incompatible.missing_keys)}")
        print(f"unexpected_keys: {list(incompatible.unexpected_keys)}")
        return True, None
    except Exception as exc:
        print("torch.load fallback failed")
        print(f"exception type: {type(exc).__name__}")
        print(f"exception: {exc}")
        print("\n--- traceback ---")
        traceback.print_exc()
        return False, load_error or exc


def extract_pts_bbox(output):
    if not isinstance(output, list) or not output:
        raise TypeError(f"expected non-empty list output, got {type(output)}")
    first = output[0]
    if not isinstance(first, dict) or "pts_bbox" not in first:
        raise KeyError(f"expected output[0]['pts_bbox'], got {summarize_recursive(output)}")
    return first["pts_bbox"]


def print_prediction_summary(output):
    print_header("prediction summary")
    print(f"output type: {type(output)}")
    pts_bbox = extract_pts_bbox(output)
    bboxes_3d = pts_bbox.get("bboxes_3d")
    scores_3d = pts_bbox.get("scores_3d")
    labels_3d = pts_bbox.get("labels_3d")
    boxes_tensor = getattr(bboxes_3d, "tensor", bboxes_3d)
    print(f"bboxes_3d tensor shape: {tuple(boxes_tensor.shape) if torch.is_tensor(boxes_tensor) else '<missing>'}")
    print(f"scores_3d shape: {tuple(scores_3d.shape) if torch.is_tensor(scores_3d) else '<missing>'}")
    print(f"labels_3d shape: {tuple(labels_3d.shape) if torch.is_tensor(labels_3d) else '<missing>'}")
    if torch.is_tensor(scores_3d):
        print(f"top 10 scores: {scores_3d[:10].detach().cpu().tolist()}")
    if torch.is_tensor(labels_3d):
        print(f"top 10 labels: {labels_3d[:10].detach().cpu().tolist()}")
    if torch.is_tensor(boxes_tensor):
        print(f"first 3 boxes tensor values: {boxes_tensor[:3].detach().cpu().tolist()}")


def run_full_forward(model, img_tensor, metadata_dict, device):
    print_header("full forward")
    img = [img_tensor.unsqueeze(0).to(device)]
    img_metas = [[metadata_dict]]
    print(f"input img type: {type(img)} len={len(img)} tensor_shape={tuple(img[0].shape)} device={img[0].device}")
    print(f"input img_metas nesting: outer={len(img_metas)} inner={len(img_metas[0])} keys={list(img_metas[0][0].keys())}")
    with torch.no_grad():
        return model(return_loss=False, img=img, img_metas=img_metas)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug BEVFormer checkpoint loading and one mini full-forward sample")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path to load instead of the first discovered candidate")
    return parser.parse_args()


def main():
    logging.disable(logging.INFO)
    torch.manual_seed(0)
    args = parse_args()

    print_header("checkpoint discovery")
    candidates = find_checkpoint_candidates()
    print_candidates(candidates)
    if not candidates and not args.checkpoint:
        print("no checkpoint found")
        print("checkpoint loaded: no")
        print("full forward succeeded: not run")
        print("predictions are checkpoint-based: no, no predictions produced")
        return 0
    checkpoint_path = choose_checkpoint(args.checkpoint, candidates)

    print_header("load config")
    print(f"config: {CONFIG_PATH}")
    cfg = Config.fromfile(CONFIG_PATH)
    apply_config_fixes(cfg)
    print("modern config fixes applied")

    _, _, img_tensor, metadata_dict = prepare_real_inputs(cfg)

    print_header("build model")
    init_default_scope("mmdet")
    print("default scope set: mmdet")
    device, device_reason = choose_device()
    print(f"selected device: {device} ({device_reason})")
    model = MMDET_MODELS.build(cfg.model)
    model.eval()
    model.to(device)

    checkpoint_loaded, load_error = load_checkpoint_modern_safe(model, checkpoint_path)
    print(f"checkpoint loaded: {'yes' if checkpoint_loaded else 'no'}")
    if load_error is not None:
        print(f"checkpoint load error: {type(load_error).__name__}: {load_error}")

    try:
        output = run_full_forward(model, img_tensor, metadata_dict, device)
    except Exception as exc:
        print("full forward failed")
        print(f"exception type: {type(exc).__name__}")
        print(f"exception: {exc}")
        print("\n--- traceback ---")
        traceback.print_exc()
        print_header("summary")
        print(f"checkpoint loaded: {'yes' if checkpoint_loaded else 'no'}")
        print("full forward succeeded: no")
        print(f"predictions are checkpoint-based: {'yes' if checkpoint_loaded else 'no, random/unloaded model'}")
        return 1

    print_prediction_summary(output)
    print_header("summary")
    print(f"checkpoint loaded: {'yes' if checkpoint_loaded else 'no'}")
    print("full forward succeeded: yes")
    print(f"predictions are checkpoint-based: {'yes' if checkpoint_loaded else 'no, random/unloaded model'}")
    print("no training run")
    print("no evaluation run")
    print("no downloads")
    print("no architecture changes")
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
