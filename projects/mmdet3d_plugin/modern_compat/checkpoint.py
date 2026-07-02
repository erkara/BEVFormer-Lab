"""Checkpoint and model-loading helpers for modernized BEVFormer entrypoints."""

import traceback
from pathlib import Path

import torch
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS as MMDET_MODELS


def load_checkpoint_modern(model, checkpoint_path):
    try:
        checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu", strict=False, logger=None)
        print("checkpoint load method: mmengine.runner.load_checkpoint")
        return checkpoint
    except Exception:
        print("mmengine.runner.load_checkpoint failed; trying torch.load fallback")
        traceback.print_exc()
        raw = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(raw, dict):
            state_dict = raw.get("state_dict") or raw.get("model") or raw.get("model_state_dict") or raw
        else:
            state_dict = raw
        incompatible = model.load_state_dict(state_dict, strict=False)
        print("checkpoint load method: torch.load + model.load_state_dict(strict=False)")
        print(f"missing_keys: {list(incompatible.missing_keys)}")
        print(f"unexpected_keys: {list(incompatible.unexpected_keys)}")
        return raw if isinstance(raw, dict) else {}


def build_model(cfg, checkpoint, no_checkpoint, device):
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
        load_checkpoint(model, str(checkpoint_path), map_location="cpu", strict=False, logger=None)
        print(f"checkpoint loaded: {checkpoint_path}")
    model.to(device)
    model.train()
    return model


def build_train_model(cfg, checkpoint, no_checkpoint, device):
    """Build a train-mode BEVFormer model and apply legacy training shims."""
    from .training import patch_legacy_assigner_and_sampler

    model = build_model(cfg, checkpoint, no_checkpoint, device)
    patch_legacy_assigner_and_sampler(model)
    return model
