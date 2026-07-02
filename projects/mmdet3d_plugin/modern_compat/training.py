"""Training shims for legacy BEVFormer heads on modern MMDetection."""

from types import SimpleNamespace

import torch
from mmengine.registry import init_default_scope

from .transforms import register_runtime_compat_transforms


def print_header(name):
    print(f"\n=== {name} ===")


def initialize_modern_compat():
    """Register runtime shims needed by legacy BEVFormer on modern OpenMMLab."""
    init_default_scope("mmdet3d")
    register_runtime_compat_transforms()


def choose_device(device_arg="auto"):
    if device_arg != "auto":
        return torch.device(device_arg), "selected by --device"
    if torch.cuda.is_available():
        try:
            torch.empty(1, device="cuda")
            return torch.device("cuda"), "CUDA is available and a test allocation succeeded"
        except Exception as exc:
            return torch.device("cpu"), f"CUDA test allocation failed: {type(exc).__name__}: {exc}"
    return torch.device("cpu"), "torch.cuda.is_available() is false"


def loss_to_scalar(value):
    if torch.is_tensor(value):
        return value.mean()
    if isinstance(value, (list, tuple)):
        tensors = [loss_to_scalar(item) for item in value]
        return sum(tensors)
    raise TypeError(f"unsupported loss value type: {type(value)}")


def parse_losses(losses):
    scalar_losses = {}
    total_loss = None
    for key, value in losses.items():
        scalar = loss_to_scalar(value)
        scalar_losses[key] = float(scalar.detach().cpu())
        if "loss" in key:
            total_loss = scalar if total_loss is None else total_loss + scalar
    if total_loss is None:
        raise RuntimeError(f"no loss keys found in losses: {list(losses.keys())}")
    return total_loss, scalar_losses


class LegacyFocalLossCost:
    def __init__(self, weight=2.0, alpha=0.25, gamma=2.0, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (1 - cls_pred).pow(self.gamma)
        return (pos_cost[:, gt_labels] - neg_cost[:, gt_labels]) * self.weight


class LegacyBBoxL1Cost:
    def __init__(self, weight=0.25):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        return torch.cdist(bbox_pred, gt_bboxes, p=1) * self.weight


def legacy_pseudo_sample(assign_result, pred_bboxes, gt_bboxes):
    gt_inds = assign_result.gt_inds
    pos_inds = torch.nonzero(gt_inds > 0, as_tuple=False).squeeze(-1).unique()
    neg_inds = torch.nonzero(gt_inds == 0, as_tuple=False).squeeze(-1).unique()
    pos_assigned_gt_inds = gt_inds[pos_inds] - 1
    if pos_inds.numel() > 0:
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long()]
    else:
        pos_gt_bboxes = gt_bboxes.new_zeros((0, gt_bboxes.size(-1)))
    return SimpleNamespace(
        pos_inds=pos_inds,
        neg_inds=neg_inds,
        pos_assigned_gt_inds=pos_assigned_gt_inds.long(),
        pos_gt_bboxes=pos_gt_bboxes,
    )


def patch_legacy_assigner_and_sampler(model):
    """Install tensor-based assigner/sampler shims expected by legacy BEVFormer."""
    head = getattr(model, "pts_bbox_head", None)
    assigner = getattr(head, "assigner", None)
    if assigner is None:
        print("legacy assigner cost patch: skipped, no assigner found")
    else:
        assigner.cls_cost = LegacyFocalLossCost(weight=2.0, alpha=0.25, gamma=2.0)
        assigner.reg_cost = LegacyBBoxL1Cost(weight=0.25)
        print("legacy assigner cost patch: tensor-based FocalLossCost and BBoxL1Cost active")
    sampler = getattr(head, "sampler", None)
    if sampler is None:
        print("legacy sampler patch: skipped, no sampler found")
    else:
        sampler.sample = legacy_pseudo_sample
        print("legacy sampler patch: tensor-based pseudo sampler active")
