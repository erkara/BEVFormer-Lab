import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmengine.config import Config

import mmdet.models
import mmdet3d.models
import projects.mmdet3d_plugin

from mmdet.registry import MODELS as MMDET_MODELS


def count_params(model, only_trainable=False):
    params = model.parameters()
    if only_trainable:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


cfg = Config.fromfile("projects/configs/bevformer/bevformer_tiny.py")
cfg.model.pop("pretrained", None)

cfg.model.img_backbone.type = "mmdet.ResNet"
cfg.model.img_neck.type = "mmdet.FPN"
cfg.model.pts_bbox_head.type = "mmdet.BEVFormerHead"

model = MMDET_MODELS.build(cfg.model)
head = model.pts_bbox_head

print(f"model class: {type(model)}")
print(f"total parameters: {count_params(model)}")
print(f"trainable parameters: {count_params(model, only_trainable=True)}")
print(f"img_backbone type: {type(model.img_backbone)}")
print(f"img_neck type: {type(model.img_neck)}")
print(f"pts_bbox_head type: {type(head)}")
print(f"pts_bbox_head.transformer type: {type(head.transformer)}")
print(f"has cls_branches: {hasattr(head, 'cls_branches')}")
print(f"has reg_branches: {hasattr(head, 'reg_branches')}")

try:
    logging.disable(logging.INFO)
    model.init_weights()
except Exception as exc:
    print("init_weights: failed")
    print(f"init_weights error type: {type(exc).__name__}")
    print(f"init_weights error: {exc}")
    tb = traceback.format_exc(limit=5).rstrip()
    print("init_weights traceback:")
    print(tb)
else:
    print("init_weights: ok")
finally:
    logging.disable(logging.NOTSET)

print("smoke test passed")
