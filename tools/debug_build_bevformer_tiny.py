import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmengine.config import Config

import mmdet.models
import mmdet3d.models
import projects.mmdet3d_plugin

from mmdet.registry import MODELS as MMDET_MODELS

cfg = Config.fromfile("projects/configs/bevformer/bevformer_tiny.py")
cfg.model.pop("pretrained", None)

cfg.model.img_backbone.type = "mmdet.ResNet"
cfg.model.img_neck.type = "mmdet.FPN"
cfg.model.pts_bbox_head.type = "mmdet.BEVFormerHead"

model = MMDET_MODELS.build(cfg.model)

print("model built")
print(type(model))
