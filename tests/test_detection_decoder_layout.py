"""Guard the checkpoint-era DETR decoder tensor convention."""

import unittest
from pathlib import Path

from mmengine.config import Config
from mmengine.registry import init_default_scope

import mmdet.models  # noqa: F401
import mmdet3d.models  # noqa: F401
import projects.mmdet3d_plugin  # noqa: F401
from mmdet.registry import MODELS

from projects.mmdet3d_plugin.modern_compat import apply_config_fixes


class DetectionDecoderLayoutTest(unittest.TestCase):
    def test_tiny_decoder_uses_sequence_first_attention(self):
        root = Path(__file__).resolve().parents[1]
        cfg = Config.fromfile(str(root / 'projects/configs/bevformer/bevformer_tiny.py'))
        apply_config_fixes(cfg)
        init_default_scope('mmdet')
        model = MODELS.build(cfg.model)
        for layer in model.pts_bbox_head.transformer.decoder.layers:
            self.assertFalse(layer.batch_first)
            self.assertFalse(layer.attentions[0].batch_first)
            self.assertFalse(layer.attentions[1].batch_first)


if __name__ == '__main__':
    unittest.main()
