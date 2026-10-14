"""Regression checks for the checkpoint-era nuScenes box convention."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mmdet3d.structures import LiDARInstance3DBoxes

from projects.mmdet3d_plugin.modern_compat.evaluation import (
    format_nuscenes_results,
    output_to_legacy_nusc_box,
)


class NuScenesExportTest(unittest.TestCase):
    def setUp(self):
        self.detection = dict(
            bboxes_3d=LiDARInstance3DBoxes(
                torch.tensor([[1.0, 2.0, 0.0, 4.0, 2.0, 2.0, 0.0, 0.5, -0.25]]), box_dim=9),
            scores_3d=torch.tensor([0.8]),
            labels_3d=torch.tensor([0]),
        )

    def test_legacy_box_size_and_yaw(self):
        box, = output_to_legacy_nusc_box(self.detection)
        np.testing.assert_allclose(box.center, [1, 2, 1])
        np.testing.assert_allclose(box.wlh, [4, 2, 2])
        np.testing.assert_allclose(box.orientation.yaw_pitch_roll[0], -np.pi / 2)
        np.testing.assert_allclose(box.velocity, [0.5, -0.25, 0])

    def test_submission_preserves_legacy_box_convention(self):
        dataset = SimpleNamespace(version='v1.0-mini', data_infos=[dict(
            token='sample', lidar2ego_rotation=[1, 0, 0, 0],
            lidar2ego_translation=[0, 0, 0], ego2global_rotation=[1, 0, 0, 0],
            ego2global_translation=[0, 0, 0])])
        cfg = SimpleNamespace(class_names=['car'], input_modality=dict(use_camera=True))
        with tempfile.TemporaryDirectory() as directory:
            path = format_nuscenes_results([dict(pts_bbox=self.detection)], dataset, cfg, directory)
            result = json.loads(Path(path).read_text())['results']['sample'][0]
        np.testing.assert_allclose(result['size'], [4, 2, 2])
        np.testing.assert_allclose(result['translation'], [1, 2, 1])
        self.assertEqual(result['detection_name'], 'car')


if __name__ == '__main__':
    unittest.main()
