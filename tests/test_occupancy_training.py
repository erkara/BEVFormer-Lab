"""Focused checks for the standalone occupancy training loop."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from mmengine.config import Config

from projects.mmdet3d_plugin.modern_compat import configure_occupancy_eval_dataset
from tools.occ_metrics import occupancy_scores
from tools.train_occ import (
    epoch_order,
    evaluate_occupancy,
    is_better_validation,
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    training_steps,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = torch.nn.Linear(2, 2)
        self.occupancy_head = torch.nn.Linear(2, 1)


class OccupancyTrainingTest(unittest.TestCase):
    def test_occupancy_scores_and_empty_masks(self):
        logits = torch.tensor([[[[10., -10.], [0., -10.]]]])
        target = torch.tensor([[[[1., 1.], [0., 0.]]]])
        scores = occupancy_scores(logits, target)
        self.assertEqual(scores['intersection'], 1)
        self.assertEqual(scores['target_cells'], 2)
        self.assertEqual(scores['predicted_cells'], 2)
        self.assertAlmostEqual(scores['iou'], 1 / 3)
        self.assertAlmostEqual(scores['dice'], 0.5)
        self.assertEqual(occupancy_scores(logits, target, 0.9)['predicted_cells'], 1)
        empty = occupancy_scores(torch.full_like(logits, -10.), torch.zeros_like(target))
        self.assertEqual((empty['iou'], empty['dice']), (1.0, 1.0))

    def test_best_checkpoint_prefers_iou_then_loss(self):
        self.assertTrue(is_better_validation({'iou': 0.0, 'loss': 0.8}, -1.0, float('inf')))
        self.assertTrue(is_better_validation({'iou': 0.4, 'loss': 0.6}, 0.3, 0.2))
        self.assertFalse(is_better_validation({'iou': 0.2, 'loss': 0.1}, 0.3, 0.6))
        self.assertTrue(is_better_validation({'iou': 0.3, 'loss': 0.5}, 0.3, 0.6))
        self.assertFalse(is_better_validation({'iou': 0.3, 'loss': 0.7}, 0.3, 0.6))

    def test_epochs_and_seeded_shuffle(self):
        indices = list(range(10))
        self.assertEqual(training_steps(10, epochs=5), 50)
        self.assertEqual(training_steps(10, max_iters=3), 3)
        first = epoch_order(indices, 7, 0)
        self.assertEqual(first, epoch_order(indices, 7, 0))
        self.assertEqual(sorted(first), indices)
        self.assertNotEqual(first, epoch_order(indices, 7, 1))

    def test_optimizer_schedule_and_checkpoint_payload(self):
        model = TinyModel()
        optimizer = make_optimizer(model, 2e-4, 0.01, 0.1)
        scheduler = make_scheduler(optimizer, 10, warmup_iters=2)
        self.assertEqual([len(group['params']) for group in optimizer.param_groups], [2, 2])
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 2e-5 / 3)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr'], 2e-4 / 3)
        for _ in range(2):
            optimizer.zero_grad()
            model.img_backbone.weight.sum().backward()
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 2e-5)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr'], 2e-4)

        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory) / 'latest.pth'
            best = Path(directory) / 'best.pth'
            save_checkpoint(latest, model, optimizer, scheduler, 10, 'cfg.py', [], epoch=1)
            save_checkpoint(best, model, optimizer, scheduler, 10, 'cfg.py', [],
                            epoch=1, val_loss=0.2, val_iou=0.3, include_optimizer=False)
            self.assertIn('optimizer', torch.load(latest, weights_only=False))
            best_payload = torch.load(best, weights_only=False)
            self.assertNotIn('optimizer', best_payload)
            self.assertEqual(best_payload['val_loss'], 0.2)

    def test_validation_config_is_deterministic_and_keeps_gt(self):
        root = Path(__file__).resolve().parents[1]
        cfg = Config.fromfile(str(root / 'projects/configs/bevformer/bevformer_tiny_occ.py'))
        val_cfg = configure_occupancy_eval_dataset(cfg)
        types = [transform['type'] for transform in val_cfg.pipeline]
        self.assertTrue(val_cfg.deterministic_history)
        self.assertFalse(val_cfg.test_mode)
        self.assertNotIn('PhotoMetricDistortionMultiViewImage', types)
        self.assertIn('LoadAnnotations3D', types)

    def test_validation_metrics_and_mode_restoration(self):
        class FakeModel(torch.nn.Module):
            def forward(self, index, **kwargs):
                logits = torch.tensor([[[[10., -10.], [-10., -10.]]]])
                if index == 1:
                    logits = torch.tensor([[[[-10., 10.], [-10., -10.]]]])
                target = torch.tensor([[[[1., 0.], [0., 0.]]]])
                return {'loss_occ': torch.tensor(0.2 + 0.2 * index)}, {
                    'logits': logits, 'targets': target,
                }

        model = FakeModel().train()
        with patch('tools.train_occ.prepare_train_batch', side_effect=lambda dataset, index, device: {'index': index}), \
             patch('tools.train_occ.save_prediction_target') as save_vis:
            result = evaluate_occupancy(model, object(), [0, 1], torch.device('cpu'),
                                        Path('/tmp'), epoch=1, iteration=2)
        self.assertTrue(model.training)
        self.assertAlmostEqual(result['loss'], 0.3)
        self.assertAlmostEqual(result['iou'], 1 / 3)
        self.assertAlmostEqual(result['dice'], 0.5)
        self.assertEqual(result['target_cells'], 2)
        self.assertEqual(result['predicted_cells'], 2)
        save_vis.assert_called_once()


if __name__ == '__main__':
    unittest.main()
