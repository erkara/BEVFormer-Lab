import copy

import numpy as np
import torch
from mmdet.registry import MODELS

from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
from projects.mmdet3d_plugin.bevformer.occupancy import (
    VEHICLE_CLASSES,
    BEVGridSpec,
    rasterize_bev_occupancy,
)


@MODELS.register_module()
class BEVFormerOccupancy(BEVFormer):
    """Experimental BEVFormer occupancy/objectness model.

    This wrapper keeps the original BEVFormer detection model untouched with inheritance.
    It reuses the image backbone/FPN and calls the existing BEVFormerHead with
    ``only_bev=True`` to run only the BEV encoder. The detection decoder and
    detection losses are not used in this model path.
    """

    def __init__(self, occupancy_head, class_names, occupancy_classes=VEHICLE_CLASSES, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.occupancy_head = MODELS.build(occupancy_head)
        self.class_names = tuple(class_names)
        self.occupancy_classes = tuple(occupancy_classes)

        if self.pts_bbox_head is None:
            raise ValueError("BEVFormerOccupancy requires pts_bbox_head for BEV encoder reuse")
        self.bev_h = int(self.pts_bbox_head.bev_h)
        self.bev_w = int(self.pts_bbox_head.bev_w)
        pc_range = [float(value) for value in self.pts_bbox_head.pc_range]
        self.occupancy_grid = BEVGridSpec(
            x_min=pc_range[0],
            y_min=pc_range[1],
            x_max=pc_range[3],
            y_max=pc_range[4],
            height=self.bev_h,
            width=self.bev_w,
        )

    def _bev_embed_to_feature_map(self, bev_embed):
        """Normalize BEV encoder output to ``[B, C, H, W]``.

        ``PerceptionTransformer.get_bev_features`` currently returns
        ``[B, H*W, C]``. The detection decoder path immediately permutes that
        tensor to ``[H*W, B, C]`` before cross-attention. This helper accepts
        either layout and documents the reshape used by the occupancy head:

        1. ensure the token dimension is ``H*W``;
        2. reshape tokens to ``[B, H, W, C]`` in row-major BEV order;
        3. permute channels first to ``[B, C, H, W]`` for Conv2d.
        """
        if bev_embed.ndim != 3:
            raise ValueError(f"expected raw BEV embed rank 3, got {tuple(bev_embed.shape)}")
        num_tokens = self.bev_h * self.bev_w
        if bev_embed.shape[1] == num_tokens:
            bev_tokens = bev_embed
            layout = "[B, H*W, C]"
        elif bev_embed.shape[0] == num_tokens:
            bev_tokens = bev_embed.permute(1, 0, 2).contiguous()
            layout = "[H*W, B, C]"
        else:
            raise ValueError(
                f"cannot identify BEV token dimension in shape {tuple(bev_embed.shape)}; expected H*W={num_tokens}")

        batch, tokens, channels = bev_tokens.shape
        if tokens != num_tokens:
            raise AssertionError(f"expected {num_tokens} BEV tokens, got {tokens}")
        bev_feats = bev_tokens.reshape(batch, self.bev_h, self.bev_w, channels).permute(0, 3, 1, 2).contiguous()
        return bev_feats, layout

    def _build_occupancy_targets(self, gt_bboxes_3d, gt_labels_3d, device, dtype):
        targets = []
        for boxes, labels in zip(gt_bboxes_3d, gt_labels_3d):
            if torch.is_tensor(labels):
                label_values = labels.detach().cpu()
            else:
                label_values = torch.as_tensor(labels, dtype=torch.long)
            mask, _, _ = rasterize_bev_occupancy(
                boxes,
                self.occupancy_grid,
                labels=label_values,
                class_names=list(self.class_names),
                selected_classes=self.occupancy_classes,
            )
            targets.append(torch.from_numpy(mask.astype(np.float32)))
        if not targets:
            raise ValueError("empty batch when building occupancy targets")
        target = torch.stack(targets, dim=0).unsqueeze(1).to(device=device, dtype=dtype)
        expected_shape = (len(gt_bboxes_3d), 1, self.bev_h, self.bev_w)
        if tuple(target.shape) != expected_shape:
            raise AssertionError(f"expected occupancy target shape {expected_shape}, got {tuple(target.shape)}")
        return target

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          return_occ_debug=False):
        #original BEVFormer head from the tiny config, just return B_{t}, no detection run
        raw_bev_embed = self.pts_bbox_head(pts_feats, img_metas, prev_bev, only_bev=True)
        # [B,HW,C] --> [B,H,W,C] --> [B,C,H,W]
        bev_feats, raw_layout = self._bev_embed_to_feature_map(raw_bev_embed)
        # shape: (B, 1, 50, 50)
        logits = self.occupancy_head(bev_feats)
        # shape: (B, 1, 50, 50)
        targets = self._build_occupancy_targets(gt_bboxes_3d, gt_labels_3d, logits.device, logits.dtype)
        losses = self.occupancy_head.loss(logits, targets)

        if not return_occ_debug:
            return losses
        debug = dict(
            raw_bev_embed_shape=tuple(raw_bev_embed.shape),
            raw_bev_embed_layout=raw_layout,
            bev_feature_shape=tuple(bev_feats.shape),
            occupancy_logits_shape=tuple(logits.shape),
            occupancy_target_shape=tuple(targets.shape),
            logits=logits.detach(),
            targets=targets.detach(),
        )
        return losses, debug

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      return_occ_debug=False):
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        was_training = self.training
        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        if not was_training:
            self.eval()

        current_img_metas = [each[len_queue - 1] for each in img_metas]
        if not current_img_metas[0]["prev_bev_exists"]:
            prev_bev = None

        img_feats = self.extract_feat(img=img, img_metas=current_img_metas)
        result = self.forward_pts_train(
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            current_img_metas,
            gt_bboxes_ignore,
            prev_bev,
            return_occ_debug=return_occ_debug,
        )
        return result
