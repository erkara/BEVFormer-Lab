import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class BEVOccupancyHead(nn.Module):
    """Map [B, C, H, W] BEV features to [B, 1, H, W] binary logits.

    ``pos_weight`` increases the BCE penalty for missed occupied cells because
    vehicle footprints cover only a small fraction of the BEV grid.
    """

    def __init__(self, in_channels=256, hidden_channels=None, pos_weight=10.0):
        super().__init__()
        hidden_channels = hidden_channels or max(in_channels // 2, 1)
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.pos_weight_value = float(pos_weight)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1),
        )
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(self, bev_feats):
        if bev_feats.ndim != 4:
            raise ValueError(f"expected BEV features [B, C, H, W], got {tuple(bev_feats.shape)}")
        if bev_feats.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {bev_feats.shape[1]}")
        return self.net(bev_feats)

    def loss(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(
                f"occupancy logits/target shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"expected logits [B, 1, H, W], got {tuple(logits.shape)}")
        target_min = float(targets.detach().min().item()) if targets.numel() else 0.0
        target_max = float(targets.detach().max().item()) if targets.numel() else 0.0
        if target_min < 0.0 or target_max > 1.0:
            raise ValueError(
                f"occupancy targets must be binary/float in [0, 1], got range [{target_min}, {target_max}]")
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weight.to(device=logits.device, dtype=logits.dtype),
        )
        return {"loss_occ": loss}
