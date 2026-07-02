import copy
import logging
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmengine.config import Config

import mmdet.models
import mmdet3d.models
import projects.mmdet3d_plugin

from mmdet.registry import MODELS as MMDET_MODELS
from projects.mmdet3d_plugin.bevformer.modules.decoder import CustomMSDeformableAttention
from tools.debug.debug_bevformer_dummy_forward import (
    CONFIG_PATH,
    IMAGE_H,
    IMAGE_W,
    NUM_CAMS,
    apply_config_fixes,
    choose_device,
    make_meta,
    tensor_shape,
)


def shape(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    return None if value is None else type(value).__name__


def print_shape(name, value):
    print(f"{name}: {shape(value)}")


def print_tensor(name, value):
    if value is None:
        print(f"{name}: None")
        return
    if torch.is_tensor(value):
        print(f"{name}: shape={tuple(value.shape)}, value={value.detach().cpu().tolist()}")
        return
    print(f"{name}: {value}")


def install_decoder_attention_probe():
    original_forward = CustomMSDeformableAttention.forward
    state = {"calls": 0}

    def wrapped_forward(self, query, key=None, value=None, identity=None,
                        query_pos=None, key_padding_mask=None,
                        reference_points=None, spatial_shapes=None,
                        level_start_index=None, flag="decoder", **kwargs):
        state["calls"] += 1
        print(f"\n=== CustomMSDeformableAttention.forward call {state['calls']} ===")
        print(f"module batch_first: {self.batch_first}")
        print(f"module num_levels: {self.num_levels}")
        print_shape("query", query)
        print_shape("key", key)
        print_shape("value", value)
        print_shape("identity", identity)
        print_shape("query_pos", query_pos)
        print_shape("reference_points", reference_points)
        print_tensor("spatial_shapes", spatial_shapes)
        print_tensor("level_start_index", level_start_index)

        effective_value = query if value is None else value
        if not self.batch_first and torch.is_tensor(effective_value):
            permuted_value = effective_value.permute(1, 0, 2)
            num_value = permuted_value.shape[1]
            print(f"value after batch_first normalization: {tuple(permuted_value.shape)}")
        elif torch.is_tensor(effective_value):
            num_value = effective_value.shape[1]
        else:
            num_value = None

        if torch.is_tensor(spatial_shapes):
            spatial_products = spatial_shapes[:, 0] * spatial_shapes[:, 1]
            spatial_sum = spatial_products.sum()
            print(f"spatial_shapes products: {spatial_products.detach().cpu().tolist()}")
            print(f"spatial_shapes product sum: {int(spatial_sum.item())}")
        else:
            spatial_sum = None
            print("spatial_shapes products: unavailable")

        print(f"num_value used by assertion: {num_value}")
        if spatial_sum is not None and num_value is not None:
            print(f"assertion comparison: {int(spatial_sum.item())} == {num_value}")

        return original_forward(
            self,
            query,
            key=key,
            value=value,
            identity=identity,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            flag=flag,
            **kwargs,
        )

    CustomMSDeformableAttention.forward = wrapped_forward
    return state


def main():
    logging.disable(logging.INFO)
    torch.manual_seed(0)

    cfg = Config.fromfile(CONFIG_PATH)
    apply_config_fixes(cfg)
    queue_length = int(getattr(cfg, "queue_length", 1))

    device, device_reason = choose_device()
    print(f"config: {CONFIG_PATH}")
    print(f"selected device: {device} ({device_reason})")
    print(f"queue_length from config: {queue_length}")
    print(f"synthetic input: B=1, num_cams={NUM_CAMS}, C=3, H={IMAGE_H}, W={IMAGE_W}")

    probe_state = install_decoder_attention_probe()

    model = MMDET_MODELS.build(cfg.model)
    model.eval()
    model.to(device)

    img = torch.randn(1, NUM_CAMS, 3, IMAGE_H, IMAGE_W, device=device)
    img_metas = [make_meta(NUM_CAMS, IMAGE_H, IMAGE_W, prev_bev_exists=False)]

    print("\n=== input shapes ===")
    print(f"img: {tuple(img.shape)}")
    print(f"img_metas keys: {sorted(img_metas[0].keys())}")
    print(f"img_shape meta: {img_metas[0]['img_shape']}")
    print(f"lidar2img length: {len(img_metas[0]['lidar2img'])}")

    with torch.no_grad():
        mlvl_feats = model.extract_feat(img=img.clone(), img_metas=copy.deepcopy(img_metas))
        print("\n=== mlvl_feats before PerceptionTransformer flattening ===")
        print(f"num feature levels: {len(mlvl_feats)}")
        print(f"mlvl_feats: {tensor_shape(mlvl_feats)}")

        bev_only = model.pts_bbox_head(
            mlvl_feats,
            copy.deepcopy(img_metas),
            prev_bev=None,
            only_bev=True,
        )
        print("\n=== BEV encoder only_bev result ===")
        print(f"bev_only: {tuple(bev_only.shape)}")

        print("\n=== full pts_bbox_head probe ===")
        try:
            model.pts_bbox_head(mlvl_feats, copy.deepcopy(img_metas), prev_bev=None)
        except Exception as exc:
            print("\n=== caught exception ===")
            print(f"exception type: {type(exc).__name__}")
            print(f"exception: {exc}")
            print("\n--- traceback ---")
            traceback.print_exc()
            print("\n=== diagnosis ===")
            print(f"decoder attention probe calls: {probe_state['calls']}")
            print(
                "The assertion fails because decoder cross-attention has "
                "batch_first=True, so it interprets the incoming value as "
                "(B, num_value, C). The value is actually sequence-first, "
                "shaped (BEV_HW, B, C), so num_value is read from dimension 1 "
                "as the batch size instead of BEV_H * BEV_W."
            )
            print(
                "Minimal next fix to test: adjust the decoder cross-attention "
                "layout contract, likely by setting the decoder "
                "CustomMSDeformableAttention to batch_first=False or otherwise "
                "passing decoder value/query tensors in the layout that its "
                "batch_first=True path expects. This looks more like a "
                "modern-port layout/config issue than a missing nuScenes sample."
            )
            return 1

    print("\nfull pts_bbox_head unexpectedly passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
