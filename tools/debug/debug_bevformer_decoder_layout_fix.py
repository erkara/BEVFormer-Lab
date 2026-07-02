import copy
import logging
import sys
import traceback
from pathlib import Path
from types import MethodType

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


def install_cuda_extension_probe():
    state = {"hook_installed": False, "forward_called": False, "install_error": None}
    try:
        from projects.mmdet3d_plugin.bevformer.modules import multi_scale_deformable_attn_function as msda

        original_forward = msda.ext_module.ms_deform_attn_forward

        def wrapped_forward(*args, **kwargs):
            state["forward_called"] = True
            return original_forward(*args, **kwargs)

        msda.ext_module.ms_deform_attn_forward = wrapped_forward
        state["hook_installed"] = True
    except Exception as exc:
        state["install_error"] = f"{type(exc).__name__}: {exc}"
    return state


def build_model(device):
    cfg = Config.fromfile(CONFIG_PATH)
    apply_config_fixes(cfg)
    model = MMDET_MODELS.build(cfg.model)
    model.eval()
    model.to(device)
    return model


def find_decoder_cross_attn_modules(model):
    modules = []
    decoder = model.pts_bbox_head.transformer.decoder
    for layer_idx, layer in enumerate(decoder.layers):
        for attn_idx, attn in enumerate(layer.attentions):
            if isinstance(attn, CustomMSDeformableAttention):
                modules.append((layer_idx, attn_idx, attn))
    return modules


def print_shape(label, value):
    if torch.is_tensor(value):
        print(f"{label}: {tuple(value.shape)}")
    elif value is None:
        print(f"{label}: None")
    else:
        print(f"{label}: {type(value).__name__}")


def summarize_exception(exc):
    print(f"exception type: {type(exc).__name__}")
    print(f"exception: {exc}")
    print("--- traceback ---")
    traceback.print_exc()


def is_decoder_spatial_assertion(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    return any(
        frame.filename.endswith("projects/mmdet3d_plugin/bevformer/modules/decoder.py")
        and frame.lineno == 331
        for frame in tb
    )


def instrument_modules(modules, mode):
    original_forwards = []
    state = {"calls": 0, "assertion_passed": False, "last_shapes": None}

    for layer_idx, attn_idx, module in modules:
        original_forward = module.forward
        original_forwards.append((module, original_forward))

        def wrapped_forward(self, query, key=None, value=None, identity=None,
                            query_pos=None, key_padding_mask=None,
                            reference_points=None, spatial_shapes=None,
                            level_start_index=None, flag="decoder",
                            _layer_idx=layer_idx, _attn_idx=attn_idx,
                            **kwargs):
            state["calls"] += 1
            effective_value = query if value is None else value
            state["last_shapes"] = {
                "query": tuple(query.shape) if torch.is_tensor(query) else None,
                "key": tuple(key.shape) if torch.is_tensor(key) else None,
                "value": tuple(value.shape) if torch.is_tensor(value) else None,
                "effective_value": tuple(effective_value.shape) if torch.is_tensor(effective_value) else None,
                "identity": tuple(identity.shape) if torch.is_tensor(identity) else None,
                "query_pos": tuple(query_pos.shape) if torch.is_tensor(query_pos) else None,
                "reference_points": tuple(reference_points.shape) if torch.is_tensor(reference_points) else None,
                "spatial_shapes": spatial_shapes.detach().cpu().tolist() if torch.is_tensor(spatial_shapes) else None,
                "level_start_index": level_start_index.detach().cpu().tolist() if torch.is_tensor(level_start_index) else None,
                "batch_first": self.batch_first,
            }
            print(f"\n--- decoder cross-attention call {state['calls']} ---")
            print(f"module location: decoder.layers[{_layer_idx}].attentions[{_attn_idx}]")
            print(f"module batch_first: {self.batch_first}")
            for name in ("query", "key", "value", "identity", "query_pos", "reference_points"):
                print_shape(name, locals()[name])
            if torch.is_tensor(spatial_shapes):
                products = spatial_shapes[:, 0] * spatial_shapes[:, 1]
                print(f"spatial_shapes: {spatial_shapes.detach().cpu().tolist()}")
                print(f"spatial_shapes product sum: {int(products.sum().item())}")
            print(f"level_start_index: {level_start_index.detach().cpu().tolist() if torch.is_tensor(level_start_index) else None}")

            if mode == "transpose_for_batch_first":
                query_in = query.transpose(0, 1) if torch.is_tensor(query) else query
                key_in = key.transpose(0, 1) if torch.is_tensor(key) else key
                value_in = value.transpose(0, 1) if torch.is_tensor(value) else value
                identity_in = identity.transpose(0, 1) if torch.is_tensor(identity) else identity
                query_pos_in = query_pos.transpose(0, 1) if torch.is_tensor(query_pos) else query_pos
                print("transposed for batch_first=True:")
                print_shape("query_in", query_in)
                print_shape("value_in", value_in)
                out = original_forward(
                    query_in,
                    key=key_in,
                    value=value_in,
                    identity=identity_in,
                    query_pos=query_pos_in,
                    key_padding_mask=key_padding_mask,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    flag=flag,
                    **kwargs,
                )
                state["assertion_passed"] = True
                return out.transpose(0, 1) if torch.is_tensor(out) else out

            out = original_forward(
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
            state["assertion_passed"] = True
            return out

        module.forward = MethodType(wrapped_forward, module)

    return state, original_forwards


def restore_forwards(original_forwards):
    for module, original_forward in original_forwards:
        module.forward = original_forward


def run_experiment(name, patch_kind, device):
    print(f"\n================ {name} ================")
    extension_probe = install_cuda_extension_probe()
    if extension_probe["install_error"]:
        print(f"cuda extension hook install error: {extension_probe['install_error']}")

    model = build_model(device)
    modules = find_decoder_cross_attn_modules(model)
    print(f"decoder CustomMSDeformableAttention modules found: {len(modules)}")
    for layer_idx, attn_idx, module in modules:
        print(f"module decoder.layers[{layer_idx}].attentions[{attn_idx}] initial batch_first={module.batch_first}")

    if patch_kind == "batch_first_false":
        for _, _, module in modules:
            module.batch_first = False
        mode = "direct"
    elif patch_kind == "transpose_for_batch_first":
        mode = "transpose_for_batch_first"
    else:
        raise ValueError(patch_kind)

    probe_state, original_forwards = instrument_modules(modules, mode)
    img = torch.randn(1, NUM_CAMS, 3, IMAGE_H, IMAGE_W, device=device)
    img_metas = [make_meta(NUM_CAMS, IMAGE_H, IMAGE_W, prev_bev_exists=False)]
    result = {
        "image_passed": False,
        "bev_encoder_passed": False,
        "decoder_assertion_passed": False,
        "full_head_passed": False,
        "cuda_extension_invoked": False,
        "failure": None,
        "last_shapes": None,
    }

    try:
        with torch.no_grad():
            mlvl_feats = model.extract_feat(img=img.clone(), img_metas=copy.deepcopy(img_metas))
            result["image_passed"] = True
            print(f"image backbone/FPN: passed, mlvl_feats={tensor_shape(mlvl_feats)}")

            bev_only = model.pts_bbox_head(
                mlvl_feats,
                copy.deepcopy(img_metas),
                prev_bev=None,
                only_bev=True,
            )
            result["bev_encoder_passed"] = True
            print(f"BEV encoder only_bev: passed, output={tuple(bev_only.shape)}")

            outputs = model.pts_bbox_head(mlvl_feats, copy.deepcopy(img_metas), prev_bev=None)
            result["full_head_passed"] = True
            print(f"full pts_bbox_head: passed, keys={sorted(outputs.keys())}")
    except Exception as exc:
        result["failure"] = exc
        result["decoder_assertion_passed"] = probe_state["assertion_passed"] or not is_decoder_spatial_assertion(exc)
        print("failure caught")
        summarize_exception(exc)
    else:
        result["decoder_assertion_passed"] = True
    finally:
        result["cuda_extension_invoked"] = extension_probe["forward_called"]
        result["last_shapes"] = probe_state["last_shapes"]
        restore_forwards(original_forwards)

    print(f"\n{name} summary:")
    print(f"image backbone/FPN passed: {result['image_passed']}")
    print(f"BEV encoder passed: {result['bev_encoder_passed']}")
    print(f"decoder cross-attention assertion passed: {result['decoder_assertion_passed']}")
    print(f"full pts_bbox_head passed: {result['full_head_passed']}")
    print(f"CUDA extension path invoked: {result['cuda_extension_invoked']}")
    print(f"last cross-attention shapes: {result['last_shapes']}")
    return result


def main():
    logging.disable(logging.INFO)
    torch.manual_seed(0)
    device, device_reason = choose_device()
    print(f"config: {CONFIG_PATH}")
    print(f"selected device: {device} ({device_reason})")
    print(f"synthetic input: B=1, num_cams={NUM_CAMS}, C=3, H={IMAGE_H}, W={IMAGE_W}")

    result_a = run_experiment(
        "Experiment A: set decoder CustomMSDeformableAttention batch_first=False",
        "batch_first_false",
        device,
    )
    result_b = run_experiment(
        "Experiment B: transpose tensors for existing batch_first=True",
        "transpose_for_batch_first",
        device,
    )

    print("\n================ final recommendation ================")
    if result_a["full_head_passed"] and not result_b["full_head_passed"]:
        print("Better candidate: Experiment A.")
        print("Reason: it matches the sequence-first tensors already used by the decoder call path and reached full head output.")
    elif result_b["full_head_passed"] and not result_a["full_head_passed"]:
        print("Better candidate: Experiment B.")
        print("Reason: it preserves the module's batch_first=True setting and only adapts the local call layout.")
    elif result_a["full_head_passed"] and result_b["full_head_passed"]:
        print("Better candidate: Experiment A.")
        print("Reason: both experiments reached full head output, but A is the smaller compatibility patch because the surrounding decoder path already passes sequence-first tensors.")
    else:
        print("Better candidate: inconclusive.")
        print("Reason: neither runtime-only experiment reached full head output; inspect the first post-assertion failure before choosing a patch.")

    return 0 if (result_a["decoder_assertion_passed"] or result_b["decoder_assertion_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
