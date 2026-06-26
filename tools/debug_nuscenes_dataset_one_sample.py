import copy
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmengine.config import Config
from mmengine.fileio import load
from mmengine.registry import build_from_cfg, init_default_scope

import mmdet.models
import mmdet3d.models
import projects.mmdet3d_plugin

from mmdet3d.registry import DATASETS, TRANSFORMS
import mmcv


CONFIG_PATH = "projects/configs/bevformer/bevformer_tiny.py"
DATA_ROOT = "data/nuscenes/"
VAL_INFO = "data/nuscenes/nuscenes_infos_temporal_val.pkl"
META_FIELDS = (
    "filename",
    "lidar2img",
    "can_bus",
    "scene_token",
    "sample_idx",
    "prev_bev_exists",
    "box_type_3d",
)


def print_header(name):
    print(f"\n=== {name} ===")


def apply_config_fixes(cfg):
    if "model" in cfg:
        cfg.model.pop("pretrained", None)
        if "img_backbone" in cfg.model:
            cfg.model.img_backbone.type = "mmdet.ResNet"
        if "img_neck" in cfg.model:
            cfg.model.img_neck.type = "mmdet.FPN"
        if "pts_bbox_head" in cfg.model:
            cfg.model.pts_bbox_head.type = "mmdet.BEVFormerHead"


class SimpleDataContainer:
    def __init__(self, data, stack=False, padding_value=0, cpu_only=False):
        self.data = data
        self._data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only


def make_data_container(data, **kwargs):
    try:
        from mmcv.parallel import DataContainer as DC
        return DC(data, **kwargs)
    except Exception:
        return SimpleDataContainer(data, **kwargs)


def register_runtime_compat_transforms():
    @TRANSFORMS.register_module(name="LoadMultiViewImageFromFiles", force=True)
    class LoadMultiViewImageFromFilesCompat:
        def __init__(self, to_float32=False, color_type="color", *args, **kwargs):
            self.to_float32 = to_float32
            self.color_type = color_type

        def __call__(self, results):
            filenames = results.get("img_filename") or results.get("filename")
            if filenames is None:
                raise KeyError("LoadMultiViewImageFromFilesCompat expected img_filename or filename")
            imgs = [mmcv.imread(name, self.color_type) for name in filenames]
            if self.to_float32:
                imgs = [img.astype(np.float32) for img in imgs]
            results["filename"] = filenames
            results["img"] = imgs
            results["img_shape"] = [img.shape for img in imgs]
            results["ori_shape"] = [img.shape for img in imgs]
            results["pad_shape"] = [img.shape for img in imgs]
            num_channels = 1 if imgs[0].ndim < 3 else imgs[0].shape[2]
            results["img_norm_cfg"] = dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False,
            )
            return results

        def __repr__(self):
            return f"LoadMultiViewImageFromFilesCompat(to_float32={self.to_float32})"

    @TRANSFORMS.register_module(name="DefaultFormatBundle3D", force=True)
    class DefaultFormatBundle3DCompat:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, results):
            if "img" in results and isinstance(results["img"], list):
                imgs = [np.ascontiguousarray(img.transpose(2, 0, 1)) for img in results["img"]]
                results["img"] = make_data_container(torch.stack([torch.from_numpy(img) for img in imgs]), stack=True)
            if "gt_labels_3d" in results and not torch.is_tensor(results["gt_labels_3d"]):
                results["gt_labels_3d"] = make_data_container(torch.as_tensor(results["gt_labels_3d"]))
            if "gt_bboxes_3d" in results:
                results["gt_bboxes_3d"] = make_data_container(results["gt_bboxes_3d"], cpu_only=True)
            return results

        def __repr__(self):
            return "DefaultFormatBundle3DCompat()"

    print("registered runtime transform alias: LoadMultiViewImageFromFiles -> legacy compat")
    print("registered runtime transform alias: DefaultFormatBundle3D -> legacy compat")


def configure_val_dataset(cfg):
    cfg.data.val.data_root = DATA_ROOT
    cfg.data.val.ann_file = VAL_INFO
    cfg.data.val.test_mode = True

    val_cfg = copy.deepcopy(cfg.data.val)
    val_cfg.ann_file = str(Path(VAL_INFO).resolve())
    val_cfg.pop("samples_per_gpu", None)
    val_cfg.pop("classes", None)
    val_cfg.lazy_init = True
    return val_cfg


def attach_legacy_infos(dataset, ann_file):
    annotations = load(ann_file)
    infos = annotations.get("infos", annotations if isinstance(annotations, list) else None)
    if infos is None:
        raise KeyError("legacy annotation file does not contain infos")
    dataset.data_infos = infos
    dataset.data_list = infos
    dataset._fully_initialized = True
    dataset.serialize_data = False
    if not hasattr(dataset, "show_ins_var"):
        dataset.show_ins_var = False
    if not hasattr(dataset, "prepare_test_data"):
        dataset.prepare_test_data = dataset.prepare_data
    metadata = annotations.get("metadata", {}) if isinstance(annotations, dict) else {}
    if "version" in metadata:
        dataset.version = metadata["version"]
    return metadata


def build_dataset(val_cfg):
    try:
        return DATASETS.build(val_cfg)
    except Exception as exc:
        print("DATASETS.build failed; falling back to build_from_cfg")
        print(f"fallback reason: {type(exc).__name__}: {exc}")
        return build_from_cfg(val_cfg, DATASETS)


def is_datacontainer(value):
    return hasattr(value, "data") and (
        hasattr(value, "cpu_only") or hasattr(value, "stack") or hasattr(value, "padding_value")
    )


def summarize(value, depth=0, max_depth=4):
    prefix = "  " * depth
    if torch.is_tensor(value):
        return f"Tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
    if isinstance(value, np.ndarray):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    if is_datacontainer(value):
        attrs = []
        for attr in ("stack", "cpu_only", "padding_value", "pad_dims"):
            if hasattr(value, attr):
                attrs.append(f"{attr}={getattr(value, attr)}")
        inner = getattr(value, "data")
        if hasattr(value, "_data"):
            inner = getattr(value, "_data")
        inner_summary = summarize(inner, depth + 1, max_depth)
        return f"DataContainer({', '.join(attrs)}) -> {inner_summary}"
    if hasattr(value, "tensor") and torch.is_tensor(value.tensor):
        return f"{type(value).__name__} tensor_shape={tuple(value.tensor.shape)} dtype={value.tensor.dtype}"
    if isinstance(value, dict):
        keys = list(value.keys())
        if depth >= max_depth:
            return f"dict keys={keys}"
        lines = [f"dict keys={keys}"]
        for key in keys:
            lines.append(f"{prefix}  {key}: {summarize(value[key], depth + 1, max_depth)}")
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return f"{type(value).__name__} len=0"
        if depth >= max_depth:
            return f"{type(value).__name__} len={len(value)} first_type={type(value[0]).__name__}"
        sample_count = min(len(value), 3)
        lines = [f"{type(value).__name__} len={len(value)}"]
        for idx in range(sample_count):
            lines.append(f"{prefix}  [{idx}]: {summarize(value[idx], depth + 1, max_depth)}")
        if len(value) > sample_count:
            lines.append(f"{prefix}  ... {len(value) - sample_count} more")
        return "\n".join(lines)
    if isinstance(value, (str, int, float, bool, type(None))):
        return repr(value)
    return type(value).__name__


def unwrap_datacontainer(value):
    if is_datacontainer(value):
        if hasattr(value, "_data"):
            return value._data
        return value.data
    return value


def print_value_summary(key, value):
    print(f"\n--- {key} ---")
    print(summarize(value))


def print_specific(item, key):
    if isinstance(item, dict) and key in item:
        print_value_summary(key, item[key])
    else:
        print(f"\n--- {key} ---")
        print("not present")


def get_primary_item(item):
    if isinstance(item, list) and len(item) == 1 and isinstance(item[0], dict):
        return item[0], "unwrapped one-element MultiScaleFlipAug3D list"
    return item, "raw item"


def get_img_metas(item):
    if not isinstance(item, dict) or "img_metas" not in item:
        return None
    return unwrap_datacontainer(item["img_metas"])


def print_img_meta_fields(img_metas):
    print_header("img_metas selected fields")
    if img_metas is None:
        print("img_metas not present")
        return
    if isinstance(img_metas, dict):
        metas = img_metas
    elif isinstance(img_metas, (list, tuple)) and img_metas and isinstance(img_metas[0], dict):
        metas = img_metas[0]
    else:
        print(f"img_metas payload is not a dict: {type(img_metas).__name__}")
        print(summarize(img_metas))
        return
    for field in META_FIELDS:
        print_value_summary(f"img_metas.{field}", metas.get(field, "<missing>"))


def main():
    logging.disable(logging.INFO)
    print_header("load config")
    print(f"config: {CONFIG_PATH}")
    cfg = Config.fromfile(CONFIG_PATH)
    apply_config_fixes(cfg)
    print("modern config fixes applied")

    print_header("register compatibility")
    init_default_scope("mmdet3d")
    print("default scope set: mmdet3d")
    register_runtime_compat_transforms()

    print_header("configure validation dataset")
    val_cfg = configure_val_dataset(cfg)
    print(f"requested cfg.data.val.data_root: {cfg.data.val.data_root}")
    print(f"requested cfg.data.val.ann_file: {cfg.data.val.ann_file}")
    print(f"requested cfg.data.val.test_mode: {cfg.data.val.test_mode}")
    print(f"dataset cfg type: {val_cfg.type}")
    print(f"build data_root: {val_cfg.data_root}")
    print(f"build ann_file: {val_cfg.ann_file}")
    print(f"build test_mode: {val_cfg.test_mode}")
    print(f"build lazy_init: {val_cfg.lazy_init}")

    print_header("build validation dataset")
    dataset = build_dataset(val_cfg)
    metadata = attach_legacy_infos(dataset, val_cfg.ann_file)
    print(f"attached legacy metadata: {metadata}")
    print(f"dataset type: {type(dataset)}")
    print(f"dataset length: {len(dataset)}")

    print_header("load one item")
    item = dataset[0]
    primary_item, primary_reason = get_primary_item(item)
    print(f"first item type: {type(item)}")
    print(f"effective item selection: {primary_reason}")
    print(f"effective item type: {type(primary_item)}")
    if isinstance(primary_item, dict):
        print(f"top-level keys: {list(primary_item.keys())}")
    else:
        print("top-level keys: <not a dict>")

    print_header("raw item summary")
    print(summarize(item))

    print_header("effective top-level value summaries")
    if isinstance(primary_item, dict):
        for key, value in primary_item.items():
            print_value_summary(key, value)
    else:
        print(summarize(primary_item))

    print_header("specific fields")
    for key in ("img", "img_metas", "gt_bboxes_3d", "gt_labels_3d"):
        print_specific(primary_item, key)

    print_img_meta_fields(get_img_metas(primary_item))
    print_header("summary")
    print("dataset one-sample probe completed")
    print("no checkpoint loaded")
    print("no model.forward run")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print("\n=== exception ===")
        print(f"exception type: {type(exc).__name__}")
        print(f"exception: {exc}")
        print("\n--- traceback ---")
        traceback.print_exc()
        raise SystemExit(1)
