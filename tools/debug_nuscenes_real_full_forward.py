import copy
import logging
import sys
import traceback
from pathlib import Path

import mmcv
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmengine.config import Config
from mmengine.fileio import load
from mmengine.registry import build_from_cfg, init_default_scope

import mmdet.models
import mmdet3d.models
import projects.mmdet3d_plugin

from mmdet.registry import MODELS as MMDET_MODELS
from mmdet3d.registry import DATASETS, TRANSFORMS


CONFIG_PATH = "projects/configs/bevformer/bevformer_tiny.py"
DATA_ROOT = "data/nuscenes/"
VAL_INFO = "data/nuscenes/nuscenes_infos_temporal_val.pkl"


def print_header(name):
    print(f"\n=== {name} ===")


def apply_config_fixes(cfg):
    cfg.model.pop("pretrained", None)
    cfg.model.img_backbone.type = "mmdet.ResNet"
    cfg.model.img_neck.type = "mmdet.FPN"
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


def is_datacontainer(value):
    return hasattr(value, "data") and (
        hasattr(value, "cpu_only") or hasattr(value, "stack") or hasattr(value, "padding_value")
    )


def unwrap_datacontainer(value):
    if is_datacontainer(value):
        if hasattr(value, "_data"):
            return value._data
        return value.data
    return value


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


def get_primary_item(item):
    if isinstance(item, list) and len(item) == 1 and isinstance(item[0], dict):
        return item[0], "unwrapped one-element MultiScaleFlipAug3D list"
    return item, "raw item"


def summarize(value):
    if torch.is_tensor(value):
        return f"Tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
    if isinstance(value, np.ndarray):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    if is_datacontainer(value):
        return f"DataContainer(stack={getattr(value, 'stack', None)}, cpu_only={getattr(value, 'cpu_only', None)}) -> {summarize(unwrap_datacontainer(value))}"
    if hasattr(value, "tensor") and torch.is_tensor(value.tensor):
        return f"{type(value).__name__} tensor_shape={tuple(value.tensor.shape)} dtype={value.tensor.dtype}"
    if isinstance(value, dict):
        return f"dict keys={list(value.keys())}"
    if isinstance(value, (list, tuple)):
        first = summarize(value[0]) if value else "<empty>"
        return f"{type(value).__name__} len={len(value)} first={first}"
    return type(value).__name__


def tensor_shape(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    if isinstance(value, np.ndarray):
        return tuple(value.shape)
    if is_datacontainer(value):
        return tensor_shape(unwrap_datacontainer(value))
    if hasattr(value, "tensor") and torch.is_tensor(value.tensor):
        return tuple(value.tensor.shape)
    if isinstance(value, dict):
        return {key: tensor_shape(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_shape(item) for item in value]
    return type(value).__name__


def print_shapes(label, value):
    print(f"{label}: {tensor_shape(value)}")


def choose_device():
    if not torch.cuda.is_available():
        return torch.device("cpu"), "torch.cuda.is_available() is false"
    try:
        torch.empty(1, device="cuda")
        return torch.device("cuda"), "CUDA is available and a test allocation succeeded"
    except Exception as exc:
        return torch.device("cpu"), f"CUDA test allocation failed: {type(exc).__name__}: {exc}"



def nesting_summary(value):
    if torch.is_tensor(value):
        return f"Tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
    if isinstance(value, np.ndarray):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    if isinstance(value, dict):
        return f"dict keys={list(value.keys())}"
    if isinstance(value, (list, tuple)):
        inner = nesting_summary(value[0]) if value else "<empty>"
        return f"{type(value).__name__} len={len(value)} first=({inner})"
    return type(value).__name__


def summarize_recursive(value, depth=0, max_depth=6):
    prefix = "  " * depth
    if torch.is_tensor(value):
        return f"Tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
    if isinstance(value, np.ndarray):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    if is_datacontainer(value):
        return f"DataContainer(stack={getattr(value, 'stack', None)}, cpu_only={getattr(value, 'cpu_only', None)}) -> {summarize_recursive(unwrap_datacontainer(value), depth + 1, max_depth)}"
    if hasattr(value, "tensor") and torch.is_tensor(value.tensor):
        return f"{type(value).__name__} tensor_shape={tuple(value.tensor.shape)} dtype={value.tensor.dtype}"
    if isinstance(value, dict):
        keys = list(value.keys())
        if depth >= max_depth:
            return f"dict keys={keys}"
        lines = [f"dict keys={keys}"]
        for key in keys:
            lines.append(f"{prefix}  {key}: {summarize_recursive(value[key], depth + 1, max_depth)}")
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        if not value:
            return f"{type(value).__name__} len=0"
        if depth >= max_depth:
            return f"{type(value).__name__} len={len(value)} first_type={type(value[0]).__name__}"
        sample_count = min(len(value), 3)
        lines = [f"{type(value).__name__} len={len(value)}"]
        for idx in range(sample_count):
            lines.append(f"{prefix}  [{idx}]: {summarize_recursive(value[idx], depth + 1, max_depth)}")
        if len(value) > sample_count:
            lines.append(f"{prefix}  ... {len(value) - sample_count} more")
        return "\n".join(lines)
    if isinstance(value, (str, int, float, bool, type(None))):
        return repr(value)
    return type(value).__name__


def print_dataset_structure(raw_item, item):
    print_header("raw dataset item structure")
    print(f"raw item type: {type(raw_item)}")
    print(f"raw item is list/tuple: {isinstance(raw_item, (list, tuple))}")
    if isinstance(raw_item, (list, tuple)):
        print(f"raw item length: {len(raw_item)}")
    print(f"effective dict keys: {list(item.keys()) if isinstance(item, dict) else '<not dict>'}")
    if isinstance(item, dict):
        print(f"img summary: {summarize_recursive(item.get('img'))}")
        print(f"img_metas summary: {summarize_recursive(item.get('img_metas'))}")


def print_real_metadata(meta):
    print_header("real metadata")
    filenames = meta.get("filename")
    lidar2img = meta.get("lidar2img")
    can_bus = meta.get("can_bus")
    print(f"filename length: {len(filenames) if filenames is not None else '<missing>'}")
    print(f"first filename: {filenames[0] if filenames else '<missing>'}")
    print(f"lidar2img length: {len(lidar2img) if lidar2img is not None else '<missing>'}")
    first_lidar2img_shape = getattr(lidar2img[0], "shape", None) if lidar2img else None
    print(f"first lidar2img shape: {first_lidar2img_shape}")
    print(f"can_bus shape: {getattr(can_bus, 'shape', '<missing>')}")
    print(f"scene_token: {meta.get('scene_token', '<missing>')}")
    print(f"sample_idx: {meta.get('sample_idx', '<missing>')}")


def prepare_real_inputs(cfg):
    print_header("register dataset compatibility")
    init_default_scope("mmdet3d")
    print("default scope set: mmdet3d")
    register_runtime_compat_transforms()

    print_header("build validation dataset")
    val_cfg = configure_val_dataset(cfg)
    print(f"requested cfg.data.val.data_root: {cfg.data.val.data_root}")
    print(f"requested cfg.data.val.ann_file: {cfg.data.val.ann_file}")
    print(f"requested cfg.data.val.test_mode: {cfg.data.val.test_mode}")
    print(f"build ann_file: {val_cfg.ann_file}")
    dataset = build_dataset(val_cfg)
    metadata = attach_legacy_infos(dataset, val_cfg.ann_file)
    print(f"attached legacy metadata: {metadata}")
    print(f"dataset type: {type(dataset)}")
    print(f"dataset length: {len(dataset)}")

    print_header("load real validation item")
    raw_item = dataset[0]
    item, item_reason = get_primary_item(raw_item)
    print(f"effective item selection: {item_reason}")
    print_dataset_structure(raw_item, item)

    img = unwrap_datacontainer(item["img"])
    img_metas = unwrap_datacontainer(item["img_metas"])
    if not torch.is_tensor(img):
        raise TypeError(f"expected img tensor after unwrap, got {type(img)}")
    if not isinstance(img_metas, dict):
        raise TypeError(f"expected img_metas dict after unwrap, got {type(img_metas)}")

    print(f"unwrapped img shape: {tuple(img.shape)}")
    print_real_metadata(img_metas)
    return raw_item, item, img, img_metas


def make_attempts(img_tensor, metadata_dict, device, raw_item):
    batched = img_tensor.unsqueeze(0).to(device)
    return [
        (
            "Attempt A: tensor BxNxCxHxW, img_metas=[[dict]]",
            lambda: (batched, [[copy.deepcopy(metadata_dict)]]),
        ),
        (
            "Attempt B: [tensor BxNxCxHxW], img_metas=[[dict]]",
            lambda: ([batched], [[copy.deepcopy(metadata_dict)]]),
        ),
        (
            "Attempt C: tensor BxT?xNxCxHxW, img_metas=[[dict]]",
            lambda: (img_tensor.unsqueeze(0).unsqueeze(0).to(device), [[copy.deepcopy(metadata_dict)]]),
        ),
        (
            "Attempt D: preserve one-aug convention with [tensor BxNxCxHxW], img_metas=[[dict]]",
            lambda: ([batched], [[copy.deepcopy(metadata_dict)]]),
        ),
    ]


def run_forward_attempt(model, name, make_inputs):
    print_header(name)
    try:
        img_arg, img_metas_arg = make_inputs()
        print(f"input img: {nesting_summary(img_arg)}")
        print(f"input img_metas: {nesting_summary(img_metas_arg)}")
        with torch.no_grad():
            output = model(return_loss=False, img=img_arg, img_metas=img_metas_arg)
    except Exception as exc:
        print("attempt failed")
        print(f"exception type: {type(exc).__name__}")
        print(f"exception: {exc}")
        print("\n--- traceback ---")
        traceback.print_exc()
        return None, exc
    print("full forward succeeded")
    print(f"output type: {type(output)}")
    print("output summary:")
    print(summarize_recursive(output))
    return output, None


def main():
    logging.disable(logging.INFO)
    torch.manual_seed(0)

    print_header("load config")
    print(f"config: {CONFIG_PATH}")
    cfg = Config.fromfile(CONFIG_PATH)
    apply_config_fixes(cfg)
    print("modern config fixes applied")

    raw_item, item, img_tensor, metadata_dict = prepare_real_inputs(cfg)

    print_header("build model")
    init_default_scope("mmdet")
    print("default scope set: mmdet")
    device, device_reason = choose_device()
    print(f"selected device: {device} ({device_reason})")
    model = MMDET_MODELS.build(cfg.model)
    model.eval()
    model.to(device)

    first_error = None
    first_error_name = None
    for name, make_inputs in make_attempts(img_tensor, metadata_dict, device, raw_item):
        output, error = run_forward_attempt(model, name, make_inputs)
        if error is None:
            print_header("summary")
            print("full forward succeeded")
            print("no checkpoint loaded")
            print("no training run")
            print("no evaluation run")
            return 0
        if first_error is None:
            first_error = error
            first_error_name = name

    print_header("summary")
    print("all full forward attempts failed")
    if first_error is not None:
        print(f"first meaningful blocker stage: {first_error_name}")
        print(f"first meaningful blocker: {type(first_error).__name__}: {first_error}")
    print("no model code modified")
    print("no checkpoint loaded")
    print("no training run")
    print("no evaluation run")
    return 1


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
