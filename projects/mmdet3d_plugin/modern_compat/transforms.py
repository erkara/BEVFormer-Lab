"""Runtime transform shims for legacy BEVFormer pipelines."""

import mmcv
import numpy as np
import torch
from mmdet3d.registry import TRANSFORMS


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
    """Register legacy transform names expected by old BEVFormer configs."""

    @TRANSFORMS.register_module(name="LoadMultiViewImageFromFiles", force=True)
    class LoadMultiViewImageFromFilesCompat:
        def __init__(self, to_float32=False, color_type="unchanged", *args, **kwargs):
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
            results["scale_factor"] = 1.0
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
                results["img"] = make_data_container(
                    torch.stack([torch.from_numpy(img) for img in imgs]), stack=True
                )
            if "gt_labels_3d" in results and not torch.is_tensor(results["gt_labels_3d"]):
                results["gt_labels_3d"] = make_data_container(torch.as_tensor(results["gt_labels_3d"]))
            if "gt_bboxes_3d" in results:
                results["gt_bboxes_3d"] = make_data_container(results["gt_bboxes_3d"], cpu_only=True)
            return results
