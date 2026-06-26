
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
# from mmcv.parallel import DataContainer as DC
try:
    from mmcv.parallel import DataContainer as DC
except ModuleNotFoundError:
    class DC:
        def __init__(self, data, *args, **kwargs):
            self.data = data
            self.stack = kwargs.get("stack", False)
            self.padding_value = kwargs.get("padding_value", 0)
            self.cpu_only = kwargs.get("cpu_only", False)


# from mmdet3d.core.bbox import BaseInstance3DBoxes
# from mmdet3d.core.points import BasePoints
# from mmdet.datasets.builder import PIPELINES
# from mmdet.datasets.pipelines import to_tensor
# from mmdet3d.datasets.pipelines import DefaultFormatBundle3D
import torch

from mmdet3d.structures import BaseInstance3DBoxes, BasePoints
from mmdet3d.registry import TRANSFORMS as PIPELINES

try:
    from mmdet.datasets.transforms import PackDetInputs  # not directly equivalent, keep as reference
except Exception:
    PackDetInputs = None

try:
    from mmdet3d.datasets.transforms import Pack3DDetInputs  # modern formatting class, not directly equivalent
except Exception:
    Pack3DDetInputs = None


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    return torch.tensor(data)


class DefaultFormatBundle3D:
    def __call__(self, results):
        return results

@PIPELINES.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        results['gt_map_masks'] = DC(
            to_tensor(results['gt_map_masks']), stack=True)

        return results