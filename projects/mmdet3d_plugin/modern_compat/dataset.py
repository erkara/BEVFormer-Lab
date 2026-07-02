"""Dataset shims for legacy BEVFormer annotations and pipelines."""

import copy
import types
from pathlib import Path

import numpy as np
import torch
from mmengine.fileio import load
from mmengine.registry import build_from_cfg
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes, get_box_type

from .config import DATA_ROOT, TRAIN_INFO, VAL_INFO
from .transforms import unwrap_datacontainer


def configure_train_dataset(cfg):
    train_cfg = copy.deepcopy(cfg.data.train)
    train_cfg.data_root = DATA_ROOT
    train_cfg.ann_file = str(Path(TRAIN_INFO).resolve())
    train_cfg.test_mode = False
    train_cfg.lazy_init = True
    train_cfg.pop("samples_per_gpu", None)
    train_cfg.pop("classes", None)
    return train_cfg


def configure_test_dataset(cfg):
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg.data_root = DATA_ROOT
    test_cfg.ann_file = str(Path(VAL_INFO).resolve())
    test_cfg.test_mode = True
    test_cfg.lazy_init = True
    test_cfg.pop("samples_per_gpu", None)
    test_cfg.pop("classes", None)
    return test_cfg


def build_train_dataset(cfg):
    """Build the legacy train dataset with modern runtime compatibility applied."""
    train_cfg = configure_train_dataset(cfg)
    dataset = build_dataset(train_cfg)
    metadata = attach_legacy_infos(dataset, train_cfg.ann_file, list(cfg.class_names))
    return dataset, metadata


def build_dataset(dataset_cfg):
    try:
        return DATASETS.build(dataset_cfg)
    except Exception as exc:
        print("DATASETS.build failed; falling back to build_from_cfg")
        print(f"fallback reason: {type(exc).__name__}: {exc}")
        return build_from_cfg(dataset_cfg, DATASETS)


def make_legacy_get_ann_info(class_names):
    def get_ann_info(self, index):
        info = self.data_infos[index]
        gt_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32)
        gt_names = np.asarray(info.get("gt_names", []))
        if gt_boxes.ndim == 1:
            gt_boxes = gt_boxes.reshape(0, 7)

        valid_flag = info.get("valid_flag")
        if valid_flag is not None and getattr(self, "use_valid_flag", False):
            valid_flag = np.asarray(valid_flag).astype(bool)
            gt_boxes = gt_boxes[valid_flag]
            gt_names = gt_names[valid_flag]

        class_to_label = {name: idx for idx, name in enumerate(class_names)}
        gt_labels = np.array([class_to_label.get(str(name), -1) for name in gt_names], dtype=np.int64)
        keep = gt_labels >= 0
        gt_boxes = gt_boxes[keep]
        gt_labels = gt_labels[keep]
        gt_names = gt_names[keep]

        if gt_boxes.shape[-1] < 9 and gt_boxes.size:
            velocity = np.zeros((gt_boxes.shape[0], 2), dtype=np.float32)
            gt_boxes = np.concatenate([gt_boxes, velocity], axis=-1)
        elif gt_boxes.size == 0:
            gt_boxes = gt_boxes.reshape(0, 9)

        boxes_3d = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))
        return dict(
            gt_bboxes_3d=boxes_3d,
            gt_labels_3d=torch.as_tensor(gt_labels, dtype=torch.long),
            gt_names=[str(name) for name in gt_names],
        )

    return get_ann_info


def make_legacy_pre_pipeline():
    def pre_pipeline(self, results):
        if not hasattr(self, "box_type_3d") or not hasattr(self, "box_mode_3d"):
            self.box_type_3d, self.box_mode_3d = get_box_type("LiDAR")
        results["img_prefix"] = ""
        results["seg_prefix"] = getattr(self, "seg_prefix", None)
        results["proposal_file"] = getattr(self, "proposal_file", None)
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    return pre_pipeline


def attach_legacy_infos(dataset, ann_file, class_names=None):
    """Attach legacy nuScenes infos fields expected by BEVFormer dataset code."""
    annotations = load(ann_file)
    infos = annotations.get("infos", annotations if isinstance(annotations, list) else None)
    if infos is None:
        raise KeyError("legacy annotation file does not contain infos")
    dataset.data_infos = infos
    dataset.data_list = infos
    dataset._fully_initialized = True
    dataset.serialize_data = False
    dataset.filter_empty_gt = False
    if not hasattr(dataset, "show_ins_var"):
        dataset.show_ins_var = False
    if not hasattr(dataset, "prepare_test_data"):
        dataset.prepare_test_data = dataset.prepare_data
    metadata = annotations.get("metadata", {}) if isinstance(annotations, dict) else {}
    if "version" in metadata:
        dataset.version = metadata["version"]
    if class_names is not None:
        if not hasattr(dataset, "box_type_3d") or not hasattr(dataset, "box_mode_3d"):
            dataset.box_type_3d, dataset.box_mode_3d = get_box_type("LiDAR")
        dataset.get_ann_info = types.MethodType(make_legacy_get_ann_info(class_names), dataset)
        dataset.pre_pipeline = types.MethodType(make_legacy_pre_pipeline(), dataset)
    return metadata


def move_labels_to_device(labels, device):
    if torch.is_tensor(labels):
        return labels.to(device)
    return torch.as_tensor(labels, device=device)


def prepare_train_batch(dataset, index, device):
    item = dataset[index]
    img = unwrap_datacontainer(item["img"])
    img_metas = unwrap_datacontainer(item["img_metas"])
    gt_bboxes_3d = unwrap_datacontainer(item["gt_bboxes_3d"])
    gt_labels_3d = unwrap_datacontainer(item["gt_labels_3d"])

    if not torch.is_tensor(img):
        raise TypeError(f"expected img tensor, got {type(img)}")
    if not isinstance(img_metas, dict):
        raise TypeError(f"expected img_metas queue dict, got {type(img_metas)}")

    return dict(
        img=img.unsqueeze(0).to(device),
        img_metas=[img_metas],
        gt_bboxes_3d=[gt_bboxes_3d],
        gt_labels_3d=[move_labels_to_device(gt_labels_3d, device)],
    )
