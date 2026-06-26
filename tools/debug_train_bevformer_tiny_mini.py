import argparse
import copy
import logging
import sys
import traceback
import types
from types import SimpleNamespace
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmengine.config import Config
from mmengine.fileio import load
from mmengine.registry import build_from_cfg, init_default_scope
from mmengine.runner import load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmdet.models  # noqa: F401,E402
import mmdet3d.models  # noqa: F401,E402
import projects.mmdet3d_plugin  # noqa: F401,E402
from mmdet.registry import MODELS as MMDET_MODELS  # noqa: E402
from mmdet3d.registry import DATASETS, TRANSFORMS  # noqa: E402
from mmdet3d.structures import LiDARInstance3DBoxes, get_box_type  # noqa: E402


CONFIG_PATH = "projects/configs/bevformer/bevformer_tiny.py"
CHECKPOINT_PATH = "ckpts/bevformer_tiny_epoch_24.pth"
DATA_ROOT = "data/nuscenes/"
TRAIN_INFO = "data/nuscenes/nuscenes_infos_temporal_train.pkl"


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


def print_header(name):
    print(f"\n=== {name} ===")


def apply_config_fixes(cfg):
    cfg.model.pop("pretrained", None)
    cfg.model.img_backbone.type = "mmdet.ResNet"
    cfg.model.img_neck.type = "mmdet.FPN"
    cfg.model.pts_bbox_head.type = "mmdet.BEVFormerHead"


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


def configure_train_dataset(cfg):
    train_cfg = copy.deepcopy(cfg.data.train)
    train_cfg.data_root = DATA_ROOT
    train_cfg.ann_file = str(Path(TRAIN_INFO).resolve())
    train_cfg.test_mode = False
    train_cfg.lazy_init = True
    train_cfg.pop("samples_per_gpu", None)
    train_cfg.pop("classes", None)
    return train_cfg


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


def attach_legacy_infos(dataset, ann_file, class_names):
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
    metadata = annotations.get("metadata", {}) if isinstance(annotations, dict) else {}
    if "version" in metadata:
        dataset.version = metadata["version"]
    if not hasattr(dataset, "box_type_3d") or not hasattr(dataset, "box_mode_3d"):
        dataset.box_type_3d, dataset.box_mode_3d = get_box_type("LiDAR")
    dataset.get_ann_info = types.MethodType(make_legacy_get_ann_info(class_names), dataset)
    dataset.pre_pipeline = types.MethodType(make_legacy_pre_pipeline(), dataset)
    return metadata


def build_dataset(train_cfg):
    try:
        return DATASETS.build(train_cfg)
    except Exception as exc:
        print("DATASETS.build failed; falling back to build_from_cfg")
        print(f"fallback reason: {type(exc).__name__}: {exc}")
        return build_from_cfg(train_cfg, DATASETS)


def choose_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg), "selected by --device"
    if torch.cuda.is_available():
        try:
            torch.empty(1, device="cuda")
            return torch.device("cuda"), "CUDA is available and a test allocation succeeded"
        except Exception as exc:
            return torch.device("cpu"), f"CUDA test allocation failed: {type(exc).__name__}: {exc}"
    return torch.device("cpu"), "torch.cuda.is_available() is false"


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

    batch = dict(
        img=img.unsqueeze(0).to(device),
        img_metas=[img_metas],
        gt_bboxes_3d=[gt_bboxes_3d],
        gt_labels_3d=[move_labels_to_device(gt_labels_3d, device)],
    )
    return batch


def loss_to_scalar(value):
    if torch.is_tensor(value):
        return value.mean()
    if isinstance(value, (list, tuple)):
        tensors = [loss_to_scalar(item) for item in value]
        return sum(tensors)
    raise TypeError(f"unsupported loss value type: {type(value)}")


def parse_losses(losses):
    scalar_losses = {}
    total_loss = None
    for key, value in losses.items():
        scalar = loss_to_scalar(value)
        scalar_losses[key] = float(scalar.detach().cpu())
        if "loss" in key:
            total_loss = scalar if total_loss is None else total_loss + scalar
    if total_loss is None:
        raise RuntimeError(f"no loss keys found in losses: {list(losses.keys())}")
    return total_loss, scalar_losses



class LegacyFocalLossCost:
    def __init__(self, weight=2.0, alpha=0.25, gamma=2.0, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (1 - cls_pred).pow(self.gamma)
        return (pos_cost[:, gt_labels] - neg_cost[:, gt_labels]) * self.weight


class LegacyBBoxL1Cost:
    def __init__(self, weight=0.25):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        return torch.cdist(bbox_pred, gt_bboxes, p=1) * self.weight


def legacy_pseudo_sample(assign_result, pred_bboxes, gt_bboxes):
    gt_inds = assign_result.gt_inds
    pos_inds = torch.nonzero(gt_inds > 0, as_tuple=False).squeeze(-1).unique()
    neg_inds = torch.nonzero(gt_inds == 0, as_tuple=False).squeeze(-1).unique()
    pos_assigned_gt_inds = gt_inds[pos_inds] - 1
    if pos_inds.numel() > 0:
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long()]
    else:
        pos_gt_bboxes = gt_bboxes.new_zeros((0, gt_bboxes.size(-1)))
    return SimpleNamespace(
        pos_inds=pos_inds,
        neg_inds=neg_inds,
        pos_assigned_gt_inds=pos_assigned_gt_inds.long(),
        pos_gt_bboxes=pos_gt_bboxes,
    )


def patch_legacy_assigner_and_sampler(model):
    head = getattr(model, "pts_bbox_head", None)
    assigner = getattr(head, "assigner", None)
    if assigner is None:
        print("legacy assigner cost patch: skipped, no assigner found")
    else:
        assigner.cls_cost = LegacyFocalLossCost(weight=2.0, alpha=0.25, gamma=2.0)
        assigner.reg_cost = LegacyBBoxL1Cost(weight=0.25)
        print("legacy assigner cost patch: tensor-based FocalLossCost and BBoxL1Cost active")
    sampler = getattr(head, "sampler", None)
    if sampler is None:
        print("legacy sampler patch: skipped, no sampler found")
    else:
        sampler.sample = legacy_pseudo_sample
        print("legacy sampler patch: tensor-based pseudo sampler active")

def build_model(cfg, checkpoint, no_checkpoint, device):
    init_default_scope("mmdet")
    model = MMDET_MODELS.build(cfg.model)
    model.to(device)
    if no_checkpoint:
        print("checkpoint loading skipped by --no-checkpoint")
        if hasattr(model, "init_weights"):
            print("running model.init_weights()")
            model.init_weights()
    else:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        load_checkpoint(model, str(checkpoint_path), map_location="cpu", strict=False, logger=None)
        print(f"checkpoint loaded: {checkpoint_path}")
    model.to(device)
    model.train()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal BEVFormer tiny nuScenes-mini training smoke test")
    parser.add_argument("--max-iters", type=int, default=2, help="maximum optimizer iterations to run")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate")
    parser.add_argument("--no-checkpoint", action="store_true", help="skip checkpoint loading and initialize weights")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="checkpoint path")
    parser.add_argument("--device", default="auto", help="device, e.g. auto, cuda, cuda:0, cpu")
    parser.add_argument("--start-index", type=int, default=0, help="first train dataset index to try")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.disable(logging.INFO)
    if args.max_iters <= 0:
        raise ValueError("--max-iters must be positive")

    torch.manual_seed(0)

    print_header("load config")
    cfg = Config.fromfile(CONFIG_PATH)
    apply_config_fixes(cfg)
    print(f"config: {CONFIG_PATH}")
    print("modern config fixes applied")

    print_header("build train dataset")
    init_default_scope("mmdet3d")
    register_runtime_compat_transforms()
    train_cfg = configure_train_dataset(cfg)
    dataset = build_dataset(train_cfg)
    metadata = attach_legacy_infos(dataset, train_cfg.ann_file, list(cfg.class_names))
    print(f"attached legacy metadata: {metadata}")
    print(f"dataset type: {type(dataset)}")
    print(f"dataset length: {len(dataset)}")

    device, device_reason = choose_device(args.device)
    print_header("device")
    print(f"selected device: {device} ({device_reason})")

    print_header("build model")
    model = build_model(cfg, args.checkpoint, args.no_checkpoint, device)
    patch_legacy_assigner_and_sampler(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    print(f"optimizer: AdamW lr={args.lr:g}")

    print_header("training smoke iterations")
    last_scalar_losses = None
    for iteration in range(args.max_iters):
        dataset_index = (args.start_index + iteration) % len(dataset)
        print(f"\n--- iteration {iteration + 1}/{args.max_iters} dataset_index={dataset_index} ---")
        batch = prepare_train_batch(dataset, dataset_index, device)
        print(f"img shape: {tuple(batch['img'].shape)}")
        print(f"img_metas queue keys: {list(batch['img_metas'][0].keys())}")
        print(f"gt boxes type: {type(batch['gt_bboxes_3d'][0])}")
        print(f"gt labels shape: {tuple(batch['gt_labels_3d'][0].shape)}")

        losses = model(return_loss=True, **batch)
        total_loss, scalar_losses = parse_losses(losses)
        last_scalar_losses = scalar_losses
        print("losses:")
        for key, value in scalar_losses.items():
            print(f"  {key}: {value:.6f}")
        print(f"  total_loss: {float(total_loss.detach().cpu()):.6f}")

        total_loss.backward()
        print("backward: ok")
        optimizer.step()
        print("optimizer step: ok")
        optimizer.zero_grad(set_to_none=True)

    print_header("summary")
    print("dataset builds: yes")
    print("one batch loads: yes")
    print("forward loss works: yes")
    print("backward works: yes")
    print("optimizer step works: yes")
    if last_scalar_losses is not None:
        print(f"last loss keys: {list(last_scalar_losses.keys())}")
    print("no evaluation run")
    print("no distributed training")
    print("no full trainval")
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
