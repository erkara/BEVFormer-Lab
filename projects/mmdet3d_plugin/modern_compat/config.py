"""Config shims for legacy BEVFormer configs on modern OpenMMLab."""

CONFIG_PATH = "projects/configs/bevformer/bevformer_tiny.py"
CHECKPOINT_PATH = "ckpts/bevformer_tiny_epoch_24.pth"
DATA_ROOT = "data/nuscenes/"
TRAIN_INFO = "data/nuscenes/nuscenes_infos_temporal_train.pkl"
VAL_INFO = "data/nuscenes/nuscenes_infos_temporal_val.pkl"


def apply_config_fixes(cfg):
    """Patch legacy model config fields for MMCV 2 / MMEngine registries."""
    cfg.model.pop("pretrained", None)
    cfg.model.img_backbone.type = "mmdet.ResNet"
    cfg.model.img_neck.type = "mmdet.FPN"
    cfg.model.pts_bbox_head.type = "mmdet.BEVFormerHead"


def prepare_config(config_path, cfg_options=None):
    """Load a config and apply the modern compatibility fixes."""
    from mmengine.config import Config

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    apply_config_fixes(cfg)
    return cfg
