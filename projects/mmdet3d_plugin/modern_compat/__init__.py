"""Compatibility shims for running legacy BEVFormer on modern OpenMMLab."""

from .checkpoint import build_model, build_train_model, load_checkpoint_modern
from .config import (
    CHECKPOINT_PATH,
    CONFIG_PATH,
    DATA_ROOT,
    TRAIN_INFO,
    VAL_INFO,
    apply_config_fixes,
    prepare_config,
)
from .dataset import (
    attach_legacy_infos,
    build_dataset,
    build_train_dataset,
    configure_occupancy_eval_dataset,
    configure_test_dataset,
    configure_train_dataset,
    prepare_train_batch,
)
from .training import (
    choose_device,
    initialize_modern_compat,
    parse_losses,
    patch_legacy_assigner_and_sampler,
    print_header,
)
from .transforms import (
    is_datacontainer,
    make_data_container,
    register_runtime_compat_transforms,
    unwrap_datacontainer,
)

__all__ = [
    'CHECKPOINT_PATH',
    'CONFIG_PATH',
    'DATA_ROOT',
    'TRAIN_INFO',
    'VAL_INFO',
    'apply_config_fixes',
    'attach_legacy_infos',
    'build_dataset',
    'build_model',
    'build_train_dataset',
    'build_train_model',
    'choose_device',
    'configure_test_dataset',
    'configure_occupancy_eval_dataset',
    'configure_train_dataset',
    'initialize_modern_compat',
    'is_datacontainer',
    'load_checkpoint_modern',
    'make_data_container',
    'parse_losses',
    'patch_legacy_assigner_and_sampler',
    'prepare_config',
    'prepare_train_batch',
    'print_header',
    'register_runtime_compat_transforms',
    'unwrap_datacontainer',
]
