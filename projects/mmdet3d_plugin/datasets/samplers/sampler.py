# from mmcv.utils.registry import Registry, build_from_cfg
try:
    from mmcv.runner import get_dist_info
except ModuleNotFoundError:
    def get_dist_info():
        return 0, 1

# SAMPLER = Registry('sampler')

from mmengine.registry import Registry
SAMPLER = Registry('sampler')




def build_sampler(cfg, default_args):
    return build_from_cfg(cfg, SAMPLER, default_args)
