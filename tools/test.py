# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
# Modified by Zhiqi Li
# Modernized locally for MMCV 2 / MMEngine / MMDet3D 1.x compatibility.
# ---------------------------------------------
import argparse
import copy
import math
import os
import os.path as osp
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from mmengine.config import Config, DictAction
from mmengine.fileio import dump, load
from mmengine.registry import build_from_cfg, init_default_scope
from mmengine.runner import load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmdet.models  # noqa: F401,E402
import mmdet3d.models  # noqa: F401,E402
import projects.mmdet3d_plugin  # noqa: F401,E402
from mmdet.registry import MODELS as MMDET_MODELS  # noqa: E402
from mmdet3d.registry import DATASETS, TRANSFORMS  # noqa: E402


DEFAULT_DATA_ROOT = 'data/nuscenes/'
DEFAULT_VAL_INFO = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'


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
    return hasattr(value, 'data') and (
        hasattr(value, 'cpu_only') or hasattr(value, 'stack') or hasattr(value, 'padding_value'))


def unwrap_datacontainer(value):
    if is_datacontainer(value):
        if hasattr(value, '_data'):
            return value._data
        return value.data
    return value


def register_runtime_compat_transforms():
    @TRANSFORMS.register_module(name='LoadMultiViewImageFromFiles', force=True)
    class LoadMultiViewImageFromFilesCompat:
        def __init__(self, to_float32=False, color_type='color', *args, **kwargs):
            self.to_float32 = to_float32
            self.color_type = color_type

        def __call__(self, results):
            filenames = results.get('img_filename') or results.get('filename')
            if filenames is None:
                raise KeyError('LoadMultiViewImageFromFilesCompat expected img_filename or filename')
            imgs = [mmcv.imread(name, self.color_type) for name in filenames]
            if self.to_float32:
                imgs = [img.astype(np.float32) for img in imgs]
            results['filename'] = filenames
            results['img'] = imgs
            results['img_shape'] = [img.shape for img in imgs]
            results['ori_shape'] = [img.shape for img in imgs]
            results['pad_shape'] = [img.shape for img in imgs]
            num_channels = 1 if imgs[0].ndim < 3 else imgs[0].shape[2]
            results['img_norm_cfg'] = dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False,
            )
            return results

    @TRANSFORMS.register_module(name='DefaultFormatBundle3D', force=True)
    class DefaultFormatBundle3DCompat:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, results):
            if 'img' in results and isinstance(results['img'], list):
                imgs = [np.ascontiguousarray(img.transpose(2, 0, 1)) for img in results['img']]
                results['img'] = make_data_container(
                    torch.stack([torch.from_numpy(img) for img in imgs]), stack=True)
            if 'gt_labels_3d' in results and not torch.is_tensor(results['gt_labels_3d']):
                results['gt_labels_3d'] = make_data_container(torch.as_tensor(results['gt_labels_3d']))
            if 'gt_bboxes_3d' in results:
                results['gt_bboxes_3d'] = make_data_container(results['gt_bboxes_3d'], cpu_only=True)
            return results


def apply_config_fixes(cfg):
    cfg.model.pop('pretrained', None)
    cfg.model.img_backbone.type = 'mmdet.ResNet'
    cfg.model.img_neck.type = 'mmdet.FPN'
    cfg.model.pts_bbox_head.type = 'mmdet.BEVFormerHead'
    cfg.model.train_cfg = None


def configure_test_dataset(cfg):
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg.data_root = DEFAULT_DATA_ROOT
    test_cfg.ann_file = str(Path(DEFAULT_VAL_INFO).resolve())
    test_cfg.test_mode = True
    test_cfg.lazy_init = True
    test_cfg.pop('samples_per_gpu', None)
    test_cfg.pop('classes', None)
    return test_cfg


def attach_legacy_infos(dataset, ann_file):
    annotations = load(ann_file)
    infos = annotations.get('infos', annotations if isinstance(annotations, list) else None)
    if infos is None:
        raise KeyError('legacy annotation file does not contain infos')
    dataset.data_infos = infos
    dataset.data_list = infos
    dataset._fully_initialized = True
    dataset.serialize_data = False
    if not hasattr(dataset, 'show_ins_var'):
        dataset.show_ins_var = False
    if not hasattr(dataset, 'prepare_test_data'):
        dataset.prepare_test_data = dataset.prepare_data
    metadata = annotations.get('metadata', {}) if isinstance(annotations, dict) else {}
    if 'version' in metadata:
        dataset.version = metadata['version']
    return metadata


def build_dataset(test_cfg):
    try:
        return DATASETS.build(test_cfg)
    except Exception as exc:
        print('DATASETS.build failed; falling back to build_from_cfg')
        print(f'fallback reason: {type(exc).__name__}: {exc}')
        return build_from_cfg(test_cfg, DATASETS)


def choose_device():
    if not torch.cuda.is_available():
        return torch.device('cpu'), 'torch.cuda.is_available() is false'
    try:
        torch.empty(1, device='cuda')
        return torch.device('cuda'), 'CUDA is available and a test allocation succeeded'
    except Exception as exc:
        return torch.device('cpu'), f'CUDA test allocation failed: {type(exc).__name__}: {exc}'


def get_primary_item(item):
    if isinstance(item, list) and len(item) == 1 and isinstance(item[0], dict):
        return item[0]
    return item


def load_frame(dataset, index):
    item = get_primary_item(dataset[index])
    img_tensor = unwrap_datacontainer(item['img'])
    metadata_dict = unwrap_datacontainer(item['img_metas'])
    if not torch.is_tensor(img_tensor):
        raise TypeError(f'expected image tensor, got {type(img_tensor)}')
    if not isinstance(metadata_dict, dict):
        raise TypeError(f'expected metadata dict, got {type(metadata_dict)}')
    return img_tensor, metadata_dict


def summarize_tensorlike(value):
    if torch.is_tensor(value):
        return f'Tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device}'
    if hasattr(value, 'tensor') and torch.is_tensor(value.tensor):
        return f'{type(value).__name__} tensor_shape={tuple(value.tensor.shape)} dtype={value.tensor.dtype}'
    if isinstance(value, dict):
        return {key: summarize_tensorlike(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        first = summarize_tensorlike(value[0]) if value else '<empty>'
        return f'{type(value).__name__} len={len(value)} first={first}'
    return type(value).__name__


def summarize_output(output):
    print(f'output type: {type(output)}')
    print(f'output summary: {summarize_tensorlike(output)}')
    if not isinstance(output, list) or not output:
        return
    first = output[0]
    if not isinstance(first, dict):
        return
    print(f'output[0] keys: {list(first.keys())}')
    pts_bbox = first.get('pts_bbox')
    if isinstance(pts_bbox, dict):
        print(f'pts_bbox keys: {list(pts_bbox.keys())}')
        for key in ('bboxes_3d', 'scores_3d', 'labels_3d'):
            print(f'{key}: {summarize_tensorlike(pts_bbox.get(key))}')


def run_model_frame(model, img_tensor, metadata_dict, device):
    img = [img_tensor.unsqueeze(0).to(device)]
    img_metas = [[metadata_dict]]
    with torch.no_grad():
        return model(return_loss=False, img=img, img_metas=img_metas)



def extract_predictions(output):
    if isinstance(output, list):
        if not output:
            raise TypeError('expected non-empty output list')
        output = output[0]
    if not isinstance(output, dict) or 'pts_bbox' not in output:
        raise KeyError(f"expected output with 'pts_bbox', got {type(output)}")
    result = output['pts_bbox']
    boxes_3d = result['bboxes_3d']
    boxes = getattr(boxes_3d, 'tensor', boxes_3d).detach().cpu()
    scores = result['scores_3d'].detach().cpu()
    labels = result['labels_3d'].detach().cpu()
    return boxes, scores, labels


def select_predictions(boxes, scores, labels, score_thr, top_k):
    keep = scores >= score_thr
    if int(keep.sum()) == 0:
        selected_idx = torch.argsort(scores, descending=True)[:min(20, int(scores.numel()))]
        fallback = True
    else:
        selected_idx = torch.nonzero(keep, as_tuple=False).flatten()
        selected_idx = selected_idx[torch.argsort(scores[selected_idx], descending=True)]
        fallback = False
    if top_k > 0:
        selected_idx = selected_idx[:top_k]
    return boxes[selected_idx], scores[selected_idx], labels[selected_idx], fallback


def extract_gt_from_dataset(dataset, index, class_names):
    info = None
    if hasattr(dataset, 'data_infos') and len(dataset.data_infos) > index:
        candidate = dataset.data_infos[index]
        if isinstance(candidate, dict) and 'gt_boxes' in candidate:
            info = candidate
    if info is None and hasattr(dataset, 'get_ann_info'):
        ann = dataset.get_ann_info(index)
        if isinstance(ann, dict):
            info = ann
    if info is None:
        return torch.empty((0, 7), dtype=torch.float32), []

    gt_boxes = np.asarray(info.get('gt_boxes', []), dtype=np.float32)
    gt_names = info.get('gt_names')
    valid_flag = info.get('valid_flag')
    if gt_boxes.ndim == 1:
        gt_boxes = gt_boxes.reshape(0, 7)
    if valid_flag is not None:
        valid_flag = np.asarray(valid_flag).astype(bool)
        gt_boxes = gt_boxes[valid_flag]
        if gt_names is not None:
            gt_names = np.asarray(gt_names)[valid_flag]
    if gt_names is None:
        names = ['<unknown>' for _ in range(len(gt_boxes))]
    else:
        names = [str(name) for name in list(gt_names)]
    return torch.as_tensor(gt_boxes[:, :7], dtype=torch.float32), names


def box_corners_bev(box):
    x, y, _, dx, dy, _, yaw = box[:7].tolist()
    hx = dx / 2.0
    hy = dy / 2.0
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], dtype=np.float32)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def label_to_class(label, class_names):
    label_int = int(label.item()) if torch.is_tensor(label) else int(label)
    return class_names[label_int] if 0 <= label_int < len(class_names) else f'label_{label_int}'


def save_bev_visualization(output, dataset, dataset_index, metadata_dict, cfg, args):
    out_dir = Path(args.show_dir or 'outputs/test_vis')
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_boxes, pred_scores, pred_labels = extract_predictions(output)
    pred_boxes, pred_scores, pred_labels, fallback = select_predictions(
        pred_boxes, pred_scores, pred_labels, args.score_thr, args.top_k)
    class_names = list(getattr(cfg, 'class_names', []))
    gt_boxes, gt_names = extract_gt_from_dataset(dataset, dataset_index, class_names)

    point_cloud_range = list(cfg.point_cloud_range)
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=False, edgecolor='0.55', linewidth=1.2, zorder=1))
    ax.scatter([0], [0], c='red', marker='*', s=220, edgecolors='black', linewidths=0.8, zorder=10)
    ax.text(1.2, 1.2, 'ego', color='red', fontsize=9, fontweight='bold', zorder=10)

    cmap = plt.get_cmap('tab10')
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        color = cmap(int(label.item()) % 10)
        corners = box_corners_bev(box)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color,
                             linewidth=3.0, linestyle='-', zorder=7))
        ax.scatter([box[0].item()], [box[1].item()], s=54, marker='o', color=color,
                   edgecolors='white', linewidths=0.7, zorder=8)
        ax.text(box[0].item(), box[1].item(), f'{label_to_class(label, class_names)} {float(score):.2f}',
                fontsize=7, color=color, zorder=9)

    for box, name in zip(gt_boxes, gt_names):
        corners = box_corners_bev(box)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor='0.15',
                             linewidth=1.8, linestyle='--', alpha=0.85, zorder=3))
        ax.scatter([box[0].item()], [box[1].item()], s=40, marker='x', color='0.15', linewidths=1.4, zorder=4)

    handles = [
        Line2D([0], [0], color='tab:blue', linestyle='-', linewidth=3.0, label='prediction box'),
        Line2D([0], [0], color='0.15', linestyle='--', linewidth=1.8, label='ground truth box'),
        Line2D([0], [0], color='red', marker='*', linestyle='None', markersize=12, label='ego'),
    ]
    ax.legend(handles=handles, loc='upper right')
    ax.set_xlim(x_min - 2, x_max + 2)
    ax.set_ylim(y_min - 2, y_max + 2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True, linewidth=0.3)
    sample_idx = metadata_dict.get('sample_idx', '<missing>')
    ax.set_title(
        f'index={dataset_index} sample={sample_idx} thr={args.score_thr:g} '
        f'preds={len(pred_scores)} gt={len(gt_boxes)}'
    )
    if fallback:
        ax.text(0.02, 0.02, 'no predictions met threshold; showing highest scores',
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='0.5', alpha=0.9),
                va='bottom', ha='left')
    fig.tight_layout()
    safe_sample = str(sample_idx)[:12]
    out_path = out_dir / f'test_sample_{dataset_index}_{safe_sample}.png'
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f'saved visualization: {out_path}')
    return out_path


def load_checkpoint_modern(model, checkpoint_path):
    try:
        checkpoint = load_checkpoint(
            model, checkpoint_path, map_location='cpu', strict=False, logger=None)
        print('checkpoint load method: mmengine.runner.load_checkpoint')
        return checkpoint
    except Exception:
        print('mmengine.runner.load_checkpoint failed; trying torch.load fallback')
        traceback.print_exc()
        raw = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(raw, dict):
            state_dict = raw.get('state_dict') or raw.get('model') or raw.get('model_state_dict') or raw
        else:
            state_dict = raw
        incompatible = model.load_state_dict(state_dict, strict=False)
        print('checkpoint load method: torch.load + model.load_state_dict(strict=False)')
        print(f'missing_keys: {list(incompatible.missing_keys)}')
        print(f'unexpected_keys: {list(incompatible.unexpected_keys)}')
        return raw if isinstance(raw, dict) else {}


def parse_args():
    parser = argparse.ArgumentParser(description='Modern BEVFormer test/inference entrypoint')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='accepted for CLI compatibility; currently skipped')
    parser.add_argument('--format-only', action='store_true', help='format outputs without evaluation')
    parser.add_argument('--eval', type=str, nargs='+', help='evaluation metrics, e.g. bbox')
    parser.add_argument('--show', action='store_true', help='save one BEV visualization to the default output directory')
    parser.add_argument('--show-dir', help='directory where one BEV visualization will be saved')
    parser.add_argument('--gpu-collect', action='store_true', help='accepted for distributed CLI compatibility')
    parser.add_argument('--tmpdir', help='tmp directory used for collecting distributed results')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='set deterministic CUDNN behavior')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config settings')
    parser.add_argument('--options', nargs='+', action=DictAction, help='deprecated eval options')
    parser.add_argument('--eval-options', nargs='+', action=DictAction, help='custom options for evaluation')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None, help='maximum validation samples to run before stopping')
    parser.add_argument('--show-index', type=int, default=0, help='dataset/output index to visualize when --show or --show-dir is set')
    parser.add_argument('--score-thr', type=float, default=0.15, help='score threshold for visualization')
    parser.add_argument('--top-k', type=int, default=10, help='maximum predictions to draw in visualization; <=0 keeps all above threshold')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.eval_options:
        raise ValueError('--options and --eval-options cannot both be specified')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()
    if args.launcher != 'none':
        raise NotImplementedError('Distributed tools/test.py is still pending; use --launcher none for this port gate.')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format-only cannot both be specified')
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print('=== load config ===')
    print(f'config: {args.config}')
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    apply_config_fixes(cfg)
    print('modern config fixes applied')

    print('\n=== build validation dataset ===')
    init_default_scope('mmdet3d')
    register_runtime_compat_transforms()
    test_cfg = configure_test_dataset(cfg)
    dataset = build_dataset(test_cfg)
    metadata = attach_legacy_infos(dataset, test_cfg.ann_file)
    print(f'attached legacy metadata: {metadata}')
    print(f'dataset type: {type(dataset)}')
    print(f'dataset length: {len(dataset)}')

    print('\n=== build model ===')
    init_default_scope('mmdet')
    device, device_reason = choose_device()
    print(f'selected device: {device} ({device_reason})')
    model = MMDET_MODELS.build(cfg.model)
    model.eval()
    model.to(device)

    checkpoint = load_checkpoint_modern(model, args.checkpoint)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    elif hasattr(dataset, 'METAINFO') and 'classes' in dataset.METAINFO:
        model.CLASSES = dataset.METAINFO['classes']
    elif hasattr(dataset, 'CLASSES'):
        model.CLASSES = dataset.CLASSES

    if args.fuse_conv_bn:
        print('fuse-conv-bn requested, but skipped in this modern minimal path')

    max_samples = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    if max_samples <= 0:
        raise ValueError('--max-samples must be positive when provided')

    print('\n=== inference ===')
    print(f'max samples: {max_samples}')
    outputs = []
    metadata_by_index = {}
    for index in range(max_samples):
        img_tensor, metadata_dict = load_frame(dataset, index)
        sample_idx = metadata_dict.get('sample_idx', '<missing>')
        scene_token = metadata_dict.get('scene_token', '<missing>')
        metadata_by_index[index] = metadata_dict
        print(f'[{index + 1}/{max_samples}] dataset_index={index} sample_idx={sample_idx} scene_token={scene_token}')
        output = run_model_frame(model, img_tensor, metadata_dict, device)
        if index == 0:
            print('\n=== first output summary ===')
            summarize_output(output)
        outputs.extend(output if isinstance(output, list) else [output])

    print('\n=== summary ===')
    print(f'outputs collected: {len(outputs)}')
    print('single-GPU tools/test.py inference: ok')

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'writing results to {out_path}')
        dump(outputs, str(out_path))
        print(f'saved output pickle: {out_path}')

    if args.show or args.show_dir:
        if args.show_index < 0 or args.show_index >= len(outputs):
            raise IndexError(f'--show-index {args.show_index} is outside collected output count {len(outputs)}')
        save_bev_visualization(
            outputs[args.show_index], dataset, args.show_index, metadata_by_index[args.show_index], cfg, args)

    kwargs = {} if args.eval_options is None else args.eval_options
    kwargs.setdefault('jsonfile_prefix', osp.join(
        'test', Path(args.config).stem, time.ctime().replace(' ', '_').replace(':', '_')))

    if args.format_only:
        if args.max_samples is not None:
            print('format-only skipped because outputs are truncated by --max-samples')
        else:
            dataset.format_results(outputs, **kwargs)

    if args.eval:
        if args.max_samples is not None:
            print('dataset.evaluate skipped because outputs are truncated by --max-samples')
        else:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            try:
                print(dataset.evaluate(outputs, **eval_kwargs))
            except Exception as exc:
                print('dataset.evaluate failed and remains pending')
                print(f'exception type: {type(exc).__name__}')
                print(f'exception: {exc}')
                traceback.print_exc()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
