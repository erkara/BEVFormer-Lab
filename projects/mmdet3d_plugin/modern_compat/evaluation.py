"""Official nuScenes detection evaluation for legacy BEVFormer outputs."""

from pathlib import Path

import numpy as np
from mmengine.fileio import dump, load
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion


def output_to_legacy_nusc_box(detection):
    """Match the MMDetection3D 0.17 box convention used by the Tiny checkpoint."""
    boxes = detection['bboxes_3d']
    scores = detection['scores_3d'].cpu().numpy()
    labels = detection['labels_3d'].cpu().numpy()
    centers = boxes.gravity_center.cpu().numpy()
    dimensions = boxes.dims.cpu().numpy()
    yaws = boxes.yaw.cpu().numpy()
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError('prediction boxes, scores, and labels must have equal lengths')
    return [NuScenesBox(
        centers[i], dimensions[i], Quaternion(axis=[0, 0, 1], radians=-yaws[i] - np.pi / 2),
        label=int(labels[i]), score=float(scores[i]),
        velocity=(*boxes.tensor[i, 7:9].cpu().tolist(), 0.0))
        for i in range(len(boxes))]


def format_nuscenes_results(outputs, dataset, cfg, output_dir):
    """Write complete validation predictions in the official submission format."""
    if len(outputs) != len(dataset.data_infos):
        raise ValueError('nuScenes formatting requires the complete validation split')
    version = dataset.version
    if version not in ('v1.0-mini', 'v1.0-trainval'):
        raise ValueError(f'unsupported nuScenes version: {version}')

    classes = list(cfg.class_names)
    eval_cfg = config_factory('detection_cvpr_2019')
    results = {}
    for info, output in zip(dataset.data_infos, outputs):
        token = info['token']
        if token in results:
            raise ValueError(f'duplicate sample token: {token}')
        boxes = output_to_legacy_nusc_box(output['pts_bbox'])
        lidar_rot = Quaternion(info['lidar2ego_rotation'])
        ego_rot = Quaternion(info['ego2global_rotation'])
        lidar_trans = np.asarray(info['lidar2ego_translation'])
        ego_trans = np.asarray(info['ego2global_translation'])
        annotations = []
        for box in boxes:
            label = int(box.label)
            if not 0 <= label < len(classes):
                raise ValueError(f'invalid prediction label: {label}')
            name = classes[label]
            box.rotate(lidar_rot)
            box.translate(lidar_trans)
            if np.linalg.norm(box.center[:2]) > eval_cfg.class_range[name]:
                continue
            box.rotate(ego_rot)
            box.translate(ego_trans)
            speed = np.linalg.norm(box.velocity[:2])
            if speed > 0.2 and name in ('car', 'construction_vehicle', 'bus', 'truck', 'trailer'):
                attribute = 'vehicle.moving'
            elif speed > 0.2 and name in ('bicycle', 'motorcycle'):
                attribute = 'cycle.with_rider'
            elif speed <= 0.2 and name == 'pedestrian':
                attribute = 'pedestrian.standing'
            elif speed <= 0.2 and name == 'bus':
                attribute = 'vehicle.stopped'
            else:
                attribute = NuScenesMetric.DefaultAttribute[name]
            annotations.append(dict(
                sample_token=token, translation=box.center.tolist(), size=box.wlh.tolist(),
                rotation=box.orientation.elements.tolist(), velocity=box.velocity[:2].tolist(),
                detection_name=name, detection_score=float(box.score), attribute_name=attribute))
        results[token] = annotations

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / 'results_nusc.json'
    dump(dict(meta=dict(cfg.input_modality), results=results), str(result_path))
    return result_path


def evaluate_nuscenes_results(result_path, dataset, output_dir):
    """Run the official nuScenes detection metrics on a complete split."""
    eval_set = {'v1.0-mini': 'mini_val', 'v1.0-trainval': 'val'}[dataset.version]
    nusc = NuScenes(version=dataset.version, dataroot=dataset.data_root, verbose=False)
    evaluator = NuScenesEval(
        nusc, config=config_factory('detection_cvpr_2019'),
        result_path=str(result_path), eval_set=eval_set,
        output_dir=str(output_dir), verbose=False)
    evaluator.main(plot_examples=0, render_curves=False)
    return load(str(Path(output_dir) / 'metrics_summary.json'))
