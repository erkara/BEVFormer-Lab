"""Prepare nuScenes validation camera frames without LiDAR or sweep downloads."""

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
from mmengine.fileio import dump
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils import splits
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.data_converter.nuscenes_converter import (  # noqa: E402
    _get_can_bus_info, obtain_sensor2top, resolve_canbus_root,
)


CAMERAS = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
)


def selected_samples(nusc, version):
    names = set(splits.val if version == 'v1.0-trainval' else splits.mini_val)
    scenes = {scene['token'] for scene in nusc.scene if scene['name'] in names}
    if len(scenes) != len(names):
        raise ValueError(f'expected {len(names)} validation scenes, found {len(scenes)}')
    samples = sorted(
        (sample for sample in nusc.sample if sample['scene_token'] in scenes),
        key=lambda sample: sample['timestamp'],
    )
    return samples, scenes


def camera_paths(nusc, samples):
    paths = set()
    for sample in samples:
        for camera in CAMERAS:
            filename = nusc.get('sample_data', sample['data'][camera])['filename']
            path = Path(filename)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != 'samples':
                raise ValueError(f'unexpected camera path in metadata: {filename}')
            paths.add(path.as_posix())
    return paths


def extract_selected(archives, root, required):
    extracted = 0
    for archive_path in archives:
        found = 0
        with tarfile.open(archive_path, mode='r|gz') as archive:
            for member in archive:
                name = member.name.removeprefix('./')
                if name not in required or not member.isfile():
                    continue
                found += 1
                destination = root / name
                if destination.is_file():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name(destination.name + '.part')
                source = archive.extractfile(member)
                if source is None:
                    raise OSError(f'cannot read {name} from {archive_path}')
                try:
                    with partial.open('wb') as target:
                        shutil.copyfileobj(source, target)
                    os.replace(partial, destination)
                finally:
                    source.close()
                    partial.unlink(missing_ok=True)
                extracted += 1
        print(f'{archive_path}: matched {found} validation images')
    print(f'extracted {extracted} new validation images')


def keyframe_archives(directory):
    directory = directory.expanduser().resolve()
    archives = [directory / f'v1.0-trainval{index:02d}_keyframes.tgz'
                for index in range(1, 11)]
    missing = [path.name for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing keyframe archives in {directory}: {", ".join(missing)}')
    return archives


def build_eval_infos(nusc, samples, can_bus_root, root):
    can_bus = NuScenesCanBus(dataroot=str(can_bus_root))
    infos = []
    frame_by_scene = {}
    for sample in samples:
        lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_calib = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
        lidar_pose = nusc.get('ego_pose', lidar['ego_pose_token'])
        lidar_rot = Quaternion(lidar_calib['rotation']).rotation_matrix
        ego_rot = Quaternion(lidar_pose['rotation']).rotation_matrix
        info = dict(
            lidar_path=str(root / lidar['filename']), token=sample['token'],
            prev=sample['prev'], next=sample['next'],
            can_bus=_get_can_bus_info(nusc, can_bus, sample),
            frame_idx=frame_by_scene.get(sample['scene_token'], 0),
            sweeps=[], cams={}, scene_token=sample['scene_token'],
            lidar2ego_translation=lidar_calib['translation'],
            lidar2ego_rotation=lidar_calib['rotation'],
            ego2global_translation=lidar_pose['translation'],
            ego2global_rotation=lidar_pose['rotation'],
            timestamp=sample['timestamp'],
        )
        frame_by_scene[sample['scene_token']] = info['frame_idx'] + 1
        for camera in CAMERAS:
            token = sample['data'][camera]
            camera_data = nusc.get('sample_data', token)
            camera_calib = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
            camera_info = obtain_sensor2top(
                nusc, token, lidar_calib['translation'], lidar_rot,
                lidar_pose['translation'], ego_rot, camera,
            )
            camera_info['data_path'] = str(root / camera_data['filename'])
            camera_info['cam_intrinsic'] = np.asarray(camera_calib['camera_intrinsic'])
            info['cams'][camera] = camera_info
        infos.append(info)
    return infos


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='root containing v1.0-trainval metadata and extracted camera images')
    parser.add_argument('--canbus-root', type=Path, default=Path('data'),
                        help='parent of can_bus pose files, or the expansion folder itself')
    parser.add_argument('--version', choices=['v1.0-trainval', 'v1.0-mini'], default='v1.0-trainval')
    parser.add_argument('--archive', type=Path, action='append', default=[],
                        help='one nuScenes blob .tgz; repeat this option for multiple archives')
    parser.add_argument('--archives-dir', type=Path,
                        help='directory containing all ten trainval keyframe archives')
    parser.add_argument('--out-info', type=Path,
                        help='output evaluation-only info pickle (default: under data root)')
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    nusc = NuScenes(version=args.version, dataroot=str(root), verbose=False)
    samples, scenes = selected_samples(nusc, args.version)
    required = camera_paths(nusc, samples)
    print(f'validation scenes: {len(scenes)}; frames: {len(samples)}; camera images: {len(required)}')
    if args.archive and args.archives_dir:
        raise ValueError('use --archive or --archives-dir, not both')
    archives = keyframe_archives(args.archives_dir) if args.archives_dir else args.archive
    if archives:
        extract_selected(archives, root, required)
    missing = [name for name in required if not (root / name).is_file()]
    print(f'missing camera images: {len(missing)}')
    if missing:
        print('Info file not written. Process the remaining blob archives and rerun this command.')
        return 0
    out_info = args.out_info or root / 'nuscenes_infos_temporal_val_camera_only.pkl'
    out_info = out_info.expanduser().resolve()
    canbus_root = resolve_canbus_root(args.canbus_root)
    print(f'CAN bus root: {canbus_root}')
    infos = build_eval_infos(nusc, samples, canbus_root, root)
    populated = sum(bool(np.any(info['can_bus'])) for info in infos)
    print(f'frames with CAN bus data: {populated}/{len(infos)}')
    if populated == 0:
        raise ValueError('all CAN bus records are zero; check the CAN bus expansion')
    out_info.parent.mkdir(parents=True, exist_ok=True)
    dump(dict(infos=infos, metadata=dict(version=args.version)), str(out_info))
    print(f'wrote evaluation info: {out_info} ({len(infos)} frames)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
