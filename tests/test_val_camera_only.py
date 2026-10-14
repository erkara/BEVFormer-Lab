"""Checks for selective nuScenes camera extraction."""

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.prepare_val_camera_only import extract_selected, keyframe_archives, resolve_canbus_root


class SelectiveExtractionTest(unittest.TestCase):
    def test_keyframe_archive_directory_requires_all_ten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [root / f'v1.0-trainval{index:02d}_keyframes.tgz'
                        for index in range(1, 11)]
            for path in expected[:-1]:
                path.touch()
            with self.assertRaisesRegex(FileNotFoundError, 'v1.0-trainval10_keyframes.tgz'):
                keyframe_archives(root)
            expected[-1].touch()
            self.assertEqual(keyframe_archives(root), expected)

    def test_resolves_nested_canbus_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose = root / 'can_bus' / 'can_bus' / 'scene-0001_pose.json'
            pose.parent.mkdir(parents=True)
            pose.touch()
            self.assertEqual(resolve_canbus_root(root), root / 'can_bus')
            self.assertEqual(resolve_canbus_root(root / 'can_bus'), root / 'can_bus')

    def test_extracts_only_expected_regular_camera_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / 'blobs.tgz'
            with tarfile.open(archive_path, 'w:gz') as archive:
                for name, payload in (
                    ('./samples/CAM_FRONT/val.jpg', b'camera'),
                    ('./samples/CAM_FRONT/train.jpg', b'other'),
                    ('./samples/LIDAR_TOP/val.bin', b'lidar'),
                ):
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
            destination = root / 'dataset'
            extract_selected([archive_path], destination, {'samples/CAM_FRONT/val.jpg'})
            self.assertEqual((destination / 'samples/CAM_FRONT/val.jpg').read_bytes(), b'camera')
            self.assertFalse((destination / 'samples/CAM_FRONT/train.jpg').exists())
            self.assertFalse((destination / 'samples/LIDAR_TOP/val.bin').exists())


if __name__ == '__main__':
    unittest.main()
