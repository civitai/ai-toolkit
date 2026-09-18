import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from toolkit.dataloader_mixins import ImageProcessingDTOMixin


class VideoFrameLoadingTests(unittest.TestCase):
    def _check_frames(
        self, source_count, frame_count, indices, *,
        pyav=False, opencv_frames=0, reported_count=None,
    ):
        source = [
            np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3) + i * 20
            for i in range(source_count)
        ]
        capture = MagicMock()
        capture.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: reported_count or source_count,
            cv2.CAP_PROP_FPS: 24,
        }[prop]
        if pyav:
            decoded = iter(source[:opencv_frames])

            def read():
                frame = next(decoded, None)
                return (False, None) if frame is None else (True, frame[:, :, ::-1].copy())

            capture.read.side_effect = read
        else:
            capture.read.side_effect = [(True, frame[:, :, ::-1].copy()) for frame in source]

        item = ImageProcessingDTOMixin()
        item.path = 'test.mp4'
        item.augments = []
        item.has_augmentations = False
        item.flip_x = True
        item.flip_y = False
        item.scale_to_width = 6
        item.scale_to_height = 4
        item.crop_x = 1
        item.crop_y = 0
        item.crop_width = 4
        item.crop_height = 4
        item.num_frames = frame_count
        item.dataset_config = SimpleNamespace(
            buckets=True, do_audio=False, auto_frame_count=False,
            shrink_video_to_frames=True, fps=24,
        )
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
        ])
        expected = torch.stack([
            transform(
                Image.fromarray(source[i])
                .transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                .crop((1, 0, 5, 4))
            )
            for i in indices
        ])
        container = MagicMock()
        container.__enter__.return_value = container
        container.decode.return_value = [
            SimpleNamespace(to_ndarray=lambda format, frame=frame: frame) for frame in source
        ]
        with (
            patch('toolkit.dataloader_mixins.cv2.VideoCapture', return_value=capture),
            patch('av.open', return_value=container),
            patch(
                'toolkit.dataloader_mixins.torch.stack',
                side_effect=AssertionError('second full video allocation'),
            ),
        ):
            item.load_and_process_video(transform)
        torch.testing.assert_close(item.tensor, expected, rtol=0, atol=0)
        self.assertTrue(item.tensor.is_contiguous())
        capture.release.assert_called_once()

    def test_video_frames_are_written_directly_to_output(self):
        self._check_frames(3, 3, [0, 1, 2])

    def test_stretched_video_preserves_repeated_frames(self):
        self._check_frames(3, 5, [0, 0, 1, 2, 2])

    def test_pyav_fallback_preserves_frames(self):
        self._check_frames(3, 5, [0, 0, 1, 2, 2], pyav=True)

    def test_pyav_fallback_preserves_already_decoded_frames(self):
        self._check_frames(3, 5, [0, 0, 1, 2, 2], pyav=True, opencv_frames=1)

    def test_pyav_fallback_repeats_last_frame_when_metadata_overshoots(self):
        self._check_frames(2, 4, [0, 1, 1, 1], pyav=True, reported_count=4)


if __name__ == '__main__':
    unittest.main()
