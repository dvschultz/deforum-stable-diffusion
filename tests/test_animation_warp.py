"""Regression: anim_frame_warp must accept BOTH a 4-dim sample buffer (what
generate() returns) and a 3-dim HWC cv2 image (the turbo/tween path), since
samples are numpy now and can no longer be told apart from cv2 images by type.
"""
from types import SimpleNamespace

import numpy as np

from helpers import animation as anim


def _keys():
    # Only the series the 2D warp path reads, indexable by frame_idx.
    return SimpleNamespace(
        angle_series=[0.0], zoom_series=[1.0],
        translation_x_series=[0.0], translation_y_series=[0.0],
    )


def _anim_args():
    return SimpleNamespace(use_depth_warping=False, animation_mode="2D",
                           flip_2d_perspective=False, border="replicate")


def test_warp_accepts_4d_sample_buffer():
    args = SimpleNamespace(W=8, H=8)
    sample = np.zeros((1, 3, 8, 8), dtype=np.float32)  # what generate() returns
    out, _ = anim.anim_frame_warp(sample, args, _anim_args(), _keys(), 0)
    assert out.shape == (8, 8, 3)  # decoded to HWC, warp succeeded


def test_warp_accepts_3d_cv2_image():
    args = SimpleNamespace(W=8, H=8)
    img = np.zeros((8, 8, 3), dtype=np.uint8)  # turbo/tween path passes these directly
    out, _ = anim.anim_frame_warp(img, args, _anim_args(), _keys(), 0)
    assert out.shape == (8, 8, 3)
