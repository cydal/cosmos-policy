"""Cheap, dependency-free (numpy + PIL only) checks for validating action-conditioned
rollouts against ground truth, without needing real object tracking.
"""

from __future__ import annotations

import numpy as np


def to_array(frame) -> np.ndarray:
    if hasattr(frame, "convert"):
        return np.asarray(frame.convert("RGB"), dtype=np.float32)
    return np.asarray(frame, dtype=np.float32)


def color_centroid(frame, rgb_0to1: tuple[float, float, float], tol: float = 60.0):
    """Pixel-space (x, y) centroid of pixels close to `rgb_0to1` (0-1 floats), or None."""
    arr = to_array(frame)
    target = np.array(rgb_0to1, dtype=np.float32) * 255.0
    dist = np.linalg.norm(arr - target, axis=-1)
    ys, xs = np.nonzero(dist < tol)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def track_centroids(frames, rgb_0to1: tuple[float, float, float], tol: float = 60.0):
    return [color_centroid(f, rgb_0to1, tol) for f in frames]


def net_displacement(centroids: list) -> tuple[float, float] | None:
    """(dx, dy) between the first and last non-None centroid."""
    valid = [c for c in centroids if c is not None]
    if len(valid) < 2:
        return None
    (x0, y0), (x1, y1) = valid[0], valid[-1]
    return x1 - x0, y1 - y0


def frame_diff_energy(frames) -> float:
    """Mean absolute pixel change per consecutive frame pair -- a cheap "how much
    motion happened" number, comparable across rollouts of the same resolution.
    """
    arrs = [to_array(f) for f in frames]
    if len(arrs) < 2:
        return 0.0
    diffs = [np.abs(arrs[i + 1] - arrs[i]).mean() for i in range(len(arrs) - 1)]
    return float(np.mean(diffs))


def diff_centroid_x(frame_a, frame_b, tol: float = 15.0) -> float | None:
    """x-centroid of pixels that changed between two frames -- a coarse "which side
    of the image did the motion happen on" signal, useful when the moving object
    can't be color-segmented directly (e.g. a gray gripper against a gray background).
    """
    a, b = to_array(frame_a), to_array(frame_b)
    diff = np.abs(b - a).mean(axis=-1)
    ys, xs = np.nonzero(diff > tol)
    if xs.size == 0:
        return None
    return float(xs.mean())
