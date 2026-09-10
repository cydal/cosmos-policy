"""Frame-level behavior-cloning dataset for the policy: one example per executed
mujoco step, `(agentview frame, current gripper aperture) -> expert primitive id`.

Split integrity is inherited, not enforced here: mujoco-env-dataset's `combo_split`
already confines every episode entirely to one of train/val/test at generation
time, so indexing per-split episode directories (never pooling frames across
splits and re-splitting) is sufficient -- see POLICY_TRAINING.md.

The model is memoryless (one frame in, one primitive out, matching the scripted
expert it imitates), so there is nothing episode-shaped about the training loop
itself: every (frame, gripper) tuple below is an independent i.i.d. example once
indexed, and ordinary shuffled batching applies.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "mujoco-env-dataset"))
from synthbot import primitives  # noqa: E402
from synthbot import photoreal  # noqa: E402

NUM_CLASSES = primitives.NUM_PRIMITIVES


class PolicyFrameDataset(Dataset):
    def __init__(
        self,
        root: pathlib.Path,
        split: str,
        transform=None,
        augment: bool = False,
        augment_seed: int = 0,
    ):
        self.root = pathlib.Path(root)
        self.split = split
        self.transform = transform
        self.augment = augment
        self._rng = np.random.default_rng(augment_seed)

        split_dir = self.root / split
        self.index: list[tuple[pathlib.Path, int]] = []  # (episode_dir, step_i)
        self._gripper_cmd: dict[pathlib.Path, np.ndarray] = {}
        self._action_label: dict[pathlib.Path, np.ndarray] = {}

        for ep_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            steps = np.load(ep_dir / "steps.npz")
            n = int(len(steps["action_label"]))
            self._gripper_cmd[ep_dir] = steps["gripper_cmd"]
            self._action_label[ep_dir] = steps["action_label"]
            self.index.extend((ep_dir, i) for i in range(n))

    def __len__(self) -> int:
        return len(self.index)

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for ep_dir, i in self.index:
            counts[self._action_label[ep_dir][i]] += 1
        return counts

    def __getitem__(self, idx: int):
        ep_dir, i = self.index[idx]
        from PIL import Image

        frame = Image.open(ep_dir / "frames" / f"{i:03d}.png").convert("RGB")
        if self.augment:
            arr = photoreal.apply(np.asarray(frame), self._rng)
            frame = Image.fromarray(arr)
        if self.transform is not None:
            frame = self.transform(frame)

        gripper = torch.tensor([self._gripper_cmd[ep_dir][i]], dtype=torch.float32)
        label = int(self._action_label[ep_dir][i])
        return frame, gripper, label


def class_weights(counts: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights, normalized to mean 1 (keeps loss magnitude stable)."""
    freq = counts / counts.sum()
    w = 1.0 / np.clip(freq, 1e-8, None)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)
