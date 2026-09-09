"""Windowed mujoco-env-dataset -> Cosmos3 action-conditioned training samples.

Turns each episode directory (`frames/NNN.png`, `steps.npz`, `meta.json`) produced by
mujoco-env-dataset's `generate_dataset.py` into (first_frame, 16-step action chunk,
17-frame ground-truth clip) windows, the exact shape `CosmosActionCondition(mode=
"forward_dynamics", ...)` consumes. Mirrors the conventions already exercised by
`validate_action_conditioning.py` (prompt template, `to_cosmos10`, grasp-window
finder) instead of inventing new ones, and deliberately feeds NATURAL (unscaled)
mujoco units into `to_cosmos10` -- no VALIDATION.md-style empirical scale factor --
because the point of a fresh domain id's action-projection weights is to learn
their own scale from data.

Also owns `register_mujoco_domain()`: the one piece of global, cross-cutting state
(monkeypatching diffusers' embodiment tables) both `train_lora_pilot.py` and
`eval_lora_pilot.py` need, so it lives in one place instead of two copy-pasted
blocks.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "mujoco-env-dataset"))
from synthbot import primitives  # noqa: E402
from synthbot.config import PALETTE  # noqa: E402
from synthbot.cosmos_action import to_cosmos10, CHUNK_SIZE, IMAGE_SIZE  # noqa: E402

MUJOCO_DOMAIN_NAME = "mujoco_pickplace"
MUJOCO_DOMAIN_ID = 21  # free slot; ids 0-20 are the shipped embodiments (see FINETUNING_SCOPE.md #3)
MUJOCO_RAW_ACTION_DIM = 10


def register_mujoco_domain() -> None:
    """Monkeypatch the fresh `mujoco_pickplace` domain into diffusers' embodiment tables.

    Must run before constructing any `CosmosActionCondition(domain_name="mujoco_pickplace")`
    -- its `__post_init__` validates `domain_name` against these exact module-level dicts.
    In-place `dict.__setitem__` (not rebinding the module attribute) so every place inside
    `pipeline_cosmos3_omni.py` that already holds a reference to these dicts sees the update
    too. Idempotent -- safe to call from both the training and eval scripts.
    """
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
        _EMBODIMENT_TO_DOMAIN_ID,
        _EMBODIMENT_TO_RAW_ACTION_DIM,
    )

    _EMBODIMENT_TO_DOMAIN_ID[MUJOCO_DOMAIN_NAME] = MUJOCO_DOMAIN_ID
    _EMBODIMENT_TO_RAW_ACTION_DIM[MUJOCO_DOMAIN_NAME] = MUJOCO_RAW_ACTION_DIM


def prompt_for(meta: dict) -> str:
    """Same template `validate_action_conditioning.py` uses -- keep train/eval prompts identical."""
    return (
        f"A parallel-jaw robot gripper on a tabletop reaches for a {meta['cube_colour']} cube "
        f"and lifts it, with a {meta['distractor_colour']} {meta['distractor_shape']} nearby "
        f"and a {meta['pad_colour']} target pad on the table."
    )


def find_grasp_start(action_label: np.ndarray, chunk_size: int) -> int:
    """Same heuristic as `validate_action_conditioning.py`: window centered a few steps
    before the first GRIPPER_CLOSE, clamped to a valid start index."""
    close_idx = np.flatnonzero(action_label == primitives.ID["GRIPPER_CLOSE"])
    if close_idx.size == 0:
        return 0
    start = int(close_idx[0]) - (chunk_size - 10)
    return max(0, min(start, len(action_label) - chunk_size))


class Episode:
    """Lazy accessor for one generated episode directory."""

    def __init__(self, ep_dir: pathlib.Path, control_hz: float):
        self.dir = ep_dir
        self.steps = np.load(ep_dir / "steps.npz")
        self.meta = json.loads((ep_dir / "meta.json").read_text())
        self.control_hz = control_hz
        self.num_steps = int(self.meta["num_steps"])

    def frame(self, i: int) -> Image.Image:
        return Image.open(self.dir / "frames" / f"{i:03d}.png").convert("RGB")

    def frames(self, start: int, end: int) -> list[Image.Image]:
        """Inclusive [start, end] pixel-frame range -> list of PIL frames."""
        return [self.frame(i) for i in range(start, end + 1)]

    def action10_window(self, start: int, chunk_size: int) -> np.ndarray:
        return to_cosmos10(self.steps["action_exec"][start : start + chunk_size])

    def prompt(self) -> str:
        return prompt_for(self.meta)

    def cube_rgb(self) -> tuple[float, float, float]:
        return PALETTE[self.meta["cube_colour"]]

    def grasp_start(self, chunk_size: int) -> int:
        return find_grasp_start(self.steps["action_label"], chunk_size)


def load_episodes(root: pathlib.Path, split: str) -> list[Episode]:
    dataset_meta = json.loads((root / "dataset_meta.json").read_text())
    control_hz = float(dataset_meta["action_space"]["control_hz"])
    split_dir = root / split
    ep_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    return [Episode(d, control_hz) for d in ep_dirs]


class WindowSample:
    """One training window: real first frame, real 16-action chunk, real 17-frame clip."""

    __slots__ = ("episode", "start", "chunk_size")

    def __init__(self, episode: Episode, start: int, chunk_size: int):
        self.episode = episode
        self.start = start
        self.chunk_size = chunk_size

    def first_frame(self) -> Image.Image:
        return self.episode.frame(self.start)

    def gt_frames(self) -> list[Image.Image]:
        return self.episode.frames(self.start, self.start + self.chunk_size)

    def action10(self) -> np.ndarray:
        return self.episode.action10_window(self.start, self.chunk_size)

    def prompt(self) -> str:
        return self.episode.prompt()


class MujocoWindowDataset:
    """Indexes every (episode, start) window at `stride` across every episode in a split.

    Uniform coverage of the whole trajectory (reach / descend / grasp / lift / carry / place),
    not just the grasp window -- the point of this pilot is teaching the model to trust our
    actions in general, not just during one phase.
    """

    def __init__(self, root: pathlib.Path, split: str, chunk_size: int = CHUNK_SIZE, stride: int = 8):
        self.chunk_size = chunk_size
        self.episodes = load_episodes(root, split)
        self.index: list[tuple[Episode, int]] = []
        for ep in self.episodes:
            last_start = ep.num_steps - chunk_size
            if last_start < 0:
                continue
            for start in range(0, last_start + 1, stride):
                self.index.append((ep, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> WindowSample:
        ep, start = self.index[i]
        return WindowSample(ep, start, self.chunk_size)


assert IMAGE_SIZE == 256, "resolution_tier=256 assumes mujoco-env-dataset's default --image-size"
