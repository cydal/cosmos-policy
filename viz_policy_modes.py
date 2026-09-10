#!/usr/bin/env python
"""Both visualization modes from POLICY_TRAINING.md, run with the CURRENT (first-
pass, not-yet-improved) policy checkpoint, so "what would more training improve"
has a concrete before-picture rather than an abstract percentage.

Mode 1 -- real policy, parallel imagination: run the policy closed-loop in the
REAL simulator (this is genuine task performance), then re-imagine each 16-step
chunk of that real trajectory through fine-tuned Cosmos, anchored fresh to the
real starting frame each chunk (no drift compounding across chunks).

Mode 2 -- pure imagination: the policy only ever sees Cosmos's own last imagined
frame after the first chunk, fully autoregressive, no real frames after frame 0.
Cosmos only accepts a full 16-action chunk at once (no intra-chunk feedback), and
the policy is a single-frame reactive model with no way to plan 16 steps ahead on
its own -- the necessary simplification here is that the policy picks ONE
primitive at each chunk boundary and that primitive is held for the whole chunk,
which is also a reasonable approximation of how the real expert behaves (a single
primitive is very often sustained for many consecutive real steps anyway).

Output: a 3-row comparison video (real / mode-1 parallel imagination / mode-2 pure
imagination), so the anchored-vs-drifting difference is visible directly.

    source env.sh
    python viz_policy_modes.py --episode ep000880 --split test --chunks 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "mujoco-env-dataset"))
from synthbot import config as C  # noqa: E402
from synthbot import primitives  # noqa: E402
from synthbot.cosmos_action import to_cosmos10  # noqa: E402
from synthbot.env import PickPlaceEnv  # noqa: E402

import config
import media
import world
from eval_policy_closed_loop import load_config, policy_primitive
from lora_data import MUJOCO_DOMAIN_NAME, prompt_for, register_mujoco_domain
from eval_lora_pilot import load_checkpoint
from policy_model import PolicyNet, imagenet_transform

DATASET_ROOT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/lora_scaleup/dataset")
POLICY_CHECKPOINT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/policy/checkpoints/latest.pt")
COSMOS_CHECKPOINT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/lora_scaleup/checkpoints/step_21483")
CHUNK_SIZE = 16
RESOLUTION_TIER = 256


def real_closed_loop(model, transform, device, cfg: C.EpisodeConfig, max_chunks: int, max_steps: int = 160):
    """Mode 1's ground truth: real frames + real 7-D actions, chunked into 16-step windows."""
    env = PickPlaceEnv(cfg, image_size=256, wrist_cam=False)
    frames, actions7 = [], []
    try:
        obs = env.reset()
        frames.append(obs.agentview)
        for _ in range(max_steps):
            primitive = policy_primitive(model, transform, device, obs.agentview, env.gripper_cmd)
            obs, act = env.step(primitive)
            frames.append(obs.agentview)
            actions7.append(act)
            if env.cube_fallen or len(actions7) >= max_chunks * CHUNK_SIZE:
                break
    finally:
        env.close()
    n_full_chunks = len(actions7) // CHUNK_SIZE
    return frames[: n_full_chunks * CHUNK_SIZE + 1], np.array(actions7[: n_full_chunks * CHUNK_SIZE]), n_full_chunks


def build_meta(action10, first_frame, prompt, fps) -> dict:
    return {
        "chunks": torch.tensor(action10, dtype=torch.float32)[None],
        "first_frame": first_frame,
        "prompt": prompt,
        "domain_name": MUJOCO_DOMAIN_NAME,
        "action_chunk_size": CHUNK_SIZE,
        "image_size": RESOLUTION_TIER,
        "fps": fps,
        "view_point": "third_person_view",
    }


def mode1_parallel_imagination(pipe, real_frames, real_actions7, prompt, fps, n_chunks, steps, seed):
    """Re-imagine each chunk fresh from the REAL starting frame -- no drift compounding."""
    imagined = [real_frames[0]]
    for c in range(n_chunks):
        start_frame = Image.fromarray(real_frames[c * CHUNK_SIZE])
        chunk7 = real_actions7[c * CHUNK_SIZE : (c + 1) * CHUNK_SIZE]
        meta = build_meta(to_cosmos10(chunk7), start_frame, prompt, fps)
        frames, _ = world.rollout(pipe, meta, num_chunks=1, steps=steps, seed=seed)
        imagined.extend(frames[1:])
    return imagined


def mode2_pure_imagination(pipe, model, transform, device, first_frame_arr, gripper0, prompt, fps, n_chunks, steps, seed):
    """Fully autoregressive: the policy only ever sees Cosmos's own last frame after
    chunk 0. One primitive per chunk boundary, held for the whole chunk (see module
    docstring for why)."""
    current_frame = first_frame_arr
    gripper_state = float(gripper0)
    imagined = [current_frame]
    for c in range(n_chunks):
        primitive = policy_primitive(model, transform, device, current_frame, gripper_state)
        action7 = primitives.to_action(primitive, gripper_state)
        gripper_state = float(action7[6])
        chunk7 = np.tile(action7, (CHUNK_SIZE, 1))
        meta = build_meta(to_cosmos10(chunk7), Image.fromarray(current_frame), prompt, fps)
        frames, _ = world.rollout(pipe, meta, num_chunks=1, steps=steps, seed=seed)
        imagined.extend(frames[1:])
        current_frame = np.asarray(frames[-1])
    return imagined


def strip_rows(rows: list[list], count: int = 8) -> Image.Image:
    n = min(len(r) for r in rows)
    idx = np.linspace(0, n - 1, count).round().astype(int)
    to_arr = lambda f: np.asarray(f.convert("RGB") if hasattr(f, "convert") else f)
    w, h = to_arr(rows[0][0]).shape[1], to_arr(rows[0][0]).shape[0]
    out = Image.new("RGB", (w * count, h * len(rows) + 20 * (len(rows) - 1)), "white")
    for r, row in enumerate(rows):
        for col, i in enumerate(idx):
            out.paste(Image.fromarray(to_arr(row[i])), (col * w, r * (h + 20)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="ep000880")
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("qa_final/policy_modes"))
    ap.add_argument("--policy-checkpoint", type=pathlib.Path, default=POLICY_CHECKPOINT)
    ap.add_argument("--dataset-root", type=pathlib.Path, default=DATASET_ROOT)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    register_mujoco_domain()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNet(pretrained=False).to(device)
    policy.load_state_dict(torch.load(args.policy_checkpoint, map_location=device))
    policy.eval()
    transform = imagenet_transform()

    ep_dir = args.dataset_root / args.split / args.episode
    cfg = load_config(ep_dir / "meta.json")
    prompt = prompt_for(json.loads((ep_dir / "meta.json").read_text()))
    fps = json.loads((args.dataset_root / "dataset_meta.json").read_text())["action_space"]["control_hz"]

    print(f"--- running real closed-loop policy rollout: {args.episode} ({args.split}) ---")
    real_frames, real_actions7, n_chunks = real_closed_loop(policy, transform, device, cfg, args.chunks)
    n_chunks = min(n_chunks, args.chunks)
    print(f"got {n_chunks} full 16-step chunks ({n_chunks * CHUNK_SIZE} real steps)")
    media.write_mp4([Image.fromarray(f) for f in real_frames], args.out / "mode0_real.mp4", fps=int(fps))

    print("--- loading fine-tuned Cosmos ---")
    pipe = world.load(config.EDGE)
    load_checkpoint(pipe, COSMOS_CHECKPOINT)

    print("--- mode 1: parallel imagination (anchored each chunk) ---")
    mode1 = mode1_parallel_imagination(pipe, real_frames, real_actions7, prompt, fps, n_chunks, args.steps, args.seed)
    media.write_mp4(mode1, args.out / "mode1_parallel_imagination.mp4", fps=int(fps))

    print("--- mode 2: pure autoregressive imagination ---")
    gripper0 = float(cfg.gripper_start)
    mode2 = mode2_pure_imagination(
        pipe, policy, transform, device, real_frames[0], gripper0, prompt, fps, n_chunks, args.steps, args.seed
    )
    media.write_mp4([Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in mode2], args.out / "mode2_pure_imagination.mp4", fps=int(fps))

    strip_rows([real_frames, mode1, mode2]).save(args.out / "comparison_strip.png")
    print(f"\nwrote mode0_real.mp4, mode1_parallel_imagination.mp4, mode2_pure_imagination.mp4, comparison_strip.png -> {args.out}/")
    print("comparison_strip.png rows: real policy rollout / mode-1 (anchored) / mode-2 (pure, drifting)")


if __name__ == "__main__":
    main()
