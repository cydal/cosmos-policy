"""One concrete, viewable example for a go/no-go read: a real mujoco frame + its
real action sequence, fed through (a) stock Cosmos3-Edge and (b) the fine-tuned
checkpoint -- same raw (unscaled) action values both times, so the only thing
that differs between (a) and (b) is the model, not the input.

Writes into --out (default qa_final/):
  mujoco_first_frame.png    the real conditioning frame
  action_sequence.csv       the 16-step action chunk, human-readable
  ground_truth.mp4          what actually happened in the mujoco sim
  cosmos_initial.mp4        stock Cosmos3-Edge, domain=bridge_orig_lerobot, raw action
  cosmos_finetuned.mp4      LoRA checkpoint, domain=mujoco_pickplace, same raw action
  comparison_strip.png      all three video rows, side by side, for a quick look

    source env.sh
    python final_comparison.py --episode ep000886 --split test
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "mujoco-env-dataset"))
from synthbot import primitives  # noqa: E402
from synthbot.cosmos_action import to_cosmos10  # noqa: E402

import analysis
import config
import media
import world
from eval_lora_pilot import load_checkpoint
from lora_data import MUJOCO_DOMAIN_NAME, load_episodes, register_mujoco_domain

DATASET_ROOT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/lora_scaleup/dataset")
CHECKPOINT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/lora_scaleup/checkpoints/step_21483")
CHUNK_SIZE = 16
RESOLUTION_TIER = 256


def build_meta(action10: np.ndarray, first_frame, prompt: str, domain: str, fps: float) -> dict:
    return {
        "chunks": torch.tensor(action10, dtype=torch.float32)[None],
        "first_frame": first_frame,
        "prompt": prompt,
        "domain_name": domain,
        "action_chunk_size": CHUNK_SIZE,
        "image_size": RESOLUTION_TIER,
        "fps": fps,
        "view_point": "third_person_view",
    }


def strip(rows: list[list], w: int, h: int, count: int = 6) -> Image.Image:
    n = min(len(r) for r in rows)
    idx = np.linspace(0, n - 1, count).round().astype(int)
    out = Image.new("RGB", (w * count, h * len(rows) + 20 * (len(rows) - 1)), "white")
    for r, row in enumerate(rows):
        for col, i in enumerate(idx):
            out.paste(row[i], (col * w, r * (h + 20)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="ep000886")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("qa_final"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    register_mujoco_domain()

    episodes = {ep.dir.name: ep for ep in load_episodes(DATASET_ROOT, args.split)}
    ep = episodes[args.episode]
    grasp_start = ep.grasp_start(CHUNK_SIZE)
    action7 = ep.steps["action_exec"][grasp_start : grasp_start + CHUNK_SIZE]
    action_label = ep.steps["action_label"][grasp_start : grasp_start + CHUNK_SIZE]
    is_noise = ep.steps["is_noise"][grasp_start : grasp_start + CHUNK_SIZE]
    action10 = to_cosmos10(action7)
    first_frame = ep.frame(grasp_start)
    gt_frames = ep.frames(grasp_start, grasp_start + CHUNK_SIZE)
    prompt = ep.prompt()

    first_frame.save(args.out / "mujoco_first_frame.png")
    with open(args.out / "action_sequence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "expert_label", "is_noise", "dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper_0to1"])
        for i in range(CHUNK_SIZE):
            w.writerow([i, primitives.NAMES[action_label[i]], bool(is_noise[i]), *[f"{v:.5f}" for v in action7[i]]])
    media.write_mp4(gt_frames, args.out / "ground_truth.mp4", fps=int(ep.control_hz))
    print(f"episode={args.episode} split={args.split} grasp_start={grasp_start}")
    print(f"prompt: {prompt}")
    print(f"wrote mujoco_first_frame.png, action_sequence.csv, ground_truth.mp4")

    # -- (a) stock Cosmos3-Edge, no fine-tuning, the pretrained domain that best matches
    # our fixed-camera setup, same raw action values we'll use for (b).
    print("\n--- loading stock Cosmos3-Edge ---")
    pipe = world.load(config.EDGE)
    meta_initial = build_meta(action10, first_frame, prompt, "bridge_orig_lerobot", ep.control_hz)
    frames_initial, _ = world.rollout(pipe, meta_initial, num_chunks=1, steps=args.steps, seed=args.seed)
    media.write_mp4(frames_initial, args.out / "cosmos_initial.mp4", fps=int(ep.control_hz))
    e_initial = analysis.frame_diff_energy(frames_initial)
    print(f"cosmos_initial: frame_diff_energy={e_initial:.3f}")
    del pipe
    torch.cuda.empty_cache()

    # -- (b) same checkpoint family, LoRA + the fine-tuned domain, same raw action values.
    print("\n--- loading fine-tuned Cosmos3-Edge (LoRA, domain=mujoco_pickplace) ---")
    pipe = world.load(config.EDGE)
    load_checkpoint(pipe, CHECKPOINT)
    meta_trained = build_meta(action10, first_frame, prompt, MUJOCO_DOMAIN_NAME, ep.control_hz)
    frames_trained, _ = world.rollout(pipe, meta_trained, num_chunks=1, steps=args.steps, seed=args.seed)
    media.write_mp4(frames_trained, args.out / "cosmos_finetuned.mp4", fps=int(ep.control_hz))
    e_trained = analysis.frame_diff_energy(frames_trained)
    print(f"cosmos_finetuned: frame_diff_energy={e_trained:.3f}")

    w, h = first_frame.size
    strip([gt_frames, frames_initial, frames_trained], w, h).save(args.out / "comparison_strip.png")
    print(f"\nwrote cosmos_initial.mp4, cosmos_finetuned.mp4, comparison_strip.png -> {args.out}/")
    print("comparison_strip.png rows: ground truth / stock Cosmos3-Edge / fine-tuned Cosmos3-Edge")


if __name__ == "__main__":
    main()
