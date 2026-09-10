#!/usr/bin/env python
"""Closed-loop evaluation: drive PickPlaceEnv with the TRAINED POLICY's own
predictions (not the scripted expert), and measure real task success. This is the
metric that actually matters -- see POLICY_TRAINING.md's "held-out accuracy is a
trap" section. No Cosmos anywhere in this script.

Reuses existing episode configs from lora_scaleup/dataset's meta.json files, so
each policy rollout is a direct, paired comparison against that exact layout's
already-recorded scripted-expert outcome (same cube/pad/distractor/camera/lighting
-- only the controller differs).

    source env.sh
    python eval_policy_closed_loop.py --split test --n-episodes 30
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
from synthbot.env import PickPlaceEnv  # noqa: E402

from policy_model import PolicyNet, imagenet_transform

DATASET_ROOT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/lora_scaleup/dataset")
CHECKPOINT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/policy/checkpoints/latest.pt")
_CFG_FIELDS = {f.name for f in dataclasses.fields(C.EpisodeConfig)}


def load_config(meta_path: pathlib.Path) -> C.EpisodeConfig:
    meta = json.loads(meta_path.read_text())
    kwargs = {k: v for k, v in meta.items() if k in _CFG_FIELDS}
    for k in ("cube_pos", "pad_pos", "distractor_pos", "ee_start", "light_pos", "table_colour"):
        kwargs[k] = tuple(kwargs[k])
    return C.EpisodeConfig(**kwargs)


@torch.no_grad()
def policy_primitive(model, transform, device, frame: np.ndarray, gripper_cmd: float) -> int:
    img = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
    g = torch.tensor([[gripper_cmd]], dtype=torch.float32, device=device)
    return int(model(img, g).argmax(dim=-1).item())


def run_episode(model, transform, device, cfg: C.EpisodeConfig, max_steps: int = 160, stay_patience: int = 3) -> dict:
    env = PickPlaceEnv(cfg, image_size=256, wrist_cam=False)
    try:
        obs = env.reset()
        stay_count = 0
        n_steps = 0
        for _ in range(max_steps):
            primitive = policy_primitive(model, transform, device, obs.agentview, env.gripper_cmd)
            if primitive == primitives.ID["STAY"]:
                stay_count += 1
                if stay_count >= stay_patience:
                    break
            else:
                stay_count = 0
            obs, _ = env.step(primitive)
            n_steps += 1
            if env.cube_fallen:
                break
        success = env.cube_on_pad
        cube_fallen = env.cube_fallen
    finally:
        env.close()
    return {"success": bool(success), "cube_fallen": bool(cube_fallen), "steps": n_steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=pathlib.Path, default=DATASET_ROOT)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=CHECKPOINT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=160)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    transform = imagenet_transform()

    ep_dirs = sorted((args.dataset_root / args.split).iterdir())[: args.n_episodes]
    results = []
    expert_success = []
    for ep_dir in ep_dirs:
        cfg = load_config(ep_dir / "meta.json")
        expert_meta = json.loads((ep_dir / "meta.json").read_text())
        r = run_episode(model, transform, device, cfg, max_steps=args.max_steps)
        r["episode"] = ep_dir.name
        r["expert_success"] = expert_meta["success"]
        results.append(r)
        expert_success.append(expert_meta["success"])
        print(f"{ep_dir.name}: policy_success={r['success']} cube_fallen={r['cube_fallen']} "
              f"steps={r['steps']}  (expert_success={expert_meta['success']})")

    policy_success = np.mean([r["success"] for r in results])
    policy_fallen = np.mean([r["cube_fallen"] for r in results])
    expert_rate = np.mean(expert_success)
    print(f"\n=== split={args.split} n={len(results)} ===")
    print(f"policy success rate: {policy_success:.3f}")
    print(f"policy cube_fallen rate: {policy_fallen:.3f}")
    print(f"expert success rate (same configs): {expert_rate:.3f}")

    out = pathlib.Path("out/policy_eval") / args.split
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "per_episode": results,
        "policy_success_rate": float(policy_success),
        "policy_cube_fallen_rate": float(policy_fallen),
        "expert_success_rate": float(expert_rate),
    }, indent=2))
    print(f"wrote {out}/results.json")


if __name__ == "__main__":
    main()
