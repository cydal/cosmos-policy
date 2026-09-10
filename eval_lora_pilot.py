#!/usr/bin/env python
"""Before/after evaluation for the LoRA fine-tuning pilot, on held-out VAL episodes.

Mirrors `validate_action_conditioning.py`'s methodology (grasp_bridge vs stay_control,
frame_diff_energy, cube net displacement) but pointed at the new `mujoco_pickplace`
domain (id 21) instead of `bridge_orig_lerobot`/`umi`, across several VAL episodes
(geometrically disjoint from train per mujoco-env-dataset's `combo_split` -- the real
generalization check per FINETUNING_SCOPE.md section 6 risk 5).

Run once with no --checkpoint (baseline: untrained domain 21, random action-projection
weights, no LoRA) and once with --checkpoint pointing at a saved step_XXXXX directory,
then diff the two summary.json files by hand -- that side-by-side is the actual
pilot verdict.

Usage:
    source env.sh
    python eval_lora_pilot.py --tag baseline
    python eval_lora_pilot.py --tag trained --checkpoint /opt/dlami/nvme/cosmos-policy/lora_pilot/checkpoints/step_00800
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

import numpy as np
import torch

import analysis
import config
import media
import world
from lora_data import MUJOCO_DOMAIN_NAME, load_episodes, register_mujoco_domain, find_grasp_start
from train_lora_pilot import build_lora_config

DATASET_ROOT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/lora_pilot/dataset")
OUT_ROOT = pathlib.Path("out/lora_pilot_eval")
CHUNK_SIZE = 16
RESOLUTION_TIER = 256
N_VAL_EPISODES = 25


def load_checkpoint(pipe, checkpoint: pathlib.Path) -> None:
    """Attach the trained LoRA adapter + domain-embedding rows to `pipe.transformer`."""
    from safetensors.torch import load_file
    from peft.utils import set_peft_model_state_dict

    lora_config = build_lora_config()
    pipe.transformer.add_adapter(lora_config)
    lora_state = load_file(str(checkpoint / "pytorch_lora_weights.safetensors"))
    set_peft_model_state_dict(pipe.transformer, lora_state, adapter_name="default")

    domain_state = torch.load(checkpoint / "domain_embedding.pt", map_location="cpu")
    pipe.transformer.action_proj_in.load_state_dict(domain_state["action_proj_in"])
    pipe.transformer.action_proj_out.load_state_dict(domain_state["action_proj_out"])
    pipe.transformer.action_proj_in.to(pipe.transformer.dtype)
    pipe.transformer.action_proj_out.to(pipe.transformer.dtype)
    log.info("loaded checkpoint from %s", checkpoint)


def build_meta(action10: np.ndarray, first_frame, prompt: str, fps: float) -> dict:
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


def build_val_cases(dataset_root: pathlib.Path, n_episodes: int, split: str = "val") -> list[dict]:
    from synthbot.cosmos_action import to_cosmos10

    episodes = load_episodes(dataset_root, split)[:n_episodes]
    cases = []
    for ep in episodes:
        grasp_start = ep.grasp_start(CHUNK_SIZE)
        action_exec = ep.steps["action_exec"]
        gripper_cmd = ep.steps["gripper_cmd"]
        grasp_action = action_exec[grasp_start : grasp_start + CHUNK_SIZE]
        stay_action = np.tile(
            [0, 0, 0, 0, 0, 0, gripper_cmd[grasp_start]], (CHUNK_SIZE, 1)
        ).astype(np.float32)
        prompt = ep.prompt()
        first_frame = ep.frame(grasp_start)
        gt = ep.frames(grasp_start, grasp_start + CHUNK_SIZE)

        cases.append(dict(
            name=f"{ep.dir.name}_grasp",
            meta=build_meta(to_cosmos10(grasp_action), first_frame, prompt, ep.control_hz),
            gt=gt,
            cube_rgb=ep.cube_rgb(),
        ))
        cases.append(dict(
            name=f"{ep.dir.name}_stay",
            meta=build_meta(to_cosmos10(stay_action), first_frame, prompt, ep.control_hz),
            gt=gt,
            cube_rgb=ep.cube_rgb(),
        ))
    return cases


def run_case(pipe, case: dict, out_dir: pathlib.Path, steps: int, seed: int) -> dict:
    frames, per_chunk = world.rollout(pipe, case["meta"], num_chunks=1, steps=steps, seed=seed)
    case_dir = out_dir / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    media.write_mp4(frames, case_dir / "predicted.mp4", fps=int(case["meta"]["fps"]))
    media.write_mp4(case["gt"], case_dir / "ground_truth.mp4", fps=int(case["meta"]["fps"]))

    metrics = {
        "seconds": round(sum(per_chunk), 1),
        "frame_diff_energy": round(analysis.frame_diff_energy(frames), 3),
        "gt_frame_diff_energy": round(analysis.frame_diff_energy(case["gt"]), 3),
    }
    pred_c = analysis.track_centroids(frames, case["cube_rgb"])
    gt_c = analysis.track_centroids(case["gt"], case["cube_rgb"])
    metrics["predicted_cube_net_disp_xy"] = analysis.net_displacement(pred_c)
    metrics["gt_cube_net_disp_xy"] = analysis.net_displacement(gt_c)
    return metrics


def summarize(results: dict) -> dict:
    """Aggregate the two headline signals from VALIDATION.md's methodology:
    (1) grasp/stay frame_diff_energy ratio -- is the model reading the action at all;
    (2) cube-displacement sign agreement with ground truth on the real-grasp case."""
    ratios, sign_matches = [], []
    for name, m in results.items():
        if not name.endswith("_grasp"):
            continue
        stay = results.get(name[: -len("_grasp")] + "_stay")
        if stay and stay["frame_diff_energy"] > 1e-6:
            ratios.append(m["frame_diff_energy"] / stay["frame_diff_energy"])
        pred, gt = m["predicted_cube_net_disp_xy"], m["gt_cube_net_disp_xy"]
        if pred is not None and gt is not None:
            dot = pred[0] * gt[0] + pred[1] * gt[1]
            sign_matches.append(dot > 0)
    return {
        "n_grasp_cases": sum(1 for n in results if n.endswith("_grasp")),
        "mean_grasp_over_stay_energy_ratio": round(float(np.mean(ratios)), 3) if ratios else None,
        "cube_direction_agreement_fraction": round(float(np.mean(sign_matches)), 3) if sign_matches else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="label for this eval run, e.g. baseline / trained")
    ap.add_argument("--checkpoint", type=pathlib.Path, default=None)
    ap.add_argument("--n-episodes", type=int, default=N_VAL_EPISODES)
    ap.add_argument("--split", default="val")
    ap.add_argument("--dataset-root", type=pathlib.Path, default=DATASET_ROOT)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    register_mujoco_domain()
    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"guard: {world.guard_memory(config.EDGE)}")
    pipe = world.load(config.EDGE)
    if args.checkpoint is not None:
        load_checkpoint(pipe, args.checkpoint)

    cases = build_val_cases(args.dataset_root, args.n_episodes, args.split)
    results = {}
    for case in cases:
        print(f"=== {case['name']} ===")
        metrics = run_case(pipe, case, out_dir, steps=args.steps, seed=args.seed)
        results[case["name"]] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    summary = summarize(results)
    print("\n=== summary:", args.tag, "===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    (out_dir / "summary.json").write_text(json.dumps({"per_case": results, "summary": summary}, indent=2))
    print(f"\nwrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
