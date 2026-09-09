"""Stress-test the mujoco -> Cosmos action conversion before trusting it for training.

One successful grasp-window rollout (run_mujoco_rollout.py) isn't enough evidence:
it doesn't tell us whether Cosmos is actually reading our action tensor (vs. just
hallucinating plausible-looking robot motion from the prompt), whether the sign/axis
conventions are right, whether the domain choice matters, or whether it was a fluke
of one scene. This runs a small test matrix, loading the pipeline once:

  grasp_bridge     the original descend->grasp->lift window, domain=bridge_orig_lerobot
  grasp_umi        same frame + action, domain=umi -- does the domain choice matter?
  stay_control      all-STAY (zero motion) action from the SAME first frame -- if this
                    moves as much as grasp_bridge, the model is ignoring our actions
  lateral           a pure-translation window (no rotation, no grasp), steps [0:16]
  lateral_reversed  the same window with dx/dy negated -- predicted motion should
                    shift the opposite way if the sign/frame convention is right
  grasp_ep2         the grasp window of a second, independently-generated episode

Metrics, since "looks right" isn't a substitute for a number:
  - cube color-centroid net displacement (predicted vs. ground truth), where a cube
    actually moves (the grasp_* cases)
  - frame_diff_energy: mean pixel change per frame, comparing stay_control against
    the real-motion cases
  - diff_centroid_x: which side of the frame changed, comparing lateral against
    lateral_reversed

    source env.sh
    python validate_action_conditioning.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "mujoco-env-dataset"))
from synthbot import primitives  # noqa: E402
from synthbot.config import PALETTE  # noqa: E402
from synthbot.cosmos_action import to_cosmos10  # noqa: E402

import config
import media
import world
import analysis

SAMPLE_ROOT = pathlib.Path("/tmp/mujoco_sample/train")
OUT_ROOT = pathlib.Path("out/validation")
CHUNK_SIZE = 16


def load_episode(ep_dir: pathlib.Path):
    steps = np.load(ep_dir / "steps.npz")
    meta = json.loads((ep_dir / "meta.json").read_text())
    control_hz = json.loads((ep_dir.parent.parent / "dataset_meta.json").read_text())["action_space"]["control_hz"]
    return steps, meta, control_hz


def frame_at(ep_dir: pathlib.Path, i: int) -> Image.Image:
    return Image.open(ep_dir / "frames" / f"{i:03d}.png").convert("RGB")


def gt_window(ep_dir: pathlib.Path, start: int, end: int) -> list[Image.Image]:
    return [frame_at(ep_dir, i) for i in range(start, end + 1)]


def find_grasp_start(action_label: np.ndarray, chunk_size: int) -> int:
    close_idx = np.flatnonzero(action_label == primitives.ID["GRIPPER_CLOSE"])
    if close_idx.size == 0:
        return 0
    start = int(close_idx[0]) - (chunk_size - 10)
    return max(0, min(start, len(action_label) - chunk_size))


def prompt_for(meta: dict) -> str:
    return (
        f"A parallel-jaw robot gripper on a tabletop reaches for a {meta['cube_colour']} cube "
        f"and lifts it, with a {meta['distractor_colour']} {meta['distractor_shape']} nearby "
        f"and a {meta['pad_colour']} target pad on the table."
    )


def make_meta(action10: np.ndarray, first_frame: Image.Image, prompt: str, domain_name: str, fps: float) -> dict:
    return {
        "chunks": torch.tensor(action10, dtype=torch.float32)[None],  # [1, 16, 10]
        "first_frame": first_frame,
        "prompt": prompt,
        "domain_name": domain_name,
        "action_chunk_size": CHUNK_SIZE,
        "image_size": 256,
        "fps": fps,
        "view_point": "third_person_view",
    }


def build_cases() -> list[dict]:
    steps0, meta0, hz0 = load_episode(SAMPLE_ROOT / "ep000000")
    steps1, meta1, hz1 = load_episode(SAMPLE_ROOT / "ep000001")
    al0 = steps0["action_label"]
    ax0 = steps0["action_exec"]

    grasp_start0 = find_grasp_start(al0, CHUNK_SIZE)
    grasp0 = ax0[grasp_start0 : grasp_start0 + CHUNK_SIZE]
    prompt0 = prompt_for(meta0)
    cube0_rgb = PALETTE[meta0["cube_colour"]]

    lateral_action = ax0[0:CHUNK_SIZE]
    lateral_reversed = lateral_action.copy()
    lateral_reversed[:, 0:2] *= -1  # flip dx, dy only

    al1 = steps1["action_label"]
    ax1 = steps1["action_exec"]
    grasp_start1 = find_grasp_start(al1, CHUNK_SIZE)
    grasp1 = ax1[grasp_start1 : grasp_start1 + CHUNK_SIZE]
    prompt1 = prompt_for(meta1)
    cube1_rgb = PALETTE[meta1["cube_colour"]]

    cases = [
        dict(
            name="grasp_bridge",
            meta=make_meta(to_cosmos10(grasp0), frame_at(SAMPLE_ROOT / "ep000000", grasp_start0), prompt0, "bridge_orig_lerobot", hz0),
            gt=gt_window(SAMPLE_ROOT / "ep000000", grasp_start0, grasp_start0 + CHUNK_SIZE),
            cube_rgb=cube0_rgb,
        ),
        dict(
            name="grasp_umi",
            meta=make_meta(to_cosmos10(grasp0), frame_at(SAMPLE_ROOT / "ep000000", grasp_start0), prompt0, "umi", hz0),
            gt=gt_window(SAMPLE_ROOT / "ep000000", grasp_start0, grasp_start0 + CHUNK_SIZE),
            cube_rgb=cube0_rgb,
        ),
        dict(
            name="stay_control",
            meta=make_meta(
                to_cosmos10(np.tile([0, 0, 0, 0, 0, 0, steps0["gripper_cmd"][grasp_start0]], (CHUNK_SIZE, 1)).astype(np.float32)),
                frame_at(SAMPLE_ROOT / "ep000000", grasp_start0),
                prompt0,
                "bridge_orig_lerobot",
                hz0,
            ),
            gt=gt_window(SAMPLE_ROOT / "ep000000", grasp_start0, grasp_start0 + CHUNK_SIZE),
            cube_rgb=cube0_rgb,
        ),
        dict(
            name="lateral",
            meta=make_meta(to_cosmos10(lateral_action), frame_at(SAMPLE_ROOT / "ep000000", 0), prompt0, "bridge_orig_lerobot", hz0),
            gt=gt_window(SAMPLE_ROOT / "ep000000", 0, CHUNK_SIZE),
            cube_rgb=None,
        ),
        dict(
            name="lateral_reversed",
            meta=make_meta(to_cosmos10(lateral_reversed), frame_at(SAMPLE_ROOT / "ep000000", 0), prompt0, "bridge_orig_lerobot", hz0),
            gt=None,
            cube_rgb=None,
        ),
        dict(
            name="grasp_ep2",
            meta=make_meta(to_cosmos10(grasp1), frame_at(SAMPLE_ROOT / "ep000001", grasp_start1), prompt1, "bridge_orig_lerobot", hz1),
            gt=gt_window(SAMPLE_ROOT / "ep000001", grasp_start1, grasp_start1 + CHUNK_SIZE),
            cube_rgb=cube1_rgb,
        ),
    ]
    return cases


def run_case(pipe, case: dict, steps: int, seed: int) -> dict:
    frames, per_chunk = world.rollout(pipe, case["meta"], num_chunks=1, steps=steps, seed=seed)
    out_dir = OUT_ROOT / case["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    media.write_mp4(frames, out_dir / "predicted.mp4", fps=int(case["meta"]["fps"]))

    metrics: dict = {"seconds": round(sum(per_chunk), 1), "frame_diff_energy": round(analysis.frame_diff_energy(frames), 3)}

    if case["gt"] is not None:
        media.write_mp4(case["gt"], out_dir / "ground_truth.mp4", fps=int(case["meta"]["fps"]))
        metrics["gt_frame_diff_energy"] = round(analysis.frame_diff_energy(case["gt"]), 3)
        strip = _strip(case["gt"], frames)
        strip.save(out_dir / "comparison_strip.png")

    if case["cube_rgb"] is not None:
        pred_c = analysis.track_centroids(frames, case["cube_rgb"])
        pred_disp = analysis.net_displacement(pred_c)
        metrics["predicted_cube_net_disp_xy"] = pred_disp
        if case["gt"] is not None:
            gt_c = analysis.track_centroids(case["gt"], case["cube_rgb"])
            metrics["gt_cube_net_disp_xy"] = analysis.net_displacement(gt_c)

    metrics["diff_centroid_x_first_last"] = analysis.diff_centroid_x(frames[0], frames[-1])
    return metrics, frames


def _strip(gt_frames, pred_frames, count: int = 6):
    idx = np.linspace(0, min(len(gt_frames), len(pred_frames)) - 1, count).round().astype(int)
    w, h = gt_frames[0].size
    strip = Image.new("RGB", (w * count, h * 2 + 20), "white")
    for col, i in enumerate(idx):
        strip.paste(gt_frames[i], (col * w, 0))
        strip.paste(pred_frames[i], (col * w, h + 20))
    return strip


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"guard: {world.guard_memory(config.DEFAULT_REPO)}")
    t0 = time.monotonic()
    pipe = world.load(config.DEFAULT_REPO)
    print(f"loaded in {(time.monotonic() - t0) / 60:.1f} min")

    cases = build_cases()
    results = {}
    for case in cases:
        print(f"\n=== {case['name']} ({case['meta']['domain_name']}) ===")
        metrics, _ = run_case(pipe, case, steps=30, seed=0)
        results[case["name"]] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    (OUT_ROOT / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_ROOT}/summary.json and per-case dirs under {OUT_ROOT}/")


if __name__ == "__main__":
    main()
