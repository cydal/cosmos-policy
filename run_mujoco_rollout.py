"""Feed one mujoco-env-dataset action chunk through Cosmos3-Edge and compare.

    source env.sh
    python run_mujoco_rollout.py --sample /path/to/cosmos_sample

`--sample` is a directory produced by mujoco-env-dataset's
`scripts/export_cosmos_sample.py`: `first_frame.png` + `action_chunks.json` (one
16-step, 10-D action chunk, converted from the dataset's native 7-D action via
`synthbot/cosmos_action.py`) + `ground_truth_frames/` (the episode's own real frames
over that same window).

Writes, into `--out` (default `out/mujoco_rollout`):
  - `cosmos_predicted.mp4`   Cosmos's rollout from the first frame + our actions
  - `ground_truth.mp4`       what actually happened in the mujoco sim
  - `comparison_strip.png`   a few frames of both, stacked, for a quick look
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time

import config
import media
import world

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def load_sample(sample_dir: pathlib.Path) -> dict:
    import torch
    from PIL import Image

    meta = json.loads((sample_dir / "action_chunks.json").read_text())
    meta["chunks"] = torch.tensor(meta["action_chunks"], dtype=torch.float32)
    meta["first_frame"] = Image.open(sample_dir / "first_frame.png").convert("RGB")
    return meta


def comparison_strip(ground_truth_frames, predicted_frames, count: int = 6) -> "Image.Image":
    from PIL import Image
    import numpy as np

    n = min(len(ground_truth_frames), len(predicted_frames))
    idx = np.linspace(0, n - 1, count).round().astype(int)
    gt = [ground_truth_frames[i] for i in idx]
    pred = [predicted_frames[i] for i in idx]

    w, h = gt[0].size
    strip = Image.new("RGB", (w * count, h * 2 + 20), "white")
    for col, (g, p) in enumerate(zip(gt, pred)):
        strip.paste(g, (col * w, 0))
        strip.paste(p, (col * w, h + 20))
    return strip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=pathlib.Path, required=True)
    ap.add_argument("--repo", default=config.DEFAULT_REPO)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out/mujoco_rollout"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta = load_sample(args.sample)
    print(f"domain={meta['domain_name']}  fps={meta['fps']}  chunk={meta['action_chunk_size']}")
    print(f"prompt: {meta['prompt']}")

    print(f"guard: {world.guard_memory(args.repo)}")
    t0 = time.monotonic()
    pipe = world.load(args.repo)
    print(f"loaded in {(time.monotonic() - t0) / 60:.1f} min")

    frames, per_chunk = world.rollout(
        pipe, meta, num_chunks=1, steps=args.steps, guidance=args.guidance, seed=args.seed
    )
    print(f"rollout: {len(frames)} frames in {sum(per_chunk):.1f}s")

    pred_path = args.out / "cosmos_predicted.mp4"
    media.write_mp4(frames, pred_path, fps=int(meta["fps"]))

    gt_dir = args.sample / "ground_truth_frames"
    from PIL import Image

    gt_frames = [Image.open(p).convert("RGB") for p in sorted(gt_dir.glob("*.png"))]
    gt_path = args.out / "ground_truth.mp4"
    media.write_mp4(gt_frames, gt_path, fps=int(meta["fps"]))

    strip = comparison_strip(gt_frames, frames)
    strip_path = args.out / "comparison_strip.png"
    strip.save(strip_path)

    print(f"wrote: {pred_path}, {gt_path}, {strip_path}")
    print("comparison_strip.png: top row = real mujoco frames, bottom row = Cosmos's prediction")


if __name__ == "__main__":
    main()
