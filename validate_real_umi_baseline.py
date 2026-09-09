"""Isolate: does Cosmos3-Edge's action conditioning do anything at all?

validate_action_conditioning.py found that a real grasp action and an all-zero
"stay" action produce almost the same amount of predicted motion for our converted
mujoco actions. Before concluding the mujoco->Cosmos conversion is wrong, check the
same real-action-vs-stay contrast using the checkpoint's OWN shipped UMI example
(known-correct units/scale/domain, ships inside the checkpoint) as a control. If
even that shows no contrast, the problem is upstream of our conversion -- e.g. how
this repo is calling the pipeline, not the action numbers themselves.

    source env.sh
    python validate_real_umi_baseline.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import time

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

import torch

import analysis
import config
import media
import world

OUT = pathlib.Path("out/validation/umi_baseline")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"guard: {world.guard_memory(config.DEFAULT_REPO)}")
    t0 = time.monotonic()
    pipe = world.load(config.DEFAULT_REPO)
    print(f"loaded in {(time.monotonic() - t0) / 60:.1f} min")

    real_meta = world.load_action_example(world.snapshot(config.DEFAULT_REPO))
    real_meta["chunks"] = real_meta["chunks"][:1]  # just chunk 0, matching validate_action_conditioning.py

    stay_meta = dict(real_meta)
    stay_meta["chunks"] = torch.zeros_like(real_meta["chunks"])
    # Zero rotation must be the IDENTITY 6-D rep (cols of I), not literally all-zero,
    # or "no rotation" is encoded as a degenerate/invalid rotation matrix instead.
    stay_meta["chunks"][..., 3] = 1.0  # r00
    stay_meta["chunks"][..., 7] = 1.0  # r11
    # Keep the real trajectory's gripper channel (last dim) rather than zeroing it too.
    stay_meta["chunks"][..., 9] = real_meta["chunks"][..., 9]

    results = {}
    for name, meta in [("umi_real_action", real_meta), ("umi_stay_action", stay_meta)]:
        print(f"\n=== {name} ===")
        frames, per_chunk = world.rollout(pipe, meta, num_chunks=1, steps=30, seed=0)
        media.write_mp4(frames, OUT / f"{name}.mp4", fps=int(meta.get("fps", 10)))
        energy = analysis.frame_diff_energy(frames)
        results[name] = {"seconds": round(sum(per_chunk), 1), "frame_diff_energy": round(energy, 3)}
        print(f"  frame_diff_energy: {energy:.3f}")

    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    ratio = results["umi_real_action"]["frame_diff_energy"] / max(results["umi_stay_action"]["frame_diff_energy"], 1e-6)
    print(f"\nreal-action / stay-action frame_diff_energy ratio: {ratio:.2f}")
    print("(near 1.0 => the pipeline isn't distinguishing real motion from no motion here either)")


if __name__ == "__main__":
    main()
