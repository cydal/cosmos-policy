"""Find the action-magnitude scale where our converted mujoco actions actually move
Cosmos's prediction, using domain="umi" (the one domain we have a real, responsive
reference for -- see validate_real_umi_baseline.py) as the search space.

mujoco's raw per-step deltas are metres (0.014 m/step @ 10 Hz translation, 0.10
rad/step yaw); the checkpoint's own UMI example moves ~4x further in translation
per step and, more importantly, tens of degrees of rotation per step (nowhere close
to ours). If Cosmos's action-projection weights expect roughly that order of
magnitude, our unscaled deltas would look close to noise-floor to the model --
consistent with validate_action_conditioning.py's stay_control ~= grasp_bridge
result. This sweeps a multiplier on the pre-conversion 7-D deltas (translation +
rotation angle; gripper is an absolute command and is never scaled) and reports
frame_diff_energy at each, against the umi_stay_action floor (~0.48) and
umi_real_action ceiling (~5.0) from that baseline run.

    source env.sh
    python validate_scale_sweep.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "mujoco-env-dataset"))
from synthbot import primitives  # noqa: E402
from synthbot.cosmos_action import to_cosmos10  # noqa: E402

import analysis
import config
import media
import world
from validate_action_conditioning import SAMPLE_ROOT, find_grasp_start, frame_at, make_meta, prompt_for, load_episode, CHUNK_SIZE

OUT = pathlib.Path("out/validation/scale_sweep")
SCALES = [1, 3, 5, 10, 20, 40]
DOMAIN = "umi"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    steps0, meta0, hz0 = load_episode(SAMPLE_ROOT / "ep000000")
    al0 = steps0["action_label"]
    ax0 = steps0["action_exec"]
    grasp_start0 = find_grasp_start(al0, CHUNK_SIZE)
    grasp0 = ax0[grasp_start0 : grasp_start0 + CHUNK_SIZE]
    prompt0 = prompt_for(meta0)
    first_frame = frame_at(SAMPLE_ROOT / "ep000000", grasp_start0)

    print(f"guard: {world.guard_memory(config.DEFAULT_REPO)}")
    t0 = time.monotonic()
    pipe = world.load(config.DEFAULT_REPO)
    print(f"loaded in {(time.monotonic() - t0) / 60:.1f} min")

    results = {}
    for scale in SCALES:
        scaled7 = grasp0.copy()
        scaled7[:, 0:6] *= scale  # translation + rotation deltas only; leave gripper (col 6) alone
        meta = make_meta(to_cosmos10(scaled7), first_frame, prompt0, DOMAIN, hz0)

        print(f"\n=== scale={scale} ===")
        frames, per_chunk = world.rollout(pipe, meta, num_chunks=1, steps=30, seed=0)
        media.write_mp4(frames, OUT / f"scale_{scale}.mp4", fps=int(hz0))
        energy = analysis.frame_diff_energy(frames)
        results[str(scale)] = {"seconds": round(sum(per_chunk), 1), "frame_diff_energy": round(energy, 3)}
        print(f"  frame_diff_energy: {energy:.3f}")

    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print("\nscale : frame_diff_energy  (umi_stay_action floor ~0.48, umi_real_action ceiling ~5.0)")
    for scale, r in results.items():
        print(f"  {scale:>4} : {r['frame_diff_energy']}")


if __name__ == "__main__":
    main()
