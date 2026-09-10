#!/usr/bin/env python
"""Sample one fresh episode config (not tied to any existing train/val/test split
data on disk) as a clean, dedicated opening frame for a longer closed-loop
CNN-policy <-> Cosmos demo. Writes a minimal dataset_root/demo/<episode>/meta.json
+ dataset_meta.json, shaped so viz_policy_modes.py can load it unchanged.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "mujoco-env-dataset"))
from synthbot import config as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--episode-id", type=int, default=0)
    ap.add_argument("--split", default="test", help="which combo pool to sample from; this episode isn't added to that split's data")
    ap.add_argument("--episode-name", default="ep_fresh")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/opt/dlami/nvme/cosmos-policy/policy_scaleup_demo/dataset"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = C.sample_config(args.episode_id, args.split, rng)

    ep_dir = args.out / "demo" / args.episode_name
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "meta.json").write_text(json.dumps(cfg.to_json(), indent=2))
    (args.out / "dataset_meta.json").write_text(json.dumps({"action_space": {"control_hz": 10.0}}))

    print(f"wrote {ep_dir / 'meta.json'}")
    print(f"cube={cfg.cube_colour} pad={cfg.pad_colour} distractor={cfg.distractor_colour} ({cfg.distractor_shape})")


if __name__ == "__main__":
    main()
