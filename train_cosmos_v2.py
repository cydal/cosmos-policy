#!/usr/bin/env python
"""Cosmos LoRA fine-tune v2: exposure-bias-aware training, per COSMOS_FINETUNE_V2.md.

Same LoRA target (112 attention modules + fresh domain embedding) and same
flow-matching training step as train_lora_pilot.py -- this script only changes
WHAT the conditioning "frame 0" is for a fraction of training windows, so the
model sees some of what mode 2's real autoregressive use actually looks like:

  --photoreal-prob     cheap tier: frame 0 is a photoreal.py-perturbed version of
                        the real frame (blur/noise/vignette/jitter). Free-ish,
                        general robustness to an imperfect input.
  --self-rollout-every  strong tier: every Nth micro-step, frame 0 is the model's
                        OWN generated last frame from the PRECEDING chunk (rolled
                        out with current weights, no_grad, eval mode) -- the direct
                        analogue of scheduled sampling, and the closest match to
                        what mode 2 actually does at inference time. Real targets
                        (frames 1-16) stay ground truth either way: the model
                        should learn to recover toward the truth, not imitate its
                        own drift.

    source env.sh
    python train_cosmos_v2.py --max-hours 18 --photoreal-prob 0.4 --self-rollout-every 300
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import pathlib
import random
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
log = logging.getLogger(__name__)

import numpy as np
import torch

import config
import world
from lora_data import MUJOCO_DOMAIN_NAME, MujocoWindowDataset, WindowSample, register_mujoco_domain
from train_lora_pilot import (
    RESOLUTION_TIER,
    VIEW_POINT,
    TRAIN_SIGMA_STEPS,
    attach_trainable_params,
    build_lora_config,
    pack_sample,
    save_checkpoint,
    training_step,
    wait_for_gpu_headroom,
)

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "mujoco-env-dataset"))
from synthbot import photoreal  # noqa: E402

DATASET_ROOT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/policy_scaleup/dataset")
CKPT_ROOT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/cosmos_v2/checkpoints")


def photoreal_override(sample: WindowSample, rng: np.random.Generator):
    """Cheap tier: perturb the real first frame, keep real targets."""
    from PIL import Image

    arr = photoreal.apply(np.asarray(sample.first_frame()), rng)
    return Image.fromarray(arr)


def self_rollout_override(pipe, sample: WindowSample, steps: int, rng: np.random.Generator):
    """Strong tier: roll out the PRECEDING chunk with current weights (no_grad, eval
    mode), return its last generated frame. None if this sample is episode-initial
    (no preceding chunk to roll out from)."""
    if sample.start < sample.chunk_size:
        return None

    chunk_a = WindowSample(sample.episode, sample.start - sample.chunk_size, sample.chunk_size)
    meta = {
        "chunks": torch.tensor(chunk_a.action10(), dtype=torch.float32)[None],
        "first_frame": chunk_a.first_frame(),
        "prompt": chunk_a.prompt(),
        "domain_name": MUJOCO_DOMAIN_NAME,
        "action_chunk_size": chunk_a.chunk_size,
        "image_size": RESOLUTION_TIER,
        "fps": chunk_a.episode.control_hz,
        "view_point": VIEW_POINT,
    }
    was_training = pipe.transformer.training
    pipe.transformer.eval()
    with torch.no_grad():
        frames, _ = world.rollout(pipe, meta, num_chunks=1, steps=steps, seed=int(rng.integers(0, 2**31 - 1)))
    if was_training:
        pipe.transformer.train()
    return frames[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000, help="ceiling; --max-hours is the real control")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-free-gib", type=float, default=28.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--max-hours", type=float, default=18.0)
    ap.add_argument("--dataset-root", type=pathlib.Path, default=DATASET_ROOT)
    ap.add_argument("--out", type=pathlib.Path, default=CKPT_ROOT)
    ap.add_argument("--photoreal-prob", type=float, default=0.4)
    ap.add_argument("--self-rollout-every", type=int, default=300, help="0 disables the strong tier")
    ap.add_argument("--self-rollout-steps", type=int, default=20, help="denoising steps for the self-rollout generation call")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    aug_rng = np.random.default_rng(args.seed)

    wait_for_gpu_headroom(args.min_free_gib)

    register_mujoco_domain()
    pipe = world.load(config.EDGE)
    device = pipe._get_execution_device()
    dtype = pipe.transformer.dtype

    lora_config = build_lora_config()
    pipe.transformer.add_adapter(lora_config)
    trainable_params = attach_trainable_params(pipe.transformer)
    pipe.transformer.train()

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    train_ds = MujocoWindowDataset(args.dataset_root, "train", stride=args.stride)
    log.info("train windows: %d", len(train_ds))

    train_scheduler = copy.deepcopy(pipe.scheduler)
    train_scheduler.set_timesteps(TRAIN_SIGMA_STEPS, device=device)
    train_sigmas = train_scheduler.sigmas.to(device)
    train_timesteps = train_scheduler.timesteps.to(device)

    args.out.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "override_kind": []}
    t0 = time.monotonic()
    order = list(range(len(train_ds)))
    random.shuffle(order)
    cursor = 0
    microstep = 0

    optimizer.zero_grad()
    stopped_early = False
    for step in range(1, args.steps + 1):
        if (time.monotonic() - t0) / 3600.0 >= args.max_hours:
            log.info("hit --max-hours=%.2f after step %d; stopping", args.max_hours, step - 1)
            stopped_early = True
            break

        for _ in range(args.grad_accum):
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            idx = order[cursor]
            cursor += 1
            sample = train_ds[idx]
            microstep += 1

            override_frame = None
            kind = "real"
            if args.self_rollout_every > 0 and microstep % args.self_rollout_every == 0:
                override_frame = self_rollout_override(pipe, sample, args.self_rollout_steps, aug_rng)
                kind = "self_rollout" if override_frame is not None else "real"
            if override_frame is None and aug_rng.random() < args.photoreal_prob:
                override_frame = photoreal_override(sample, aug_rng)
                kind = "photoreal"

            packed = pack_sample(pipe, sample, device, dtype, fps=sample.episode.control_hz, override_first_frame=override_frame)
            sigma_idx = random.randrange(TRAIN_SIGMA_STEPS)
            sigma = train_sigmas[sigma_idx].to(dtype=torch.float32)
            timestep_value = train_timesteps[sigma_idx].item()

            loss = training_step(pipe, packed, sigma, timestep_value, dtype)
            (loss / args.grad_accum).backward()
            history["train_loss"].append(float(loss.detach().cpu()))
            history["override_kind"].append(kind)

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % args.log_every == 0 or step == 1:
            recent = history["train_loss"][-args.log_every * args.grad_accum :]
            free, total = torch.cuda.mem_get_info()
            log.info(
                "step %d/%d  loss=%.5f (last %d avg)  elapsed=%.1fmin  vram_used=%.1fGiB",
                step, args.steps, sum(recent) / len(recent), len(recent),
                (time.monotonic() - t0) / 60, (total - free) / 2**30,
            )
        if step % args.ckpt_every == 0 or step == args.steps:
            save_checkpoint(pipe, args.out, step)
            (args.out / "loss_history.json").write_text(json.dumps(history))

    if stopped_early:
        save_checkpoint(pipe, args.out, step - 1)
    (args.out / "loss_history.json").write_text(json.dumps(history))
    log.info(
        "done: %d steps run, %.1f min total%s, checkpoints at %s",
        (step - 1) if stopped_early else args.steps, (time.monotonic() - t0) / 60,
        " (stopped on --max-hours)" if stopped_early else "", args.out,
    )


if __name__ == "__main__":
    main()
