#!/usr/bin/env python
"""Minimal LoRA fine-tuning pilot for Cosmos3-Edge on mujoco-env-dataset.

Implements the plan in FINETUNING_SCOPE.md: LoRA (r=8, lora_alpha=16) on the 112
`self_attn.{add_q_proj,add_k_proj,add_v_proj,to_add_out}` modules across the
transformer's 28 layers, plus a fresh domain id's `action_proj_in`/`action_proj_out`
rows, trained with a flow-matching / rectified-flow velocity-prediction loss at the
noisy vision-token positions only.

Reuses the pipeline's own segment-builders and encode helpers (`_encode_video`,
`_prepare_action_video_conditioning`, `_prepare_text_segment`, `_prepare_vision_segment`,
`_prepare_action_segment`, `_mask_velocity_predictions`, `tokenize_prompt`,
`prepare_latents`) instead of reimplementing sequence packing -- see FINETUNING_SCOPE.md
section 2 and the reuse note in section 4. The only genuinely new numerical code is the
flow-matching interpolation itself (`x_t = (1-sigma)*x0 + sigma*noise`, target velocity
`v = noise - x0`), derived from `UniPCMultistepScheduler`'s own `convert_model_output`
for `prediction_type="flow_prediction"` (`x0_pred = sample - sigma_t * model_output`) --
so the loss is inference-consistent by construction, not eyeballed.

Batch size is 1 (the pipeline itself is single-sample-per-call: "the pipeline runs one
sample per call" -- packing multiple items into one joint sequence for a real batch>1
is exactly the sequence-packing complexity this script avoids reimplementing). An
effective batch of `--grad-accum` is emulated by accumulating `.backward()` over that
many single-sample micro-steps before each `optimizer.step()`.

Usage:
    source env.sh
    python train_lora_pilot.py --steps 750 --grad-accum 4 --lr 1e-4
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
import torch.nn.functional as F

import config
import world
from lora_data import (
    MUJOCO_DOMAIN_NAME,
    MujocoWindowDataset,
    WindowSample,
    register_mujoco_domain,
)

DATASET_ROOT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/lora_pilot/dataset")
CKPT_ROOT = pathlib.Path("/opt/dlami/nvme/cosmos-policy/lora_pilot/checkpoints")
LORA_TARGET_NAMES = ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
RESOLUTION_TIER = 256
VIEW_POINT = "third_person_view"
TRAIN_SIGMA_STEPS = 1000  # granularity of the discretized train-time noise schedule


def wait_for_gpu_headroom(min_free_gib: float, poll_s: float = 30.0, max_wait_s: float = 3600.0) -> None:
    """Block until at least `min_free_gib` is free, per the GPU-coordination note: a
    concurrent Cosmos3-Nano test may be using most of this shared L40S. Checks the real
    `nvidia-smi` free-memory number, not process names."""
    waited = 0.0
    while True:
        free, total = torch.cuda.mem_get_info()
        free_gib = free / 2**30
        if free_gib >= min_free_gib:
            log.info("GPU headroom OK: %.1f GiB free (need %.1f)", free_gib, min_free_gib)
            return
        if waited >= max_wait_s:
            raise RuntimeError(f"waited {waited:.0f}s, still only {free_gib:.1f} GiB free; giving up")
        log.info("only %.1f GiB free (<%.1f needed) -- another job likely holds the GPU; waiting", free_gib, min_free_gib)
        time.sleep(poll_s)
        waited += poll_s


def build_lora_config():
    from peft import LoraConfig

    target_modules = [f"layers.{i}.self_attn.{name}" for i in range(28) for name in LORA_TARGET_NAMES]
    return LoraConfig(r=8, lora_alpha=16, target_modules=target_modules, lora_dropout=0.0, bias="none")


def attach_trainable_params(transformer) -> list[torch.nn.Parameter]:
    """LoRA adapter (112 modules) + the fresh domain's action-projection embeddings.

    Freezes everything else explicitly rather than trusting peft's default
    trainable-marking alone, so it's obvious from this function what can move.
    `action_proj_in`/`action_proj_out` are `DomainAwareLinear` (an `nn.Embedding` per
    domain) -- unfreezing the whole embedding is safe because gradients only ever
    populate the `mujoco_pickplace` (id 21) rows given our data never uses another
    domain id in a forward pass (verified empirically in FINETUNING_SCOPE.md section 3);
    weight_decay=0 on the optimizer (see main()) is what keeps a nonzero global decay
    from leaking into the untouched rows regardless.
    """
    for p in transformer.parameters():
        p.requires_grad_(False)

    trainable = []
    for name, p in transformer.named_parameters():
        if "lora_" in name or name.startswith("action_proj_in.") or name.startswith("action_proj_out."):
            p.requires_grad_(True)
            trainable.append(p)
    n_lora = sum(p.numel() for n, p in transformer.named_parameters() if "lora_" in n)
    n_domain = sum(
        p.numel() for n, p in transformer.named_parameters()
        if n.startswith("action_proj_in.") or n.startswith("action_proj_out.")
    )
    log.info("trainable: %d LoRA params + %d domain-embedding params (%d tensors total)", n_lora, n_domain, len(trainable))
    return trainable


def encode_gt_latents(pipe, sample: WindowSample, device, dtype, override_first_frame=None):
    """Real ground-truth vision latents for ALL 17 frames of the window (not just frame
    0) -- the piece `prepare_latents` doesn't expose (it only returns a noise-mixed
    `latents` tensor meant for inference's first denoising step). Reuses the pipeline's
    own conditioning-frame preprocessing + VAE encode, just called directly instead of
    through `prepare_latents`, since training needs real targets at every noisy position
    to build `x_t` at an arbitrary sampled timestep.

    `override_first_frame`: if given (a PIL image), replaces frame 0 -- the anchor the
    model conditions on -- while frames 1-16 stay the REAL targets. This is the exposure-
    bias fix in `COSMOS_FINETUNE_V2.md`: train some fraction of examples to reach the
    correct real continuation starting from an imperfect (perturbed or self-generated)
    frame 0, rather than always the pristine real one, since mode 2's autoregressive use
    conditions on exactly the kind of imperfect frame training never otherwise sees."""
    gt_frames = sample.gt_frames()
    if override_first_frame is not None:
        gt_frames = [override_first_frame] + list(gt_frames[1:])
    vision_tensor, action_image_size, _, _ = pipe._prepare_action_video_conditioning(
        gt_frames, RESOLUTION_TIER, sample.chunk_size + 1, device=device, dtype=dtype
    )
    x0_vision = pipe._encode_video(vision_tensor).contiguous().float()
    x0_vision = pipe._remove_action_video_padding_from_latent(x0_vision, action_image_size)
    return x0_vision  # [1, C, T_latent, H, W], float32


def pack_sample(pipe, sample: WindowSample, device, dtype, fps: float, override_first_frame=None):
    """Build every static (non-timestep) field a forward-dynamics training step needs,
    reusing the pipeline's text/vision/action segment builders exactly as `__call__`
    does. Returns a dict consumed by `training_step`.

    `override_first_frame`: see `encode_gt_latents`. Threaded into the
    `CosmosActionCondition.image` too, so the action-segment bookkeeping stays
    consistent with what the vision encoder actually saw."""
    from diffusers import CosmosActionCondition

    first_frame = override_first_frame if override_first_frame is not None else sample.first_frame()
    action10 = torch.from_numpy(sample.action10())
    action_cond = CosmosActionCondition(
        mode="forward_dynamics",
        chunk_size=sample.chunk_size,
        domain_name=MUJOCO_DOMAIN_NAME,
        resolution_tier=RESOLUTION_TIER,
        raw_actions=action10,
        image=first_frame,
        view_point=VIEW_POINT,
    )

    cond_input_ids, _ = pipe.tokenize_prompt(
        sample.prompt(),
        None,
        num_frames=sample.chunk_size + 1,
        height=RESOLUTION_TIER,
        width=RESOLUTION_TIER,
        fps=fps,
        action_mode="forward_dynamics",
        action_view_point=VIEW_POINT,
    )
    text_segment = pipe._prepare_text_segment(cond_input_ids, device=device)

    x0_vision = encode_gt_latents(pipe, sample, device, dtype, override_first_frame=override_first_frame)
    latent_t = x0_vision.shape[2]
    vision_condition_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=dtype)
    vision_condition_mask[0, 0, 0] = 1.0  # frame 0 is the real anchor -- never noised

    vision_segment = pipe._prepare_vision_segment(
        input_vision_tokens=x0_vision,
        has_image_condition=True,
        mrope_offset=text_segment["vision_start_temporal_offset"],
        vision_fps=fps,
        curr=text_segment["und_len"],
        device=device,
        condition_frame_indexes=[0],
    )

    # Action fields (real actions, all steps marked as conditioning per forward_dynamics
    # -- "actions give forward dynamics, they aren't predicted"): reuse `prepare_latents`
    # for this piece since, unlike vision, its returned `action_latents` for
    # forward_dynamics IS already the real (zero-padded) action tensor -- the condition
    # mask covers every step, so no noise is ever mixed in.
    (
        _vision_latents_unused,
        _sound_latents,
        action_latents,
        _fps_v,
        _fps_s,
        _vision_mask_unused,
        _sound_mask,
        action_condition_mask,
        action_domain_id,
        _action_image_size,
        raw_action_dim_resolved,
        action_condition_frame_indexes,
    ) = pipe.prepare_latents(
        image=None,
        video=None,
        num_frames=None,
        height=None,
        width=None,
        fps=fps,
        device=device,
        dtype=dtype,
        enable_sound=False,
        action=action_cond,
    )

    action_segment = pipe._prepare_action_segment(
        input_action_tokens=action_latents,
        condition_frame_indexes=action_condition_frame_indexes,
        mrope_offset=text_segment["vision_start_temporal_offset"],
        action_fps=fps,
        curr=text_segment["und_len"] + vision_segment["num_vision_tokens"],
        device=device,
    )

    position_ids = torch.cat(
        [text_segment["text_mrope_ids"], vision_segment["vision_mrope_ids"], action_segment["action_mrope_ids"]], dim=1
    )
    sequence_length = text_segment["und_len"] + vision_segment["num_vision_tokens"] + action_segment["action_len"]

    return {
        "text_segment": text_segment,
        "vision_segment": vision_segment,
        "action_segment": action_segment,
        "position_ids": position_ids,
        "sequence_length": sequence_length,
        "x0_vision": x0_vision.to(device=device),
        "vision_condition_mask": vision_condition_mask,
        "action_latents": action_latents,
        "action_domain_id": action_domain_id,
    }


def training_step(pipe, packed: dict, sigma: torch.Tensor, timestep_value: float, dtype) -> torch.Tensor:
    """One forward+loss for one packed sample at one sampled noise level `sigma`.

    Flow-matching interpolation derived from `UniPCMultistepScheduler.convert_model_output`'s
    `prediction_type="flow_prediction"` branch (`x0_pred = sample - sigma_t * model_output`,
    `alpha_t = 1 - sigma`, `sigma_t = sigma` for `use_flow_sigmas=True`) => `x_t = (1-sigma)*x0
    + sigma*noise`, target velocity `v = noise - x0`. `sigma` is drawn from the same
    discretized `scheduler.set_timesteps(N)` schedule inference uses (see `main()`) --
    NVIDIA's own training-time timestep distribution is undisclosed (FINETUNING_SCOPE.md
    section 2/6), so this is the documented safe fallback, not a guess.
    """
    x0 = packed["x0_vision"]
    mask = packed["vision_condition_mask"]
    noise = torch.randn_like(x0)
    x_t = (1.0 - sigma) * x0 + sigma * noise
    vision_tokens = (mask * x0 + (1.0 - mask) * x_t).to(dtype=dtype)

    v_target = (noise - x0).to(dtype=dtype)

    vseg = packed["vision_segment"]
    aseg = packed["action_segment"]
    tseg = packed["text_segment"]
    device = vision_tokens.device

    vision_timesteps = torch.full((vseg["num_noisy_vision_tokens"],), timestep_value, device=device)
    action_timesteps = torch.full((aseg["num_noisy_action_tokens"],), timestep_value, device=device)

    preds_vision, _preds_sound, _preds_action = pipe.transformer(
        input_ids=tseg["input_ids"],
        text_indexes=tseg["text_indexes"],
        position_ids=packed["position_ids"],
        und_len=tseg["und_len"],
        sequence_length=packed["sequence_length"],
        vision_tokens=[vision_tokens],
        vision_token_shapes=vseg["vision_token_shapes"],
        vision_sequence_indexes=vseg["vision_sequence_indexes"],
        vision_mse_loss_indexes=vseg["vision_mse_loss_indexes"],
        vision_timesteps=vision_timesteps,
        vision_noisy_frame_indexes=vseg["vision_noisy_frame_indexes"],
        sound_tokens=None,
        action_tokens=[packed["action_latents"].to(dtype=dtype)],
        action_token_shapes=aseg["action_token_shapes"],
        action_sequence_indexes=aseg["action_sequence_indexes"],
        action_mse_loss_indexes=aseg["action_mse_loss_indexes"],
        action_timesteps=action_timesteps,
        action_noisy_frame_indexes=aseg["action_noisy_frame_indexes"],
        action_domain_ids=[packed["action_domain_id"]],
        return_dict=False,
    )

    velocity_vision, _, _ = pipe._mask_velocity_predictions(
        preds_vision, None, vision_condition_mask=[mask], sound_condition_mask=None,
        preds_action=None, action_condition_mask=None, raw_action_dim=None,
    )

    # Loss over exactly the noisy latent frames -- indexing instead of a plain `.mean()`
    # over the (already correctly zeroed) full tensor, so conditioned-frame zero-vs-zero
    # entries don't dilute the loss scale (see FINETUNING_SCOPE.md section 2: "MSE against
    # predicted velocity at the noisy vision-token positions only").
    noisy_idx = (mask.squeeze(-1).squeeze(-1) < 0.5).nonzero(as_tuple=True)[0]
    pred_noisy = velocity_vision.index_select(2, noisy_idx).float()
    target_noisy = v_target.index_select(2, noisy_idx).float()
    return F.mse_loss(pred_noisy, target_noisy)


def save_checkpoint(pipe, out_dir: pathlib.Path, step: int) -> None:
    ckpt_dir = out_dir / f"step_{step:05d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pipe.transformer.save_lora_adapter(str(ckpt_dir))
    domain_state = {
        "action_proj_in": pipe.transformer.action_proj_in.state_dict(),
        "action_proj_out": pipe.transformer.action_proj_out.state_dict(),
    }
    torch.save(domain_state, ckpt_dir / "domain_embedding.pt")
    log.info("checkpoint written to %s", ckpt_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=750)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--stride", type=int, default=8, help="window stride in mujoco control steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-free-gib", type=float, default=28.0, help="GPU headroom to wait for before starting")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=CKPT_ROOT)
    ap.add_argument("--dataset-root", type=pathlib.Path, default=DATASET_ROOT)
    ap.add_argument(
        "--max-hours", type=float, default=None,
        help="stop (after checkpointing) once this much wall-clock time has elapsed, "
        "even if --steps hasn't been reached -- the actual control knob for an "
        "unattended overnight run, where --steps should be set generously high",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    wait_for_gpu_headroom(args.min_free_gib)

    register_mujoco_domain()
    pipe = world.load(config.EDGE)
    device = pipe._get_execution_device()
    dtype = pipe.transformer.dtype
    if args.grad_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
        log.info("gradient checkpointing enabled")

    lora_config = build_lora_config()
    pipe.transformer.add_adapter(lora_config)
    trainable_params = attach_trainable_params(pipe.transformer)
    pipe.transformer.train()

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    train_ds = MujocoWindowDataset(args.dataset_root, "train", stride=args.stride)
    log.info("train windows: %d", len(train_ds))

    # Discretized train-time noise schedule, matched to inference's own sigma/timestep
    # convention (FINETUNING_SCOPE.md section 2's documented safe fallback for NVIDIA's
    # undisclosed training-time timestep distribution).
    train_scheduler = copy.deepcopy(pipe.scheduler)
    train_scheduler.set_timesteps(TRAIN_SIGMA_STEPS, device=device)
    train_sigmas = train_scheduler.sigmas.to(device)
    train_timesteps = train_scheduler.timesteps.to(device)

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.monotonic()
    order = list(range(len(train_ds)))
    random.shuffle(order)
    cursor = 0

    optimizer.zero_grad()
    stopped_early = False
    for step in range(1, args.steps + 1):
        if args.max_hours is not None and (time.monotonic() - t0) / 3600.0 >= args.max_hours:
            log.info("hit --max-hours=%.2f at step %d; checkpointing and stopping", args.max_hours, step - 1)
            stopped_early = True
            break
        for micro in range(args.grad_accum):
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            idx = order[cursor]
            cursor += 1
            sample = train_ds[idx]

            packed = pack_sample(pipe, sample, device, dtype, fps=sample.episode.control_hz)
            sigma_idx = random.randrange(TRAIN_SIGMA_STEPS)
            sigma = train_sigmas[sigma_idx].to(dtype=torch.float32)
            timestep_value = train_timesteps[sigma_idx].item()

            loss = training_step(pipe, packed, sigma, timestep_value, dtype)
            (loss / args.grad_accum).backward()
            history.append(float(loss.detach().cpu()))

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % args.log_every == 0 or step == 1:
            recent = history[-args.log_every * args.grad_accum :]
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
        "done: %d/%d steps, %.1f min total%s, final ckpt at %s",
        (step - 1) if stopped_early else args.steps, args.steps,
        (time.monotonic() - t0) / 60, " (stopped early on --max-hours)" if stopped_early else "", args.out,
    )


if __name__ == "__main__":
    main()
