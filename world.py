"""Load Cosmos 3 and roll it forward, on a single discrete GPU.

Split out of any pipeline/orchestration code so it can be exercised directly from
smoke_test.py -- the fast iteration loop.

This differs from the DGX Spark version of this module
(world_models/ai-build-and-learn/topics/cosmos/world.py) in exactly one important
way: that box has ONE 119.7 GiB pool shared by the GPU, the OS and everything else,
so its guard reads /proc/meminfo's MemAvailable. This box has a dedicated 46 GiB
VRAM pool that nothing else touches, so the honest number is
`torch.cuda.mem_get_info()` directly -- no host-RAM indirection needed or wanted.

device_map="cuda" streams shards straight to the device instead of materializing the
whole model on the host first and then copying it (`from_pretrained(...).to("cuda")`
would need 2x the model resident at once). Never add a `.to()` after this call.
"""

from __future__ import annotations

import json
import logging
import os
import time

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

ACTION_FRAME = "assets/example_action_fd_agibotworld_first_frame.png"
ACTION_CHUNKS = "assets/example_action_fd_agibotworld_action_chunks.json"

# Rough BF16-resident + VAE-decode-spike budgets per checkpoint. EDGE is the only one
# with real headroom on a 46 GiB card; NANO is here so guard_memory fails fast with an
# actionable message instead of hanging deep into a load.
NEEDS_GIB = {
    "nvidia/Cosmos3-Edge": 20.0,
    "nvidia/Cosmos3-Nano": 46.0,
    "nvidia/Cosmos3-Super": 200.0,
}
MAX_FRACTION = 0.90
HEADROOM_GIB = 2.0


def guard_memory(repo: str = "nvidia/Cosmos3-Edge") -> str:
    """Cap this process against actual free VRAM, or refuse to start.

    Raises RuntimeError rather than silently proceeding with too little budget --
    failing in two seconds with an actionable message beats an eight-minute load that
    ends in a bare CUDA OOM.
    """
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device"

    free, total = torch.cuda.mem_get_info()
    free_gib, total_gib = free / 2**30, total / 2**30
    needs_gib = NEEDS_GIB.get(repo, 20.0)
    budget_gib = free_gib - HEADROOM_GIB

    if budget_gib < needs_gib:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB free VRAM ({budget_gib:.1f} GiB after headroom), "
            f"and {repo} needs ~{needs_gib:.0f} GiB. Refusing to start rather than OOM "
            f"partway through a load. Check `nvidia-smi` for other processes holding "
            f"the GPU."
        )

    fraction = min(MAX_FRACTION, budget_gib / total_gib)
    torch.cuda.set_per_process_memory_fraction(fraction)
    line = f"{free_gib:.1f} GiB free of {total_gib:.1f} GiB; capped this process at {fraction:.2f} ({fraction * total_gib:.1f} GiB)"
    log.info("memory guard: %s", line)
    return line


def snapshot(repo: str) -> str:
    """Return the local path to the (already-fetched) checkpoint."""
    from huggingface_hub import snapshot_download

    t0 = time.monotonic()
    path = snapshot_download(repo_id=repo, max_workers=4)
    log.info("snapshot %s -> %s (%.1f min)", repo, path, (time.monotonic() - t0) / 60)
    return path


def load(repo: str):
    """Build a Cosmos3OmniPipeline on the GPU, ready to call.

    `enable_safety_checker=False`: the default True constructs a CosmosSafetyChecker,
    which pulls a separate, GATED Llama Guard checkpoint and fails at construction on
    any box whose HF token hasn't accepted that licence. These runs have no content
    guardrail -- fine for local experimentation, not for anything user-facing.
    """
    import torch
    from diffusers import Cosmos3OmniPipeline

    path = snapshot(repo)
    guard_memory(repo)

    t0 = time.monotonic()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,  # the only precision NVIDIA tests for Cosmos 3
        device_map="cuda",
        enable_safety_checker=False,
    )
    log.info("loaded %s in %.1f min", repo, (time.monotonic() - t0) / 60)
    return pipe


def generate(
    pipe,
    prompt: str,
    *,
    image=None,
    negative_prompt: str | None = None,
    num_frames: int = 45,
    height: int = 480,
    width: int = 832,
    fps: int = 24,
    steps: int = 20,
    guidance: float = 6.0,
    seed: int | None = 0,
):
    """Text-to-video, or image-to-video when `image` is given.

    `num_frames == 1` is text-to-image, an `image` anchors frame 0 (image-to-video),
    otherwise it's text-to-video. Mode comes from the inputs, not a flag.
    """
    import torch

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    t0 = time.monotonic()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=float(fps),
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        enable_safety_check=False,
    )
    secs = time.monotonic() - t0
    log.info("generated %s frames in %.1fs (%.1fs/step)", num_frames, secs, secs / steps)
    return result, secs


def load_action_example(model_path: str) -> dict:
    """Read the action-conditioning example that ships inside the checkpoint.

    Everything the rollout needs is self-describing in that JSON: prompt, embodiment
    domain, viewpoint, fps, resolution tier, chunk size, and the chunks themselves as
    [num_chunks, chunk_size, action_dim].
    """
    import torch
    from PIL import Image

    chunks_path = os.path.join(model_path, ACTION_CHUNKS)
    frame_path = os.path.join(model_path, ACTION_FRAME)
    if not os.path.exists(chunks_path):
        available = sorted(os.listdir(os.path.join(model_path, "assets"))) if os.path.isdir(
            os.path.join(model_path, "assets")
        ) else []
        raise FileNotFoundError(
            f"no {ACTION_CHUNKS!r} in {model_path}. assets/ contains: {available}"
        )

    meta = json.loads(open(chunks_path).read())
    meta["chunks"] = torch.tensor(meta["action_chunks"], dtype=torch.float32)
    meta["first_frame"] = Image.open(frame_path).convert("RGB")
    log.info(
        "action example: %s, domain=%s, chunks=%s",
        meta.get("prompt"), meta.get("domain_name"), tuple(meta["chunks"].shape),
    )
    return meta


def rollout(
    pipe,
    meta: dict,
    *,
    num_chunks: int = 1,
    steps: int = 20,
    guidance: float = 6.0,
    seed: int | None = 0,
):
    """Action-conditioned forward dynamics: first frame + actions -> future video.

    Chunks are rolled AUTOREGRESSIVELY: chunk 0 conditions on the real observed frame;
    every chunk after it conditions on the last frame the model itself predicted.
    Each chunk of `chunk_size` actions yields `chunk_size + 1` frames.

    height/width/num_frames must all be left at their defaults (None) for an action
    run -- resolution comes from `action.resolution_tier`, frame count from
    `action.chunk_size`; passing them raises.
    """
    import torch
    from diffusers import CosmosActionCondition

    chunks = meta["chunks"]
    frame = meta["first_frame"]
    frames: list = []
    per_chunk: list[float] = []

    for i in range(min(num_chunks, chunks.shape[0])):
        generator = torch.Generator().manual_seed(seed + i) if seed is not None else None
        t0 = time.monotonic()
        result = pipe(
            prompt=meta["prompt"],
            action=CosmosActionCondition(
                mode="forward_dynamics",
                chunk_size=int(meta["action_chunk_size"]),
                domain_name=meta["domain_name"],
                resolution_tier=int(meta["image_size"]),
                raw_actions=chunks[i],
                image=frame,
                view_point=meta.get("view_point", "ego_view"),
            ),
            fps=float(meta.get("fps", 10)),
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            use_system_prompt=False,
            enable_safety_check=False,
        )
        secs = time.monotonic() - t0
        per_chunk.append(secs)
        chunk_frames = list(result.video)
        frames.extend(chunk_frames if i == 0 else chunk_frames[1:])
        frame = chunk_frames[-1]
        log.info("chunk %s: %s frames in %.1fs", i, len(chunk_frames), secs)

    return frames, per_chunk
