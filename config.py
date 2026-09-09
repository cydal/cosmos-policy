"""Shared constants for the Cosmos 3 setup on this box.

This box: x86_64, one discrete NVIDIA L40S (46 GiB VRAM), CUDA 13.2 driver, 30 GiB
host RAM, root disk nearly full (~11 GiB free). That is a different shape of machine
from the DGX Spark (arm64, GB10, 119.7 GiB unified memory) that
world_models/ai-build-and-learn/topics/cosmos was written for, so nothing here is
copied verbatim from there -- same diffusers-based approach, adapted constants.

Big, regenerable things (HF cache, venv) live on /opt/dlami/nvme (217 GiB free
ephemeral instance storage) rather than the root disk. If that data disappears
across a stop/start cycle, re-run setup.sh / fetch.sh; nothing there is precious.
"""

from __future__ import annotations

import pathlib

DATA_ROOT = pathlib.Path("/opt/dlami/nvme/cosmos-policy")
HF_HOME = str(DATA_ROOT / "hf")

# Checkpoints, smallest first. Start with EDGE: 4B params vs NANO's 16B, on a single
# 46 GiB discrete GPU with no unified-memory offload to lean on. NANO was measured at
# a ~46 GiB peak (weights + VAE decode) on a 119.7 GiB shared pool; on a 46 GiB
# dedicated card that leaves ~0 headroom, so EDGE first is the safe default. All three
# are ungated on HF, per the earlier Cosmos 3 setup notes.
EDGE = "nvidia/Cosmos3-Edge"    # 4B, no video-to-video transfer, no sound
NANO = "nvidia/Cosmos3-Nano"    # 16B, ~35 GB on disk -- likely too tight on this GPU
SUPER = "nvidia/Cosmos3-Super"  # 64B, does not fit this box; reference only

DEFAULT_REPO = EDGE
