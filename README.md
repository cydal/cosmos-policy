# cosmos-policy

Freeze NVIDIA Cosmos 3 as a virtual environment (action-conditioned world model) and
train a ResNet18-backed CNN policy against it, bootstrapped from the synthetic
pick-and-place data in [`../mujoco-env-dataset`](../mujoco-env-dataset).

## Plan

1. **Environment + Cosmos 3** (this step) -- get Cosmos 3 running on this box and
   prove it with a smoke test.
2. **Action-space check** -- confirm `mujoco-env-dataset`'s 7-D
   `[dx, dy, dz, droll, dpitch, dyaw, gripper]` action matches what Cosmos 3's
   `CosmosActionCondition(mode="forward_dynamics", ...)` expects, or work out the
   mapping if not. Pass a real sample image + action sequence from the mujoco dataset
   through Cosmos and confirm the rollout is sane.
3. **Policy training** -- ResNet18 (ImageNet-pretrained) CNN policy, Cosmos 3 frozen
   as the environment it's trained/evaluated against.

## This box vs. the DGX Spark demo

`world_models/ai-build-and-learn/topics/cosmos` already has a working Cosmos 3 setup,
but it's hard-pinned to a DGX Spark: arm64, CUDA 13.0, one 119.7 GiB pool shared
between GPU/OS/everything, Flyte/K8s orchestration. This box is x86_64 with one
dedicated 46 GiB L40S and a Flyte-free local workflow, so `setup.sh`/`world.py` here
are rewritten for that, not copied.

Root disk has ~11 GiB free, so the venv and the HF cache both live on
`/opt/dlami/nvme` (217 GiB free ephemeral instance storage) -- see `config.py`.
Nothing there is precious; re-run `setup.sh`/`fetch.sh` if it's ever wiped.

## Model size

Starting with `nvidia/Cosmos3-Edge` (4B params) rather than `Cosmos3-Nano` (16B).
Nano was measured elsewhere at a ~46 GiB peak (weights + VAE decode spike) on a
119.7 GiB *shared* pool; this box's entire GPU *is* 46 GiB with nothing else to lean
on, so Edge is the box that actually has headroom. `world.py`'s `NEEDS_GIB` table has
the per-checkpoint numbers.

## Usage

```bash
./setup.sh                                            # venv + torch + diffusers, once
./fetch.sh                                            # pull Cosmos3-Edge, once
/opt/dlami/nvme/cosmos-policy/.venv/bin/python smoke_test.py --image   # fastest check
/opt/dlami/nvme/cosmos-policy/.venv/bin/python smoke_test.py --action  # action-conditioned
```
