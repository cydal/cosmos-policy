# Feasibility scoping: LoRA fine-tuning Cosmos3-Edge on our mujoco data

Written by a research agent tasked with scoping (not implementing) whether
fine-tuning Cosmos3-Edge is a tractable fix for the visual-domain-gap problem in
[`VALIDATION.md`](VALIDATION.md), given the constraint that Cosmos must stay
Edge-sized (4B) for the downstream policy-training loop's throughput. The riskiest
technical claims below (gradient flow, LoRA injection, domain-embedding isolation)
were verified empirically against the real checkpoint on the actual GPU, not just
read off the source -- see the probe description in each section.

**Bottom line: go, with LoRA (not full fine-tuning -- doesn't fit this GPU's
optimizer-state budget), targeting the action-conditioned attention pathway plus a
fresh, isolated domain-embedding slot.**

## 1. Is `Cosmos3OmniTransformer` trainable via normal `backward()`?

Yes, confirmed by reading the code and by running it.

`Cosmos3OmniTransformer` is a plain `nn.Module`-based class (`ModelMixin,
ConfigMixin, PeftAdapterMixin, AttentionMixin`), `_supports_gradient_checkpointing =
True`. Every op from actions/images to prediction is standard differentiable
PyTorch (`nn.Linear`, `nn.Embedding` for the domain-aware action projection,
RMSNorm, rotary embeddings, SDPA attention -- `flash_attn` isn't installed, so it
falls back to native `torch.nn.functional.scaled_dot_product_attention`, no custom
CUDA extension). The only `@torch.no_grad()` in the stack is on
`Cosmos3OmniPipeline.__call__` itself, not on the transformer -- a training loop
calls `pipe.transformer(...)` directly, reusing the pipeline's own public segment
builders (`tokenize_prompt`, `prepare_latents`, `_prepare_vision_segment`,
`_prepare_action_segment`) instead of reimplementing sequence packing.

**Probe**: built one packed forward-dynamics step by hand with the pipeline's own
segment-builders, ran `transformer(...)`, summed a toy MSE loss over
`preds_vision` + `preds_action`, called `.backward()`. Gradients populated
correctly on `proj_in.weight`, layer-0 attention projections, and the action tables
-- gradient flows end-to-end, no graph-breaking ops.

## 2. Training objective

Scheduler config confirms this is a rectified-flow / flow-matching model
(`prediction_type: "flow_prediction"`, `predict_x0: true`). Natural loss (not a
reusable diffusers helper -- write it, ~30 lines): sample `t`, VAE-encode the
`chunk_size+1` ground-truth frames the same way `_encode_video` already does at
inference, construct `x_t = (1-t)x0 + t*eps`, target velocity `v = eps - x0`,
forward with the real first frame and real action chunk marked as conditioning
(never noised -- forward_dynamics gives actions, doesn't predict them), MSE against
predicted velocity at the noisy vision-token positions only. Pure video-prediction
loss; no action-prediction loss needed for "teach it to trust our actions."

No reusable training script exists in the installed `diffusers` (no `examples/` in
the wheel). NVIDIA's cosmos-cookbook docs mention "LoRA Post-Training" for the
Cosmos family generally but the agent couldn't retrieve raw training source to
confirm the exact recipe. **Open uncertainty**: exact timestep-sampling
distribution NVIDIA trained with is undisclosed; sampling from the same
discretized schedule `scheduler.set_timesteps(N)` already produces is a safe,
train/inference-consistent default even without their original recipe.

## 3. LoRA feasibility and target modules

Confirmed via the same probe (with `peft` installed to an isolated dir, cleaned up
after): `Cosmos3OmniTransformer` already mixes in `PeftAdapterMixin`, so
`model.add_adapter(LoraConfig(...))` works with no custom plumbing.

**Architecture detail that matters here**: this is a dual-pathway "Mixture of
Transformers" -- a causal text pathway, and a bidirectional "generation" pathway
that vision, action, and sound tokens are all packed into and jointly attend
within. `self_attn.{add_q_proj,add_k_proj,add_v_proj,to_add_out}` (4 modules x 28
layers = 112) are the **only** modules through which action tokens can influence
the vision prediction -- exactly the mechanism the decisive test in VALIDATION.md
identified as broken for our render style. LoRA (`r=8`) on all 112: **3.2M
trainable params out of 3.37B (0.095%)**, verified forward+backward works, every
LoRA param gets a nonzero gradient, frozen base weights get none.

**A second, even cheaper, fully-isolated lever**: the domain-aware action
projection (`DomainAwareLinear`) is an `nn.Embedding(32, ...)` indexed by domain
id; only ids 0-20 are used by the shipped domains (bridge/droid/umi/fractal/...).
**Ids 21-31 are free.** Probe: ran forward+backward with a fresh unused domain id
(25) -- gradients appeared *only* in that domain's embedding rows; every other
domain's rows (bridge, droid, umi, ...) had exactly zero gradient. Allocating a new
domain id and training just its ~0.5M-param rows is a fully isolated fix for the
*action-scale/calibration* problem (removing the empirical scale-guessing in
VALIDATION.md sections 3-4) with **no risk of degrading the pretrained
bridge/droid/umi/fractal weights** -- provided weight decay is 0 on those rows
(AdamW decay would otherwise nudge untouched rows even with zero gradient). This
alone doesn't fix the *visual*-style problem; LoRA on the gen-pathway is needed for
that.

**Recommended target set, tiered by risk:**
1. Safest/cheapest: new domain id's action-projection rows only -- fixes
   calibration, not the visual gap.
2. **Recommended starting point**: (1) + LoRA on the 112 attention modules above.
3. If (2) is insufficient: add `mlp_moe_gen` LoRA (84 more modules).
4. Higher risk, not recommended first: LoRA on `proj_in`/`proj_out` (shared by
   every domain and by plain t2v/i2v -- broader blast radius).

## 4. Data pipeline shape

mujoco-env-dataset's existing layout maps almost directly onto what training needs:

| Cosmos needs | mujoco-env-dataset has |
|---|---|
| `raw_actions [16, 10]` | `action_exec (T,7)` in `steps.npz` -> `cosmos_action.to_cosmos10` on any 16-step slice |
| first frame + 17 ground-truth frames | `frames/000.png...NNN.png`, 256x256 -- `CHUNK_SIZE`/`IMAGE_SIZE` in `cosmos_action.py` were already chosen to match, no resize/re-tier glue needed |
| VAE-encoded target latents | not present -- call `pipe._encode_video` per 17-frame slice (frozen, `.detach()`) |
| domain id | not present -- a constant, our new custom id |

Glue code needed is a data loader (windowing episodes, collating, calling
`to_cosmos10` + VAE-encode + the pipeline's own packing helpers), not new modeling
code. ~90 frames/episode average x 420 episodes already generated -> ~2,200
non-overlapping training windows already available, more generatable in seconds.
Data volume is not the constraint.

## 5. Rough compute/time budget (explicitly uncertain -- extrapolated, not measured end-to-end)

**Measured directly** (chunk_size=4, one 256x256 sample, batch=1, no gradient
checkpointing, idle 44.4GiB-usable L40S):

| config | reserved VRAM (fwd+bwd) |
|---|---|
| load only | 8.30 GiB |
| full-parameter fine-tune (all 3.37B get grad) | 15.49 GiB forward+backward alone |
| LoRA (r=8, 112 modules) | 8.36 GiB -- barely above load |

**Why LoRA, concretely, not just "it's smaller":** full fine-tuning needs Adam
optimizer state for 3.37B params. Even in bf16 that's ~27GiB just for optimizer
state before activations; in fp32 optimizer state (more numerically standard for
Adam) it's over 50GiB -- **exceeds this card's 44GiB entirely.** LoRA's optimizer
state for 3.2M params is ~25MB. This is the decisive argument, independent of the
"keep Cosmos small for inference" framing.

Scaling chunk_size 4 -> 16 (the real target) roughly triples activation memory
(more latent frames after temporal VAE compression) -- still comfortably inside the
LoRA case's headroom, likely without needing gradient checkpointing at batch 1-2.

**Step count / wall-clock -- a rough estimate, flagged as such**: LoRA-adapting a
frozen backbone to one narrow, low-diversity visual domain (one table, one
gripper, fixed camera) is a much smaller problem than pretraining. Plausible
budget: **~1,000-5,000 steps at batch 1-4, ~1-3s/step, i.e. roughly 1-4 GPU-hours**
for a first measurable-improvement checkpoint -- could be off 2-3x either way since
it's extrapolated from inference timing, not measured training throughput.
Recommended: run a ~200-500 step pilot and check it against the existing
`analysis.py` metrics (`frame_diff_energy` real-vs-stay ratio, `track_centroids`
direction-correctness) before committing to a longer run.

## 6. Risks and recommendation

**Recommendation: go**, tiered LoRA approach from §3, evaluated against the
existing VALIDATION.md metrics before scaling up.

**Risks, ranked:**
1. **Unknown whether attention-only LoRA is *enough*** to close the visual gap,
   vs. needing the MLP tier or even `proj_in`/`proj_out` too. This is the central
   empirical unknown -- the point of the pilot. No amount of reading resolves it.
2. **Undisclosed exact training recipe** (timestep sampling, loss weighting,
   whether action-conditioning used classifier-free-guidance-style dropout).
   Mitigated by matching the inference-time sigma schedule; real risk of a subtle
   train/inference mismatch vs. NVIDIA's own recipe.
3. **Weight-decay leakage** into untouched domain rows if the optimizer isn't
   configured carefully (0 decay on the new domain-embedding params) -- avoidable,
   but a real footgun if missed.
4. **Shared GPU**: memory math above assumes near-exclusive access; reuse the
   `NEEDS_GIB`/`guard_memory` pattern from `world.py` for a training script too.
5. **Overfitting risk**: small, low-diversity dataset (one table/lighting/camera
   setup) could make the LoRA learn "trust actions in this exact scene" rather
   than something that generalizes. mujoco-env-dataset already has proper
   geometrically-disjoint val/test splits -- use them to check this, not just
   train-set `frame_diff_energy`.

**Minimal viable pilot**: new domain id (e.g. 21) with fresh `action_proj_in`/
`action_proj_out` rows + LoRA (`r=8`, `lora_alpha=16`) on the 112
`self_attn.{add_q_proj,add_k_proj,add_v_proj,to_add_out}` modules, frozen backbone
otherwise, ~1,000-2,000 sixteen-step windows from the existing train split, batch
1-4, no gradient checkpointing initially, Adam with `weight_decay=0` on the
domain-embedding rows, ~500-1,000 step pilot (~1 GPU-hour), evaluated with the
existing `analysis.py` metrics before deciding whether to scale up or add the MLP
tier.
