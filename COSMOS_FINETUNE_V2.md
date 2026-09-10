# Cosmos fine-tune v2: closing the autoregressive drift gap

Motivated by `VALIDATION.md` section 8: the closed-loop demo shows mode 1
(anchored) holds up well but mode 2 (pure autoregressive) drifts, and the
diagnosis is that this is specifically a Cosmos train/inference mismatch --
training has only ever shown Cosmos clean real conditioning frames, never a
frame carrying its own generation artifacts, which is exactly what mode 2
requires it to handle. Budget: 20 hours total for the event this demos in, ~1h
reserved for final demo generation/documentation, **19h for fine-tuning**.

## Levers, in the order they matter

### 1. Exposure-bias-aware training data (the headline fix, new this round)

Mix a fraction of training windows so the conditioning "first frame" is not
pristine ground truth:

- **Cheap tier (do this regardless)**: run `mujoco-env-dataset/synthbot/
  photoreal.py`'s perturbation (blur/noise/vignette/color-jitter -- already
  built, already used for policy-training augmentation) on the conditioning
  frame for some fraction of windows (propose ~40%), while keeping the *target*
  frames (the loss target) clean. Teaches the model "produce a coherent
  continuation even when the input frame isn't pristine" without needing any
  new generation infrastructure.
- **Stronger tier (if time allows)**: periodically (e.g. every ~200-500 steps)
  generate a real self-conditioning example: take a real frame N chunks back,
  roll the *current* model forward (no grad, eval mode, ~4s/chunk) to produce
  an actually Cosmos-generated frame, then train the *next* chunk against real
  target frames using that generated frame as the conditioning input. This is
  the direct analogue of scheduled sampling / DAgger, applied to the world
  model instead of a policy -- the closest thing to actually training on mode
  2's real operating distribution. More engineering than the cheap tier, more
  directly targeted at the actual failure mode.

Recommend: cheap tier as the default for every run from now on (it's nearly
free), stronger tier as the main new thing this 19h buys us, since it's the
one lever aimed squarely at the diagnosed problem rather than a general
"more/better" hope.

### 2. More / more diverse data -- already paid for, just reuse it

`policy_scaleup/dataset` (2,000 train / 150 val / 150 test, generated for the
policy scale-up) is sitting on disk already. Reuse it directly for Cosmos
fine-tuning too -- zero additional generation time, and more visual diversity
(colors/lighting/camera pose) than the 800-episode set the last Cosmos run
used. Unlike the policy's generalization problem, Cosmos fine-tuning isn't
gated by `combo_split`'s train/test combo split in the same discrete sense, so
this is a softer "more grounding" argument than a hard requirement -- but it's
free, so take it.

### 3. More steps -- already open-ended, still true

The last run's loss was still trending down at 21,483 steps (10.75h), not
plateaued. Simply running longer on the same LoRA target continues to help
along whatever it's already learning. This is the "do nothing new, just run
longer" baseline improvement, folded into the plan by construction (more
budget this time).

### 4. Expand LoRA capacity: add the `mlp_moe_gen` tier -- orthogonal, optional

`FINETUNING_SCOPE.md`'s tier-3 fallback (84 more modules: `gate_proj/up_proj/
down_proj` across 28 layers). This targets a *different* symptom
(`LORA_SCALEUP_RESULTS.md`'s magnitude-insensitivity finding -- energy ratio
flat despite 26x more steps, "looks like a real capacity/objective ceiling for
attention-only LoRA") than the autoregressive-drift problem this round is
about. Worth doing given the time budget allows both, but it's a genuinely
separate bet, not a second attempt at the same fix -- flagging so scope stays
clear, and because it needs its own quick memory/throughput smoke test before
committing 19h to it blind (more modules = more compute per step; unknown
by how much until measured).

## Proposed plan for the 19h

1. Reuse `policy_scaleup/dataset` (no generation time spent).
2. Implement the cheap exposure-bias tier (photoreal-perturbed conditioning
   frames, ~40% of windows) -- always on.
3. Implement the stronger tier (periodic self-rollout-conditioned training
   examples) if the smoke test shows it fits comfortably in the memory/time
   budget alongside everything else.
4. Add the `mlp_moe_gen` LoRA tier, after a quick smoke test confirms the
   per-step cost is acceptable within a `--max-hours`-capped run (same
   safety pattern as every long run so far -- wall-clock cap, not a step-count
   guess, best-checkpoint tracking).
5. Run for the remaining budget after setup/smoke-testing, capped well under
   19h to leave real margin for the reserved 1h of demo/documentation time.
6. Re-run the same closed-loop demo (mode 0/1/2, same fresh opening frame or a
   new one) at the end for a direct before/after comparison against
   `qa/policy_scaleup/closed_loop_demo/`.

## Open question before building this

Items 1 and 3 are free/default. Item 2 (cheap exposure-bias tier) is cheap and
clearly worth doing. Items 3-strong (self-rollout conditioning) and 4
(`mlp_moe_gen`) are both real, separate engineering efforts that compete for
the same 19h and the same "does this actually work" uncertainty -- worth
deciding deliberately rather than trying to cram both in and risking neither
getting a fair, well-measured shot within the budget.
