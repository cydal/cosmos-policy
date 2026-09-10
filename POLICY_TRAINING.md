# Policy training: design

Trains the ResNet18 CNN policy from the original project brief, via behavior
cloning on mujoco-env-dataset's scripted-expert episodes. Cosmos (fine-tuned, per
`LORA_SCALEUP_RESULTS.md`) is **not** part of training -- it's used afterward, for
closed-loop visualization of the trained policy. See the conversation that led
here for the reasoning; this doc is the settled design.

## Action space -- no ambiguity, no Cosmos dependency

Three different "action" representations exist in this project and they must not
be confused:

1. **Primitive id** (`0`-`10`, `synthbot/primitives.py`) -- what the policy
   predicts. 11-way classification.
2. **7-D mujoco action** (`[dx, dy, dz, droll, dpitch, dyaw, gripper]`) -- what
   `primitives.to_action(primitive_id, current_gripper)` deterministically produces,
   and what actually drives the simulator. This is mujoco's native, unchanging
   action space.
3. **10-D Cosmos action** (`synthbot/cosmos_action.py:to_cosmos10`) -- a conversion
   applied ONLY when feeding an action sequence into Cosmos for visualization.
   Never something the policy sees or produces.

The policy's output space is (1). Its actions drive real mujoco physics via (2).
Conversion to (3) happens only at the visualization boundary, reusing code already
validated in `final_comparison.py`. There is no dimension mismatch risk because the
policy is never trained toward, or evaluated against, Cosmos's action format.

## Model

- Backbone: `torchvision.models.resnet18(weights=IMAGENET1K_V1)`, final `fc`
  replaced.
- Auxiliary input: current commanded gripper aperture (`gripper_cmd`, float in
  `[0, 1]`) -- required because `to_action`'s mapping is deterministic in
  `(primitive, current_gripper)`, not primitive alone (a `"keep"`-type primitive's
  resulting 7-D action depends on what the gripper is currently doing). Fused by
  concatenating a small embedding of this scalar with the ResNet's pooled features
  before the final classification layer.
- Output: 11-way softmax over `primitives.NAMES`.
- Input resolution: resized to 224x224 (torchvision's pretrained-weight transform
  default) rather than kept at the dataset's native 256x256, so the ImageNet
  normalization statistics apply as intended. Worth revisiting only if closed-loop
  eval suggests the policy is missing fine positional detail.
- No temporal state: the scripted expert this imitates is itself memoryless
  (env.py's docstring: "memoryless scripted expert"), fully reactive to the current
  frame + gripper state. One frame in, one primitive out.

## Data

- **Split integrity**: split at the *episode* level, never frames -- already
  guaranteed by mujoco-env-dataset's `combo_split` (episodes land in `train/`,
  `val/`, `test/` directories directly; the loader must index per-split episode
  directories, never pool-then-resplit at the frame level).
- **Batching**: no episode-awareness needed *within* a split -- each
  `(frame, gripper_cmd) -> primitive_id` tuple is an independent example (memoryless
  model), so ordinary shuffled i.i.d. batching applies once the per-episode frames
  are indexed.
- **Target**: `action_label` (the expert's *intended* primitive), not the primitive
  implied by `action_exec`/`is_noise` -- noisy steps are deliberate exploration
  perturbations in the recorded trajectory; the label is still "what the expert
  would actually do here," which is what we want the policy to learn.
- **Class imbalance**: `MOVE_*` primitives dominate (long transit phases);
  `GRIPPER_OPEN`/`CLOSE` are rare but failure there is catastrophic (task fails
  outright). Use class-weighted loss (inverse frequency, computed from the train
  split) rather than plain cross-entropy.
- **Augmentation**: standard image augmentation, plus reusing
  `mujoco-env-dataset/synthbot/photoreal.py`'s camera-sensor simulation
  (blur/noise/vignette/chromatic aberration/exposure jitter) at train time -- free
  synergy, since it's the same lever that measurably helped Cosmos's domain gap
  (`VALIDATION.md` section 6), and it should make the policy more robust to exactly
  the kind of visual variation Cosmos's imagined frames introduce during Mode-1/2
  visualization.
- **Volume**: prototype on the existing `lora_scaleup/dataset` (800/80/80 episodes,
  ~72k frames) -- reused as-is, no leakage concern from also having been used for
  Cosmos fine-tuning, since it's a different model/objective entirely. Decide
  whether to scale up from the prototype's own train/val gap (see Evaluation),
  not pre-emptively.

## Training

- Loss: class-weighted cross-entropy against `action_label`.
- Optimizer: AdamW, modest LR (e.g. `1e-4` for the new head, optionally a lower LR
  or partial freeze for the pretrained backbone early on -- standard transfer-
  learning practice, to be tuned against the prototype run).
- This is a small model (11M params) on a simple visual task with strong
  pretrained features -- expect this to be fast (minutes to tens of minutes on the
  L40S, not hours), unlike the Cosmos fine-tuning run.

## Evaluation -- closed-loop success rate is the real metric, not held-out accuracy

Held-out per-step classification accuracy is a proxy and a known trap: a policy
can look good under teacher-forcing and still fail closed-loop, because one small
mistake moves it off the expert's state distribution and errors compound. Primary
evaluation must be:

1. **Closed-loop rollout in the real simulator** (cheap, exact): drive
   `PickPlaceEnv` with the *trained policy's own predictions*, step by step, from
   fresh episode configs in each split. Success criterion is exactly
   `env.cube_on_pad` (`env.py`: resting, within 4cm of the pad center, not held) --
   the same definition `expert.py`'s own rollout uses, so the number is directly
   comparable to the scripted expert's ~98-100% success rate (the ceiling).
   Report success rate and `cube_fallen` rate on train/val/test configs -- the
   train-vs-test gap is the generalization number the whole `combo_split` design
   exists to measure.
2. **Confusion matrix** on held-out frames, to catch a policy that's quietly
   ignoring rare-but-critical classes (`GRIPPER_OPEN`/`CLOSE`) while still scoring
   well on aggregate accuracy.
3. Qualitative rollout videos (success and failure cases) from the real simulator.

Cosmos does not appear anywhere in this evaluation -- it's a visualization layer
on top of an already-evaluated policy, not part of judging whether the policy
works.

## Visualization (after training, Cosmos re-enters here only)

**Mode 1 -- real policy, parallel imagination (primary).** Run the trained policy
closed-loop in the real simulator for a full episode (exact, cheap, this is the
actual performance video). Then, per 16-step chunk of that real trajectory, feed
the real starting frame + the policy's real actions through fine-tuned Cosmos to
imagine that same chunk, anchored fresh to a real frame each time (so Cosmos's own
drift never compounds chunk-to-chunk). Two-row video: real mujoco on top, Cosmos's
imagination of the same actions on bottom, for the whole episode. Generalizes
`final_comparison.py` from one fixed action window to a full policy-driven episode.

**Mode 2 -- pure imagination (bonus, deliberately shown with its limits intact).**
Policy predicts a chunk from the real first frame -> Cosmos imagines 17 frames ->
policy predicts the *next* chunk from Cosmos's own last imagined frame (not a real
one) -> repeat, fully autoregressive, no real frames after the first. Given the
magnitude-insensitivity finding, expect visible drift after a few chunks -- shown
honestly, not cherry-picked, since the point is to see how far pure imagination
currently holds up, not to oversell it.

## Open items to settle empirically, not by guessing

- Whether the backbone needs partial freezing/lower LR vs. full fine-tuning at
  `1e-4` -- decide from the prototype run's loss curve.
- Whether 224x224 loses task-relevant detail vs. native 256x256 -- decide from
  closed-loop success rate, not held-out accuracy alone.
- Dataset scale-up -- decide from the prototype's train/val/closed-loop gap.
