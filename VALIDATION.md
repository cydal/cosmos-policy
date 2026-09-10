# Validating the mujoco -> Cosmos action bridge, before trusting it for training

`run_mujoco_rollout.py`'s first result (one grasp window, visually similar to ground
truth) was not enough evidence on its own. This is the test matrix that followed,
and what it actually shows. Scripts: `validate_action_conditioning.py`,
`validate_real_umi_baseline.py`, `validate_scale_sweep.py`, plus a few one-off
snippets referenced inline below. Metrics/helpers in `analysis.py`.

## 1. The first result was likely not a real action-conditioning effect

`validate_action_conditioning.py`'s test matrix, domain `bridge_orig_lerobot`,
unscaled (raw metres/radians) actions:

| case | frame_diff_energy | note |
|---|---|---|
| `grasp_bridge` (real grasp+lift action) | 0.631 | |
| `stay_control` (all-zero action, same frame) | 0.638 | **~identical to grasp_bridge** |
| `lateral` (dx/dy forward) | diff_centroid_x 132.2 | |
| `lateral_reversed` (dx/dy negated) | diff_centroid_x 132.6 | **~identical to lateral** |

A real action and a no-op produce the same amount of predicted motion; negating an
action's sign doesn't change where the motion happens. That's not evidence of
working action conditioning -- more likely the model generating generic,
plausible-looking motion regardless of the actual numbers, which happened to
resemble the real trajectory in the very first (visually-judged) test.

## 2. The pipeline itself does respond to actions -- calibration proof

`validate_real_umi_baseline.py`, using the checkpoint's own real UMI example
(known-correct units, ships inside Cosmos3-Edge):

| case | frame_diff_energy |
|---|---|
| real UMI action | 5.017 |
| same frame, all-zero (identity-rotation) action | 0.479 |

10.5x. So `world.rollout` / the pipeline call is correct; the earlier null result is
specific to *our* action values, not a bug in how this repo calls Cosmos.

## 3. Root cause #1: our raw action magnitudes are far too small

Two independent methods converge on the same answer:

- **Forward sweep** (`validate_scale_sweep.py`): multiplying our converted action's
  translation+rotation deltas by a scalar before feeding it to domain `umi`, energy
  stays flat at the noise floor (~0.5) through 10x, then rises sharply at 20-40x.
- **Inverse dynamics** (one-off snippet, see below): ran Cosmos's own
  `mode="inverse_dynamics"` on our *real* ground-truth video (known physical motion)
  for domain `bridge_orig_lerobot`, and read back what action values Cosmos itself
  infers for that motion. Inferred z-translation magnitude was **~12.5x** our raw
  value (0.131 vs. 0.0105 mean |z|) -- independently landing in the same 10-40x band.

mujoco-env-dataset's actions are honest SI units (metres, radians); Cosmos's
domain-aware action-projection weights evidently expect a much larger numeric range
for at least this checkpoint. There is no published normalization spec, so this
scale factor is empirical, not derived.

## 4. Root cause #2: domain framing, and it matters more than the scale

- `umi` is **egocentric** (camera mounted on the hand/wrist). Scaled into a
  responsive range, it makes the *whole scene warp/tilt* like a moving camera --
  physically sensible for UMI's own footage, wrong for our fixed third-person
  camera. (`out/validation/scale60_umi_strip.png` -- table surface visibly bends.)
- `bridge_orig_lerobot` is the fixed-camera, in-frame-arm domain that actually
  matches our setup. But scaled into the same range it stays weak and points the
  wrong way (cube net displacement consistently *down*, real motion is *up* --
  `bridge_precise_scale_18_strip.png` etc.), and past ~scale 100 it produces
  outright hallucinated geometry, not a stronger version of the same motion
  (`bridge_scale_150_strip.png`: a red floating structure appears out of nowhere).

## 5. Decisive test: the action math is fine; our render style is the blocker

Held domain (`umi`) and action conversion logic fixed, swapped only the
conditioning image: the checkpoint's own **real** UMI photo instead of our MuJoCo
render, same converted mujoco action.

| case | frame_diff_energy |
|---|---|
| real UMI image + our action, scale 1 | 0.846 |
| real UMI image + our action, scale 40 | 3.478 |
| real UMI image + our action, scale 60 | 4.341 |
| (reference) real image + real action | 5.017 |
| (reference) real image + zero action | 0.479 |

On an in-distribution (real camera) image, our action conversion produces a strong,
cleanly-scaling response approaching the model's own real-action ceiling -- with
*no changes to the conversion math*, only the conditioning frame. That isolates the
failure: it's not primarily our 7D->10D conversion, and not purely a scale constant.
It's that every one of Cosmos 3's action-conditioned training domains
(bridge/droid/umi/fractal/...) is real camera footage, and this dataset's flat-shaded
synthetic MuJoCo renders are visually far enough outside that distribution that
Cosmos3-Edge (a 4B, distilled checkpoint) can't reliably apply its learned action
dynamics to them, independent of how well the action numbers are calibrated.

## 6. Tried: closing the gap with camera-sensor post-processing

Added `mujoco-env-dataset/synthbot/photoreal.py` (gaussian blur, sensor noise,
vignette, chromatic aberration, brightness/contrast/saturation jitter -- applied
post-render, opt-in via `--photoreal`, never touching physics or the recorded
action/state arrays) plus a skybox and mild specular/shininess tweaks in
`scene.py`. Re-ran the same grasp-window test against a `--photoreal` episode:

| domain | scale | energy (stay_control ~1.53 on this render) | cube_disp (GT: (-4.95, -4.79)) |
|---|---|---|---|
| `bridge_orig_lerobot` | 1 | 1.517 | (0.69, 0.20) |
| `bridge_orig_lerobot` | 18 | 1.665 | (1.26, **-1.78**) |
| `bridge_orig_lerobot` | 60 | 2.512 | (1.92, **-2.11**) -- but hallucinates (see below) |
| `umi` | 40 | 7.277 | (20.4, 9.2) -- scene-zoom, not a real match |
| `umi` | 60 | 9.168 | (10.9, 9.8) -- same |

The one genuine improvement: on the clean render, `bridge_orig_lerobot`'s predicted
cube motion was *wrong-direction* (positive/downward) at every scale tried. On the
photoreal render it's *correct-direction* (negative/upward, matching the real lift)
from scale 18 upward. But magnitude still undershoots ground truth by ~2-3x even at
the best point, and past a moderate scale both domains degrade: `bridge_orig_lerobot`
at scale 60 hallucinates a phantom red/purple object out of nothing
(`qa/photoreal_bridge_hallucination.png`); `umi` at scale 40-60 turns into an extreme
whole-scene zoom-out rather than in-frame gripper motion, worse than on the clean
render.

Net: camera-sensor post-processing is a real, measurable, and cheap improvement
(better sign-correctness on the right-framing domain) but does not close the gap by
itself. MuJoCo's rasterizer ceiling (flat shading, no global illumination, hard
shadow edges) is still there underneath the post-process, and the model still
breaks down at the action magnitudes needed to get a strong response. Worth keeping
(`--photoreal` is opt-in, zero cost when off) as a complement to, not a replacement
for, fine-tuning -- see the separate fine-tuning feasibility scoping.

## 7. Is it a model-size problem? Tried Cosmos3-Nano (16B, not distilled)

Not to use for policy training (too big for the rollout-throughput the training
loop needs -- see the fine-tuning scoping doc) but purely to check whether Cosmos3-
Edge's small, distilled 4B is itself the reason our actions don't land, or whether
a bigger, non-distilled checkpoint behaves the same way.

**It barely fits on this GPU at all**, which is itself informative: loading takes
30.6 GiB (vs. Edge's much smaller footprint), leaving only 13.3 GiB free. Our own
`world.py` memory guard actually refuses Nano outright (its `NEEDS_GIB=46` estimate,
carried over from a different, shared-memory box, is conservative but not
groundless) -- bypassing it worked for the 10-D domains here, but Nano's own native
29-D AgiBotWorld example left only ~6.5 GiB free after one rollout, i.e. Nano is
right at this card's ceiling for real generation, not comfortably inside it.

Same grasp-window test as Edge's (clean render, domain `bridge_orig_lerobot`):

| scale | energy (stay_control: 0.731) | cube_disp (GT: (-2.67, -24.23)) |
|---|---|---|
| 1 | 0.649 | (2.86, -4.48) |
| 18 | 0.885 | (1.78, 0.01) |

Same character as Edge's numbers (VALIDATION.md §1/§4): energy barely above the
stay-control floor, no clean directional signal. Visually, scale 18's prediction
isn't a coherent grasp+lift *or* Edge's phantom-object hallucination -- it's a
third failure mode, the gripper's base detaching and floating up out of frame
entirely (`qa/nano_bridge_scale18_detach.png`).

**Conclusion: size is not the fix.** A 4x bigger, non-distilled checkpoint shows
the same weak/incoherent response to our converted actions on this rendering style.
This corroborates rather than contradicts section 5's diagnosis -- the visual
domain gap is a property of what these models were trained on (real camera
footage, across the board), not of how big any single checkpoint is. It also means
there's no shortcut available by "just use the bigger model": fine-tuning (or
closing the visual gap some other way) is the path, not a size upgrade.

## 8. Closed-loop policy<->Cosmos demo: where fidelity actually breaks down

With the scaled-up policy in hand (`LORA_SCALEUP_RESULTS.md`'s sibling,
`POLICY_TRAINING.md`), ran the literal closed loop the original brief asked for:
policy -> action -> Cosmos -> predicted observation -> policy -> action ->
Cosmos -> ..., for 8 chunks / 128 steps, alongside the real mujoco rollout and
the anchored (mode 1) version, all from the same fresh opening frame
(`qa/policy_scaleup/closed_loop_demo/`, `viz_policy_modes.py`).

**Result**: mode 1 (anchored -- Cosmos re-grounds to a real frame every chunk)
tracks the real rollout closely for the full 128 steps, one small transient
glitch aside. Mode 2 (pure autoregressive -- no real frames after frame 0)
degrades progressively: tracks reasonably for ~3 chunks, then both geometric
distortion (already seen at a shorter 4-chunk horizon) and a new symptom that
only the longer horizon revealed -- **exposure/color drift**, the whole scene's
lighting washing out into a flatter, paler palette by chunk 4+.

**Diagnosis**: since the same policy and the same action sequence drive both
modes, and the only variable that changes is whether Cosmos conditions on a real
frame or its own last generated frame, the divergence is attributable to Cosmos's
autoregressive stability, not the policy. This is corroborated by the closed-loop
success-rate numbers, which were already measured independently of Cosmos
entirely (`POLICY_TRAINING.md`'s eval): 86-90% real-simulator success across
splits. The policy is not the bottleneck here.

**Why "more of the same training data" likely isn't sufficient on its own**:
`train_lora_pilot.py`'s training data has only ever shown Cosmos clean, real
conditioning frames (`encode_gt_latents` always reads real `sample.gt_frames()`).
Mode 2 puts Cosmos in a situation it was never trained for: conditioning on a
frame that already carries its own generation artifacts. That's a train/inference
distribution mismatch (exposure bias), not a data-volume problem -- closing it
needs training examples that specifically include imperfect/self-referential
conditioning frames, not just more clean ones. See `COSMOS_FINETUNE_V2.md` for
the resulting plan.

## Where this leaves the project

Cosmos3-Edge is not, right now, a trustworthy frozen "virtual environment" for this
dataset's specific rendering style. Calibrating the action scale further won't fix
a visual domain gap, and (§7) neither does a bigger checkpoint. Remaining options:

1. ~~Try Cosmos3-Nano~~ -- tried (§7). Same weak/incoherent response; ruled out.
2. **Push the mujoco rendering toward photorealism** -- tried (§6). Real,
   measurable, but partial improvement; not sufficient alone.
3. **Fine-tune Cosmos3-Edge** (LoRA, keeping it Edge-sized for the eventual policy-
   training loop) -- scoped in `FINETUNING_SCOPE.md`, piloted at 800 steps
   (`LORA_PILOT_RESULTS.md`), then scaled to a 21,483-step / 10.75h overnight run
   on 800 episodes (`LORA_SCALEUP_RESULTS.md`). Confirmed at scale, on 25 held-out
   test episodes: direction-correctness improved 60% -> 84% and the untrained-
   domain's outright hallucinations are gone, replaced with stable, plausible
   rollouts. Action-magnitude sensitivity still hasn't moved even with 26x more
   steps -- looks like a capacity/objective limit of attention-only LoRA rather
   than an undertrained one; `mlp_moe_gen` LoRA (FINETUNING_SCOPE.md's tier-3
   fallback) is the next lever if that turns out to matter for policy training.
4. **Descope Cosmos's role**: keep MuJoCo as the actual training/eval environment
   (it already works, is exact, and is free), and treat Cosmos-conditioned rollouts
   as an exploratory/auxiliary signal rather than the environment the policy is
   trained or evaluated against.
