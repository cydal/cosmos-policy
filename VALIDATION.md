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

## Where this leaves the project

Cosmos3-Edge is not, right now, a trustworthy frozen "virtual environment" for this
dataset's specific rendering style. Calibrating the action scale further won't fix
a visual domain gap. Options, roughly in order of effort:

1. **Try Cosmos3-Nano** (16B, not distilled) -- may generalize better to stylized
   renders; tight on this GPU's 46 GiB VRAM (~46 GiB peak measured on the DGX Spark's
   *shared* pool), so this needs its own headroom check, not an assumption it fits.
2. **Push the mujoco rendering toward photorealism** (materials/lighting/textures)
   to shrink the domain gap -- real engineering effort in mujoco-env-dataset, no
   guarantee of how much it'd help.
3. **Descope Cosmos's role**: keep MuJoCo as the actual training/eval environment
   (it already works, is exact, and is free), and treat Cosmos-conditioned rollouts
   as an exploratory/auxiliary signal rather than the environment the policy is
   trained or evaluated against.
