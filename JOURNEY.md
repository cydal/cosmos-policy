# The journey: mujoco -> Cosmos 3 -> a working policy

A narrative reference for presenting this project live. Each section is one
beat: what we tried, what we found, and which file/image/video to pull up while
talking about it. Full technical detail lives in the linked docs; this is the
story, not the appendix.

**One-line summary**: we set out to freeze NVIDIA's Cosmos 3 as a "virtual
environment" for a robot policy. Getting there required catching a false-positive
result, diagnosing *why* a foundation world model failed on our data, fixing what
was fixable, training a policy that actually works (86-90% closed-loop success),
and being honest about what's still an open problem (Cosmos's long-horizon
stability) rather than papering over it.

---

## Act 1 -- Setup, and the mismatch nobody warns you about

**Goal**: mujoco-generated synthetic pick-and-place data + Cosmos 3 (frozen,
action-conditioned world model) + a ResNet18 CNN policy.

Getting Cosmos 3 running locally was its own small saga: a cuDNN library that
loads fine but dies on first real use (`env.sh` fixes it -- a linker search-path
problem, not a missing file), picking `Cosmos3-Edge` (4B) over `Cosmos3-Nano`
(16B) because this box's 46 GiB GPU has no unified-memory pool to lean on the way
the reference setup did.

Then the first real technical puzzle: mujoco's action space is 7-D
(`[dx,dy,dz,droll,dpitch,dyaw,gripper]`), but Cosmos's own action-conditioned
domains all expect a 10-D "unified" format (3-D translation + a 6-D rotation
representation + gripper). Bridged it with one small, well-justified conversion
(`to_cosmos10`) -- translation and gripper pass through unchanged, only the
rotation dimensions needed expanding.

*Show*: `README.md`'s "Step 2" section, `mujoco-env-dataset/synthbot/
cosmos_action.py`.

## Act 2 -- The first result looked great. It was wrong.

Fed one real mujoco frame + one real action sequence through Cosmos, got back a
video that looked like a plausible continuation. Success, right?

**No** -- and catching this is the most important beat in the whole project.
Before trusting it, we built a proper stress test: fed Cosmos a "do nothing"
action instead of the real one. It produced *statistically the same amount of
motion* as the real action. Reversing an action's sign didn't change which way
things moved either. The first "success" was noise dressed up as a result.

Diagnosed two real, separate problems:
1. Our action magnitudes (honest physical units -- metres, radians) were
   **10-40x too small** for what Cosmos's weights respond to. Confirmed two
   independent ways: a scale sweep, and asking Cosmos's own `inverse_dynamics`
   mode what action it infers from our real video.
2. Even scaled correctly, Cosmos still struggled -- because every one of its
   action-conditioned domains was trained on **real camera footage**, and our
   renders are flat-shaded synthetic. Proved this wasn't the action math by
   feeding the *same* converted action alongside Cosmos's own real reference
   photo instead of our render: strong, correctly-scaling response. Swap back to
   our render, same action: weak or hallucinated. The renderer, not the numbers,
   was the blocker.

*Show*: `VALIDATION.md` sections 1-5. `qa/bridge_domain_hallucination.png` (a
phantom object materializing out of nowhere) is the single most convincing image
in this project for "why we don't trust a result until we've tried to break it."

## Act 3 -- Ruling things out, in parallel

Two honest questions, two clean experiments:

- **"Is this just a model-size problem?"** Tried `Cosmos3-Nano` (16B, 4x bigger,
  not distilled). Same weak/incoherent response, a *third*, different failure
  mode (the gripper's base detaching and floating out of frame). Barely fit the
  GPU at all. **Size wasn't it.**
- **"Would more photorealistic rendering help?"** Added a camera-sensor
  simulation (blur, sensor noise, vignette, chromatic aberration) to the mujoco
  renderer. Real, measurable, but partial improvement -- direction-correctness
  improved, but the model still broke down at the action magnitudes needed for
  a strong response. **Necessary, not sufficient.**

*Show*: `VALIDATION.md` sections 6-7, `qa/nano_bridge_scale18_detach.png`.

## Act 4 -- Fine-tuning: scoped rigorously, then piloted small

Scoped LoRA fine-tuning as the real fix -- but scoped it properly first: verified
gradients actually flow through the frozen checkpoint, verified a fresh,
unused "domain slot" trains in complete isolation from the pretrained domains
(zero gradient leakage, checked empirically), sized the memory budget before
committing GPU-hours to it. (`FINETUNING_SCOPE.md`)

An 800-step pilot on 40 episodes: the untrained baseline hallucinated outright
(a phantom humanoid appearing mid-rollout). After 800 steps, no more
hallucination -- direction-correctness roughly doubled. Promising, not
conclusive. (`LORA_PILOT_RESULTS.md`)

Scaled to a real overnight run: 800 episodes, stride 4, **21,483 steps over
10.75 hours**, fully unattended (detached from the session so it survived
disconnection). On 25 held-out test episodes: direction-correctness **60% ->
84%**. Magnitude-sensitivity didn't move -- flagged honestly as a likely
capacity ceiling of attention-only LoRA, not an undertrained one.
(`LORA_SCALEUP_RESULTS.md`)

*Show*: `qa/lora_scaleup_ep899_before_after.png`, `qa/lora_scaleup_ep886_before_after.png`.

## Act 5 -- One example you can just watch

Numbers are convincing; a video is more convincing. Built one clean,
self-contained artifact: a real frame, its real action sequence (readable CSV,
including which steps were the scripted expert's noise-injection
perturbations), and the resulting rollout from **stock** Cosmos vs. the
**fine-tuned** checkpoint -- identical inputs, only the model differs.

*Show*: `qa_final/comparison_strip.png`, `qa_final/cosmos_initial.mp4` vs.
`qa_final/cosmos_finetuned.mp4`.

## Act 6 -- Training the actual policy (deliberately without Cosmos)

Decision point: should Cosmos be *part of* policy training, or just used to
*visualize* it afterward? Cosmos is orders of magnitude slower than the real
simulator (~4s to imagine one chunk vs. ~1.7s to generate an entire real
episode) and, per Act 2-4, still imprecise. Chose behavior cloning directly on
real mujoco data -- fast, exact, and the dataset already had everything needed
(the CNN predicts a discrete primitive id, not raw actions -- a design already
baked into `mujoco-env-dataset` from the start). Cosmos's role: visualization
and evaluation, never a gradient source. (`POLICY_TRAINING.md`)

First pass, 800 episodes: **87% train success, 60-63% val/test** -- a real
generalization gap, plus a mid-run bug worth mentioning live (GPU sitting at 0%
utilization for a very CPU-limited reason: an image-augmentation step costing
15ms/image on a 4-core box, dwarfing the actual model step -- fixed by applying
it to half the samples instead of all, +40% throughput for free).

Diagnosis, not a guess: the *specific* confusions (yaw vs. z-move,
left/right vs. forward/back) were tied to per-episode camera-angle
randomization -- a data-density problem. Scaled to **2,000 episodes**, retrained:
**86% / 82% / 90%** train/val/test. The generalization gap that was the whole
problem essentially closed -- test even edged out train.

*Show*: `qa/policy_scaleup/success_rate_comparison.png`,
`qa/policy_scaleup/confusion_matrix_comparison.png`.

## Act 7 -- Closing the loop, and a question that caught a real gap

Built the actual closed loop the original brief asked for: policy -> action ->
Cosmos -> predicted observation -> policy -> ... Two natural conditions:
**anchored** (Cosmos re-grounds to a real frame every 16-step chunk -- stays
close to real behavior throughout) and **pure imagination** (Cosmos only ever
sees its own prior output -- drifts after a few chunks: first geometric
distortion, then, at a longer 128-step horizon we specifically ran to find out,
a *second* failure mode -- progressive exposure/color drift).

Then a sharper question: **are we sure this is Cosmos's fault, not the
policy's?** Mode 1 and pure-imagination don't actually hold the *actions*
constant -- the policy reacts to whatever it currently sees, so once Cosmos
drifts even slightly, the policy might make different choices than it would
from a real frame. That's a real confound. Isolated it with a third condition:
Cosmos self-conditioning autoregressively, but driven by the **real,
ground-truth-correct** action sequence throughout -- no policy involved at all.
**Result: drifts just as badly.** Perfect actions don't save it. The
instability is Cosmos's own self-conditioning, confirmed rather than assumed.

*Show*: `qa/policy_scaleup/closed_loop_demo/comparison_strip_4row.png` -- all
four conditions, one image. This is the best "we didn't just believe our own
diagnosis, we tested it" beat in the project.

## Act 8 -- A targeted fix, run overnight, and an honest negative result

Given the diagnosis (an exposure-bias / train-inference mismatch -- Cosmos had
only ever been trained on pristine real conditioning frames, never one carrying
its own generation artifacts), built a targeted fix: mix in photoreal-perturbed
frames (cheap) and, periodically, frames the model *generated itself* moments
earlier (the direct analogue of scheduled sampling). Smoke-tested the mechanism
before committing real time to it. Launched an **18-hour** fully unattended run
-- **34,315 steps**, zero crashes, stopped cleanly on schedule.

Measured it properly across 3 episodes rather than trusting one. **No
measurable improvement** -- if anything, marginally worse on this sample. An
honest negative result: it rules out "a sparse self-conditioning signal fixes
this" rather than confirming a win, and points at either a stronger
self-conditioning ratio or more LoRA capacity (the `mlp_moe_gen` tier, already
scoped and waiting) as the next real lever -- not a dead end, a ruled-out path.

*Show*: the drift-curve numbers in this doc's companion analysis, or just say it
plainly -- this is the moment to talk about intellectual honesty in an ML
project, not spin a null result into a win.

## Where things actually stand

- **The policy works.** 86-90% closed-loop success across train/val/test, in
  the real simulator, never drops the object even when it fails.
- **Fine-tuned Cosmos (21K-step checkpoint) is a real, working short-horizon
  world model.** Anchored, chunk-by-chunk visualization is trustworthy.
- **Long-horizon pure imagination is still an open problem.** Characterized
  honestly, root-caused correctly, one fix attempted and ruled out -- not yet
  solved.
- **The throughline**: every claim in this project was tested before it was
  believed, including our own. That's the actual deliverable, as much as any
  single number.
