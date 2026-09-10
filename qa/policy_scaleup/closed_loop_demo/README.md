# Closed-loop CNN-policy <-> Cosmos demo, at a longer horizon

This is the literal loop requested: **policy -> action -> Cosmos -> predicted
observation -> policy -> action -> Cosmos -> ...** -- which is exactly Mode 2
from `qa/policy_scaleup/modes/`, just run for 8 chunks (128 steps) instead of 4
(64), on a freshly-sampled opening frame (`make_fresh_demo_episode.py`, seed
20260910, not tied to any existing train/val/test episode) rather than reusing
an eval episode.

## Why go longer

The 4-chunk demos didn't run long enough to show the full shape of Mode 2's
failure mode. This one does.

## What's here

- `mode0_real.mp4` -- the trained policy's real closed-loop rollout, 128 steps,
  real physics.
- `mode1_parallel_imagination.mp4` -- Cosmos re-imagining each chunk anchored to
  a real frame every time (no drift compounding). Tracks the real rollout
  closely for all 8 chunks, with one small transient visual glitch around
  chunk 6.
- `mode2_pure_imagination.mp4` -- the actual policy<->Cosmos closed loop, no
  real frames after frame 0. Diverges progressively:
  - Chunks 1-3: tracks reasonably well.
  - Chunks 4+: **exposure/color drift** -- the whole scene's lighting washes
    out into a flatter, paler palette. This is a new failure mode the shorter
    (4-chunk) demos didn't run long enough to reveal -- on top of the
    geometric distortion (the arm bending into a thin bent shape, object
    identity blurring into indistinct colored blobs) already seen at the
    shorter horizon.
- `comparison_strip.png` -- all three, stacked.

## Read

Worth keeping as the canonical "how far does pure imagination actually hold up"
demo -- it's more informative than the shorter version specifically because it
ran long enough to show that Mode 2's drift isn't just geometric, it's
photometric too. Mode 1 (anchored) remains the trustworthy visualization mode
for anything beyond a couple of chunks; Mode 2 is honestly a curiosity/limits
demo at the current level of Cosmos fine-tuning, not yet a reliable long-horizon
simulator.
