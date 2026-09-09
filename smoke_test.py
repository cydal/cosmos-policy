"""Prove Cosmos 3 works on this host.

    $VENV/bin/python smoke_test.py              # one short 480p clip (text-to-video)
    $VENV/bin/python smoke_test.py --image      # 1-frame text-to-image, fast
    $VENV/bin/python smoke_test.py --action     # one action-conditioned chunk

Writes into out/ and prints a luminance probe of what it produced: "it ran without
an error" is not the same as "it rendered something".
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import time

import config
import world

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

os.environ.setdefault("HF_HOME", config.HF_HOME)


def probe(path: pathlib.Path) -> str:
    """Cheap non-black-frame check: mean pixel value of the first frame."""
    from PIL import Image

    if path.suffix == ".jpg":
        im = Image.open(path)
    else:
        import av

        container = av.open(str(path))
        im = next(container.decode(video=0)).to_image()
    import numpy as np

    arr = np.asarray(im.convert("L"))
    return f"{path.name}: {arr.shape}, mean luminance {arr.mean():.1f}/255"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=config.DEFAULT_REPO)
    ap.add_argument("--image", action="store_true", help="1 frame (text-to-image)")
    ap.add_argument("--action", action="store_true", help="action-conditioned rollout")
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)

    print(f"guard: {world.guard_memory(args.repo)}")
    t0 = time.monotonic()
    pipe = world.load(args.repo)
    print(f"loaded in {(time.monotonic() - t0) / 60:.1f} min")

    if args.action:
        meta = world.load_action_example(world.snapshot(args.repo))
        frames, per_chunk = world.rollout(pipe, meta, num_chunks=1, steps=args.steps)
        import media

        mp4_path = outdir / "action.mp4"
        media.write_mp4(frames, mp4_path, fps=int(meta.get("fps", 10)))
        print(f"action rollout: {sum(per_chunk):.1f}s -> {mp4_path}")
        print(probe(mp4_path))
        return

    prompt = (
        "A red cube sits on a wooden tabletop under soft studio lighting. "
        "A robot gripper descends from above and approaches the cube."
    )
    result, secs = world.generate(
        pipe,
        prompt,
        num_frames=1 if args.image else args.frames,
        steps=args.steps,
    )

    if args.image:
        img_path = outdir / "sample.jpg"
        result.video[0].save(img_path, format="JPEG", quality=90)
        print(f"text-to-image: {secs:.1f}s -> {img_path}")
        print(probe(img_path))
        return

    import media

    mp4_path = outdir / "sample.mp4"
    media.write_mp4(result.video, mp4_path, fps=24)
    print(f"text-to-video: {secs:.1f}s ({secs / args.steps:.1f}s/step) -> {mp4_path}")
    print(probe(mp4_path))


if __name__ == "__main__":
    main()
