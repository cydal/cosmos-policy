"""Minimal PyAV mp4 writer for smoke_test.py -- no report/HTML machinery needed here."""

from __future__ import annotations

import pathlib


def write_mp4(frames, path: pathlib.Path, *, fps: int = 24) -> None:
    import av
    import numpy as np

    first = np.asarray(frames[0])
    height, width = first.shape[0], first.shape[1]

    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"

    for frame in frames:
        arr = np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame)
        video_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
