#!/usr/bin/env python3
"""Generate a synthetic clip for smoke-testing the pipeline plumbing.

Honest about what this is: YOLO will find **no people** in a moving rectangle,
so this clip produces an empty memory graph. That is the point — it exercises
the parts that have nothing to do with the models:

  * the capture loop and the MOG2 motion gate (the rectangle moves, so frames
    get through; the still tail at the end gets skipped),
  * the settled-keyframe detector,
  * the dashboard, the MJPEG stream and the zone editor.

If you want to see the memory graph fill up, point `--video` at real footage of
real people. Example 1 (`examples/01_memory_graph`) shows the graph itself with
no camera at all.

    python examples/03_video_pipeline/make_clip.py         # writes clip.mp4
    python examples/03_video_pipeline/make_clip.py --seconds 20 --out /tmp/x.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 640, 480, 20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path(__file__).with_name("clip.mp4")))
    p.add_argument("--seconds", type=float, default=12.0)
    args = p.parse_args()

    n = int(args.seconds * FPS)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"OpenCV could not open a writer for {args.out} "
                         "(no mp4v encoder?). Try an .avi path instead.")

    for i in range(n):
        frame = np.full((H, W, 3), 30, np.uint8)
        cv2.rectangle(frame, (40, 380), (600, 420), (60, 60, 60), -1)   # "floor"

        # Moves for the first 75% of the clip, then holds still — so you can
        # watch the motion gate stop letting frames through.
        moving = i < int(n * 0.75)
        x = 60 + int((W - 200) * (i / max(1, int(n * 0.75)))) if moving else W - 140
        cv2.rectangle(frame, (x, 220), (x + 80, 400), (0, 140, 220), -1)

        label = "moving — motion gate open" if moving else "still — frames skipped"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(frame, f"synthetic clip  {i + 1}/{n}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    size = Path(args.out).stat().st_size
    print(f"wrote {args.out}  ({n} frames, {args.seconds:g}s, {size / 1024:.0f} KB)")
    print("\nNext:")
    print(f"  python -m intelligence_os.web --video {args.out} --port 8000")
    print("  → http://localhost:8000")


if __name__ == "__main__":
    main()
