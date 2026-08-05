#!/usr/bin/env python3
"""Sample frames from the drone footage for still-image detection.

Requires ffmpeg on PATH. Existing frame_*.jpg are removed first so frames from a
previous run at a different interval can't linger and be counted with the new set.

Usage:
    scripts/extract_frames.py              # one frame every 3 s
    scripts/extract_frames.py 5
    scripts/extract_frames.py --src other.mp4 --dest data/other_frames
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw", "drone-footage.mp4")
DEST = os.path.join(ROOT, "data", "frames")
# %02d keeps names sorting in temporal order, which count_birds.py relies on when
# it picks up a directory with sorted(). Padding is only correct below 100 frames.
PATTERN = "frame_%02d.jpg"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("interval", nargs="?", type=float, default=3.0,
                   help="seconds between frames (default 3)")
    p.add_argument("--src", default=SRC, help="input video")
    p.add_argument("--dest", default=DEST, help="output directory")
    a = p.parse_args()

    if a.interval <= 0:
        sys.exit("interval must be positive")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (brew install ffmpeg)")
    if not os.path.exists(a.src):
        sys.exit(f"video not found: {a.src}")

    os.makedirs(a.dest, exist_ok=True)
    stale = glob.glob(os.path.join(a.dest, "frame_*.jpg"))
    for f in stale:
        os.remove(f)
    if stale:
        print(f"removed {len(stale)} frame(s) from a previous run")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", a.src,
        "-vf", f"fps=1/{a.interval}",
        "-q:v", "2",
        os.path.join(a.dest, PATTERN),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg failed (exit {e.returncode})")

    n = len(glob.glob(os.path.join(a.dest, "frame_*.jpg")))
    print(f"wrote {n} frames to {a.dest}")
    if n > 99:
        print("warning: more than 99 frames, so frame_100+ no longer sorts in "
              "temporal order", file=sys.stderr)


if __name__ == "__main__":
    main()
