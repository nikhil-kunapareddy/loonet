"""Reassemble tiles named <base><row><col>.jpg into the full mosaic.

Replaces tools/stitcher.py from AUBIRDSTEST (which used per-pixel Python loops
and hardcoded Windows paths). Row/col are the last two digits before the suffix.

Usage:
    python src/stitch_tiles.py outputs/aubirds_test out.jpg --suffix _det
"""
import argparse
import os
import re
import sys

from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tile_dir")
    p.add_argument("output")
    p.add_argument("--suffix", default="", help="text between the row/col digits and the extension")
    p.add_argument("--scale", type=float, default=1.0, help="downscale the mosaic")
    a = p.parse_args()

    pat = re.compile(r"^(?P<base>.+?)(?P<row>\d)(?P<col>\d)" + re.escape(a.suffix) + r"\.(jpg|jpeg|png)$", re.I)
    tiles = {}
    for f in os.listdir(a.tile_dir):
        m = pat.match(f)
        if m:
            tiles[(int(m["row"]), int(m["col"]))] = os.path.join(a.tile_dir, f)
    if not tiles:
        sys.exit(f"no tiles matching <base><row><col>{a.suffix}.jpg in {a.tile_dir}")

    nrow = max(r for r, _ in tiles) + 1
    ncol = max(c for _, c in tiles) + 1
    tw, th = Image.open(next(iter(tiles.values()))).size
    mosaic = Image.new("RGB", (ncol * tw, nrow * th))
    for (r, c), path in sorted(tiles.items()):
        mosaic.paste(Image.open(path), (c * tw, r * th))

    if a.scale != 1.0:
        mosaic = mosaic.resize((int(mosaic.width * a.scale), int(mosaic.height * a.scale)), Image.LANCZOS)
    mosaic.save(a.output, quality=90)
    print(f"{nrow}x{ncol} tiles of {tw}x{th} -> {mosaic.width}x{mosaic.height}  {a.output}")


if __name__ == "__main__":
    main()
