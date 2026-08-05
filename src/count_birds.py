"""Bird counting with the AUBIRDSTEST Faster R-CNN (Akcay et al. 2020, Animals 10:1207).

Reimplementation of Object_detection_autest.py + tools/splitter.py, minus the
hardcoded Windows paths and the TF1 API. Tiling geometry (1024x600, no overlap,
zero-padded) and the 0.8 score threshold are taken from their scripts.

Usage:
    python src/count_birds.py data/frames --rgb --out outputs/frames
    python src/count_birds.py data/raw/images -t 0.9 --out outputs/images
"""
import argparse
import csv
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "aubirds", "frozen_inference_graph.pb")
TILE_W, TILE_H = 1024, 600          # tools/splitter.py
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff"}


def load_model(path=MODEL):
    if not os.path.exists(path):
        sys.exit(f"model not found at {path}\nrun scripts/fetch_aubirds.py to download it")
    gd = tf.compat.v1.GraphDef()
    with open(path, "rb") as f:
        gd.ParseFromString(f.read())
    wrapped = tf.compat.v1.wrap_function(
        lambda: tf.compat.v1.import_graph_def(gd, name=""), []
    )
    return wrapped.prune(
        "image_tensor:0",
        ["detection_boxes:0", "detection_scores:0", "num_detections:0"],
    )


def tiles(img):
    """Yield (tile, x0, y0), zero-padding up to a whole number of tiles.

    An image that already fits inside one tile is passed through untouched: padding
    it out to 1024x600 would only make the graph's 600x600 resizer shrink the real
    content, costing resolution on exactly the small images that need it most.
    """
    h, w = img.shape[:2]
    if h <= TILE_H and w <= TILE_W:
        yield img, 0, 0
        return
    nx, ny = -(-w // TILE_W), -(-h // TILE_H)
    padded = np.zeros((ny * TILE_H, nx * TILE_W, 3), dtype=img.dtype)
    padded[:h, :w] = img
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * TILE_H, ix * TILE_W
            yield padded[y0:y0 + TILE_H, x0:x0 + TILE_W], x0, y0


def detect(fn, img, thresh, whole):
    """Return absolute-pixel boxes [(x1,y1,x2,y2,score)] over the full image."""
    out = []
    chunks = [(img, 0, 0)] if whole else tiles(img)
    for tile, x0, y0 in chunks:
        th, tw = tile.shape[:2]
        boxes, scores, _ = fn(tf.constant(tile[None, ...]))
        boxes, scores = boxes.numpy()[0], scores.numpy()[0]
        keep = scores >= thresh
        for (ymin, xmin, ymax, xmax), s in zip(boxes[keep], scores[keep]):
            out.append((xmin * tw + x0, ymin * th + y0,
                        xmax * tw + x0, ymax * th + y0, float(s)))
    return out


def annotate(img, dets, path):
    from PIL import ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for x1, y1, x2, y2, _ in dets:
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    im.save(path, quality=88)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("-t", "--thresh", type=float, default=0.8)
    p.add_argument("--whole", action="store_true", help="no tiling; whole image at once")
    p.add_argument("--rgb", action="store_true", help="feed RGB (their script fed BGR)")
    p.add_argument("--out", default=None, help="write annotated images here")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    if os.path.isdir(a.target):
        files = sorted(
            os.path.join(a.target, f) for f in os.listdir(a.target)
            if os.path.splitext(f)[1].lower() in EXTS
        )
    else:
        files = [a.target]
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit(f"no images found in {a.target}")

    if a.out:
        os.makedirs(a.out, exist_ok=True)

    print(f"loading model ({os.path.getsize(MODEL) / 1e6:.0f} MB)...", flush=True)
    fn = load_model()

    rows = []
    print(f"{'image':<46} {'megapixels':>10} {'birds':>7} {'sec':>7}")
    print("-" * 74)
    for path in files:
        try:
            rgb = np.array(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"{os.path.basename(path):<46} {'SKIP':>10} {type(e).__name__}")
            continue
        img = rgb if a.rgb else rgb[:, :, ::-1]      # their script fed cv2 BGR
        t0 = time.time()
        dets = detect(fn, np.ascontiguousarray(img), a.thresh, a.whole)
        dt = time.time() - t0
        h, w = rgb.shape[:2]
        name = os.path.basename(path)
        print(f"{name:<46} {w * h / 1e6:>10.1f} {len(dets):>7} {dt:>7.1f}")
        rows.append((name, w, h, len(dets), round(dt, 2)))
        if a.out:
            annotate(rgb, dets, os.path.join(a.out, os.path.splitext(name)[0] + "_det.jpg"))

    print("-" * 74)
    print(f"{'TOTAL':<46} {'':>10} {sum(r[3] for r in rows):>7} {sum(r[4] for r in rows):>7.1f}")

    if a.out:
        with open(os.path.join(a.out, "counts.csv"), "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["image", "width", "height", "birds", "seconds"])
            wtr.writerows(rows)


if __name__ == "__main__":
    main()
