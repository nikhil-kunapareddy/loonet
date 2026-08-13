#!/usr/bin/env python3
"""Quality-screen a folder of images and flag duplicates before annotation.

Two reports, in this order:

  1. QUALITY  - images that should not go into the training set (unreadable,
                too small, out of focus, badly exposed, or near-flat), each
                with the reason and the measured value.
  2. DUPLICATES - exact duplicates (identical bytes) and near-duplicates
                (same scene at a different size, quality, or crop-free
                re-save), grouped.

All metrics are computed on the EXIF-upright grayscale image, downscaled so the
long side is --work-side (default 1024). Normalising the scale first is what
makes the blur number comparable between a 24 MP DSLR frame and a 1080p video
grab; without it, the same photo scores wildly differently at each size.

Focus and contrast are measured on the best region, not the whole frame. On
this dataset the whole-frame versions rejected a pin-sharp bird on flat water
and another on open sky, because 95% of those frames is featureless background
and it drags every full-image average down. A frame where nothing at all is in
focus still fails, since then no region scores well either.

Thresholds are defaults, not truths. Run with --csv first, look at the spread
for your own footage, then set the flags. Nothing is deleted or moved.

Duplicate matching is scale- and re-compression-invariant but not rotation-
invariant: a 90-degree-rotated copy reads as a different image.

Usage:
    scripts/check_images.py data/raw/images
    scripts/check_images.py data/frames --csv outputs/qc.csv
    scripts/check_images.py data/raw/images --focus 300 --hamming 8 -r
"""
import argparse
import csv
import hashlib
import os
import sys

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".webp"}
HASH_SIZE = 8                      # dHash grid -> 64-bit fingerprint
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
FLAT = 5.0                         # a featureless frame has no structure to
                                   # fingerprint: every pixel ties with its
                                   # neighbour, so the hash comes out all zeros
                                   # and pure black, white and grey all "match"


# ---------------------------------------------------------------- metrics

def laplacian_var(g):
    """Variance of the 3x3 Laplacian: the standard focus measure.

    Low variance means few sharp edges, i.e. blur.
    """
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
           - 4.0 * g[1:-1, 1:-1])
    return float(lap.var())


def tone_range(g):
    """1st-99th percentile spread, i.e. dynamic range ignoring clipped outliers."""
    p1, p99 = np.percentile(g, [1, 99])
    return float(p99 - p1)


def offsets(size, tile, step):
    """Tile origins along one axis, with the last one snapped to the edge."""
    o = list(range(0, size - tile + 1, step))
    if o[-1] != size - tile:
        o.append(size - tile)
    return o


def region_stats(g, tile):
    """Focus and contrast of the best region: (max tile blur, max tile range).

    Half-overlapping tiles, so a subject straddling a tile border still lands
    whole inside some tile.
    """
    h, w = g.shape
    if h < tile or w < tile:
        return laplacian_var(g), tone_range(g)
    step = max(1, tile // 2)
    focus = contrast = 0.0
    for y in offsets(h, tile, step):
        for x in offsets(w, tile, step):
            patch = g[y:y + tile, x:x + tile]
            focus = max(focus, laplacian_var(patch))
            contrast = max(contrast, tone_range(patch))
    return focus, contrast


def dhash(gray):
    """64-bit difference hash, packed into 8 bytes.

    Compares each pixel with its right-hand neighbour on a 9x8 thumbnail, so it
    keys on structure and ignores resolution, re-compression and overall
    brightness -- the things that differ between two copies of one photo.
    """
    small = gray.resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
    a = np.asarray(small, dtype=np.int16)
    return np.packbits(a[:, 1:] > a[:, :-1])


def analyze(path, work_side):
    """Return a metrics dict for one file, or {"error": ...} if it won't open."""
    if os.path.getsize(path) == 0:
        return {"error": "empty file (0 bytes)"}
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)   # score what a loader would see
            w, h = im.size
            gray = im.convert("L")             # forces a full decode: truncated
                                               # files raise here, as intended
            scale = work_side / max(w, h)
            if scale < 1.0:
                gray_small = gray.resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))),
                    Image.LANCZOS,
                )
            else:
                gray_small = gray
            g = np.asarray(gray_small, dtype=np.float32)
            fingerprint = dhash(gray)
    except Exception as e:
        # Pillow embeds the full path in its message; the name is already printed
        return {"error": f"{type(e).__name__}: {str(e).replace(path, os.path.basename(path))}"}

    focus, contrast = region_stats(g, max(32, work_side // 8))
    return {
        "width": w,
        "height": h,
        "megapixels": w * h / 1e6,
        "focus": focus,                    # sharpest region
        "contrast": contrast,              # most tonally varied region
        "focus_global": laplacian_var(g),   # whole-frame versions, for the CSV
        "contrast_global": tone_range(g),   # only: too harsh to reject on
        "brightness": float(g.mean()),
        "dark_frac": float((g < 16).mean()),
        "blown_frac": float((g > 240).mean()),
        "hash": fingerprint,
    }


def grade(m, a):
    """Return the list of reasons this image fails, empty if it passes."""
    bad = []
    if min(m["width"], m["height"]) < a.min_side:
        bad.append(f"too small ({m['width']}x{m['height']})")
    if m["focus"] < a.focus:
        bad.append(f"out of focus (focus {m['focus']:.0f} < {a.focus:g})")
    if m["brightness"] < a.dark:
        bad.append(f"underexposed (brightness {m['brightness']:.0f})")
    elif m["brightness"] > a.bright:
        bad.append(f"overexposed (brightness {m['brightness']:.0f})")
    if m["blown_frac"] > a.blown:
        bad.append(f"blown highlights ({m['blown_frac']:.0%} of pixels)")
    if m["dark_frac"] > a.crushed:
        bad.append(f"crushed shadows ({m['dark_frac']:.0%} of pixels)")
    if m["contrast"] < a.contrast:
        bad.append(f"low contrast (range {m['contrast']:.0f} < {a.contrast:g})")
    return bad


# ------------------------------------------------------------- duplicates

class Union:
    """Union-find, so A~B and B~C land in one group instead of two pairs."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)

    def groups(self):
        out = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return [g for g in out.values() if len(g) > 1]


def near_duplicate_groups(hashes, max_dist):
    """Cluster fingerprints within max_dist bits of each other.

    Compares row i against the whole tail at once; O(n^2) bit work, but it is
    one vectorised XOR per image, which stays quick well past any hand-labelled
    dataset size.
    """
    u = Union(len(hashes))
    dist = {}
    for i in range(len(hashes) - 1):
        d = POPCOUNT[hashes[i] ^ hashes[i + 1:]].sum(axis=1)
        for j in np.nonzero(d <= max_dist)[0]:
            k = i + 1 + int(j)
            u.join(i, k)
            dist[(i, k)] = int(d[j])
    return u.groups(), dist


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def keeper(names, metrics):
    """Pick the copy worth keeping: most pixels, then sharpest.

    Ties break toward the shortest name, which keeps IMG_9497.HEIC over
    IMG_9497(1).HEIC -- copies are the ones that pick up a suffix.
    """
    return min(names, key=lambda n: (-metrics[n]["megapixels"], -metrics[n]["focus"],
                                     len(n), n))


# ------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("folder", help="directory of images to check")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="also descend into subdirectories")
    p.add_argument("--csv", metavar="PATH",
                   help="write every measured metric here (use this to pick thresholds)")

    q = p.add_argument_group("quality thresholds")
    q.add_argument("--min-side", type=int, default=480,
                   help="reject if the shorter side is under this (default 480)")
    q.add_argument("--focus", type=float, default=150.0,
                   help="reject below this sharpest-region Laplacian variance "
                        "(default 150; raise toward 300 to also cut soft frames)")
    q.add_argument("--dark", type=float, default=40.0,
                   help="reject below this mean brightness, 0-255 (default 40)")
    q.add_argument("--bright", type=float, default=215.0,
                   help="reject above this mean brightness, 0-255 (default 215)")
    q.add_argument("--blown", type=float, default=0.25,
                   help="reject if more than this fraction is clipped white (default 0.25)")
    q.add_argument("--crushed", type=float, default=0.50,
                   help="reject if more than this fraction is clipped black (default 0.50)")
    q.add_argument("--contrast", type=float, default=30.0,
                   help="reject below this 1st-99th percentile spread in the "
                        "most varied region (default 30)")
    q.add_argument("--work-side", type=int, default=1024,
                   help="long side the metrics are measured at (default 1024)")

    d = p.add_argument_group("duplicate detection")
    d.add_argument("--hamming", type=int, default=5,
                   help="near-duplicate if fingerprints differ by <= this many "
                        "of 64 bits (default 5; 0 disables, 10 is loose)")
    a = p.parse_args()

    if not os.path.isdir(a.folder):
        sys.exit(f"not a directory: {a.folder}")

    if a.recursive:
        files = [os.path.join(root, f)
                 for root, _, fs in os.walk(a.folder) for f in fs
                 if os.path.splitext(f)[1].lower() in EXTS]
    else:
        files = [os.path.join(a.folder, f) for f in os.listdir(a.folder)
                 if os.path.splitext(f)[1].lower() in EXTS]
    files = sorted(f for f in files if os.path.isfile(f))
    if not files:
        sys.exit(f"no images found in {a.folder}")

    def label(path):
        return os.path.relpath(path, a.folder) if a.recursive else os.path.basename(path)

    print(f"checking {len(files)} images in {a.folder}", flush=True)

    metrics, unreadable, ok_paths = {}, [], []
    for path in files:
        m = analyze(path, a.work_side)
        name = label(path)
        if "error" in m:
            unreadable.append((name, m["error"]))
            continue
        metrics[name] = m
        ok_paths.append(path)

    # ------------------------------------------------------------ report 1
    failures = [(label(p), grade(metrics[label(p)], a)) for p in ok_paths]
    failures = [(n, r) for n, r in failures if r]

    print()
    print("=" * 72)
    print(f"QUALITY: {len(failures) + len(unreadable)} of {len(files)} images not fit for training")
    print("=" * 72)
    if not failures and not unreadable:
        print("(none - all images passed)")
    for name, err in unreadable:
        print(f"  {name}\n      unreadable: {err}")
    for name, reasons in failures:
        print(f"  {name}\n      " + "; ".join(reasons))

    # ------------------------------------------------------------ report 2
    by_digest = {}
    for path in ok_paths:
        by_digest.setdefault(sha256(path), []).append(label(path))
    exact = sorted((sorted(v) for v in by_digest.values() if len(v) > 1))
    exact_dupes = {n for g in exact for n in g}

    names = [label(p) for p in ok_paths]
    # Featureless frames are excluded: their fingerprints are all-zero, so they
    # would all cluster together and bury the real matches. They are already in
    # the quality report, and identical ones still surface as exact duplicates.
    fingerprinted = [n for n in names if metrics[n]["contrast"] >= FLAT]
    flat_skipped = len(names) - len(fingerprinted)

    near = []
    if a.hamming > 0 and fingerprinted:
        stacked = np.array([metrics[n]["hash"] for n in fingerprinted], dtype=np.uint8)
        groups, dist = near_duplicate_groups(stacked, a.hamming)
        for g in groups:
            members = sorted(fingerprinted[i] for i in g)
            # An exact-byte group is already reported above; only surface a
            # perceptual group if it says something new.
            if set(members) <= exact_dupes:
                continue
            gs = set(g)
            spread = max((v for k, v in dist.items()
                          if k[0] in gs and k[1] in gs), default=0)
            near.append((members, spread))
        near.sort()

    print()
    print("=" * 72)
    print(f"DUPLICATES: {len(exact)} exact group(s), {len(near)} near-duplicate group(s)")
    print("=" * 72)
    if flat_skipped and a.hamming > 0:
        print(f"({flat_skipped} featureless image(s) left out of near-duplicate matching)")
    if not exact and not near:
        print("(none)")
    for i, g in enumerate(exact, 1):
        keep = keeper(g, metrics)
        print(f"  exact group {i} (identical bytes):")
        for n in g:
            print(f"      {'KEEP  ' if n == keep else 'drop  '}{n}")
    for i, (g, spread) in enumerate(near, 1):
        keep = keeper(g, metrics)
        # "linked", not "differ": grouping is transitive, so two members at
        # opposite ends of a chain can sit further apart than this.
        print(f"  near group {i} (linked within {spread}/64 bits):")
        for n in g:
            m = metrics[n]
            marker = "KEEP  " if n == keep else "drop  "
            print(f"      {marker}{n}  ({m['width']}x{m['height']}, focus {m['focus']:.0f})")

    if a.csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        cols = ["width", "height", "megapixels", "focus", "contrast",
                "focus_global", "contrast_global", "brightness",
                "dark_frac", "blown_frac"]
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image"] + cols + ["verdict"])
            for name, err in unreadable:
                w.writerow([name] + [""] * len(cols) + [f"unreadable: {err}"])
            for name in names:
                m = metrics[name]
                reasons = grade(m, a)
                w.writerow([name] + [round(m[c], 3) for c in cols]
                           + ["; ".join(reasons) if reasons else "ok"])
        print(f"\nmetrics written to {a.csv}")


if __name__ == "__main__":
    main()
