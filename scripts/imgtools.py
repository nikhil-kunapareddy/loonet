#!/usr/bin/env python3
"""Shared metrics, fingerprints and staging helpers for the data pipeline.

The pipeline is four copy-only hops under data/:

    raw -> renamed -> no_dups -> quality_check -> un_annotated

One script owns each hop. Everything two of them have to agree on lives here:
the quality thresholds and their defaults, the focus measure, and the duplicate
fingerprint. check_images.py reports the same numbers without touching a file,
so what the report says and what the stages do cannot drift apart.

Not a command; nothing here runs on its own.
"""
import hashlib
import os
import shutil

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".webp"}
HASH_SIZE = 8                      # dHash grid -> 64-bit fingerprint
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
FLAT = 5.0                         # a featureless frame has no structure to
                                   # fingerprint: every pixel ties with its
                                   # neighbour, so the hash comes out all zeros
                                   # and pure black, white and grey all "match"

# EXIF tag numbers, so the pipeline reads capture data without a tag-name table
MAKE, MODEL, DATETIME, ORIENTATION = 271, 272, 306, 274
EXIF_IFD = 0x8769                  # DateTimeOriginal lives in this sub-IFD
DATETIME_ORIGINAL = 36867


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
    """Return a metrics dict for one file, or {"error": ...} if it won't open.

    All metrics are computed on the EXIF-upright grayscale image, downscaled so
    the long side is work_side. Normalising the scale first is what makes the
    blur number comparable between a 24 MP DSLR frame and a 1080p video grab;
    without it, the same photo scores wildly differently at each size.

    Focus and contrast are measured on the best region, not the whole frame. On
    this dataset the whole-frame versions rejected a pin-sharp bird on flat
    water and another on open sky, because 95% of those frames is featureless
    background and it drags every full-image average down. A frame where
    nothing at all is in focus still fails, since then no region scores well
    either.
    """
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


def grade(m, t):
    """Return the list of reasons this image fails, empty if it passes.

    t is any object carrying the thresholds added by add_quality_args, i.e. the
    parsed argparse namespace of whichever script is asking.
    """
    bad = []
    if min(m["width"], m["height"]) < t.min_side:
        bad.append(f"too small ({m['width']}x{m['height']})")
    if m["focus"] < t.focus:
        bad.append(f"out of focus (focus {m['focus']:.0f} < {t.focus:g})")
    if m["brightness"] < t.dark:
        bad.append(f"underexposed (brightness {m['brightness']:.0f})")
    elif m["brightness"] > t.bright:
        bad.append(f"overexposed (brightness {m['brightness']:.0f})")
    if m["blown_frac"] > t.blown:
        bad.append(f"blown highlights ({m['blown_frac']:.0%} of pixels)")
    if m["dark_frac"] > t.crushed:
        bad.append(f"crushed shadows ({m['dark_frac']:.0%} of pixels)")
    if m["contrast"] < t.contrast:
        bad.append(f"low contrast (range {m['contrast']:.0f} < {t.contrast:g})")
    return bad


def add_quality_args(p):
    """Add the quality thresholds to a parser.

    Thresholds are defaults, not truths. Run check_images.py --csv first, look
    at the spread for your own footage, then set the flags.
    """
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
    return q


def add_hamming_arg(p):
    """Add the near-duplicate distance flag.

    Matching is scale- and re-compression-invariant but not rotation-invariant:
    a 90-degree-rotated copy reads as a different image.
    """
    p.add_argument("--hamming", type=int, default=5,
                   help="near-duplicate if fingerprints differ by <= this many "
                        "of 64 bits (default 5; 0 disables, 10 is loose)")


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


def near_duplicate_groups(hashes, max_dist, union=None):
    """Cluster fingerprints within max_dist bits of each other.

    Compares row i against the whole tail at once; O(n^2) bit work, but it is
    one vectorised XOR per image, which stays quick well past any hand-labelled
    dataset size.

    Pass union to keep joining an existing Union (of the same length), so
    byte-identical and merely-similar matches can land in one grouping.
    """
    u = union if union is not None else Union(len(hashes))
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


def keeper(names, metrics, pinned=()):
    """Pick the copy worth keeping: most pixels, then sharpest.

    Ties break toward the shortest name, which keeps IMG_9497.HEIC over
    IMG_9497(1).HEIC -- copies are the ones that pick up a suffix.

    Names in pinned win outright. A stage passes the files it has already
    copied forward, because a name that is downstream may already be annotated:
    once a copy has been chosen, a better one arriving later does not get to
    replace it.
    """
    best = [n for n in names if n in pinned] or list(names)
    return min(best, key=lambda n: (-metrics[n]["megapixels"], -metrics[n]["focus"],
                                    len(n), n))


# ---------------------------------------------------------------- staging

def list_images(folder, recursive=False):
    """Sorted paths of the images in folder.

    The extension filter also keeps a stage's manifest CSV out of the way, so
    the bookkeeping can live in the same folder as the images it describes.
    """
    if recursive:
        files = [os.path.join(root, f)
                 for root, _, fs in os.walk(folder) for f in fs
                 if os.path.splitext(f)[1].lower() in EXTS]
    else:
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in EXTS]
    return sorted(f for f in files if os.path.isfile(f))


def rel(path, folder):
    """Name to print and record for a file: its path relative to its stage."""
    return os.path.relpath(path, folder)


def copy_into(src, dest, dry_run=False):
    """Copy src to dest, preserving mtime.

    copy2 rather than copy: rename_images.py dates undated web downloads from
    the file mtime, so a stage that reset it would change those names on a
    later re-run.
    """
    if dry_run:
        return
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    shutil.copy2(src, dest)


def require_dir(path, what="directory"):
    if not os.path.isdir(path):
        raise SystemExit(f"{what} not found: {path}")
    return path
