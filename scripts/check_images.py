#!/usr/bin/env python3
"""Quality-screen a folder of images and flag duplicates. Reports only.

Two reports, in this order:

  1. QUALITY  - images that should not go into the training set (unreadable,
                too small, out of focus, badly exposed, or near-flat), each
                with the reason and the measured value.
  2. DUPLICATES - exact duplicates (identical bytes) and near-duplicates
                (same scene at a different size, quality, or crop-free
                re-save), grouped.

Nothing is deleted or moved. This is the calibration tool: run it with --csv,
look at the spread for your own footage, then set the thresholds on
quality_filter.py, which acts on the same numbers. Point it at any pipeline
stage to see what that stage is holding.

All metrics are computed on the EXIF-upright grayscale image, downscaled so the
long side is --work-side (default 1024). Normalising the scale first is what
makes the blur number comparable between a 24 MP DSLR frame and a 1080p video
grab; without it, the same photo scores wildly differently at each size.

Focus and contrast are measured on the best region, not the whole frame. On
this dataset the whole-frame versions rejected a pin-sharp bird on flat water
and another on open sky, because 95% of those frames is featureless background
and it drags every full-image average down. A frame where nothing at all is in
focus still fails, since then no region scores well either.

Duplicate matching is scale- and re-compression-invariant but not rotation-
invariant: a 90-degree-rotated copy reads as a different image. The two tests
are reported separately here; dedupe_images.py merges them, so a byte-identical
pair that is also near a third image is two groups below and one group there.

Usage:
    scripts/check_images.py data/raw
    scripts/check_images.py data/no_dups --csv outputs/qc.csv
    scripts/check_images.py data/renamed --focus 300 --hamming 8 -r
"""
import argparse
import csv
import os
import sys

import numpy as np

import imgtools as it


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
    it.add_quality_args(p)
    dups = p.add_argument_group("duplicate detection")
    it.add_hamming_arg(dups)
    a = p.parse_args()

    if not os.path.isdir(a.folder):
        sys.exit(f"not a directory: {a.folder}")

    files = it.list_images(a.folder, a.recursive)
    if not files:
        sys.exit(f"no images found in {a.folder}")

    print(f"checking {len(files)} images in {a.folder}", flush=True)

    metrics, unreadable, ok_paths = {}, [], []
    for path in files:
        m = it.analyze(path, a.work_side)
        name = it.rel(path, a.folder)
        if "error" in m:
            unreadable.append((name, m["error"]))
            continue
        metrics[name] = m
        ok_paths.append(path)

    # ------------------------------------------------------------ report 1
    failures = [(it.rel(p, a.folder), it.grade(metrics[it.rel(p, a.folder)], a))
                for p in ok_paths]
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
        by_digest.setdefault(it.sha256(path), []).append(it.rel(path, a.folder))
    exact = sorted((sorted(v) for v in by_digest.values() if len(v) > 1))
    exact_dupes = {n for g in exact for n in g}

    names = [it.rel(p, a.folder) for p in ok_paths]
    # Featureless frames are excluded: their fingerprints are all-zero, so they
    # would all cluster together and bury the real matches. They are already in
    # the quality report, and identical ones still surface as exact duplicates.
    fingerprinted = [n for n in names if metrics[n]["contrast"] >= it.FLAT]
    flat_skipped = len(names) - len(fingerprinted)

    near = []
    if a.hamming > 0 and fingerprinted:
        stacked = np.array([metrics[n]["hash"] for n in fingerprinted], dtype=np.uint8)
        groups, dist = it.near_duplicate_groups(stacked, a.hamming)
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
        keep = it.keeper(g, metrics)
        print(f"  exact group {i} (identical bytes):")
        for n in g:
            print(f"      {'KEEP  ' if n == keep else 'drop  '}{n}")
    for i, (g, spread) in enumerate(near, 1):
        keep = it.keeper(g, metrics)
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
                reasons = it.grade(m, a)
                w.writerow([name] + [round(m[c], 3) for c in cols]
                           + ["; ".join(reasons) if reasons else "ok"])
        print(f"\nmetrics written to {a.csv}")


if __name__ == "__main__":
    main()
