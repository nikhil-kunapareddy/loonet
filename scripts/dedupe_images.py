#!/usr/bin/env python3
"""Pipeline stage 2: data/renamed -> data/no_dups.

Two tests, in one grouping:

  - identical bytes (SHA-256), which catches the download you saved twice and
    the "(1)" copy the browser made;
  - a 64-bit perceptual fingerprint, which catches the same photo at a
    different size or JPEG quality, where the bytes differ but the picture
    does not.

Everything linked by either test lands in one group, and exactly one member of
each group is copied forward: most pixels, then sharpest, then shortest name.
(check_images.py reports the two tests as separate lists, so a byte-identical
pair that is also near a third image shows up there as two groups. Here it is
one group with one keeper, which is what copying forward requires.)

Featureless frames -- flat water, blank sky -- are left out of fingerprint
matching. Their hashes come out all zeros, so they would all match each other
and bury the real duplicates. Identical ones are still caught by SHA-256, and
the quality stage is where they get dropped for being featureless.

Files that will not open are copied forward rather than judged here; stage 3 is
the one place that rejects on quality, and its report is where you want to read
about them.

Re-running is safe. A file already in data/no_dups wins its group even if a
larger copy turns up later, because a name that has moved downstream may
already be annotated. Nothing in data/renamed is modified or removed: the
duplicates stay there, and data/no_dups/duplicates.csv says which keeper each
one lost to.

Matching is scale- and re-compression-invariant but not rotation-invariant: a
90-degree-rotated copy reads as a different image.

Usage:
    scripts/dedupe_images.py
    scripts/dedupe_images.py -n                  # print the plan, change nothing
    scripts/dedupe_images.py --hamming 8         # looser: catches crops and edits
    scripts/dedupe_images.py --hamming 0         # identical bytes only
"""
import argparse
import csv
import os

import numpy as np

import imgtools as it

SRC = os.path.join(it.DATA, "renamed")
DEST = os.path.join(it.DATA, "no_dups")
REPORT = "duplicates.csv"
COLS = ["dropped", "kept", "reason", "bits_from_keeper"]


def bit_distance(a, b):
    """Fingerprint distance in bits, 0-64."""
    return int(it.POPCOUNT[a ^ b].sum())


def reason(name, keep, metrics, digests):
    """Why this file lost, and how far it sits from the keeper."""
    if digests[name] == digests[keep]:
        return "identical bytes", ""
    ha, hb = metrics[name].get("hash"), metrics[keep].get("hash")
    if ha is None or hb is None:
        return "grouped by an identical copy", ""
    return "near-duplicate", bit_distance(ha, hb)


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src", default=SRC, help=f"renamed images (default {it.rel(SRC, it.ROOT)})")
    p.add_argument("--dest", default=DEST, help=f"unique output (default {it.rel(DEST, it.ROOT)})")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="also descend into subdirectories of --src")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="print the plan and change nothing on disk")
    p.add_argument("--work-side", type=int, default=1024,
                   help="long side the metrics are measured at (default 1024)")
    it.add_hamming_arg(p)
    a = p.parse_args()

    it.require_dir(a.src, "source folder")
    files = it.list_images(a.src, a.recursive)
    if not files:
        raise SystemExit(f"no images found in {a.src}")

    already = {it.rel(f, a.dest) for f in it.list_images(a.dest, a.recursive)} \
        if os.path.isdir(a.dest) else set()
    print(f"checking {len(files)} image(s) in {a.src} "
          f"({len(already)} already in {a.dest})", flush=True)

    names, metrics, digests, unreadable = [], {}, {}, []
    for path in files:
        name = it.rel(path, a.src)
        m = it.analyze(path, a.work_side)
        if "error" in m:
            unreadable.append((name, m["error"]))
            # No pixels to compare, so it can only match on bytes. The zeros
            # make keeper() prefer any readable copy in the same group.
            m = {"width": 0, "height": 0, "megapixels": 0.0, "focus": 0.0,
                 "contrast": 0.0, "hash": None}
        names.append(name)
        metrics[name] = m
        digests[name] = it.sha256(path)

    index = {n: i for i, n in enumerate(names)}
    u = it.Union(len(names))

    by_digest = {}
    for n in names:
        by_digest.setdefault(digests[n], []).append(n)
    for group in by_digest.values():
        for n in group[1:]:
            u.join(index[group[0]], index[n])

    fingerprinted = [n for n in names
                     if metrics[n]["hash"] is not None
                     and metrics[n]["contrast"] >= it.FLAT]
    flat_skipped = len(names) - len(fingerprinted)
    if a.hamming > 0 and fingerprinted:
        stacked = np.array([metrics[n]["hash"] for n in fingerprinted], dtype=np.uint8)
        groups, _ = it.near_duplicate_groups(stacked, a.hamming)
        for g in groups:
            for j in g[1:]:
                u.join(index[fingerprinted[g[0]]], index[fingerprinted[j]])

    dropped = {}                      # name -> (keeper, reason, bits)
    groups = []
    for g in u.groups():
        members = sorted(names[i] for i in g)
        keep = it.keeper(members, metrics, already)
        groups.append((members, keep))
        for n in members:
            if n != keep:
                why, bits = reason(n, keep, metrics, digests)
                dropped[n] = (keep, why, bits)
    groups.sort()

    print()
    print("=" * 72)
    print(f"DUPLICATES: {len(groups)} group(s), {len(dropped)} file(s) dropped")
    print("=" * 72)
    if flat_skipped and a.hamming > 0:
        print(f"({flat_skipped} featureless image(s) left out of fingerprint matching)")
    if unreadable:
        print(f"({len(unreadable)} unreadable image(s) passed through to the "
              f"quality stage)")
    if not groups:
        print("(none - every image is unique)")
    for i, (members, keep) in enumerate(groups, 1):
        print(f"  group {i}:")
        for n in members:
            m = metrics[n]
            if n == keep:
                pin = " (already downstream)" if n in already else ""
                print(f"      KEEP  {n}  ({m['width']}x{m['height']}, "
                      f"focus {m['focus']:.0f}){pin}")
            else:
                why, bits = dropped[n][1:]
                detail = f"{why}, {bits}/64 bits" if bits != "" else why
                print(f"      drop  {n}  ({m['width']}x{m['height']}) - {detail}")

    keepers = [n for n in names if n not in dropped]
    new = [n for n in keepers if n not in already]
    for n in new:
        it.copy_into(os.path.join(a.src, n), os.path.join(a.dest, n), a.dry_run)

    print()
    print(f"{'would copy' if a.dry_run else 'copied'} {len(new)} unique image(s) "
          f"-> {a.dest}")
    print(f"  {len(keepers)} unique of {len(names)}; "
          f"{len(keepers) - len(new)} were already there")

    if a.dry_run:
        print(f"\ndry run: {os.path.join(a.dest, REPORT)} not written")
        return

    os.makedirs(a.dest, exist_ok=True)
    report = os.path.join(a.dest, REPORT)
    with open(report, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for n in sorted(dropped):
            keep, why, bits = dropped[n]
            w.writerow([n, keep, why, bits])
    print(f"{report} lists the {len(dropped)} dropped file(s)")


if __name__ == "__main__":
    main()
