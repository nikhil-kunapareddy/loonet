#!/usr/bin/env python3
"""Pipeline stage 3: data/no_dups -> data/quality_check.

Copies forward the images fit to annotate and leaves the rest behind. An image
is rejected when it will not open, is too small, is out of focus, is badly
exposed, or is near-flat -- the same tests and the same defaults that
check_images.py reports, so a file the report cleared cannot be dropped here.

Focus and contrast are measured on the sharpest and most varied region rather
than the whole frame, because a pin-sharp bird on open water is 95%
featureless background and every whole-image average rejects it.

Thresholds are defaults, not truths. Run

    scripts/check_images.py data/no_dups --csv outputs/qc.csv

first, look at the spread for your own footage, then set the flags here to
match. Every threshold is a flag.

Nothing in data/no_dups is modified or removed: rejects stay there, and
data/quality_check/rejected.csv records what failed and by how much.
Re-running only copies what is missing, so raising a threshold does not pull
back an image already passed forward -- delete it from data/quality_check (and
from data/un_annotated) to retract it.

Usage:
    scripts/quality_filter.py
    scripts/quality_filter.py -n                 # print the verdicts, copy nothing
    scripts/quality_filter.py --focus 300        # also cut merely soft frames
"""
import argparse
import csv
import os

import imgtools as it

SRC = os.path.join(it.DATA, "no_dups")
DEST = os.path.join(it.DATA, "quality_check")
REPORT = "rejected.csv"
COLS = ["image", "width", "height", "focus", "contrast", "brightness",
        "dark_frac", "blown_frac", "reasons"]


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src", default=SRC, help=f"unique images (default {it.rel(SRC, it.ROOT)})")
    p.add_argument("--dest", default=DEST, help=f"passing output (default {it.rel(DEST, it.ROOT)})")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="also descend into subdirectories of --src")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="print the verdicts and change nothing on disk")
    it.add_quality_args(p)
    a = p.parse_args()

    it.require_dir(a.src, "source folder")
    files = it.list_images(a.src, a.recursive)
    if not files:
        raise SystemExit(f"no images found in {a.src}")

    already = {it.rel(f, a.dest) for f in it.list_images(a.dest, a.recursive)} \
        if os.path.isdir(a.dest) else set()
    print(f"screening {len(files)} image(s) in {a.src} "
          f"({len(already)} already in {a.dest})", flush=True)

    passed, rejected = [], []
    for path in files:
        name = it.rel(path, a.src)
        m = it.analyze(path, a.work_side)
        if "error" in m:
            rejected.append((name, None, [f"unreadable: {m['error']}"]))
            continue
        reasons = it.grade(m, a)
        if reasons:
            rejected.append((name, m, reasons))
        else:
            passed.append((name, m))

    print()
    print("=" * 72)
    print(f"QUALITY: {len(rejected)} of {len(files)} not fit for training")
    print("=" * 72)
    if not rejected:
        print("(none - every image passed)")
    for name, _, reasons in rejected:
        print(f"  {name}\n      " + "; ".join(reasons))

    new = [(n, m) for n, m in passed if n not in already]
    for name, _ in new:
        it.copy_into(os.path.join(a.src, name), os.path.join(a.dest, name), a.dry_run)

    print()
    print(f"{'would copy' if a.dry_run else 'copied'} {len(new)} image(s) -> {a.dest}")
    print(f"  {len(passed)} passed of {len(files)}; "
          f"{len(passed) - len(new)} were already there")

    if a.dry_run:
        print(f"\ndry run: {os.path.join(a.dest, REPORT)} not written")
        return

    os.makedirs(a.dest, exist_ok=True)
    report = os.path.join(a.dest, REPORT)
    with open(report, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for name, m, reasons in rejected:
            row = [round(m[c], 3) for c in COLS[1:-1]] if m else [""] * (len(COLS) - 2)
            w.writerow([name] + row + ["; ".join(reasons)])
    print(f"{report} lists the {len(rejected)} rejected file(s)")


if __name__ == "__main__":
    main()
