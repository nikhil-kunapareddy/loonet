#!/usr/bin/env python3
"""Pipeline stage 1: data/raw -> data/renamed.

Anything new lands in data/raw under whatever name its camera, phone or
download gave it. This gives each file a pipeline name that says where it came
from and when it was taken:

    dji_20260822_143820_0001.jpg
    nikon_20150708_140141_0003.jpg
    web_20260818_0005.jpg
    ^source     ^capture time  ^sequence

source comes from the EXIF Make/Model -- DJI -> dji, NIKON CORPORATION ->
nikon, Apple iPhone -> iphone -- and any camera not in the table is tagged with
its make. A file with no EXIF Make is a web download, tagged web.

The time is EXIF DateTimeOriginal, or DateTime if that is missing. A file with
neither -- most web downloads, whose metadata the site stripped -- falls back
to its modification time and gets a date with no time of day. That is
deliberate: a name without a time is the signal that the date is when the file
reached this machine, not when the shutter fired. The two tags are independent,
so a stripped-of-camera download that kept its date is still tagged web, and
still gets a real time.

The sequence is assigned in capture order, so sorting the folder by name sorts
it chronologically. Only within a batch, though -- see below.

Names are stable across runs. data/renamed/rename_map.csv records every
raw -> renamed pair; a re-run skips raw files already listed and numbers new
arrivals after the highest sequence used so far, so a file that is already
annotated downstream never changes name underneath it. The cost is that a
second batch sorts after the first even if it was shot earlier: the name orders
by batch, then by time. The captured column in the map is the real answer.

Raw files are copied, not moved: data/raw stays the archive. A raw file is
matched by name, so replacing one with different bytes under the same name
reads as already done and is skipped.

Usage:
    scripts/rename_images.py
    scripts/rename_images.py -n                      # print the plan, write nothing
    scripts/rename_images.py --src data/incoming --dest data/renamed
"""
import argparse
import csv
import os
import re
from datetime import datetime

from PIL import Image

import imgtools as it

SRC = os.path.join(it.DATA, "raw")
DEST = os.path.join(it.DATA, "renamed")
MAP = "rename_map.csv"
COLS = ["renamed", "raw", "source", "captured", "captured_from"]

# Matched against "make model" lowercased, so the specific wins over the vague:
# an Apple phone is iphone, not apple. First hit wins, hence the order.
SOURCES = ["dji", "iphone", "ipad", "pixel", "gopro", "nikon", "canon", "sony",
           "fujifilm", "olympus", "panasonic", "autel", "parrot", "skydio"]
WEB = "web"                      # no EXIF Make at all: a download, not a capture

# .jpeg and .jpg are one format under two names, and so are .heic/.heif and
# .tif/.tiff. Collapsing them means a glob for *.jpg finds every JPEG.
SAME_FORMAT = {".jpeg": ".jpg", ".heif": ".heic", ".tiff": ".tif"}
SEQ = re.compile(r"_(\d{4,})$")  # trailing sequence of a pipeline name


def clean(s):
    """Lowercase alphanumeric run of a maker string, for use in a filename."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def read_exif(path):
    """Return (source, timestamp, captured_from) for one file.

    captured_from is "exif" when the timestamp is the camera's own, or "mtime"
    when it is the file's modification time.
    """
    make = model = stamp = None
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            make, model = ex.get(it.MAKE), ex.get(it.MODEL)
            sub = ex.get_ifd(it.EXIF_IFD)
            stamp = sub.get(it.DATETIME_ORIGINAL) or ex.get(it.DATETIME)
    except Exception:
        pass                     # unreadable here is not fatal: it still gets a
                                 # name, and check_images.py is what reports it

    blob = f"{make or ''} {model or ''}".lower()
    source = next((s for s in SOURCES if s in blob), None) or clean(make) or WEB

    when = None
    if stamp:
        try:
            when = datetime.strptime(str(stamp).strip(), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            when = None          # cameras do write 0000:00:00 00:00:00
    if when:
        return source, when, "exif"
    return source, datetime.fromtimestamp(os.path.getmtime(path)), "mtime"


def target_name(source, when, captured_from, seq, ext):
    """Build the pipeline filename. No time of day when the date is an mtime."""
    stamp = when.strftime("%Y%m%d_%H%M%S" if captured_from == "exif" else "%Y%m%d")
    ext = ext.lower()
    return f"{source}_{stamp}_{seq:04d}{SAME_FORMAT.get(ext, ext)}"


def load_map(path):
    """Existing raw -> renamed rows, keyed by raw name."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["raw"]: r for r in csv.DictReader(f) if r.get("raw")}


def highest_seq(rows, dest):
    """Largest sequence already handed out.

    Read from the map and from the names in dest, so a deleted map cannot
    restart numbering on top of files that are already there.
    """
    names = [r["renamed"] for r in rows.values()]
    if os.path.isdir(dest):
        names += [it.rel(p, dest) for p in it.list_images(dest)]
    seqs = [int(m.group(1)) for m in
            (SEQ.search(os.path.splitext(n)[0]) for n in names) if m]
    return max(seqs, default=0)


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src", default=SRC, help=f"incoming images (default {it.rel(SRC, it.ROOT)})")
    p.add_argument("--dest", default=DEST, help=f"renamed output (default {it.rel(DEST, it.ROOT)})")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="also descend into subdirectories of --src")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="print the plan and change nothing on disk")
    a = p.parse_args()

    it.require_dir(a.src, "source folder")
    mapping = os.path.join(a.dest, MAP)
    done = load_map(mapping)
    seq = highest_seq(done, a.dest)

    files = it.list_images(a.src, a.recursive)
    fresh = [f for f in files if it.rel(f, a.src) not in done]

    print(f"{len(files)} image(s) in {a.src}: "
          f"{len(files) - len(fresh)} already named, {len(fresh)} new")

    # A video is not this stage's job, but silently ignoring the one file in the
    # folder that matters most would be worse than a line of output.
    others = [f for f in sorted(os.listdir(a.src))
              if os.path.splitext(f)[1].lower() not in it.EXTS
              and os.path.isfile(os.path.join(a.src, f))
              and not f.startswith(".") and f != MAP]
    if others:
        print(f"skipped {len(others)} non-image file(s): {', '.join(others[:6])}"
              + (" ..." if len(others) > 6 else ""))
        if any(f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")) for f in others):
            print("  (video: run extract_frames.py, then drop the frames in --src)")

    if not fresh:
        print("nothing to do")
        return

    read = [(f, *read_exif(f)) for f in fresh]
    read.sort(key=lambda r: (r[2], it.rel(r[0], a.src)))   # capture order

    rows, collisions = [], []
    for path, source, when, captured_from in read:
        raw = it.rel(path, a.src)
        seq += 1
        name = target_name(source, when, captured_from, seq,
                           os.path.splitext(raw)[1])
        out = os.path.join(a.dest, name)
        if os.path.exists(out):
            # Only reachable if dest holds a name the map does not explain
            collisions.append((raw, name))
            continue
        it.copy_into(path, out, a.dry_run)
        rows.append({
            "renamed": name,
            "raw": raw,
            "source": source,
            # Date only for an mtime, same as the name: claiming a time of day
            # would dress up "when this file was written" as "when it was shot"
            "captured": (when.isoformat(timespec="seconds") if captured_from == "exif"
                         else when.date().isoformat()),
            "captured_from": captured_from,
        })

    print()
    for r in rows:
        print(f"  {r['raw']}\n      -> {r['renamed']}  ({r['captured_from']})")

    for raw, name in collisions:
        print(f"  {raw}\n      SKIPPED: {name} already exists and is not in {MAP}")

    tally = {}
    for r in rows:
        tally[r["source"]] = tally.get(r["source"], 0) + 1
    print()
    print(f"{'would rename' if a.dry_run else 'renamed'} {len(rows)} image(s) -> {a.dest}")
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())))

    if a.dry_run:
        print(f"\ndry run: {mapping} not written")
        return

    os.makedirs(a.dest, exist_ok=True)
    with open(mapping, "w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        for r in sorted(list(done.values()) + rows,
                        key=lambda r: r["renamed"]):
            w.writerow({c: r.get(c, "") for c in COLS})
    print(f"{mapping} now covers {len(done) + len(rows)} file(s)")


if __name__ == "__main__":
    main()
