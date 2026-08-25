#!/usr/bin/env python3
"""Pipeline stage 4: data/quality_check -> data/un_annotated.

The last hop makes the screened images openable by an annotation tool and
records where each one came from.

  - Everything becomes JPEG. HEIC is the reason: an iPhone shoots it, and CVAT,
    Label Studio and LabelImg will not read it.
  - EXIF rotation is baked into the pixels and the orientation tag is dropped.
    A tool that honours the tag and a tool that ignores it disagree about which
    way up the image is, and boxes drawn against one are wrong in the other.
    After this stage there is nothing left to disagree about.
  - All other metadata goes too, since re-saving without EXIF is what clears
    the orientation. The manifest keeps the part that matters.

A file that is already JPEG and already upright is copied byte for byte instead
of re-saved, because re-encoding a JPEG costs quality for nothing.

data/un_annotated/manifest.csv maps every final image back to the raw file it
came from, with the source and capture time read out of
data/renamed/rename_map.csv. It is rebuilt from a scan of the folder on every
run, so it stays true even if you delete images by hand.

Re-running only writes what is missing. Nothing in data/quality_check is
modified or removed.

Usage:
    scripts/prep_annotation.py
    scripts/prep_annotation.py -n                # print the plan, write nothing
    scripts/prep_annotation.py --quality 100     # near-lossless re-saves
"""
import argparse
import csv
import os

from PIL import Image, ImageOps

import imgtools as it

SRC = os.path.join(it.DATA, "quality_check")
DEST = os.path.join(it.DATA, "un_annotated")
RENAME_MAP = os.path.join(it.DATA, "renamed", "rename_map.csv")
MANIFEST = "manifest.csv"
COLS = ["final", "raw", "source", "captured", "width", "height"]
UPRIGHT = (None, 0, 1)           # orientation values that need no rotation
KEEP_AS_IS = (".jpg", ".jpeg")


def to_rgb(im):
    """Drop the image to plain RGB, flattening any transparency onto white.

    White, not the default black: a transparent PNG composited onto black
    becomes a dark blob, which is harder to draw a box on than a pale one.
    """
    if im.mode == "RGB":
        return im
    if im.mode in ("RGBA", "LA", "P"):
        rgba = im.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[-1])
        return flat
    return im.convert("RGB")


def can_pass_through(path, ext):
    """True if the file is already an upright JPEG, so copying beats re-saving."""
    if ext not in KEEP_AS_IS:
        return False
    try:
        with Image.open(path) as im:      # header only, no decode
            return (im.getexif().get(it.ORIENTATION) in UPRIGHT
                    and im.mode in ("RGB", "L"))
    except Exception:
        return False


def convert(path, out, quality, dry_run):
    """Write path to out as an upright, metadata-free JPEG."""
    if dry_run:
        return
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)  # rotation into the pixels
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        # No exif= argument, so the orientation tag does not survive
        to_rgb(im).save(out, "JPEG", quality=quality, optimize=True)


def load_lineage(path):
    """Stage-1 rows keyed by filename stem.

    Keyed by stem rather than name because this stage changes the extension;
    the stem carries the sequence number and is what stays constant.
    """
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {os.path.splitext(r["renamed"])[0]: r
                for r in csv.DictReader(f) if r.get("renamed")}


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src", default=SRC, help=f"screened images (default {it.rel(SRC, it.ROOT)})")
    p.add_argument("--dest", default=DEST,
                   help=f"annotation-ready output (default {it.rel(DEST, it.ROOT)})")
    p.add_argument("--rename-map", default=RENAME_MAP,
                   help="stage-1 map, for the manifest's raw/source/captured columns")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="also descend into subdirectories of --src")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="print the plan and change nothing on disk")
    p.add_argument("--quality", type=int, default=95,
                   help="JPEG quality for re-saved images, 1-100 (default 95)")
    a = p.parse_args()

    it.require_dir(a.src, "source folder")
    files = it.list_images(a.src, a.recursive)
    if not files:
        raise SystemExit(f"no images found in {a.src}")

    already = {it.rel(f, a.dest) for f in it.list_images(a.dest, a.recursive)} \
        if os.path.isdir(a.dest) else set()
    print(f"preparing {len(files)} image(s) from {a.src} "
          f"({len(already)} already in {a.dest})", flush=True)

    copied, converted, skipped, claimed = [], [], [], {}
    for path in files:
        name = it.rel(path, a.src)
        stem, ext = os.path.splitext(name)
        final = stem + ".jpg"
        if final in claimed:
            print(f"  {name}\n      SKIPPED: {claimed[final]} already claims {final}")
            continue
        claimed[final] = name
        if final in already:
            skipped.append(final)
            continue
        out = os.path.join(a.dest, final)
        if can_pass_through(path, ext.lower()):
            it.copy_into(path, out, a.dry_run)
            copied.append(final)
        else:
            convert(path, out, a.quality, a.dry_run)
            converted.append((name, final))

    print()
    for name, final in converted:
        print(f"  {name}\n      -> {final}  (re-saved as JPEG)")
    print()
    verb = "would prepare" if a.dry_run else "prepared"
    print(f"{verb} {len(copied) + len(converted)} image(s) -> {a.dest}")
    print(f"  {len(copied)} copied unchanged, {len(converted)} converted, "
          f"{len(skipped)} already there")

    if a.dry_run:
        print(f"\ndry run: {os.path.join(a.dest, MANIFEST)} not written")
        return

    lineage = load_lineage(a.rename_map)
    if not lineage:
        print(f"note: {a.rename_map} not found, so the manifest's raw, source "
              f"and captured columns will be blank")

    os.makedirs(a.dest, exist_ok=True)
    manifest = os.path.join(a.dest, MANIFEST)
    rows = 0
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        for path in it.list_images(a.dest, a.recursive):
            final = it.rel(path, a.dest)
            src_row = lineage.get(os.path.splitext(final)[0], {})
            with Image.open(path) as im:
                width, height = im.size
            w.writerow({
                "final": final,
                "raw": src_row.get("raw", ""),
                "source": src_row.get("source", ""),
                "captured": src_row.get("captured", ""),
                "width": width,
                "height": height,
            })
            rows += 1
    print(f"{manifest} covers {rows} image(s), ready to annotate")


if __name__ == "__main__":
    main()
