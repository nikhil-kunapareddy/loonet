#!/usr/bin/env python3
"""Fetch ground-level photos of invasive lake plants into data/vegnet_raw.

The source set for vegnet, the harmful-vegetation detector. Four species that
crowd out the shoreline and shallows loons nest and feed in, and that show up
in an ordinary photograph taken from the shore or a boat:

    loosestrife     purple loosestrife      Lythrum salicaria
    phragmites      common reed (invasive)  Phragmites australis australis
    water_chestnut  water chestnut          Trapa natans
    yellow_iris     yellow flag iris        Iris pseudacorus

Phragmites is pinned to the introduced subspecies: the native americanus looks
nearly the same, and a species-level query would mix the two. Eurasian
watermilfoil is left out on purpose -- it grows under the surface, and its
photos are mostly a pulled stem held up to the camera, not a plant in place.

Source is iNaturalist, research grade only, so a second observer has agreed on
the species. That matters more here than on Commons: the class label comes from
the observation, not from anyone looking at the photo later. Only CC0, CC BY
and CC BY-SA photos are taken, captive/cultivated plants are skipped, and the
search is limited to North America. One photo per observation, so a single
patch photographed twenty times does not fill a class.

Files are named <slug>_<photo id>.jpg. The name is the identity: a re-run skips
anything already on disk or listed in rejected.txt, so deleting a bad photo and
adding its photo id to rejected.txt gets a different one next time.
manifest.csv records the observation, licence and attribution of every file.

Usage:
    scripts/fetch_vegnet.py                  # fetch up to --per per species
    scripts/fetch_vegnet.py --per 60         # widen the pool
    scripts/fetch_vegnet.py --only phragmites
    scripts/fetch_vegnet.py --dry-run
"""
import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imgtools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "vegnet_raw")
MANIFEST = os.path.join(DEST, "manifest.csv")
REJECTS = os.path.join(DEST, "rejected.txt")

API = "https://api.inaturalist.org/v1/observations"
UA = "loonet-vegnet/0.1 (invasive plant detector research; one-off dataset fetch)"

# iNaturalist asks for no more than one request a second across API and media.
MIN_INTERVAL = 1.0
_last_call = [0.0]

LICENSES = ("cc0", "cc-by", "cc-by-sa")
NORTH_AMERICA = 97394
MIN_SIDE = 480          # matches imgtools.add_quality_args --min-side default
JPEG_QUALITY = 92
HAMMING = 5             # imgtools' near-duplicate default

# slug -> (iNaturalist taxon id, common name)
SPECIES = {
    "loosestrife":    (61321,  "purple loosestrife"),
    "phragmites":     (556361, "common reed"),
    "water_chestnut": (169900, "water chestnut"),
    "yellow_iris":    (47779,  "yellow flag iris"),
}


def get(url, tries=4):
    """One throttled GET, patient about 429s."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    delay = 2.0
    for attempt in range(tries):
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == tries - 1:
                raise
            pause = float(e.headers.get("Retry-After") or 0) or delay
            print(f"    HTTP {e.code}, waiting {pause:.0f}s", flush=True)
            time.sleep(min(pause, 60))
            delay = min(delay * 2, 60)
        except (urllib.error.URLError, OSError):
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
        finally:
            _last_call[0] = time.time()


def observations(taxon_id, page):
    """One page of research-grade, free-licensed, wild observations.

    Ordered by faves: on a plant this common, the most-faved photos are the
    ones in focus with the plant filling a fair part of the frame, which is
    what a detector needs; newest-first is mostly blurry phone shots.
    """
    q = urllib.parse.urlencode({
        "taxon_id": taxon_id, "quality_grade": "research", "captive": "false",
        "photos": "true", "photo_license": ",".join(LICENSES),
        "place_id": NORTH_AMERICA, "order_by": "votes", "order": "desc",
        "per_page": 100, "page": page,
    })
    return json.loads(get(f"{API}?{q}"))["results"]


def first_free_photo(obs):
    for p in obs.get("photos", []):
        if p.get("license_code") in LICENSES and p.get("url"):
            return p
    return None


def save_jpeg(raw, path):
    """Write raw bytes as a baseline RGB JPEG; return (dhash, width, height)."""
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        fingerprint = imgtools.dhash(im.convert("L"))
        im.convert("RGB").save(path, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return fingerprint, im.width, im.height


def load_lines(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {s for s in (l.split("#")[0].strip() for l in f) if s}


def load_manifest():
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows):
    cols = ["file", "species", "taxon", "observation", "photo_id", "license",
            "attribution", "observed_on", "place", "width", "height"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["file"]))


def collect(slug, want, seen, rejected, args):
    taxon_id, common = SPECIES[slug]
    out_dir = os.path.join(DEST, slug)
    have = {f for f in os.listdir(out_dir)} if os.path.isdir(out_dir) else set()
    kept = len([f for f in have if f.endswith(".jpg")])
    rows = []
    page = 1
    while kept < want and page <= 5:
        batch = observations(taxon_id, page)
        if not batch:
            break
        for obs in batch:
            if kept >= want:
                break
            p = first_free_photo(obs)
            if not p:
                continue
            pid = str(p["id"])
            name = f"{slug}_{pid}.jpg"
            if name in have or pid in rejected:
                continue
            if args.dry_run:
                print(f"    would fetch {slug}/{name}  <- obs {obs['id']}")
                kept += 1
                continue

            url = p["url"].replace("/square.", "/large.")
            try:
                raw = get(url)
            except (urllib.error.URLError, OSError) as e:
                print(f"    FAILED {name}: {e}")
                continue
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, name)
            tmp = path + ".part"
            try:
                fingerprint, w, h = save_jpeg(raw, tmp)
            except Exception as e:
                os.path.exists(tmp) and os.remove(tmp)
                print(f"    unreadable {name}: {type(e).__name__}")
                continue
            if min(w, h) < MIN_SIDE:
                os.remove(tmp)
                continue
            if seen and imgtools.POPCOUNT[fingerprint ^ np.array(seen)].sum(axis=1).min() <= HAMMING:
                os.remove(tmp)
                continue
            seen.append(fingerprint)
            os.replace(tmp, path)

            rows.append({
                "file": f"{slug}/{name}",
                "species": common,
                "taxon": obs["taxon"]["name"],
                "observation": f"https://www.inaturalist.org/observations/{obs['id']}",
                "photo_id": pid,
                "license": p["license_code"].upper(),
                "attribution": p.get("attribution", "")[:120],
                "observed_on": obs.get("observed_on") or "",
                "place": (obs.get("place_guess") or "")[:60],
                "width": w,
                "height": h,
            })
            kept += 1
            print(f"    {slug}/{name}", flush=True)
        page += 1
    if kept < want:
        print(f"  {slug}: only {kept} of {want} wanted")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--per", type=int, default=45,
                   help="photos per species (default 45: ~33 survive review)")
    p.add_argument("--only", action="append", choices=sorted(SPECIES))
    p.add_argument("--dry-run", action="store_true", help="list, do not download")
    a = p.parse_args()

    print(f"fetching invasive plant photos into {DEST}")
    existing = {r["file"]: r for r in load_manifest()}
    rejected = load_lines(REJECTS)
    seen = []
    fresh = []
    for slug in a.only or sorted(SPECIES):
        print(f"  {slug}")
        fresh += collect(slug, a.per, seen, rejected, a)
    if a.dry_run:
        return

    for r in fresh:
        existing[r["file"]] = r
    kept = {k: v for k, v in existing.items() if os.path.exists(os.path.join(DEST, k))}
    write_manifest(list(kept.values()))

    per = {}
    for k in kept:
        per[k.split("/")[0]] = per.get(k.split("/")[0], 0) + 1
    print(f"\n{len(kept)} images")
    for s in sorted(per):
        print(f"  {s:15s} {per[s]:3d}")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
