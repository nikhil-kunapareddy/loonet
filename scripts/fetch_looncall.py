#!/usr/bin/env python3
"""Fetch loon and not-loon sound recordings into data/looncall_raw.

The source set for looncall, the loon-call detector. Half the recordings are
Common Loon; the other half are what a microphone on a northern lake hears
instead, weighted toward what a careless detector would call a loon:

    loon                 Common Loon               the positive class
    barred_owl           Barred Owl                hoots, at night, on the same lakes
    gray_wolf            Gray Wolf                 a howl is the classic wail mix-up
    pied_billed_grebe    Pied-billed Grebe         loud, tonal, on the water
    sandhill_crane       Sandhill Crane            rolling call, close to a tremolo
    belted_kingfisher    Belted Kingfisher         rattle, close to a tremolo
    canada_goose ...     everyday lake chorus: goose, mallard, heron,
                         red-winged blackbird, bullfrog, spring peeper

plus drone/, the audio track of data/raw/drone-footage.mp4: propeller noise
from our own camera, which is what every clip we record will sit on top of.

Source is iNaturalist, research grade, CC0 / CC BY / CC BY-SA, North America,
one sound per observation -- the same rules as fetch_vegnet.py. Every file is
re-encoded to 48 kHz mono MP3 (BirdNET's rate) and cut to the first --max-s
seconds. Most uploads are phone m4a or mp3 already, so MP3 loses little, and it
keeps the folder small enough to version.

Files are named <slug>_<sound id>.mp3; a re-run skips anything on disk or in
rejected.txt. manifest.csv records the observation, licence and attribution.

Usage:
    scripts/fetch_looncall.py                  # fetch what is missing
    scripts/fetch_looncall.py --only loon --per 80
    scripts/fetch_looncall.py --dry-run
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_vegnet import LICENSES, NORTH_AMERICA, get, load_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "looncall_raw")
MANIFEST = os.path.join(DEST, "manifest.csv")
REJECTS = os.path.join(DEST, "rejected.txt")
DRONE_SRC = os.path.join(ROOT, "data", "raw", "drone-footage.mp4")

API = "https://api.inaturalist.org/v1/observations"
SR = 48000
MIN_S = 3.0             # one BirdNET window

# slug -> (iNaturalist taxon id, common name, recordings wanted). Loon gets as
# many recordings as all the others together, so the classes come out even.
SPECIES = {
    "loon":                 (4626,   "Common Loon",           66),
    "barred_owl":           (19893,  "Barred Owl",             6),
    "gray_wolf":            (42048,  "Gray Wolf",              6),
    "pied_billed_grebe":    (4246,   "Pied-billed Grebe",      6),
    "sandhill_crane":       (508048, "Sandhill Crane",         6),
    "belted_kingfisher":    (2548,   "Belted Kingfisher",      6),
    "canada_goose":         (7089,   "Canada Goose",           6),
    "mallard":              (6930,   "Mallard",                6),
    "great_blue_heron":     (4956,   "Great Blue Heron",       6),
    "red_winged_blackbird": (9744,   "Red-winged Blackbird",   6),
    "bullfrog":             (65979,  "American Bullfrog",      6),
    "spring_peeper":        (24268,  "Spring Peeper",          6),
}


def observations(taxon_id, page):
    q = urllib.parse.urlencode({
        "taxon_id": taxon_id, "quality_grade": "research", "captive": "false",
        "sounds": "true", "sound_license": ",".join(LICENSES),
        "place_id": NORTH_AMERICA, "order_by": "votes", "order": "desc",
        "per_page": 100, "page": page,
    })
    return json.loads(get(f"{API}?{q}"))["results"]


def first_free_sound(obs):
    for s in obs.get("sounds", []):
        if s.get("license_code") in LICENSES and s.get("file_url"):
            return s
    return None


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def to_mp3(src, dst, max_s):
    """Decode anything ffmpeg reads; write 48 kHz mono MP3, first max_s seconds."""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", str(SR),
         "-t", str(max_s), "-c:a", "libmp3lame", "-b:a", "128k", dst],
        capture_output=True, text=True)
    return r.returncode == 0


def load_manifest():
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows):
    cols = ["file", "species", "taxon", "observation", "sound_id", "license",
            "attribution", "observed_on", "place", "source_s", "kept_s"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["file"]))


def collect(slug, want, rejected, args):
    taxon_id, common, _ = SPECIES[slug]
    out_dir = os.path.join(DEST, slug)
    have = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
    kept = len([f for f in have if f.endswith(".mp3")])
    rows = []
    page = 1
    while kept < want and page <= 5:
        batch = observations(taxon_id, page)
        if not batch:
            break
        for obs in batch:
            if kept >= want:
                break
            s = first_free_sound(obs)
            if not s:
                continue
            sid = str(s["id"])
            name = f"{slug}_{sid}.mp3"
            if name in have or sid in rejected:
                continue
            if args.dry_run:
                print(f"    would fetch {slug}/{name}  <- obs {obs['id']}")
                kept += 1
                continue

            try:
                raw = get(s["file_url"])
            except (urllib.error.URLError, OSError) as e:
                print(f"    FAILED {name}: {e}")
                continue
            ext = os.path.splitext(urllib.parse.urlparse(s["file_url"]).path)[1] or ".bin"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(raw)
            try:
                src_s = duration(tmp.name)
                if src_s < MIN_S:
                    continue
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, name)
                if not to_mp3(tmp.name, path, args.max_s):
                    print(f"    undecodable {name}")
                    os.path.exists(path) and os.remove(path)
                    continue
            finally:
                os.remove(tmp.name)

            rows.append({
                "file": f"{slug}/{name}",
                "species": common,
                "taxon": obs["taxon"]["name"],
                "observation": f"https://www.inaturalist.org/observations/{obs['id']}",
                "sound_id": sid,
                "license": s["license_code"].upper(),
                "attribution": (s.get("attribution") or "")[:120],
                "observed_on": obs.get("observed_on") or "",
                "place": (obs.get("place_guess") or "")[:60],
                "source_s": f"{src_s:.1f}",
                "kept_s": f"{min(src_s, args.max_s):.1f}",
            })
            kept += 1
            print(f"    {slug}/{name}  {src_s:.0f}s", flush=True)
        page += 1
    if kept < want:
        print(f"  {slug}: only {kept} of {want} wanted")
    return rows


def drone(args):
    """Our own propeller noise, from the drone footage's audio track."""
    name = "drone/drone_footage.mp3"
    path = os.path.join(DEST, name)
    if os.path.exists(path) or args.dry_run or not os.path.exists(DRONE_SRC):
        return []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not to_mp3(DRONE_SRC, path, args.max_s):
        print("    drone audio: ffmpeg failed")
        return []
    s = duration(DRONE_SRC)
    print(f"    {name}  {s:.0f}s")
    return [{"file": name, "species": "drone propeller noise", "taxon": "",
             "observation": "data/raw/drone-footage.mp4", "sound_id": "",
             "license": "own recording", "attribution": "", "observed_on": "2026-08-22",
             "place": "", "source_s": f"{s:.1f}", "kept_s": f"{min(s, args.max_s):.1f}"}]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--per", type=int, default=None,
                   help="recordings per species (default: the SPECIES table)")
    p.add_argument("--only", action="append", choices=sorted(SPECIES) + ["drone"])
    p.add_argument("--max-s", type=float, default=60.0,
                   help="keep the first N seconds of each recording (default 60)")
    p.add_argument("--dry-run", action="store_true", help="list, do not download")
    a = p.parse_args()

    print(f"fetching sound recordings into {DEST}")
    existing = {r["file"]: r for r in load_manifest()}
    rejected = load_lines(REJECTS)
    fresh = []
    for slug in a.only or list(SPECIES) + ["drone"]:
        print(f"  {slug}")
        if slug == "drone":
            fresh += drone(a)
        else:
            fresh += collect(slug, a.per or SPECIES[slug][2], rejected, a)
    if a.dry_run:
        return

    for r in fresh:
        existing[r["file"]] = r
    kept = {k: v for k, v in existing.items() if os.path.exists(os.path.join(DEST, k))}
    write_manifest(list(kept.values()))

    per = {}
    for k in kept:
        per[k.split("/")[0]] = per.get(k.split("/")[0], 0) + 1
    print(f"\n{len(kept)} recordings")
    for s in per:
        print(f"  {s:22s} {per[s]:3d}")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
