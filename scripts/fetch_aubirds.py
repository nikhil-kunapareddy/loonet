#!/usr/bin/env python3
"""Fetch the pretrained Faster R-CNN from Akcay et al. 2020 (Animals 10:1207).

Weights are gitignored (52 MB); run this after a fresh clone.

Usage:
    scripts/fetch_aubirds.py
    scripts/fetch_aubirds.py --force        # re-download files already present
"""
import argparse
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "models", "aubirds")
BASE = "https://raw.githubusercontent.com/melihoz/AUBIRDSTEST/master/inference_graph"
FILES = ["frozen_inference_graph.pb", "labelmap.pbtxt", "pipeline.config"]


def human(n):
    for unit, div in (("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.0f} {unit}"
    return f"{n} B"


def fetch(name, dest_dir, force=False):
    """Download one file, writing to <name>.part first so a failed or interrupted
    transfer never leaves a truncated graph behind for TensorFlow to choke on."""
    path = os.path.join(dest_dir, name)
    if os.path.exists(path) and not force:
        print(f"  {name}: already present ({human(os.path.getsize(path))}), skipping")
        return path

    tmp = path + ".part"
    print(f"  fetching {name} ...", end="", flush=True)
    try:
        with urllib.request.urlopen(f"{BASE}/{name}") as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  fetching {name} ... {got * 100 // total}%",
                          end="", flush=True)
    except (urllib.error.URLError, OSError) as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        sys.exit(f"\nfailed to fetch {name}: {e}")

    os.replace(tmp, path)
    print(f"\r  fetching {name} ... {human(os.path.getsize(path))}  ")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true", help="re-download files already present")
    p.add_argument("--dest", default=DEST, help=f"where to put the model (default {DEST})")
    a = p.parse_args()

    os.makedirs(a.dest, exist_ok=True)
    print(f"fetching AUBIRDSTEST inference graph into {a.dest}")
    for name in FILES:
        fetch(name, a.dest, a.force)

    # 52,243,181 bytes as of the master commit this was written against
    graph = os.path.join(a.dest, FILES[0])
    print(f"done: {human(os.path.getsize(graph))} {graph}")


if __name__ == "__main__":
    main()
