#!/usr/bin/env python3
"""Run the four data-pipeline stages in order.

    raw -> renamed -> no_dups -> quality_check -> un_annotated
        rename       dedupe     quality          prep

Each stage copies rather than moves, so data/raw stays the archive and every
folder shows what reached that point. Re-running the pipeline only handles what
is new: names already assigned stay assigned, and images already passed forward
stay passed forward.

This is the convenience path with every default taken. To tune a threshold, run
that stage on its own -- scripts/quality_filter.py --help -- since the flags
belong to the stages, not here.

Usage:
    scripts/run_pipeline.py
    scripts/run_pipeline.py -n                   # plan the whole run, write nothing
    scripts/run_pipeline.py --from dedupe        # redo everything after renaming
    scripts/run_pipeline.py --only quality
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    ("rename", "rename_images.py", "data/raw -> data/renamed"),
    ("dedupe", "dedupe_images.py", "data/renamed -> data/no_dups"),
    ("quality", "quality_filter.py", "data/no_dups -> data/quality_check"),
    ("prep", "prep_annotation.py", "data/quality_check -> data/un_annotated"),
]
NAMES = [s[0] for s in STAGES]


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--from", dest="start", choices=NAMES, default=NAMES[0],
                   help="first stage to run (default rename)")
    p.add_argument("--to", dest="stop", choices=NAMES, default=NAMES[-1],
                   help="last stage to run (default prep)")
    p.add_argument("--only", choices=NAMES, help="run just this stage")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="pass -n to every stage: nothing is written")
    a = p.parse_args()

    start, stop = (a.only, a.only) if a.only else (a.start, a.stop)
    lo, hi = NAMES.index(start), NAMES.index(stop)
    if lo > hi:
        sys.exit(f"--from {start} comes after --to {stop}")

    # Under -n the later stages read folders the earlier ones would have filled,
    # so they report on what is there now, not on what the run would produce.
    if a.dry_run and hi > lo:
        print("dry run: each stage sees the folders as they are now, so the "
              "later ones cannot preview what an earlier one would add\n")

    for name, script, flow in STAGES[lo:hi + 1]:
        print("#" * 72)
        print(f"# {name}: {flow}")
        print("#" * 72, flush=True)
        cmd = [sys.executable, os.path.join(HERE, script)]
        if a.dry_run:
            cmd.append("-n")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"\n{name} failed (exit {r.returncode}); stopping here")
        print()

    print(f"pipeline done: {start} -> {stop}")


if __name__ == "__main__":
    main()
