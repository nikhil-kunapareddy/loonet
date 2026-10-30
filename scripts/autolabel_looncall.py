#!/usr/bin/env python3
"""Draft 3-second loon / not-loon windows from data/looncall_raw.

A recording on iNaturalist says "there is a loon in here", not when. looncall
works on 3-second windows, BirdNET's own unit, so each window needs a label,
and this script drafts them:

    loon recordings     windows BirdNET scores >= --pos for Common Loon, up to
                        --per-rec of the strongest. Weaker windows are left
                        out entirely -- a quiet stretch between calls may still
                        hold a faint one, so it is neither label.
    other species       windows BirdNET scores < --neg for loon, up to
                        --per-rec of the loudest (the loudest window is the
                        one with the owl or the goose in it). A recording with
                        any window >= --flag is dropped whole: a goose filmed
                        on a loon lake can have a loon behind it.
    drone               every window. Propeller noise is never a loon.

So positives are drafted by BirdNET and negatives come from the observation's
species. That has a cost worth stating: BirdNET recalls every positive at
--pos by construction, so a BirdNET baseline is flattered on recall. Its false
positives on the negatives are a fair test.

Writes:
    data/annotated/looncall_v1/windows.csv      file, start_s, end_s, label, ...
    data/annotated/looncall_v1/labels.yaml      names, window, sample rate
    outputs/looncall_autolabel/*.png            spectrogram sheets for review

Review the sheets; add the sound id of any bad recording to
data/looncall_raw/rejected.txt and re-run. The notebook does the split.

Usage:
    scripts/autolabel_looncall.py
    scripts/autolabel_looncall.py --pos 0.6 --per-rec 6
"""
import argparse
import csv
import os
import sys

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_looncall import DEST as RAW, REJECTS, SPECIES, SR
from fetch_vegnet import load_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "data", "annotated", "looncall_v1")
OUT = os.path.join(ROOT, "outputs", "looncall_autolabel")

LOON = "Gavia immer_Common Loon"
WINDOW = 3.0
SHEET_COLS, SHEET_ROWS = 8, 8


def load_birdnet():
    import birdnet
    return birdnet.load("acoustic", "2.4", "tf", library="litert")


def loon_scores(model, paths):
    """{path: [(start, end, conf), ...]} for every full 3 s window."""
    r = model.predict(paths, top_k=None, default_confidence_threshold=None,
                      custom_species_list=[LOON], n_workers=4, batch_size=8)
    out = {}
    for row in r.to_structured_array():
        s, e = float(row["start_time"]), float(row["end_time"])
        if e - s >= WINDOW - 1e-3:
            out.setdefault(str(row["input"]), []).append((s, e, float(row["confidence"])))
    return out


def rms_db(audio, start, end):
    seg = audio[int(start * SR):int(end * SR)]
    return float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9))


def sheet(rows, path, title):
    """Spectrograms of the chosen windows, 0-6 kHz, one tile each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    fig, axes = plt.subplots(SHEET_ROWS, SHEET_COLS, figsize=(SHEET_COLS * 2.2, SHEET_ROWS * 1.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, rows):
        audio, _ = sf.read(os.path.join(RAW, r["file"]), start=int(r["start_s"] * SR),
                           stop=int(r["end_s"] * SR), dtype="float32", always_2d=False)
        f, t, S = spectrogram(audio, SR, nperseg=1024, noverlap=768)
        keep = f <= 6000
        ax.pcolormesh(t, f[keep] / 1000, 10 * np.log10(S[keep] + 1e-12), shading="auto",
                      cmap="magma", vmin=-110, vmax=-40)
        sid = os.path.basename(r["file"]).rsplit("_", 1)[-1][:-4]
        ax.set_title(f"{sid} @{r['start_s']:.0f}s  {r['birdnet_loon']:.2f}", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # 0.4, not lower: on review, windows BirdNET scored 0.1-0.4 were mostly faint
    # or empty, and a positive with no audible call teaches the wrong thing
    p.add_argument("--pos", type=float, default=0.4, help="loon score floor for a positive (0.4)")
    p.add_argument("--neg", type=float, default=0.02, help="loon score ceiling for a negative (0.02)")
    p.add_argument("--flag", type=float, default=0.5,
                   help="drop a not-loon recording if any window reaches this (0.5)")
    p.add_argument("--per-rec", type=int, default=4, help="windows kept per recording (4)")
    a = p.parse_args()

    rejected = load_lines(REJECTS)
    slugs = [s for s in list(SPECIES) + ["drone"] if os.path.isdir(os.path.join(RAW, s))]
    paths = sorted(os.path.join(RAW, s, f) for s in slugs
                   for f in os.listdir(os.path.join(RAW, s))
                   if f.endswith(".mp3") and f.rsplit("_", 1)[-1][:-4] not in rejected)
    print(f"scoring {len(paths)} recordings with BirdNET 2.4")
    scores = loon_scores(load_birdnet(), paths)

    rows, flagged = [], []
    for path in paths:
        rel = os.path.relpath(path, RAW)
        slug = rel.split("/")[0]
        wins = scores.get(path, [])
        if not wins:
            continue
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        assert sr == SR, (rel, sr)
        cand = [{"file": rel, "start_s": s, "end_s": e, "source": slug,
                 "birdnet_loon": c, "rms_db": rms_db(audio, s, e)} for s, e, c in wins]
        if slug == "loon":
            pick = sorted((w for w in cand if w["birdnet_loon"] >= a.pos),
                          key=lambda w: -w["birdnet_loon"])[:a.per_rec]
            label = 1
        elif slug == "drone":
            pick, label = cand, 0
        else:
            top = max(w["birdnet_loon"] for w in cand)
            if top >= a.flag:
                flagged.append((rel, top))
                continue
            pick = sorted((w for w in cand if w["birdnet_loon"] < a.neg),
                          key=lambda w: -w["rms_db"])[:a.per_rec]
            label = 0
        for w in sorted(pick, key=lambda w: w["start_s"]):
            rows.append({**w, "label": label})

    os.makedirs(DS, exist_ok=True)
    cols = ["file", "start_s", "end_s", "label", "source", "birdnet_loon", "rms_db"]
    with open(os.path.join(DS, "windows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "start_s": f"{r['start_s']:.1f}", "end_s": f"{r['end_s']:.1f}",
                        "birdnet_loon": f"{r['birdnet_loon']:.4f}", "rms_db": f"{r['rms_db']:.1f}"})
    with open(os.path.join(DS, "labels.yaml"), "w") as f:
        yaml.safe_dump({"path": DS, "audio": RAW, "windows": "windows.csv",
                        "names": {0: "no loon", 1: "common loon"},
                        "window_s": WINDOW, "sample_rate": SR,
                        "embedding": "BirdNET 2.4 (birdnet 1.3.1, litert), 1024-d"},
                       f, sort_keys=False)

    os.makedirs(OUT, exist_ok=True)
    for old in os.listdir(OUT):
        if old.endswith(".png"):
            os.remove(os.path.join(OUT, old))
    per_sheet = SHEET_COLS * SHEET_ROWS
    for name, lab in (("loon", 1), ("not_loon", 0)):
        sel = [r for r in rows if r["label"] == lab]
        for i in range(0, len(sel), per_sheet):
            sheet(sel[i:i + per_sheet], os.path.join(OUT, f"{name}_{i // per_sheet + 1}.png"),
                  f"{name} windows {i + 1}-{min(i + per_sheet, len(sel))} of {len(sel)}")

    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], set()).add(r["file"])
    for s in slugs:
        n_rec = len(by_src.get(s, ()))
        n_win = sum(1 for r in rows if r["source"] == s)
        print(f"  {s:22s} {n_rec:3d} recordings  {n_win:4d} windows")
    print(f"\n{sum(r['label'] for r in rows)} loon / {sum(1 - r['label'] for r in rows)} not-loon windows")
    for rel, top in flagged:
        print(f"  dropped {rel}: a window scores {top:.2f} for loon")
    print(f"windows: {os.path.join(DS, 'windows.csv')}")
    print(f"sheets:  {OUT}")


if __name__ == "__main__":
    main()
