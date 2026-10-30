"""Find loon calls in audio with looncall (models/looncall/v1).

Every file is cut into 3 s windows, BirdNET 2.4 turns each window into a
1024-d embedding, and looncall's logistic regression turns that into
P(loon call). Anything ffmpeg can decode works -- wav, mp3, flac, m4a, and the
audio track of a video such as the drone footage.

Writes one CSV row per window, and prints per file how many windows heard a
loon and when the first one did.

Needs requirements-audio.txt (BirdNET on ai-edge-litert) and ffmpeg on PATH.
BirdNET's model is CC BY-NC-SA 4.0, so this is for non-commercial use.

Usage:
    python src/detect_loon_calls.py recordings/ --out outputs/loon_calls.csv
    python src/detect_loon_calls.py data/raw/drone-footage.mp4 -t 0.7
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "looncall", "v1")
SR = 48000
NATIVE = {".wav", ".mp3", ".flac", ".ogg"}          # soundfile reads these itself
EXTS = NATIVE | {".m4a", ".aac", ".mp4", ".mov", ".m4v", ".wma", ".opus"}


def load_weights(folder=MODEL):
    path = os.path.join(folder, "weights.npz")
    if not os.path.exists(path):
        sys.exit(f"looncall weights not found at {path}")
    return dict(np.load(path))


def p_loon(w, emb):
    z = ((emb - w["mean"]) / w["scale"]) @ w["coef"] + w["intercept"]
    return 1 / (1 + np.exp(-z))


def collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for d, _, files in os.walk(p):
                out += [os.path.join(d, f) for f in sorted(files)
                        if os.path.splitext(f)[1].lower() in EXTS]
        elif os.path.splitext(p)[1].lower() in EXTS:
            out.append(p)
        else:
            print(f"skip {p}: not an audio or video file")
    return out


def decodable(paths, tmp):
    """BirdNET reads wav/mp3/flac/ogg; convert the rest to 48 kHz mono wav first."""
    mapping = {}
    for p in paths:
        if os.path.splitext(p)[1].lower() in NATIVE:
            mapping[os.path.abspath(p)] = p
            continue
        dst = os.path.join(tmp, f"{len(mapping):05d}.wav")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", p, "-vn", "-ac", "1",
                            "-ar", str(SR), dst], capture_output=True, text=True)
        if r.returncode:
            print(f"skip {p}: ffmpeg could not decode it")
            continue
        mapping[dst] = p
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="audio/video files or folders")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "loon_calls.csv"))
    ap.add_argument("-t", "--thresh", type=float, default=None,
                    help="P(loon) to count a window as a call (default: the model's, 0.5)")
    ap.add_argument("--model", default=MODEL, help="folder holding weights.npz")
    a = ap.parse_args()

    w = load_weights(a.model)
    thresh = float(w["threshold"]) if a.thresh is None else a.thresh
    files = collect(a.inputs)
    if not files:
        sys.exit("no audio found")

    import birdnet
    model = birdnet.load("acoustic", "2.4", "tf", library="litert")
    with tempfile.TemporaryDirectory() as tmp:
        mapping = decodable(files, tmp)
        enc = model.encode(list(mapping), n_workers=4, batch_size=8).to_structured_array()

    rows = []
    for r in enc:
        src = mapping[str(r["input"])]
        p = float(p_loon(w, r["embedding"]))
        rows.append({"file": src, "start_s": f"{float(r['start_time']):.1f}",
                     "end_s": f"{float(r['end_time']):.1f}", "p_loon": f"{p:.3f}",
                     "loon": int(p >= thresh)})

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "start_s", "end_s", "p_loon", "loon"])
        wr.writeheader()
        wr.writerows(rows)

    for src in files:
        mine = [r for r in rows if r["file"] == src]
        if not mine:
            continue
        hits = [r for r in mine if r["loon"]]
        first = f"first at {hits[0]['start_s']}s" if hits else ""
        print(f"{len(hits):4d}/{len(mine):<4d} windows with a loon  {first:16s} {src}")
    print(f"threshold {thresh}   csv: {a.out}")


if __name__ == "__main__":
    main()
