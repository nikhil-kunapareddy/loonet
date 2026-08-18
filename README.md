# loonet

Training an object detector that finds and counts loons in top-down drone
footage.

This repo is the shared workspace for that: every model we train gets a version,
an evaluation on the same held-out split, and a row in the table below — so runs
are comparable and findings are in one place instead of scattered across
screenshots.

## Model versions

| version | model | trained on | mAP@50 | count MAE | verdict |
| --- | --- | --- | --- | --- | --- |
| v0 | Faster R-CNN, pretrained (Akçay et al. 2020) | not ours — distant flocks | — | — | **ruled out**, see [Baseline](#baseline-v0) |
| v1 | YOLOv11 + tiled inference | *pending annotation* | — | — | *not started* |

v0 has no numbers because there was no annotated ground truth to score it
against; it was ruled out qualitatively. From v1 on, every row gets measured.

### How a version is recorded

```
runs/<version>/
  config.yaml     exact training config (committed)
  metrics.json    eval on the held-out split (committed)
  notes.md        what changed vs. the previous version, and why (committed)
  weights/        checkpoints (gitignored — too big, re-trainable)
```

Report at minimum **mAP@50**, **mAP@50–95**, **precision/recall at the threshold
you'd actually run at**, and **count MAE per frame**. That last one is what the
project is judged on: a detector with respectable mAP that splits one bird into
six boxes is useless here, and mAP alone hides it.

Evaluate on the same held-out split every time, or the table above means
nothing. Pick that split once, before training v1, and don't touch it.

## Pipeline

1. **Frames** — `scripts/extract_frames.py 3` samples one frame every 3 s from a
   video into `data/frames/`.
2. **Screen** — `scripts/check_images.py` flags images unfit for training and
   duplicate groups, before anyone spends time annotating them.
3. **Annotate** — YOLO-format labels beside the images
   (`data/annotated/<set>/images`, `.../labels`).
4. **Train** — on tiles, not resized full frames. See the baseline for why.
5. **Evaluate** — held-out split → `runs/<version>/metrics.json` → new row above.
6. **Run** — tiled inference over new footage.

Steps 1–2 and the inference side of 6 exist today. Steps 3–5 are the current
work.

## Setup

Needs **Python 3.11** (TensorFlow 2.21) and `ffmpeg` on PATH.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The training stack (torch / ultralytics) is not pinned yet — it lands with v1.

The clone is **~350 MB**: imagery (`data/`), the v0 weights (`models/`), run
outputs (`outputs/`), and the paper (`references/`) are all versioned, so there's
nothing to fetch or ask for before running anything. `scripts/fetch_aubirds.py`
still works if the weights ever need re-downloading.

## Usage

```bash
scripts/extract_frames.py 3                              # video -> data/frames/
scripts/check_images.py data/raw/images --csv outputs/qc.csv
python src/count_birds.py data/frames --out outputs/     # detect + count (v0)
python src/stitch_tiles.py outputs/ mosaic.jpg           # reassemble tiles
```

`count_birds.py` splits images into 1024×600 tiles, scores at 0.8, and writes
annotated `*_det.jpg` files plus a `counts.csv`. Flags: `-t/--thresh`,
`--whole` (no tiling), `--rgb`, `--limit`.

`check_images.py` prints two lists — images unfit for training (unreadable, too
small, out of focus, badly exposed, featureless) and duplicate groups (identical
bytes, plus near-duplicates by perceptual hash) — and changes nothing on disk.
Focus and contrast are scored on the sharpest region rather than the whole
frame, so a small sharp bird on flat water isn't rejected as blurry. Every
threshold is a flag; `--csv` dumps raw metrics so they can be calibrated per
dataset.

Note: `extract_frames.py` deletes existing `data/frames/frame_*.jpg` first.

## Baseline (v0)

The pretrained aerial-bird detector from
[Akçay et al. 2020](https://doi.org/10.3390/ani10071207)
([weights](https://github.com/melihoz/AUBIRDSTEST)), run over loon imagery.

It doesn't work for this task. It finds loons, but fragments a single bird into
~5–6 boxes and fires on algae and sun glint — it was trained on distant flocks
and resizes every input to fit 600×600. Threshold tuning doesn't fix it; raising
the score suppresses real birds first. **Its numbers are box counts, not bird
counts** — don't quote them as counts.

This is settled; it's not worth re-testing. The code stays so the negative
result is reproducible, and because the tiling and counting scaffolding carries
over to v1.

## Layout

```
src/count_birds.py     tiled inference + annotation + counts.csv
src/stitch_tiles.py    reassemble tiles into a mosaic
scripts/               fetch v0 weights, extract frames, screen training images
runs/<version>/        per-version config, metrics, notes (weights gitignored)
data/                  source imagery and extracted frames
models/                v0 detector weights
outputs/               detections, counts.csv, mosaics
references/            papers — start with the v0 one below
assets/                gitignored (samples duplicated from outputs/)
```

## Citation

> Akçay, H.G.; Kabasakal, B.; Aksu, D.; Demir, N.; Öz, M.; Erdoğan, A.
> Automated Bird Counting with Deep Learning for Regional Bird Distribution
> Mapping. *Animals* **2020**, 10, 1207.

## License

Code is MIT — see [LICENSE](LICENSE). The pretrained detector and third-party
imagery keep their own terms.
