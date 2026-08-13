# loonet

Detecting and counting loons in top-down drone footage.

## Status

This is a **baseline evaluation, not a working loon counter.** It runs the
pretrained aerial-bird detector from
[Akçay et al. 2020](https://doi.org/10.3390/ani10071207)
([weights](https://github.com/melihoz/AUBIRDSTEST)) over loon imagery.

The model doesn't work for this task. It finds loons, but fragments a single
bird into ~5–6 boxes and fires on algae and sun glint — it was trained on
distant flocks and resizes every input to fit 600×600. Threshold tuning doesn't
fix it; raising the score suppresses real birds first.

So the numbers it reports are **box counts, not bird counts.** The code is kept
so the negative result stays reproducible. Next step: a modern detector
(YOLOv11 / RT-DETR) with tiled inference, trained on annotated loon data.

## Setup

Needs **Python 3.11** (TensorFlow 2.21) and `ffmpeg` on PATH.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

scripts/fetch_aubirds.py    # downloads the 52 MB detector into models/aubirds/
```

## Usage

```bash
scripts/extract_frames.py 3                              # video -> data/frames/
python src/count_birds.py data/frames --out outputs/     # detect + count
python src/stitch_tiles.py outputs/ mosaic.jpg           # reassemble tiles
```

`count_birds.py` splits images into 1024×600 tiles and scores at 0.8, then
writes annotated `*_det.jpg` files and a `counts.csv`. Flags: `-t/--thresh`,
`--whole` (no tiling), `--rgb`, `--limit`.

Note: `extract_frames.py` deletes existing `data/frames/frame_*.jpg` first.

### Screening training data

```bash
scripts/check_images.py data/raw/images --csv outputs/qc.csv
```

Prints two lists — images unfit for training (unreadable, too small, out of
focus, badly exposed, featureless) and duplicate groups (identical bytes, plus
near-duplicates found by perceptual hash) — and changes nothing on disk.
Focus and contrast are scored on the sharpest region rather than the whole
frame, so a small sharp bird on flat water isn't rejected as blurry. Thresholds
are all flags; `--csv` dumps the raw metrics so they can be calibrated per
dataset.

## Layout

```
src/count_birds.py     tiled inference + annotation + counts.csv
src/stitch_tiles.py    reassemble tiles into a mosaic
scripts/               fetch weights, extract frames, screen training images
data/ models/ outputs/ assets/ references/    all gitignored
```

A fresh clone has no inputs or weights — the repo is code and docs only.

## Citation

> Akçay, H.G.; Kabasakal, B.; Aksu, D.; Demir, N.; Öz, M.; Erdoğan, A.
> Automated Bird Counting with Deep Learning for Regional Bird Distribution
> Mapping. *Animals* **2020**, 10, 1207.

## License

Code is MIT — see [LICENSE](LICENSE). The pretrained detector and third-party
imagery keep their own terms.
