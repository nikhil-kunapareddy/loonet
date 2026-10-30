# loonet

Training an object detector to find and count loons in top-down drone footage,
and a second one, vegnet, for the invasive plants that crowd their shoreline.

## Layout

```
src/count_birds.py     tiled inference + annotation + counts.csv
src/stitch_tiles.py    reassemble tiles into a mosaic
scripts/               the data pipeline, plus v0 weights and frame extraction
data/raw/              everything new lands here; never modified afterwards
data/renamed/          stage 1 + rename_map.csv
data/no_dups/          stage 2 + duplicates.csv
data/quality_check/    stage 3 + rejected.csv
data/un_annotated/     stage 4 + manifest.csv — annotate these
data/aubirds_data/     25 AUBIRDSTEST tiles, for reproducing the v0 result
data/vegnet_raw/       vegnet source photos (iNaturalist) + manifest.csv
models/aubirds/        v0 weights
models/vegnet/<ver>/   vegnet weights
outputs/               detections, counts.csv, mosaics
references/            Akçay et al. 2020, Animals 10:1207 (CC BY)
runs/<version>/        config, metrics, notes per version (weights not versioned)
```

## Usage

Needs **Python 3.11** (TensorFlow 2.21) and `ffmpeg` on PATH.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data pipeline

Drop new imagery in `data/raw`; four stages copy it forward to annotation-ready.

```
raw ─▶ renamed ─▶ no_dups ─▶ quality_check ─▶ un_annotated
   rename     dedupe     quality           prep
```

```bash
scripts/run_pipeline.py                  # all four stages
scripts/run_pipeline.py -n               # print the plan, write nothing
scripts/quality_filter.py --focus 300    # one stage on its own, to tune it
```

Files are renamed from their EXIF — `dji_20260822_143820_0137.jpg` is source,
capture time, sequence — and keep that name for good. Stages copy rather than
move, so re-running only handles what is new and nothing is deleted: rejects stay
in the folder before, and each stage writes a CSV saying what it left and why.
`--help` on a stage explains its logic and its flags.

`check_images.py` reports quality and duplicates for any folder without touching
it; `--csv` dumps the metrics behind `quality_filter.py`'s thresholds.

`extract_frames.py` samples `data/raw/drone-footage.mp4` into `data/frames/`,
deleting existing `frame_*.jpg` first. The pipeline ignores video; move the
frames you want into `data/raw` to feed them in.

## Detection

```bash
python src/count_birds.py data/frames --out outputs/    # detect + count
python src/stitch_tiles.py outputs/ mosaic.jpg          # reassemble tiles
```

`count_birds.py` tiles at 1024×600 and scores at 0.8. Flags: `-t/--thresh`,
`--whole` (no tiling), `--rgb`, `--limit`. Its numbers are **box counts, not bird
counts** — v0 over-splits, so don't quote them as counts.

## vegnet

A separate detector for four invasive plants, in ground-level photos: purple
loosestrife, common reed (the introduced *Phragmites*), water chestnut and
yellow flag iris, one class each. Training needs `requirements-train.txt`.

```bash
scripts/fetch_vegnet.py         # iNaturalist photos -> data/vegnet_raw
scripts/autolabel_vegnet.py     # draft boxes -> data/annotated/vegnet_v1
notebooks/vegnet_v1.ipynb       # split, train, copy best.pt to models/vegnet/v1
```

The boxes are **drafted, not hand-drawn**. The species comes from the
observation, which is research grade, so a second observer agreed on it.
Grounding DINO drafts where the plant is, and the contact sheets in
`outputs/vegnet_autolabel/` were reviewed by eye. A box is a patch, one
contiguous stand, not one plant. `data/vegnet_raw/rejected.txt` lists what
review pruned and why, mostly water chestnut *nuts* rather than the plant.

v1 is 132 images, 33 per species, sized and trained like loonnet_v1. On 28 val
images it scores mAP50 0.77 and mAP50-95 0.57. Water chestnut scores highest
(mAP50 0.96) and loosestrife lowest (0.65). The val boxes are drafts too, so
this measures agreement with the drafts, not with an annotator. Correcting them
by hand is the v2 step.

MIT — see [LICENSE](LICENSE).
