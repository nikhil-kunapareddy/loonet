# loonet

Training an object detector to find and count loons in top-down drone footage.

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
models/aubirds/        v0 weights
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

MIT — see [LICENSE](LICENSE).

## Frontend: run locally

1. Open a terminal in the frontend folder:

   ```bash
   cd Loonet/loonet
   ```

2. Install the frontend dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```

4. Open the local URL printed in the terminal, usually `http://localhost:5173`.
