# loonet

Training an object detector to find and count loons in top-down drone footage.

## Layout

```
src/count_birds.py     tiled inference + annotation + counts.csv
src/stitch_tiles.py    reassemble tiles into a mosaic
scripts/               fetch v0 weights, extract frames, screen training images
data/raw/              source stills and drone footage
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

```bash
scripts/extract_frames.py 3                             # video -> data/frames/
scripts/check_images.py data/raw --csv outputs/qc.csv   # screen before annotating
python src/count_birds.py data/frames --out outputs/    # detect + count
python src/stitch_tiles.py outputs/ mosaic.jpg          # reassemble tiles
```

`count_birds.py` tiles at 1024×600 and scores at 0.8. Flags: `-t/--thresh`,
`--whole` (no tiling), `--rgb`, `--limit`. Its numbers are **box counts, not bird
counts** — v0 over-splits, so don't quote them as counts.

`check_images.py` lists images unfit for training (unreadable, too small, out of
focus, badly exposed, featureless) and duplicate groups, and changes nothing on
disk. Every threshold is a flag; `--csv` dumps raw metrics to calibrate them.

`extract_frames.py` deletes existing `data/frames/frame_*.jpg` first.

MIT — see [LICENSE](LICENSE).
