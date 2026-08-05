# CLAUDE.md

Loon detection/counting from top-down drone footage. See README.md for setup and usage.

## Environment

Use `.venv/bin/python` directly — Python 3.11 is required (TensorFlow 2.21 constraint).

## Status: the AUBIRDSTEST baseline is a closed question

The pretrained Faster R-CNN from Akçay et al. 2020 is **not usable for loons** and
this is settled — don't re-litigate it or propose threshold tuning as a fix. It
finds loons but fragments one bird into 5–6 boxes and fires on algae and sun glint,
because `pipeline.config` resizes everything to fit 600×600 and it was trained on
distant flocks, not close subjects.

The code is kept so the negative result stays reproducible. The next step is a
modern detector (YOLOv11 / RT-DETR) with tiled inference, trained on annotated
loon data.

Consequence: numbers out of `count_birds.py` are **box counts, not bird counts**.
Never report them as bird counts.

## Conventions

- Never commit media. `data/`, `models/`, `outputs/`, `assets/`, and `references/`
  are all gitignored; the repo is code and docs only.
- Don't vendor model weights — `scripts/fetch_aubirds.py` downloads them.
- Artifacts get their own top-level directory. Don't nest weights, code, or
  outputs under `data/`.
- Scripts are stdlib-only Python; `ffmpeg` is the one external binary.
- Keep tiling geometry (1024×600) and the 0.8 threshold as-is — they come from the
  original paper's scripts.

## Gotcha

`scripts/extract_frames.py` deletes existing `data/frames/frame_*.jpg` before
writing.
