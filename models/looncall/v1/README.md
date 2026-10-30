# looncall v1

Hears Common Loon calls. Audio is cut into 3-second windows, and each window
gets P(loon call).

| File | What |
|---|---|
| `weights.npz` | The classifier as plain arrays, 14 kB: `mean`, `scale`, `coef`, `intercept`, `threshold`. Loading it needs only numpy and cannot run code. |
| `labels.yaml` | Class names, window length, sample rate, embedding model. |

looncall is a small head on top of BirdNET 2.4. BirdNET turns each window
into a 1024-number embedding, and looncall turns that into a probability:

```
p = sigmoid(((embedding - mean) / scale) · coef + intercept)
```

## Run it

```bash
uv venv ~/.venvs/loonet-audio --python 3.11
uv pip install --python ~/.venvs/loonet-audio/bin/python -r requirements-audio.txt
~/.venvs/loonet-audio/bin/python src/detect_loon_calls.py recordings/ --out loon_calls.csv
```

It accepts wav, mp3, flac, m4a, or a video's audio track (it needs ffmpeg on
PATH). It writes one CSV row per window: `file, start_s, end_s, p_loon, loon`.
The BirdNET model downloads on first use.

## How it was trained

The recordings are research-grade iNaturalist sounds, licensed CC0, CC BY or CC BY-SA:

- **Loon:** 62 Common Loon recordings.
- **Not loon:** 66 recordings of 11 species heard on the same lakes, chosen to
  include sounds easily mistaken for a loon (barred owl, gray wolf, pied-billed
  grebe, sandhill crane, belted kingfisher and others).
- **Drone noise:** the propeller noise from our own drone footage.

That gives 153 loon windows and 211 not-loon windows, split 80/20 by recording.
The classifier is a logistic regression, with its strength (C) picked by 5-fold
cross-validation grouped by recording. The pipeline is
`scripts/fetch_looncall.py`, then `scripts/autolabel_looncall.py`, then
`notebooks/looncall_v1.ipynb`.

**The labels are drafted.** Loon windows are the ones BirdNET scored at 0.4 or
above, checked by eye on spectrograms. Not-loon windows come from the other
species' recordings and the drone noise.

## Scores

**The clean val set (65 windows) doesn't tell you much.** BirdNET and looncall
both score 1.000 on it, because the windows were picked by BirdNET's own score
(above 0.4 for loon, below 0.02 for not loon). That makes the set easy by
construction.

**The test that matters** mixes each val window with our drone's propeller
noise. Recall is the share of loon windows caught at threshold 0.5:

| SNR (dB) | BirdNET recall | looncall recall | False alarms (both models) |
|---|---|---|---|
| clean | 1.00 | 1.00 | 0 |
| 10 | 0.75 | 0.86 | 0 |
| 5 | 0.54 | 0.68 | 0 |
| 0 | 0.36 | 0.46 | 0 |
| −5 | 0.07 | 0.14 | 0 |

Two caveats: there are only 28 loon windows, so each point of recall is about
one window. And the drone noise also appears in training, as not-loon windows.

**Known confusion:** one gray wolf recording (iNaturalist observation 327373890)
scores 0.52 in one window. BirdNET also hears a loon there, so it may be a
howl, or a loon calling in the background.

## Licence

The looncall weights are this repo's (MIT). But you cannot use them without
BirdNET 2.4, whose model is **CC BY-NC-SA 4.0: non-commercial only**.
Every training recording's author and licence are in
`data/looncall_raw/manifest.csv`.
