# Models

Each folder holds the weights and a README covering what the model detects,
how to run it, how it was trained, and its scores.

| Model | Detects | In | File | Size | Needs | Licence |
|---|---|---|---|---|---|---|
| [loonnet v1](loonnet/v1/) | Common Loons | photos, drone frames | `best.pt` | 20 MB | `ultralytics` | AGPL-3.0 (from YOLO26s) |
| [vegnet v1](vegnet/v1/) | 4 invasive lake plants | ground-level photos | `best.pt` | 20 MB | `ultralytics` | AGPL-3.0 (from YOLO26s) |
| [looncall v1](looncall/v1/) | loon calls | audio, per 3 s | `weights.npz` | 14 kB | `birdnet`, ffmpeg | MIT, but BirdNET is CC BY-NC-SA 4.0 |
| [aubirds](aubirds/) | birds (v0 baseline, over-splits loons) | drone frames | `frozen_inference_graph.pb` | 50 MB | TensorFlow | none stated; from github.com/melihoz/AUBIRDSTEST |

## Quick start

```bash
yolo predict model=models/loonnet/v1/best.pt source=frames/           # loons
yolo predict model=models/vegnet/v1/best.pt  source=shoreline_photos/ # plants
python src/detect_loon_calls.py recordings/ --out loon_calls.csv      # calls
```

The YOLO models need `requirements-train.txt`; looncall needs
`requirements-audio.txt`.

To download one model without cloning the repo, open its file on GitHub and
click "Download raw file".
