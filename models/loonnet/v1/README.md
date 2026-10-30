# loonnet v1

Finds Common Loons in photos and drone frames. One class, `common loon`.

| File | What |
|---|---|
| `best.pt` | Ultralytics YOLO26s weights, 20 MB. Class names are inside it. |
| `data.yaml` | The training config the weights came from. |

## Run it

```bash
pip install ultralytics==8.4.138
yolo predict model=models/loonnet/v1/best.pt source=path/to/images conf=0.25
```

Or from Python:

```python
from ultralytics import YOLO
boxes = YOLO("models/loonnet/v1/best.pt").predict("frame.jpg", conf=0.25)[0].boxes
```

## How it was trained

Fine-tuned from YOLO26s on `data/annotated/loonnet_v1`: 132 hand-labelled
images (drone frames, phone and camera stills, web photos), 80/20 split.
Settings: 640 px, 100 epochs, batch 16, seed 0. The run is in
`runs/loonnet_v1_yolo26s/`.

## Scores (26 val images, epoch 100)

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.90 | 0.88 | 0.93 | 0.64 |

Train and val come from the same footage, so these scores say it generalises
across frames, not to new lakes. `scripts/eval_others.py` scores it on
loon-free hard cases.

## Licence

Fine-tuned from Ultralytics YOLO26s, which is AGPL-3.0. The weights inherit
that licence unless you hold an Ultralytics Enterprise licence.
