# vegnet v1

Finds four invasive lake plants in ground-level photos, one class each:

| id | class |
|---|---|
| 0 | purple loosestrife |
| 1 | common reed (introduced *Phragmites australis australis*) |
| 2 | water chestnut |
| 3 | yellow flag iris |

A box marks a **patch**, one contiguous stand of a species, not one plant.

| File | What |
|---|---|
| `best.pt` | Ultralytics YOLO26s weights, 20 MB. Class names are inside it. |
| `data.yaml` | The training config the weights came from. |

## Run it

```bash
pip install ultralytics==8.4.138
yolo predict model=models/vegnet/v1/best.pt source=path/to/photos conf=0.25
```

## How it was trained

132 research-grade iNaturalist photos, 33 per species, licensed CC0, CC BY or
CC BY-SA. Fine-tuned from YOLO26s with the same settings as loonnet v1. The
pipeline is `scripts/fetch_vegnet.py`, then `scripts/autolabel_vegnet.py`, then
`notebooks/vegnet_v1.ipynb`.

**The boxes are drafted, not hand-drawn.** Grounding DINO drafted them and a
person reviewed contact sheets. `data/vegnet_raw/rejected.txt` lists the 69
photos dropped in review.

## Scores (28 val images, 7 per species)

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.82 | 0.74 | 0.77 | 0.57 |
| water chestnut | 0.96 | 0.88 | 0.96 | 0.78 |
| common reed | 0.85 | 0.81 | 0.77 | 0.56 |
| yellow flag iris | 0.92 | 0.67 | 0.71 | 0.50 |
| purple loosestrife | 0.53 | 0.62 | 0.65 | 0.44 |

The val boxes are drafts too, so these scores measure agreement with the drafts.
With 7 images per species, one image moves a species' score a lot.

## Licence

Fine-tuned from Ultralytics YOLO26s (AGPL-3.0), so the weights inherit AGPL-3.0.
Every training photo's author and licence are in `data/vegnet_raw/manifest.csv`.
