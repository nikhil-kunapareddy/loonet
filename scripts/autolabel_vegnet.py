#!/usr/bin/env python3
"""Draft YOLO boxes for data/vegnet_raw and stage them as data/annotated/vegnet_v1.

Nobody has hand-drawn these boxes. The species comes from the iNaturalist
observation, which a second observer confirmed, so the class is trustworthy;
only *where* the plant is in the frame is drafted here, by Grounding DINO (an
open-vocabulary detector, IDEA-Research/grounding-dino-base from the Hugging
Face hub) prompted with a few descriptions of each species.
Treat vegnet_v1 as auto-labelled and spot-checked, not annotated -- the boxes
are good enough to train a first model on and to correct by hand for v2.

What a box means. A loon is one bird, one box. A plant is not like that:
loosestrife comes as dozens of spikes, phragmites as a wall of stems, water
chestnut as a mat. So a box here is a *patch* -- one contiguous stand of the
species -- and the detector's per-spike or per-leaf boxes are merged into
patches before writing. Merging is by overlap plus a small gap allowance, so two
stands either side of open water stay two boxes.

Each species is prompted on its own photos only. That is what makes the class
id free: whatever it boxes in a loosestrife photo is labelled loosestrife.
Photos where it boxes nothing above --conf are left out, not padded.

Why not YOLOE, which ultralytics already ships: its text prompts need OpenAI's
clip package, installable only from GitHub, and the prompt-free variant's tag
names are misaligned in ultralytics 8.4.138 (loosestrife comes back as
"Golden Gate Bridge"). Grounding DINO needs only transformers, from PyPI.

Writes, per run:
    data/annotated/vegnet_v1/images, labels    the staged dataset
    data/annotated/vegnet_v1/data.yaml         names in SPECIES order
    outputs/vegnet_autolabel/<slug>.jpg        contact sheets, boxes drawn
    outputs/vegnet_autolabel/autolabel.csv     per-photo box count and conf

Review the contact sheets; add the photo id of any bad one to
data/vegnet_raw/rejected.txt, re-run fetch_vegnet.py for replacements, then
re-run this. Train/val lists are written by the notebook, as for loonnet.

Usage:
    scripts/autolabel_vegnet.py                 # label, stage up to --per each
    scripts/autolabel_vegnet.py --conf 0.4      # stricter drafts
"""
import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_vegnet import DEST as RAW, REJECTS, SPECIES, load_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "data", "annotated", "vegnet_v1")
OUT = os.path.join(ROOT, "outputs", "vegnet_autolabel")
MODEL_ID = "IDEA-Research/grounding-dino-base"   # cached under ~/.cache/huggingface

# Several phrasings per species; a box from any of them counts. Latin names do
# poorly -- the text encoder has seen "yellow iris" far more often than "Iris
# pseudacorus" -- so each list says what the plant looks like from the shore,
# whole plant first, so the merged patch covers the stand and not one flower.
PROMPTS = {
    "loosestrife":    ["purple loosestrife plant", "purple flower spikes"],
    "phragmites":     ["reeds", "tall reed grass with plumes"],
    "water_chestnut": ["floating plant rosettes", "floating leaves"],
    "yellow_iris":    ["yellow iris plant", "yellow iris flower"],
}
CLASSES = sorted(SPECIES)          # class id = index; matches data.yaml

GAP = 0.02           # merge boxes closer than this fraction of the image side
MIN_AREA = 0.005     # drop merged patches under this fraction of the frame


def merge_patches(boxes, w, h):
    """Union boxes that overlap or nearly touch, until nothing changes."""
    gx, gy = GAP * w, GAP * h
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            a = boxes.pop()
            i = 0
            while i < len(boxes):
                b = boxes[i]
                if (a[0] - gx <= b[2] and b[0] - gx <= a[2]
                        and a[1] - gy <= b[3] and b[1] - gy <= a[3]):
                    a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(a)
        boxes = out
    return [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) >= MIN_AREA * w * h]


def contact_sheet(items, path, cols=6, cell=300):
    """Grid of thumbnails with patches drawn and the photo id underneath."""
    tiles = []
    for img_path, patches in items:
        im = cv2.imread(img_path)
        h, w = im.shape[:2]
        for x1, y1, x2, y2 in patches:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255),
                          max(2, w // 200))
        s = cell / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        tile = np.full((cell + 22, cell, 3), 30, np.uint8)
        tile[:im.shape[0], :im.shape[1]] = im
        pid = os.path.basename(img_path).rsplit("_", 1)[-1][:-4]
        colour = (255, 255, 255) if patches else (80, 80, 255)
        cv2.putText(tile, f"{pid}  {len(patches)}", (4, cell + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 30))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--conf", type=float, default=0.3, help="box score floor (default 0.3)")
    p.add_argument("--per", type=int, default=33,
                   help="photos staged per species (default 33: 132 total, loonnet_v1's size)")
    p.add_argument("--device", default=None, help="mps / cuda / cpu (default: best available)")
    a = p.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    device = a.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    rejected = load_lines(REJECTS)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()

    def detect(path, text):
        im = Image.open(path).convert("RGB")
        inp = proc(images=im, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
        r = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=a.conf, text_threshold=0.25,
            target_sizes=[im.size[::-1]])[0]
        return r["boxes"].cpu().numpy().tolist(), r["scores"].cpu().numpy().tolist(), im.size

    for sub in ("images", "labels"):
        shutil.rmtree(os.path.join(DS, sub), ignore_errors=True)
        os.makedirs(os.path.join(DS, sub))
    os.makedirs(OUT, exist_ok=True)

    log = []
    for cls, slug in enumerate(CLASSES):
        folder = os.path.join(RAW, slug)
        photos = sorted(f for f in os.listdir(folder) if f.endswith(".jpg")
                        and f.rsplit("_", 1)[-1][:-4] not in rejected)
        text = " ".join(f"{t}." for t in PROMPTS[slug])     # Grounding DINO's phrase syntax
        sheet, staged = [], 0
        for name in photos:
            src = os.path.join(folder, name)
            raw, scores, (w, h) = detect(src, text)
            patches = merge_patches(raw, w, h)
            top = max(scores, default=0.0)
            take = bool(patches) and staged < a.per
            sheet.append((src, patches))
            log.append({"file": f"{slug}/{name}", "class": slug, "raw_boxes": len(raw),
                        "patches": len(patches), "max_conf": f"{top:.3f}",
                        "staged": int(take)})
            if not take:
                continue
            shutil.copy2(src, os.path.join(DS, "images", name))
            with open(os.path.join(DS, "labels", name[:-4] + ".txt"), "w") as f:
                for x1, y1, x2, y2 in patches:
                    f.write(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
            staged += 1
        contact_sheet(sheet, os.path.join(OUT, f"{slug}.jpg"))
        empty = sum(1 for _, pt in sheet if not pt)
        print(f"  {slug:15s} {len(photos):3d} photos  {empty:2d} with no box  {staged:3d} staged")

    with open(os.path.join(OUT, "autolabel.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(log[0]))
        wr.writeheader()
        wr.writerows(log)
    with open(os.path.join(DS, "data.yaml"), "w") as f:
        yaml.safe_dump({"path": DS, "train": "train.txt", "val": "val.txt",
                        "names": {i: SPECIES[s][1] for i, s in enumerate(CLASSES)}},
                       f, sort_keys=False)
    print(f"\nstaged {sum(r['staged'] for r in log)} images in {DS}")
    print(f"contact sheets: {OUT}")


if __name__ == "__main__":
    main()
