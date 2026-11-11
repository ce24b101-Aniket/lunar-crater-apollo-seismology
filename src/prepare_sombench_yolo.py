"""Convert official SomBench NAC COCO boxes to YOLO format for a baseline run."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

SRC = Path("work/data/Sombench-NAC-Crater-Detection")
OUT = Path("work/data/sombench_nac_yolo")
SPLITS = ("train", "val", "test")

summary = {}
for split in SPLITS:
    src = json.loads((SRC / f"annotations_{split}.json").read_text(encoding="utf-8"))
    id_to_anns = {}
    for ann in src["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)
    counts = {"images": 0, "standard_images": 0, "boxes_ge5px": 0, "excluded_large_images": 0}
    for image in src["images"]:
        image_id = image["id"]
        anns = id_to_anns.get(image_id, [])
        # The benchmark's "large" boxes are only complete above 10 m. Exclude
        # those tiles from this all-size detector comparison to avoid unlabeled craters.
        if image.get("label_set") != "standard":
            counts["excluded_large_images"] += 1
            continue
        arr = np.load(SRC / image["file_name"])
        if arr.shape != (256, 256) or arr.dtype != np.uint8:
            raise ValueError(f"Unexpected image shape/type: {image['file_name']} {arr.shape} {arr.dtype}")
        stem = Path(image["file_name"]).stem
        img_dest = OUT / "images" / split / f"{stem}.png"
        lab_dest = OUT / "labels" / split / f"{stem}.txt"
        img_dest.parent.mkdir(parents=True, exist_ok=True)
        lab_dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, mode="L").save(img_dest)
        rows = []
        for ann in anns:
            if float(ann.get("diam_px", 0)) < 5.0 or ann.get("truncated", False):
                continue
            x, y, w, h = map(float, ann["bbox"])
            xc = min(max((x + w / 2) / 256, 0), 1)
            yc = min(max((y + h / 2) / 256, 0), 1)
            wn = min(max(w / 256, 0), 1)
            hn = min(max(h / 256, 0), 1)
            if wn > 0 and hn > 0:
                rows.append(f"0 {xc:.8f} {yc:.8f} {wn:.8f} {hn:.8f}")
        lab_dest.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        counts["images"] += 1
        counts["standard_images"] += 1
        counts["boxes_ge5px"] += len(rows)
    summary[split] = counts

root = OUT.resolve().as_posix()
yaml = f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: crater\n"
(OUT / "sombench_nac.yaml").write_text(yaml, encoding="utf-8")
(OUT / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("YAML:", OUT / "sombench_nac.yaml")
