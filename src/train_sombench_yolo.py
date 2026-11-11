"""Train and evaluate a pretrained YOLO11n crater baseline on SomBench NAC."""
from pathlib import Path
import json, os, platform
import torch

ROOT = Path("work/data/sombench_nac_yolo").resolve()
RUNS = Path("work/runs").resolve()
RUNS.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("work/yolo_config").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("work/mpl_config").resolve()))
torch.set_num_threads(6)
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
train = model.train(
    data=str(ROOT / "sombench_nac.yaml"),
    project=str(RUNS), name="yolo11n_sombench_nac", exist_ok=True,
    epochs=20, imgsz=640, batch=4, workers=0, device=0,
    patience=6, seed=42, deterministic=True, plots=True,
    cache=True, pretrained=True, amp=False, close_mosaic=3,
    save=True, verbose=True,
)
best = RUNS / "yolo11n_sombench_nac" / "weights" / "best.pt"
if not best.exists():
    raise FileNotFoundError(best)
model = YOLO(str(best))
metrics = model.val(
    data=str(ROOT / "sombench_nac.yaml"), split="test", imgsz=640,
    batch=4, workers=0, device=0, plots=True, project=str(RUNS),
    name="yolo11n_sombench_nac_test", exist_ok=True,
)
result = {
    "model": "Ultralytics YOLO11n, COCO pretrained, fine-tuned on SomBench NAC",
    "dataset": "NASA-IBM AI4Science/Sombench-NAC-Crater-Detection (CC-BY-4.0)",
    "data_split": "official box-level train/validation/test; standard label_set only",
    "label_filter": "diameter >= 5 px; truncated annotations excluded; large-only boxes excluded",
    "input_size": 640,
    "seed": 42,
    "epochs_requested": 20,
    "device": "CPU",
    "torch": torch.__version__,
    "ultralytics": __import__("ultralytics").__version__,
    "platform": platform.platform(),
    "best_checkpoint": str(best),
    "test_metrics": {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class_map50_95": [float(x) for x in metrics.box.maps],
    },
}
(RUNS / "yolo11n_sombench_nac_test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
