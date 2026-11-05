"""Fine-tune YOLO26n on NEU-DET and export the weights the API serves.

Starts from COCO-pretrained YOLO26n weights, fine-tunes on the train split,
picks the best epoch on the val split, then reports final numbers on the
held-out test split (never seen during training or model selection).

Outputs:
    models/defect_yolo26n.pt   weights loaded by the API
    models/metrics.json        test-set metrics shown in the README

Usage:
    python scripts/download_data.py
    python scripts/train.py [--epochs 60] [--imgsz 640] [--device mps]
"""

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "neu-det" / "data.yaml"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"
# COCO-pretrained starting point, downloaded here on first run
BASE_WEIGHTS = MODELS_DIR / "base" / "yolo26n.pt"
OUTPUT_WEIGHTS = MODELS_DIR / "defect_yolo26n.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps", help="mps, cpu, or a CUDA index")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit("dataset missing: run python scripts/download_data.py first")

    # A full path makes Ultralytics download the base weights into the
    # project instead of the current directory; global settings stay untouched.
    BASE_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=15,
        seed=42,
        deterministic=True,
        project=str(RUNS_DIR),
        name="neu-det",
        exist_ok=True,
        plots=True,
    )

    best = RUNS_DIR / "neu-det" / "weights" / "best.pt"
    shutil.copy(best, OUTPUT_WEIGHTS)

    # Final evaluation on the untouched test split.
    test = YOLO(OUTPUT_WEIGHTS).val(
        data=str(DATA_YAML),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        project=str(RUNS_DIR),
        name="neu-det-test",
        exist_ok=True,
    )

    names = test.names
    metrics = {
        "model": "YOLO26n",
        "dataset": "NEU-DET (KeenForgeAI/NEU-DET-corrected)",
        "split": "test",
        "images": 180,
        "imgsz": args.imgsz,
        "epochs_trained": args.epochs,
        "mAP50": round(float(test.box.map50), 4),
        "mAP50_95": round(float(test.box.map), 4),
        "precision": round(float(test.box.mp), 4),
        "recall": round(float(test.box.mr), 4),
        "per_class_mAP50": {
            names[int(c)]: round(float(v), 4)
            for c, v in zip(test.box.ap_class_index, test.box.ap50, strict=True)
        },
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
