from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "weights" / "v2_best.pt"
DATA_YAML = ROOT / "data" / "finetune_v3.yaml"
RUNS_DIR = ROOT / "runs"

model = YOLO(str(MODEL_PATH))
model.train(data=str(DATA_YAML), epochs=40, imgsz=640, batch=32, device=0, workers=0, optimizer="AdamW", lr0=0.0002, lrf=0.05, weight_decay=0.0005, warmup_epochs=2.0, warmup_momentum=0.8, cos_lr=True, patience=15, hsv_h=0.01, hsv_s=0.35, hsv_v=0.25, degrees=3.0, translate=0.08, scale=0.25, shear=1.0, fliplr=0.5, flipud=0.0, mosaic=0.5, close_mosaic=10, mixup=0.0, amp=True, cache="ram", seed=42, project=str(RUNS_DIR), name="BAP_YOLOV8S_PERSON_FINE_TUNE_V3", exist_ok=False, save=True, plots=True, verbose=True)
