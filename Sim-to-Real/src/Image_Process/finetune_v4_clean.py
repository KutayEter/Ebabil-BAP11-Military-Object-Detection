from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "weights" / "v3_best.pt"
DATA_YAML = ROOT / "data" / "finetune_v4_clean.yaml"
RUNS_DIR = ROOT / "runs"
RUN_NAME = "BAP_YOLOV8S_REAL_FINETUNE_V4_CLEAN"

model = YOLO(str(MODEL_PATH))
model.train(data=str(DATA_YAML), epochs=30, imgsz=640, batch=32, device=0, workers=0, optimizer="AdamW", lr0=0.0001, lrf=0.05, weight_decay=0.0005, warmup_epochs=2.0, warmup_momentum=0.8, cos_lr=True, patience=12, hsv_h=0.01, hsv_s=0.30, hsv_v=0.25, degrees=3.0, translate=0.08, scale=0.25, shear=1.0, fliplr=0.5, flipud=0.0, mosaic=0.35, close_mosaic=5, mixup=0.0, amp=True, cache="ram", seed=42, deterministic=True, project=str(RUNS_DIR), name=RUN_NAME, exist_ok=False, save=True, plots=True, verbose=True)
