from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "data.yaml"
RUNS_DIR = ROOT / "runs"

model = YOLO("yolov8s.yaml")
model.train(data=str(DATA_YAML), epochs=200, imgsz=640, batch=32, device=0, workers=0, optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005, warmup_epochs=5.0, warmup_momentum=0.8, cos_lr=True, patience=40, hsv_h=0.015, hsv_s=0.50, hsv_v=0.35, degrees=5.0, translate=0.10, scale=0.35, shear=2.0, fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=15, mixup=0.0, amp=True, cache="ram", seed=42, pretrained=False, project=str(RUNS_DIR), name="BAP_YOLOV8S_SCRATCH_640", exist_ok=False, save=True, plots=True, verbose=True)
