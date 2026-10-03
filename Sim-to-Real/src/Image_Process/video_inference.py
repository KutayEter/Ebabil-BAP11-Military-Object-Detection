from ultralytics import YOLO
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"weights"/"best.pt"
VIDEO_PATH=ROOT/"samples"/"test_video.mp4"
OUTPUT_DIR=ROOT/"runs"/"VIDEO_TEST"
model=YOLO(str(MODEL_PATH))
model.predict(source=str(VIDEO_PATH), conf=0.25, imgsz=640, device=0, save=True, project=str(OUTPUT_DIR.parent), name=OUTPUT_DIR.name, exist_ok=True, show=False, verbose=True)
