from pathlib import Path
from ultralytics import YOLO
from collections import Counter
import csv

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "weights" / "best.pt"
TEST_ROOT = ROOT / "real_test_bulk"
OUTPUT_ROOT = ROOT / "runs" / "REAL_BULK_TEST"
CLASS_IDS = {"tank":0, "hava_savunma":1, "askeri_personel":2, "askeri_arac":3}
CLASS_NAMES = {v:k for k,v in CLASS_IDS.items()}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
model = YOLO(str(MODEL_PATH))
overall=Counter(); class_stats={k:Counter() for k in CLASS_IDS}; rows=[]
for folder_name, expected_id in CLASS_IDS.items():
    folder=TEST_ROOT/folder_name
    if not folder.exists(): continue
    out_dir=OUTPUT_ROOT/folder_name; out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        result=model.predict(source=str(image_path), conf=0.25, imgsz=640, device=0, verbose=False)[0]
        detected_ids=[int(x.item()) for x in result.boxes.cls] if result.boxes is not None else []
        detected_names=[CLASS_NAMES.get(x, f"class_{x}") for x in detected_ids]
        if not detected_ids: status="NONE"; overall["none"]+=1; class_stats[folder_name]["none"]+=1
        elif expected_id in detected_ids: status="CORRECT"; overall["correct"]+=1; class_stats[folder_name]["correct"]+=1
        else: status="WRONG"; overall["wrong"]+=1; class_stats[folder_name]["wrong"]+=1
        overall["total"]+=1; class_stats[folder_name]["total"]+=1
        result.save(filename=str(out_dir/image_path.name))
        rows.append([folder_name,image_path.name,CLASS_NAMES[expected_id],",".join(detected_names),status])
with open(OUTPUT_ROOT/"real_test_results.csv","w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["folder","image","expected_class","detected_classes","status"]); w.writerows(rows)
for k,s in class_stats.items():
    total=s["total"]; rate=(s["correct"]/total*100) if total else 0; print(k, dict(s), f"%{rate:.1f}")
