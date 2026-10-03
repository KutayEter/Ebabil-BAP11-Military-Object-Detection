from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data"/"dataset"
CLASS_NAMES={0:"tank",1:"hava_savunma",2:"askeri_personel",3:"askeri_arac"}
exts={".jpg",".jpeg",".png",".webp",".bmp"}
for split in ["train","val"]:
    images_dir=BASE/"images"/split; labels_dir=BASE/"labels"/split
    images=[p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]; labels=list(labels_dir.glob("*.txt"))
    bbox=Counter(); imcnt=Counter(); unknown=Counter()
    for lbl in labels:
        seen=set()
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts=line.split()
            if len(parts)<5: continue
            try: cls=int(float(parts[0]))
            except: continue
            if cls in CLASS_NAMES: bbox[cls]+=1; seen.add(cls)
            else: unknown[cls]+=1
        for cls in seen: imcnt[cls]+=1
    print(split.upper(),"images",len(images),"labels",len(labels))
    for cls in range(4): print(cls,CLASS_NAMES[cls],"image",imcnt[cls],"bbox",bbox[cls])
    print("unknown",dict(unknown))
