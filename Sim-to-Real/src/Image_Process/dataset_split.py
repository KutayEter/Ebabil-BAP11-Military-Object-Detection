from pathlib import Path
import random, shutil
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data"/"pool"; DEST=ROOT/"data"/"dataset"
SRC_IMAGES=SOURCE/"images"; SRC_LABELS=SOURCE/"labels"
TRAIN_IMAGES=DEST/"images"/"train"; TRAIN_LABELS=DEST/"labels"/"train"
VAL_IMAGES=DEST/"images"/"val"; VAL_LABELS=DEST/"labels"/"val"
for p in [TRAIN_IMAGES,TRAIN_LABELS,VAL_IMAGES,VAL_LABELS]: p.mkdir(parents=True,exist_ok=True)
random.seed(42); exts={".jpg",".jpeg",".png",".webp",".bmp"}; samples=[]
for img in SRC_IMAGES.iterdir():
    if not img.is_file() or img.suffix.lower() not in exts: continue
    lbl=SRC_LABELS/f"{img.stem}.txt"
    if not lbl.exists(): continue
    classes=set()
    for line in lbl.read_text(encoding="utf-8").splitlines():
        parts=line.split()
        if len(parts)>=5:
            try: classes.add(int(float(parts[0])))
            except: pass
    if classes: samples.append((img,lbl,classes))
by=defaultdict(list); multi=[]
for s in samples:
    (by[next(iter(s[2]))] if len(s[2])==1 else multi).append(s)
single_total=sum(len(by[c]) for c in range(4)); targets={c:int(len(by[c])/single_total*300) for c in range(4)}
remaining=300-sum(targets.values()); fr={c:(len(by[c])/single_total*300)-targets[c] for c in range(4)}
for c in sorted(fr,key=fr.get,reverse=True):
    if remaining<=0: break
    targets[c]+=1; remaining-=1
val=[]; used=set()
for c in range(4):
    items=by[c].copy(); random.shuffle(items); chosen=items[:targets[c]]; val.extend(chosen); used.update(x[0].stem for x in chosen)
train=[s for s in samples if s[0].stem not in used]
for group,idst,ldst in [(train,TRAIN_IMAGES,TRAIN_LABELS),(val,VAL_IMAGES,VAL_LABELS)]:
    for img,lbl,_ in group: shutil.copy2(img,idst/img.name); shutil.copy2(lbl,ldst/lbl.name)
print("Train",len(train),"Val",len(val))
