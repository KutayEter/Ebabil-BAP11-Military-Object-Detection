from pathlib import Path
SOURCE_LABELS=Path("data/source/labels")
OUTPUT_LABELS=Path("data/remapped/labels")
SOURCE_CLASS_IDS={"1","3"}
TARGET_CLASS_ID="3"
OUTPUT_LABELS.mkdir(parents=True,exist_ok=True)
for label_path in SOURCE_LABELS.glob("*.txt"):
    if label_path.name.lower()=="classes.txt": continue
    new=[]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts=line.strip().split()
        if len(parts)>=5 and parts[0] in SOURCE_CLASS_IDS:
            parts[0]=TARGET_CLASS_ID; new.append(" ".join(parts))
    if new: (OUTPUT_LABELS/label_path.name).write_text("\n".join(new)+"\n",encoding="utf-8")
