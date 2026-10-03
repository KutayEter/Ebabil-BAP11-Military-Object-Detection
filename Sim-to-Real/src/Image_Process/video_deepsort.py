from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path
import cv2, torch
ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"weights"/"best.pt"
VIDEO_PATH=ROOT/"samples"/"test_video.mp4"
OUTPUT_PATH=ROOT/"runs"/"video_deepsort.mp4"
model=YOLO(str(MODEL_PATH))
tracker=DeepSort(max_age=20,n_init=2,max_iou_distance=0.7,max_cosine_distance=0.3,nn_budget=100,embedder="mobilenet",half=True,bgr=True,embedder_gpu=torch.cuda.is_available())
cap=cv2.VideoCapture(str(VIDEO_PATH))
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
writer=cv2.VideoWriter(str(OUTPUT_PATH),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
while True:
    ret,frame=cap.read()
    if not ret: break
    result=model.predict(source=frame,conf=0.25,imgsz=640,device=0 if torch.cuda.is_available() else "cpu",verbose=False)[0]
    detections=[]
    if result.boxes is not None:
        for box in result.boxes:
            x1,y1,x2,y2=box.xyxy[0].cpu().numpy(); conf=float(box.conf[0]); cls_id=int(box.cls[0]); cls_name=model.names[cls_id]
            detections.append(([float(x1),float(y1),float(x2-x1),float(y2-y1)],conf,cls_name))
    tracks=tracker.update_tracks(detections,frame=frame)
    for track in tracks:
        if not track.is_confirmed(): continue
        x1,y1,x2,y2=map(int,track.to_ltrb(orig=False)); cls_name=track.get_det_class() or "object"
        label=f"{cls_name} ID:{track.track_id}" + (" KALMAN" if track.time_since_update>0 else "")
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2); cv2.putText(frame,label,(x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2,cv2.LINE_AA)
    writer.write(frame)
cap.release(); writer.release()
