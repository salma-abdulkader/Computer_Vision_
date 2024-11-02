import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

model = YOLO(r'c:\\Users\\dell\\Downloads\\best.pt') 
names = model.model.names

cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(frame, line_width=2,font_size=2)
    
    """
    """
    results = model.track(frame ,iou=0.5, show=False ,tracker="bytetrack.yaml", persist=True) #botsort.yaml	
    
    
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss=results[0].boxes.cls.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu()
        conf=results[0].boxes.conf.tolist() 
        
        for box, track_id ,cof,c in zip(boxes, track_ids,conf,clss):
            x1,y1,x2,y2 =box.int().tolist()
            # if cof > 0.4 :
            annotator.box_label(box, label=f"{names[c]}"), color=(255,0,0))
            annotator.text(xy=(x2,y1),text=str(track_id), box_style=True,txt_color=(255, 0, 0))
            annotator.display_objects_labels(
                frame,str(names[c]) , (104, 31, 17), (255, 255, 255), x1 + (x2-x1)//2, y1 +(y2-y1)//2, 10
            )
            
    # Break the loop if 'q' is pressed
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()