import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO(r'C:/Users/mshah/Desktop/YOLO/best.pt')

# Open the video
video_path = 0
cap = cv2.VideoCapture(video_path)
names = model.model.names

# Loop through video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Convert color to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLOv8 on the frame
    results = model(frame_rgb)
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  
        cls = names[int(result.cls[0])]  
        conf = round(float(result.conf[0]), 2) 
        
        # Draw rectangle and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{cls} {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

        
            
            
            
            
            
            
            
            
            
            
            