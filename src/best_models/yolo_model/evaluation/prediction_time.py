import cv2
from ultralytics import YOLO
import time

# Load your trained YOLOv8 model
model = YOLO("../model/model_yolo_1000.pt")

# Open a video file or a stream
#video_path = "/root/src/data/video/data_cut_1080p.mp4"

video_path = "/root/src/dataset/real_data/whole _video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read() 
    if not ret:
        break
    
    try:
        frame = frame[100:930, 500:1350]
        # Get detections from YOLO model
        start_time = time.time()
        results = model(frame)
        detections = results[0].boxes.xywh  # assuming xywh format for the bounding boxes
        conf = results[0].boxes.conf
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Detection time: {elapsed_time} seconds")
        with open('prediction_time.txt', 'a') as file:
            file.write(str(elapsed_time) + "\n")
        
        x_center, y_center, width, height = detections[0]
        
        cv2.rectangle(frame, (int(x_center-width/2), int(y_center-height/2)), (int(x_center+width/2), int(y_center+height/2)), (255, 0, 0), 2)
    except:
        print("Error during tracking")
    
    # Display the frame
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()