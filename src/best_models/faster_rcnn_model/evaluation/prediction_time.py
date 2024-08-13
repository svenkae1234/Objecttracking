import torch
import time
import cv2
from PIL import Image

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)

NUM_CLASSES = 2

def get_faster_rcnn_model(num_classes):
    """return model and preprocessing transform"""
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    )
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
        in_features=model.roi_heads.box_predictor.cls_score.in_features,
        out_features=num_classes,
        bias=True,
    )
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
        in_features=model.roi_heads.box_predictor.bbox_pred.in_features,
        out_features=num_classes * 4,
        bias=True,
    )
    preprocess = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT.transforms()
    return model, preprocess

def inference(img, device, model):
    with torch.no_grad():
        pred = model([img.to(device)]) # forward pass
    x1, y1, x2, y2 = pred[0]["boxes"].cpu().detach().numpy()[0]
    return [x1,y1,x2,y2]

model, preprocess = get_faster_rcnn_model(num_classes=NUM_CLASSES)
model.to(device)

model = torch.load("../model/faster_rcnn_batch_32_epochs_40.pth")

model.eval()

# Open the video file
video_path = '/root/src/dataset/real_data/whole _video.mp4'
cap = cv2.VideoCapture(video_path)

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    try: 
        frame = frame[100:930, 500:1350]
        frame_pil = Image.fromarray(frame)
        input_frame = preprocess(frame_pil)

        start_time = time.time()
        # Predict the bounding box
        coord = inference(input_frame, device, model)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        with open('prediction_time.txt', 'a') as file:
            file.write(str(elapsed_time) + "\n")
        
        # Draw a rectangle around the detected ball
        cv2.rectangle(frame, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), 1000, 2)
    except:
        print("Error during tracking!!!")

    # Display the frame with the detected bounding box
    cv2.imshow('Ball Detection', frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
