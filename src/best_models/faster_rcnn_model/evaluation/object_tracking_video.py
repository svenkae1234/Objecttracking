import torch
import cv2
import math
import numpy as np
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)

# Function to calculate the center of a bounding box
def calculate_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Get the appropriate device (CPU, GPU, or MPS)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

NUM_CLASSES = 2  # background=0 included, target object=1

def get_faster_rcnn_model(num_classes):
    """Return model and preprocessing transform."""
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
        pred = model([img.to(device)])  # Forward pass
    if len(pred[0]["boxes"]) > 0:  # Check if any bounding boxes were detected
        x1, y1, x2, y2 = pred[0]["boxes"].cpu().detach().numpy()[0]
        return [x1, y1, x2, y2]
    else:
        return None

# Load the model and preprocessing transform
model, preprocess = get_faster_rcnn_model(num_classes=NUM_CLASSES)
model.to(device)

# Load the trained model weights
model = torch.load("../model/faster_rcnn_batch_16_epochs_40.pth")
model.eval()

# Open a connection to the camera
cap = cv2.VideoCapture(1)  # Use 0 for the default camera
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from the camera.")
        break

    # Convert the frame to RGB and then to a PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    input_tensor = preprocess(frame_pil)

    # Perform inference
    coord = inference(input_tensor, device, model)

    # Draw the bounding box if detection is successful
    if coord is not None:
        x1, y1, x2, y2 = map(int, coord)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

        # Calculate and display the center
        center_x, center_y = calculate_center([x1, y1, x2, y2])
        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)  # Red center

    # Display the frame
    cv2.imshow("Object Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
