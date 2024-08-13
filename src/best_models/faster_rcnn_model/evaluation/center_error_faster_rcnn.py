import torch
import pandas as pd
import numpy as np
import math
from PIL import Image
import os
import cv2

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

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

NUM_CLASSES = 2  # background=0 included, Suzanne = 1

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

model = torch.load("../model/faster_rcnn_batch_16_epochs_40.pth")

# Load ground truth data from CSV
ground_truth_df = pd.read_csv('/root/src/dataset/real_data/labels/labels.csv')
image_folder = "/root/src/dataset/real_data/frames/"

# Initialize a list to store center errors
center_errors = []

model.eval()

# Iterate through each row in the ground truth dataframe
for index, row in ground_truth_df.iterrows():
    try:
        # Get ground truth bounding box and filename
        gt_bbox = row[['x1', 'y1', 'x2', 'y2']].to_numpy()
        image_path = image_folder + row['filename']
        
        image_pil = Image.open(image_path)
        input_image = preprocess(image_pil)
        coord = inference(input_image, device, model)
        predicted_bbox = [int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])]
        
        # Calculate the center of the ground truth and predicted bounding boxes
        gt_center = calculate_center(gt_bbox)
        pred_center = calculate_center(predicted_bbox)
        
        # Calculate the center error (Euclidean distance)
        error = euclidean_distance(gt_center, pred_center)
        
        print("Center Error: ", error)
        
            # Append the error to the list
        center_errors.append(error)
    except:
        print("Error during tracking")
    


# Calculate the mean center error
mean_center_error = np.mean(center_errors)
print(f'Mean Center Error: {mean_center_error}')

with open('mean_center_error.txt', 'w') as file:
    file.write(str(mean_center_error))
