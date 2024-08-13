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

def calculate_points_of_bbox(x1, y1, x2, y2):
    points_inside_square = set()
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            points_inside_square.add((x, y))
        # Convert the list of points to a NumPy array
    return points_inside_square

def calculate_area_of_overlap(ground_truth_points, prediction_points):
    overlapping_points = ground_truth_points & prediction_points

    # Count the number of overlapping points
    num_of_overlap = len(overlapping_points)    
    return num_of_overlap
                
def calculate_region_overlap(ground_truth, prediction):

    ground_truth_points = calculate_points_of_bbox(ground_truth[0], ground_truth[1], ground_truth[2], ground_truth[3])
    prediction_points = calculate_points_of_bbox(prediction[0], prediction[1], prediction[2], prediction[3])

    area_of_overlap = calculate_area_of_overlap(ground_truth_points, prediction_points)
    area_of_union = len(ground_truth_points) + len(prediction_points)
    
    return area_of_overlap/(area_of_union-area_of_overlap)

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

model = torch.load("../model/faster_rcnn_batch_32_epochs_20.pth")

# Load ground truth data from CSV
ground_truth_df = pd.read_csv('/root/src/dataset/real_data/labels/labels.csv')
image_folder = "/root/src/dataset/real_data/frames/"

model.eval()

# Iterate through each row in the ground truth dataframe
for index, row in ground_truth_df.iterrows():
    try:
        # Get ground truth bounding box and filename
        gt_bbox = row[['x1', 'y1', 'x2', 'y2']].to_numpy()
        image_path = image_folder + row['filename']
        
        image = cv2.imread(image_path)
        image_pil = Image.open(image_path)
        input_image = preprocess(image_pil)
        coord = inference(input_image, device, model)
        predicted_bbox = [int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])]

        region_overlap = calculate_region_overlap(gt_bbox, predicted_bbox)
        print("Region Overlap: ", region_overlap)
        
        with open('region_overlap.txt', 'a') as file:
            file.write(str(region_overlap) + "\n")
        
    except:
        print("Error during tracking")