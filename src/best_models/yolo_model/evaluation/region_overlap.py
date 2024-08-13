import numpy as np
from ultralytics import YOLO
import pandas as pd
import cv2


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

# Load your trained YOLOv8 model
model = YOLO("../model/model_yolo_1000.pt")

# Load ground truth data from CSV
ground_truth_df = pd.read_csv('/root/src/dataset/real_data/labels/labels.csv')
image_folder = "/root/src/dataset/real_data/frames/"

# Initialize a list to store center errors
center_errors = []

# Iterate through each row in the ground truth dataframe
for index, row in ground_truth_df.iterrows():
    try:
        # Get ground truth bounding box and filename
        gt_bbox = row[['x1', 'y1', 'x2', 'y2']].to_numpy()
        image_path = image_folder + row['filename']
        
        image = cv2.imread(image_path)
        
        # Load the image and get the model's predicted bounding box
        results = model(image)
        detections = results[0].boxes.xywh  # assuming xywh format for the bounding boxes
        
        x_center, y_center, width, height = detections[0]
        predicted_bbox = [int(x_center-width/2), int(y_center-height/2), int(x_center+width/2), int(y_center+height/2)]

        region_overlap = calculate_region_overlap(gt_bbox, predicted_bbox)
        print("Region Overlap: ", region_overlap)
        
        with open('region_overlap.txt', 'a') as file:
            file.write(str(region_overlap) + "\n")
        
    except:
        print("Error during tracking")