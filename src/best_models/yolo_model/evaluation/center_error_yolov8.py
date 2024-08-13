import pandas as pd
import numpy as np
import math
from ultralytics import YOLO
import cv2

# Function to calculate the center of a bounding box
def calculate_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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
        # Calculate the center of the ground truth and predicted bounding boxes
        gt_center = calculate_center(gt_bbox)
        pred_center = calculate_center(predicted_bbox)
        
        # Calculate the center error (Euclidean distance)
        error = euclidean_distance(gt_center, pred_center)
        center_errors.append(error)
        
        print("Center Error: ", error)
    except:
        print("Error during tracking")

# Calculate the mean center error
mean_center_error = np.mean(center_errors)
print(f'Mean Center Error: {mean_center_error}')

with open('mean_center_error.txt', 'w') as file:
    file.write(str(mean_center_error))
