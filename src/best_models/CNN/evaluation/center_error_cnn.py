from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import cv2

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Function to calculate the center of a bounding box
def calculate_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def preprocess_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = image / 255  # Normalize to [0, 1]
    return image

def predict_ball_position(model, image_paths):
    # Preprocess the images
    images = [preprocess_image(image_path) for image_path in image_paths]
    images = tf.stack(images)  # Stack images into a single tensor
    
    # Ensure the batch dimension is correct
    images = tf.expand_dims(images, axis=0) if len(images.shape) == 3 else images

    # Predict coordinates
    predictions = model.predict(images)
    
    return predictions

model = load_model('../model/model_10.h5')
model.compile()

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
        
        image = preprocess_image(image_path)
        image = tf.expand_dims(image, axis=0)
        input_image = np.array(image)
        coord = model.predict(input_image)
        predicted_bbox = [int(coord[0,0]), int(coord[0,1]), int(coord[0,2]), int(coord[0,3])]

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
