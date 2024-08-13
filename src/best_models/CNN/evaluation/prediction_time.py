import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import time

# Load your trained model
model = tf.keras.models.load_model('../model/model_10.h5')

# Open the video file
video_path = '/root/src/dataset/real_data/whole _video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
    
def preprocess_image(image):
        # Encode the numpy array to JPEG format using OpenCV
    _, jpeg_encoded_image = cv2.imencode('.jpg', image)

    # Convert the JPEG-encoded image to a byte string
    jpeg_encoded_image = jpeg_encoded_image.tobytes()

    # Decode the JPEG-encoded byte string to a tensor using TensorFlow
    image = tf.io.decode_jpeg(jpeg_encoded_image, channels=3)
    image = image / 255  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)
    return image


# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Resize frame to match the input shape of the model
    frame = frame[100:930, 500:1350]
    input_frame = preprocess_image(frame)

    start_time = time.time()
    # Predict the bounding box
    coord = model.predict(input_frame)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    with open('prediction_time.txt', 'a') as file:
        file.write(str(elapsed_time) + "\n")
    
    # Draw a rectangle around the detected ball
    cv2.rectangle(frame, (int(coord[0,0]), int(coord[0,1])), (int(coord[0,2]), int(coord[0,3])), 1000, 2)

    # Display the frame with the detected bounding box
    cv2.imshow('Ball Detection', frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
