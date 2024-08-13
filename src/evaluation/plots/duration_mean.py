import matplotlib.pyplot as plt

txt_paths = ["/root/src/own_model/best_model/prediction_time.txt",
             "/root/src/own_model/faster_rcnn_model/prediction_time.txt",
             "/root/src/own_model/yolo_model/prediction_time.txt"]

# Function to calculate the mean of values in a text file
def calculate_mean_from_file(file_path):
    # Initialize an empty list to store the values
    values = []
    
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read each line in the file
        for line in file:
            # Strip any leading/trailing whitespace and convert to float
            value = float(line.strip())
            # Append the value to the list
            values.append(value)
    
    # Calculate the mean
    mean_value = sum(values) / len(values) if values else 0
    
    return mean_value

# Example usage
mean_values = []
for txt_path in txt_paths:
    mean_values.append(calculate_mean_from_file(txt_path))

labels = ["CNN", "Faster RCNN", "Yolov8"]

for i in range(3):
    print(labels[i] + ": " + str(mean_values[i]))

# Create the bar chart
plt.bar(labels, mean_values, width=0.3)

# Add a title and labels for the axes (optional)
plt.title('Durchschnittliche Zeit für eine Prediction')
plt.xlabel('Model')
plt.ylabel('Mean in Sekunden')
plt.grid()

# Display the plot
plt.show()
