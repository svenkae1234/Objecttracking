import matplotlib.pyplot as plt

txt_paths = ["/root/src/own_model/best_model/evaluation/mean_center_error.txt",
             "/root/src/own_model/faster_rcnn_model/evaluation/mean_center_error.txt",
             "/root/src/own_model/yolo_model/evaluation/mean_center_error.txt"]

def read_files(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace and convert to float
            value = float(line.strip())
    return value

center_error_mean_values = []
for txt_path in txt_paths:
    center_error_mean_values.append(read_files(txt_path))

labels = ["CNN", "Faster RCNN", "Yolov8"]

for i in range(3):
    print(labels[i] + ": " + str(center_error_mean_values[i]))

# Create the bar chart
plt.bar(labels, center_error_mean_values, width=0.3)

# Add a title and labels for the axes (optional)
plt.title('Durchschnittlicher Center Error')
plt.xlabel('Model')
plt.ylabel('Center Error in Pixeln')
plt.grid()

# Display the plot
plt.show()
        