import matplotlib.pyplot as plt
# Define the thresholds
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
models = ['batch_32_epochs_40', 'batch_32_epochs_20', 'batch_16_epochs_40']
colors = ['r', 'b', 'g']

txt_paths = ["region_overlap_batch_32_epochs_40.txt",
             "region_overlap_batch_32_epochs_20.txt",
             "region_overlap_batch_16_epochs_40.txt"]

i = 0
for txt_path in txt_paths:
# Read the values from the text file
    with open(txt_path, 'r') as file:
        values = [float(line.strip()) for line in file]

    # Initialize a dictionary to store the counts for each threshold
    counts = {threshold: 0 for threshold in thresholds}

    # Count the number of values that are higher than each threshold
    for value in values:
        for threshold in thresholds:
            if value >= threshold:
                counts[threshold] += 1

    scores = []

        # Print the results
    print(models[i])
    for threshold in thresholds:
        scores.append(counts[threshold]/len(values))
        print(f"Number of values higher than {threshold}: {counts[threshold]}")
        
    plt.plot(thresholds, scores, color=colors[i], label=models[i])
    i+=1

plt.xlabel("Overlap threshold")
plt.ylabel("Success rate")
plt.title("Success plot")
plt.grid()
plt.legend()
plt.show()