import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import pandas as pd
import os
import matplotlib.pyplot as plt

from model import create_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)
#gpus = tf.config.list_logical_devices('GPU')

def load_dataset(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = image / 255  # Normalize to [0, 1]
    return image, label

model = create_model()
model.summary()
print(model)

labels_file = '/root/src/dataset/real_data/labels/labels.csv'
image_dir = '/root/src/dataset/real_data/frames'
num_of_samples = 1000

labels_df = pd.read_csv(labels_file)
labels_df = labels_df.head(num_of_samples)

# Create TensorFlow datasets
filenames = [os.path.join(image_dir, fname) for fname in labels_df['filename']]
labels = labels_df[['x1','y1','x2','y2']]

# Create a dataset from the filenames and labels
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(lambda x, y: load_dataset(x, y), num_parallel_calls=tf.data.AUTOTUNE)

# Split into training and validation datasets
train_size = int(0.8 * len(filenames))
train_dataset = dataset.take(train_size).shuffle(buffer_size=train_size).batch(1).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = dataset.skip(train_size).batch(4).prefetch(buffer_size=tf.data.AUTOTUNE)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=['mae'])

history = model.fit(train_dataset,
                    epochs=1000,
                    validation_data=val_dataset,
                    callbacks=[early_stopping])

# Save the weights
model.save('../model/model.h5')

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()