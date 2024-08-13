from tensorflow.keras import layers, models

def create_model():

    model = models.Sequential()
    model.add(layers.Input([830, 850, 3]))

    model.add(layers.Conv2D(512, (6, 6), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (6, 6), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (6, 6), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (4, 4), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(4))

    return model