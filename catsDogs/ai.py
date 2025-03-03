import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, models

class_names = ["cat", "dog"]
img_height = 32
img_width = 32
batch_size = 2

DS_TRAIN = tf.keras.preprocessing.image_dataset_from_directory(
    'sub/',
    labels='inferred',
    label_mode = "int", #binary
    #class_names=['0', '1']
    color_mode='grayscale',
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle=True,
    seed = 123,
    validation_split = 0.1,
    subset="training",
)

DS_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'sub/',
    labels='inferred',
    label_mode = "int", #binary
    #class_names=['0', '1']
    color_mode='grayscale',
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle=True,
    seed = 123,
    validation_split = 0.1,
    subset="validation",
)

model = keras.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(DS_TRAIN, epochs=10, validation_data=DS_validation)

loss, accuracy = model.evaluate(DS_validation)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")