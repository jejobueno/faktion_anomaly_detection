import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf

import cv2

import os

from glob import glob

anomalous_data_dir = os.path.join('.', 'data', 'anomalous_dice', '*.jpg')
normal_data_dir = os.path.join('.', 'data', 'normal_dice', '*')

anomalous_files = glob(anomalous_data_dir)
print(f"{len(anomalous_files)} anomalous files found")

normal_files_folders = glob(normal_data_dir)
normal_files_paths = []
labels = []
for folder in normal_files_folders:
    files = glob(folder + '\\*.jpg')
    normal_files_paths += files
    labels += [int(os.path.basename(folder))] * len(files)

print(f"{len(normal_files_paths)} normal files found")
print(f"{len(labels)} normal files labels found")
# for x in range(len(anomalous_files)):
#    print(anomalous_files[x])

img_size = 224
normal_images = []

for img in normal_files_paths:
    try:
        img_arr = cv2.imread(img)[..., ::-1]  # convert BGR to RGB format
        resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
        normal_images.append(resized_arr)
    except Exception as e:
        print(e)

plt.figure(figsize=(5, 5))
plt.imshow(normal_images[1000])
plt.title(labels[1000])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(normal_images, labels, test_size=0.33, random_state=42)

# Normalize the data
X_train = np.array(X_train) / 255
X_test = np.array(X_test) / 255

X_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

X_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(224, 224, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(11, activation="softmax"))

model.summary()

opt = Adam(lr=0.000001)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(500)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions = model.predict_classes(X_test)
predictions = predictions.reshape(1, -1)[0]
print(classification_report(y_test, predictions, target_names=['Rugby (Class 0)', 'Soccer (Class 1)']))
