import tensorflow as tf
from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import matplotlib.image as img
from scipy import signal
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time
import pandas as pd

import split_train_val_test

split_train_val_test.split()

train_folder = os.path.join('.', 'data', 'normal_dice', 'train')
# Flow training images in batches of 32 using train_datagen generator
train_generator = tf.keras.utils.image_dataset_from_directory(
    directory=train_folder,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    shuffle=True,
    seed=42
)

validation_folder = os.path.join('.', 'data', 'normal_dice', 'val')
# Flow training images in batches of 32 using train_datagen generator
validation_generator = tf.keras.utils.image_dataset_from_directory(
    directory=validation_folder,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    shuffle=True,
    seed=42
)

test_folder = os.path.join('.', 'data', 'normal_dice', 'test')
# Flow training images in batches of 32 using train_datagen generator
test_generator = tf.keras.utils.image_dataset_from_directory(
    directory=test_folder,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=1,
    shuffle=False,
    seed=42
)

anomalous_folder = os.path.join('.', 'data', 'anomalous_dice')
# Flow training images in batches of 32 using train_datagen generator
anomalous_generator = tf.keras.utils.image_dataset_from_directory(
    directory=anomalous_folder,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=1,
    shuffle=False,
    seed=42
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_generator.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = validation_generator.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 11

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(128, 128, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds,
                    epochs=3, batch_size=32)

test_images = []
test_labels = []
for images, labels in test_generator.take(-1):  # -1 take all the images.
    test_images.append(images.numpy())
    test_labels.append(labels.numpy())

test_images = np.squeeze(test_images)

class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def plot_image(predictions_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("Label predicted : {}".format(class_names[predicted_label]) + "\n" +
               "Similarity value : {:2.0f}".format(np.max(predictions_array)) + "\n" +
               "Image test label : {}".format(class_names[true_label]), color=color)


def plot_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(len(class_names)))
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    min_value = min(predictions_array) * 0.1
    max_value = max(predictions_array) * 0.1
    plt.ylim([min(predictions_array) + min_value, max(predictions_array) + max_value])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_class(img_class, class_value):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_class)
    plt.xlabel("Class value: {}".format(class_value))


train_images = []
train_class = []
for images, labels in train_ds.take(1):
    train_images.append(images.numpy())
    train_class.append(labels.numpy())

train_images = np.squeeze(train_images)

predictions = model.predict(test_generator)
anomalous_predictions = model.predict(anomalous_generator)

anomalous_images = []
anomalous_labels = []
for images, labels in anomalous_generator.take(-1):  # -1 take all the images.
    anomalous_images.append(images.numpy())
    anomalous_labels.append(labels.numpy())

anomalous_images = np.squeeze(anomalous_images)


def final_plot_normal(i):
    for j in range(len(train_class[0])):
        if train_class[0][j] == test_labels[i][0]:
            train_img_index = j
        else:
            next

    plt.figure(figsize=(10, 7))
    plt.subplot(1, 3, 1)
    plt.gca().set_title('Image to predict')
    plot_image(predictions[i], test_labels[i][0], test_images[i])
    plt.subplot(1, 3, 2)
    plt.gca().set_title('Similarity values')
    plot_value_array(predictions[i], test_labels[i][0])
    plt.subplot(1, 3, 3)
    plt.gca().set_title('Random image with same class')
    plot_class(train_images[train_img_index], test_labels[i][0])
    plt.show()


def final_plot_anomalous(i):
    for j in range(len(train_class[0])):
        if train_class[0][j] == np.argmax(anomalous_predictions[i]):
            train_img_index = j
        else:
            next
    plt.figure(figsize=(10, 7))
    plt.subplot(1, 3, 1)
    plt.gca().set_title('Image to predict')
    plot_image(anomalous_predictions[i], anomalous_labels[i][0], anomalous_images[i])
    plt.subplot(1, 3, 2)
    plt.gca().set_title('Similarity values')
    plot_value_array(anomalous_predictions[i], anomalous_labels[i][0])
    plt.subplot(1, 3, 3)
    plt.gca().set_title('Random image with same class')
    plot_class(train_images[train_img_index], np.argmax(anomalous_predictions[i]))
    plt.show()


final_plot_normal(10)
final_plot_anomalous(4)