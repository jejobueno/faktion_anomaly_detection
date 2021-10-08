import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import joblib

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import load_model

import tensorflow as tf

import cv2

import os

from glob import glob

anomalous_data_dir = os.path.join('.','assets',  'data', 'anomalous_dice', '*.jpg')
normal_data_dir = os.path.join('.','assets',  'data', 'normal_dice', '*')

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

model = load_model('./model/classification_model.h5')