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
thresh = []

for img in normal_files_paths:
    try:
        img_arr = cv2.imread(img)[..., ::-1]  # convert BGR to RGB format
        resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
        _, thresh = cv2.threshold(resized_arr, 80, 255, cv2.THRESH_BINARY_INV)
        normal_images.append(resized_arr)
        thresh.append(thresh)
    except Exception as e:
        print(e)