import os

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(
    validation_split=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_folder = os.path.abspath('./normal_dice/train')
# Flow training images in batches of 32 using train_datagen generator
train_generator = tf.keras.utils.image_dataset_from_directory(
    directory=train_folder,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    shuffle=True,
    seed=42
)

validation_generator = data_generator.flow_from_directory(
    directory=train_folder,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    shuffle=True,
    seed=42
)

history = classifier.fit_generator(
    train_generator,
    steps_per_epoch=(8000 / 86),
    epochs=2,
    validation_data=validation_generator,
    validation_steps=8000 / 86,
    callbacks=[learning_rate_reduction]
)
