import os
import cv2
from glob import glob


def load_good_dices():
    normal_data_dir = os.path.join('.', 'assets', 'data', 'normal_dice', '*')
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

    return load_images(normal_files_paths), labels


def load_anomalous_dices():
    anomalous_data_dir = os.path.join('.', 'assets',  'data', 'anomalous_dice', '*.jpg')

    anomalous_files = glob(anomalous_data_dir)
    print(f"{len(anomalous_files)} anomalous files found")
    return load_images(anomalous_files), [11]*len(anomalous_files)


def load_images(images_path):
    img_size = 224
    images = []
    for img in images_path:
        try:
            img_arr = cv2.imread(img, 0)  # convert BGR to RGB format
            resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
            images.append(resized_arr)
        except Exception as e:
            print(e)
    return images
