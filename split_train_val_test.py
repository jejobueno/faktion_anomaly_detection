"""
This scripts will split all the files into three folder train, test and val inside the data folder.
"""
import os
import shutil

from glob import glob


normal_data_dir = os.path.join('.', 'assets', 'data', 'normal_dice', '*')
anomalous_data_dir = os.path.join('.', 'assets', 'data', 'anomalous_dice', '*.jpg')
normal_train_dir = os.path.join('.', 'data', 'normal_dice', 'train')
normal_val_dir = os.path.join('.', 'data', 'normal_dice', 'val')
normal_test_dir = os.path.join('.', 'data', 'normal_dice', 'test')

normal_files_folders = glob(normal_data_dir)
normal_files_paths = []
if not os.path.exists(normal_train_dir):
    os.mkdir(os.path.join('.', 'data'))
    os.mkdir(os.path.join('.', 'data', 'normal_dice'))
    os.mkdir(os.path.join('.', 'data', 'anomalous_dice'))
    os.mkdir(normal_train_dir)
    os.mkdir(normal_val_dir)
    os.mkdir(normal_test_dir)

for folder in normal_files_folders:
    # If the folders does not exist we create them
    if not os.path.exists(os.path.join(normal_train_dir, os.path.basename(folder))):
        os.mkdir(os.path.join(normal_train_dir, os.path.basename(folder)))
        os.mkdir(os.path.join(normal_val_dir, os.path.basename(folder)))
        os.mkdir(os.path.join(normal_test_dir, os.path.basename(folder)))
    # recollect the paths of all the images
    files = glob(os.path.join(folder, '*.jpg'))
    i = 0
    for img in files:
        if i < 2:
            # First two images to the test folder
            shutil.copyfile(img, os.path.join(normal_test_dir, os.path.basename(folder), os.path.basename(img)))
        if 2 < i < 20:
            # Next 18 images to the test folder
            shutil.copyfile(img, os.path.join(normal_val_dir, os.path.basename(folder), os.path.basename(img)))
        else:
            # Rest of the files to the train folder
            shutil.copyfile(img, os.path.join(normal_train_dir, os.path.basename(folder), os.path.basename(img)))
        i += 1

for img in glob(anomalous_data_dir):
    shutil.copyfile(img, os.path.join('.', 'data', 'anomalous_dice', os.path.basename(img)))
