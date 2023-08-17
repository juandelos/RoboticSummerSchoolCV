# import os
# import numpy as np


# dir_list = ['ALU_p40', 'ALU_p80', 'ALU_p120', 'ALU_p180', 'ALU_p240', 'ALU_p400', 'ALU_p500']

# for x in dir_list:
#     len = len(os.listdir('vision_project/ALU_p40'))
    

import os
import random
import shutil

# Path to the main folder containing class folders
main_folder = 'vision_project'

# List of class names (folder names)
class_names = ['STEEL_p40', 'STEEL_p80', 'STEEL_p120', 'STEEL_p180', 'STEEL_p240', 'STEEL_p400', 'STEEL_p500', 'ALU_p40', 'ALU_p80', 'ALU_p120', 'ALU_p180', 'ALU_p240', 'ALU_p400', 'ALU_p500']

# Create folders for train, test, and validation sets
train_folder = 'train'
test_folder = 'test'
val_folder = 'val'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Percentage of data to allocate to each set
train_ratio = 0.75
test_ratio = 0.15
val_ratio = 0.1

# Loop through each class folder
for class_name in class_names:
    class_folder = os.path.join(main_folder, class_name)
    images = os.listdir(class_folder)
    num_images = len(images)
    
    # Shuffle the images randomly
    random.shuffle(images)
    
    # Calculate the number of images for each set
    num_train = int(train_ratio * num_images)
    num_test = int(test_ratio * num_images)
    num_val = num_images - num_train - num_test
    
    # Split the images into sets
    train_images = images[:num_train]
    test_images = images[num_train:num_train+num_test]
    val_images = images[num_train+num_test:]
    
    # Move images to respective set folders
    for image in train_images:
        src = os.path.join(class_folder, image)
        dest = os.path.join(train_folder, class_name, image)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(src, dest)
    
    for image in test_images:
        src = os.path.join(class_folder, image)
        dest = os.path.join(test_folder, class_name, image)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(src, dest)
    
    for image in val_images:
        src = os.path.join(class_folder, image)
        dest = os.path.join(val_folder, class_name, image)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(src, dest)