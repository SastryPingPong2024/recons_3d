import os
import cv2
import numpy as np
import glob
import shutil
import random

def convert_to_yolo_format(corners, img_width, img_height):
    x_min = min(corners[:,0]) / img_width
    y_min = min(corners[:,1]) / img_height
    x_max = max(corners[:,0]) / img_width
    y_max = max(corners[:,1]) / img_height
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return f"0 {x_center} {y_center} {width} {height}\n"

def process_dataset(image_folder, label_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_folder, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "labels", split), exist_ok=True)
    
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    random.shuffle(image_paths)
    
    split_ratio = [0.8, 0.1, 0.1]
    train_split = int(len(image_paths) * split_ratio[0])
    val_split = train_split + int(len(image_paths) * split_ratio[1])
    
    for idx, image_path in enumerate(image_paths):
        filename = os.path.basename(image_path).split(".")[0]
        label_path = os.path.join(label_folder, f"{filename}.csv")
        
        if not os.path.exists(label_path):
            print(f"Warning: Label not found for {image_path}")
            continue
        
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]
        
        corners = np.loadtxt(label_path)
        
        if len(corners) != 10:
            print(f"Skipping {label_path}: Incorrect label format")
            continue
        
        yolo_label = convert_to_yolo_format(corners, img_width, img_height)
        
        if idx < train_split:
            split = "train"
        elif idx < val_split:
            split = "val"
        else:
            split = "test"
        
        cv2.imwrite(os.path.join(output_folder, "images", split, f"{filename}.jpg"), image)
        with open(os.path.join(output_folder, "labels", split, f"{filename}.txt"), "w") as f:
            f.write(yolo_label)
    
    print("Dataset conversion and split complete!")

# Set paths
image_folder = "train/images_aug/"
label_folder = "train/labels_aug/"
output_folder = "dataset_yolo/"

# Run conversion
process_dataset(image_folder, label_folder, output_folder)
