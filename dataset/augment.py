import os
import cv2
import numpy as np
import pandas as pd
import random
import math

# Input and output directories
image_dir = "train/images/"
label_dir = "train/labels/"
aug_image_dir = "train/images_aug/"
aug_label_dir = "train/labels_aug/"

# Ensure output directories exist
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

# Rotation center point (px, py) - Set as per requirement
n_per_image = 10  # Number of augmentations per image
angle_range = 30

# Function to rotate a point around (px, py)
def rotate_point(x, y, px, py, angle_rad):
    x_shifted, y_shifted = x - px, y - py
    x_rot = x_shifted * math.cos(angle_rad) - y_shifted * math.sin(angle_rad) + px
    y_rot = x_shifted * math.sin(angle_rad) + y_shifted * math.cos(angle_rad) + py
    return x_rot, y_rot

# Function to horizontally flip an image and label points
def flip_image_and_labels(image, points):
    flipped_image = cv2.flip(image, 1)
    flipped_points = [(image.shape[1] - x, y) for x, y in points]
    flipped_points = [flipped_points[i] for i in [2, 1, 0, 5, 4, 3, 9, 8, 7, 6]]  # Reorder points
    return flipped_image, flipped_points

# Function to rotate an image and label points
def rotate_image_and_labels(image, points, angle, px, py):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((px, py), angle, 1.0)
    print(px,py,M)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # Rotate each (x, y) point
    angle_rad = math.radians(angle)
    rotated_points = [rotate_point(x, y, px, py, -angle_rad) for x, y in points]
    
    return rotated_image, rotated_points

# Process each image
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".csv").replace(".png", ".csv"))
        
        if not os.path.exists(label_path):
            print(f"Label file missing for {filename}, skipping...")
            continue
        
        # Load image and label points
        image = cv2.imread(img_path)
        label_points = np.loadtxt(label_path)  # Shape (10, 2)
        px = np.mean(label_points[:, 0])
        py = np.mean(label_points[:, 1])
        for i in range(n_per_image):
            angle = random.uniform(-angle_range, angle_range)
            aug_image, aug_points = rotate_image_and_labels(image, label_points, angle, px, py)

            k = random.randint(0, 1)
            if k == 1:
                aug_image, aug_points = flip_image_and_labels(aug_image, aug_points)
            
            # Save augmented image
            aug_img_name = f"{os.path.splitext(filename)[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(aug_image_dir, aug_img_name), aug_image)

            # Save augmented label points
            aug_label_name = aug_img_name.replace(".jpg", ".csv")
            np.savetxt(os.path.join(aug_label_dir, aug_label_name), np.array(aug_points))

print("Data augmentation completed!")
