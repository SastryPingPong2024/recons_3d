import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys

FOLDER = 'test'

# Go through each folder in the directory
for folder in os.listdir(FOLDER):
    if os.path.isdir(os.path.join(FOLDER, folder)):
        print('Processing folder:', folder)
        if os.path.exists(os.path.join(FOLDER, folder, folder)):
            print("Valid folder")
            # Run command python3 process-video.py --folder os.path.join(FOLDER, folder)
            os.system('python3 process-video.py --folder ' + os.path.join(FOLDER, folder))
        else:
            print("Invalid folder")

print("Reconstructing 3d in all videos now")
# Go through each folder in the directory
for folder in os.listdir(FOLDER):
    if os.path.isdir(os.path.join(FOLDER, folder)):
        print('Processing folder:', folder)
        if os.path.exists(os.path.join(FOLDER, folder, folder)):
            print("Valid folder")
            # Run command python3 process-video.py --folder os.path.join(FOLDER, folder)
            os.system('python3 3d_recons.py --folder ' + os.path.join(FOLDER, folder))
        else:
            print("Invalid folder")