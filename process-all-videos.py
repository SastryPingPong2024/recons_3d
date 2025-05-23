import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import tqdm
import random

FOLDER = '/clifford-data/home/pingpong-daniel/robot_table_tennis/pipeline_outputs/shared/dataset_sample'
SAMPLE_RATIO = 1.

def list_directories_by_modification_time(directory):
    # Get a list of directories
    dir_list = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    # Sort directories by modification time
    sorted_dirs = sorted(dir_list, key=os.path.getmtime)
    
    return sorted_dirs


# for folder in os.listdir(FOLDER):
#     if os.path.isdir(os.path.join(FOLDER, folder)):
#         print('Processing folder:', folder)
#         for folder_ in os.listdir(os.path.join(FOLDER, folder)):
#             if os.path.exists(os.path.join(FOLDER, folder, folder_, 'video')):
#                 print("Valid folder")
#                 # Run command python3 process-video.py --folder os.path.join(FOLDER, folder)
#                 os.system('python3 classify_camera_pos.py --folder ' + os.path.join(FOLDER, folder, folder_))
#                 # exit(0)
#             else:
#                 print("Invalid folder")



# Go through each folder in the directory
n = 0
# Sort by last modified
print("Before")
folder_list = os.listdir(FOLDER)
print("After")
folder_list.sort()
print("Sorted")
random.shuffle(folder_list)
print("Shuffled")
# LIST = ['87TgS5kIP-M_116205_116820_0_6_1_8',
# '6b8DKaBOGdM_166996_167551_2_0_0_2',
# '6b8DKaBOGdM_158241_158826_1_4_0_1',
# '5TAlvsdEiYQ_43220_43865_0_4_0_5',
# '5TAlvsdEiYQ_38025_38525_0_1_0_1',
# '5CpoadRv32Q_339426_339936_0_6_0_4',
# '5CEapFU9lXI_476924_477729_0_4_2_2',
# '5CEapFU9lXI_470954_471609_0_6_1_10']
LIST = ['-DNXFkdAMcM_211490_212125_1_8_0_5', '-7_7O3dusC8_994038_994603_0_2_0_2']
os.system('mkdir -p ' + 'sampled_videos4')
for folder in tqdm.tqdm(folder_list):
    n += 1
    if os.path.isdir(os.path.join(FOLDER, folder)):
        print('Processing folder:', folder)
        source_path = os.path.join(FOLDER, folder) +'/3d_recons__/'+folder+'_3d_recons.mp4'
        target_path = 'sampled_videos4' +'/'+folder+'_3d_recons.mp4'
        if os.path.exists(os.path.join(FOLDER, folder, 'human_pose_tracker')):
            print("Valid folder")
            # Run command python3 process-video.py --folder os.path.join(FOLDER, folder)
            os.system('python3 process-video.py --folder ' + os.path.join(FOLDER, folder))
            print("Camera calibrated")
            os.system('python3 3d_recons.py --folder ' + os.path.join(FOLDER, folder))
            print("3d reconstructed")
            if random.random() < SAMPLE_RATIO :
                print("Saving video")
                os.system('cp -r ' + source_path + ' ' + target_path)
        else:
            print("Invalid folder")
    # if n >= 100 :
    #     n = 0
    #     break


# print("Reconstructing 3d in all videos now")
# # Go through each folder in the directory
# for folder in os.listdir(FOLDER):
#     if os.path.isdir(os.path.join(FOLDER, folder)):
#         print('Processing folder:', folder)
#         if os.path.exists(os.path.join(FOLDER, folder, folder)):
#             print("Valid folder")
#             # Run command python3 process-video.py --folder os.path.join(FOLDER, folder)
#             os.system('python3 3d_recons.py --folder ' + os.path.join(FOLDER, folder))
#         else:
#             print("Invalid folder")