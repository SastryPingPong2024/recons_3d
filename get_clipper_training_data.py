import numpy as np
import os
import cv2
import argparse
import json

THRES = 0
MAX_NO = 8
parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='_0pjrW7Viek', help='Location of target video file')

args = parser.parse_args()
FOLDER = args.folder

for _folder in tqdm.tqdm(os.listdir(FOLDER)):
    n += 1
    folder = os.path.join(FOLDER, _folder)
    try :
        VIDEO_ID = video_path.split('/')[-1]
        json_file = os.path.join(folder, '2d_pose_rtmpose_gpu', '2d_pose.json')

        # Read json file and save into a dictionary
        with open(json_file, 'r') as f:
            data = json.load(f)

        keypts = np.zeros((len(data), MAX_NO, 17, 2))
        for frame in data:
            i = 0
            for target in frame['targets']:
                if i < MAX_NO:
                    keypts[frame['frame'],i] = np.array(target['keypoints'])
                    i += 1

        # Save the keypoints into a numpy file
        np.save(os.path.join(folder, 'keypoints.npy'), keypts)
    except Exception as e:
        print("Error in processing folder", folder, ":", e)
        continue
