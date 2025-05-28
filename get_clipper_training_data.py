import numpy as np
import os
import cv2
import argparse
import json
import tqdm
from utils import save_info, load_info, go2frame, show_image, nextframe
from ultralytics import YOLO
import torch

THRES = 0
MAX_NO = 8
parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='/clifford-data/home/pingpong-daniel/robot_table_tennis/pipeline_outputs/shared/dataset_sample_em3', help='Location of target video file')

args = parser.parse_args()
FOLDER = args.folder
# Example usage
model_path = "dataset/runs/detect/train/weights/best.pt"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained YOLO model
model_bb = YOLO(model_path).to(DEVICE)

def run_inference(model, image):
    
    # image = torch.tensor(image).to(DEVICE)
    # Run inference without printing anything
    results = model(image, verbose=False)
    
    
    # Extract bounding boxes
    bboxes = []
    for result in results:
        for box in result.boxes.xyxy:
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            bboxes.append((x_min, y_min, x_max, y_max))
    
    return bboxes



n=0
margin = 15
for _folder in tqdm.tqdm(os.listdir(FOLDER)):
    n += 1
    folder = os.path.join(FOLDER, _folder)
    print("Processing folder", folder)

    try :
        VIDEO_ID = folder.split('/')[-1]
        cap = cv2.VideoCapture(os.path.join(folder, 'video', VIDEO_ID + '.mp4'))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        json_file = os.path.join(folder, '2d_pose_rtmpose_gpu', '2d_pose.json')

        # Read json file and save into a dictionary
        with open(json_file, 'r') as f:
            data = json.load(f)

        keypts = np.zeros((len(data), MAX_NO, 17, 2))
        image = nextframe(cap)
        bounding_boxes = run_inference(model_bb, image)
        l = 0
        r = image.shape[1]
        t = 0
        d = image.shape[0]
        if len(bounding_boxes) > 0:
            l = bounding_boxes[0][0] - margin
            r = bounding_boxes[0][2] + margin
            t = bounding_boxes[0][1] - margin
            d = bounding_boxes[0][3] + margin
            
        for frame in data:
            i = 0
            for target in frame['targets']:
                n_intersections = 0
                for pt in target['keypoints']:
                    x = pt[0]
                    y = pt[1]
                    if x > l and x < r and y > t and y < d:
                        n_intersections += 1
                    
                if i < MAX_NO and n_intersections >= THRES:
                    keypts[int(frame['frame_index']),i] = np.array(target['keypoints'])
                    i += 1
            # print(i)

        # Save the keypoints into a numpy file
        np.save(os.path.join(folder, 'keypoints.npy'), keypts)
    except Exception as e:
        print("Error in processing folder", folder, ":", e)
        continue
