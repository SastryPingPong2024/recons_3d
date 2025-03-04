import os
import cv2
import sys, getopt
import parser 
from utils import save_info, load_info, go2frame, show_image
import numpy as np
import argparse
import torch
from models import modify_resnet
from ultralytics import YOLO
import cv2
# read a json file
import json

margin = 15
L = 2.74  # Length
B = 1.525  # Breadth
H = 0.76  # Height
d1 = 0.1525  # Distance from end to net
h = 0.1525  # Height of net
VIDEO_ID = '_0pjrW7Viek'


def calibrate_camera(image_points, world_points, image_size):
    """
    Calibrates a camera given 2D-3D point correspondences, assuming no distortion.
    
    :param image_points: List of 2D image points [(x, y), ...]
    :param world_points: List of corresponding 3D world points [(X, Y, Z), ...]
    :param image_size: Tuple (width, height) of the image
    :return: Camera matrix, rotation vectors, and translation vectors
    """
    # Convert lists to numpy arrays
    object_points = np.array(world_points, dtype=np.float32).reshape(-1, 1, 3)
    img_points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    
    # Prepare input data format for OpenCV
    object_points_list = [object_points]  # List of object points
    img_points_list = [img_points]  # List of image points
    
    # Provide an initial guess for the camera matrix
    focal_length = max(image_size)  # Approximate focal length
    camera_matrix = np.array([[focal_length, 0, image_size[0] / 2],
                               [0, focal_length, image_size[1] / 2],
                               [0, 0, 1]], dtype=np.float32)
    
    # Perform camera calibration assuming no distortion
    ret, camera_matrix, _, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list, img_points_list, image_size, camera_matrix, None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |
              cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    
    if not ret:
        raise RuntimeError("Camera calibration failed.")
    
    return camera_matrix, rvecs, tvecs

def project_3d_to_2d(camera_matrix, rvecs, tvecs, world_points):
    """
    Projects a 3D world point onto the image plane.
    
    :param camera_matrix: Intrinsic camera matrix
    :param rvecs: Rotation vectors
    :param tvecs: Translation vectors
    :param world_point: A single 3D world point (X, Y, Z)
    :return: 2D image coordinates (x, y)
    """
    world_points = np.array(world_points, dtype=np.float32).reshape(-1, 1, 3)
    image_points, _ = cv2.projectPoints(world_points, rvecs[0], tvecs[0], camera_matrix, None)
    return image_points

def world_to_camera_coordinates(rvecs, tvecs, world_point):
    """
    Transforms a 3D point from world coordinates to camera coordinates.
    
    :param rvecs: Rotation vectors
    :param tvecs: Translation vectors
    :param world_point: A single 3D world point (X, Y, Z)
    :return: 3D point in camera coordinates (X', Y', Z')
    """
    R, _ = cv2.Rodrigues(rvecs[0])
    world_point = np.array(world_point, dtype=np.float32).reshape(3, 1)
    camera_point = R @ world_point + tvecs[0]
    camera_position = - R.T @ tvecs[0]
    print(camera_position)
    return tuple(camera_point.ravel())

def draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, point1, point2,color=(0, 0, 255)):
    """
    Draws a projected 3D line segment onto an image.
    
    :param image: cv2 read image
    :param camera_matrix: Intrinsic camera matrix
    :param rvecs: Rotation vectors
    :param tvecs: Translation vectors
    :param point1: First 3D world point (X, Y, Z)
    :param point2: Second 3D world point (X, Y, Z)
    """
    
    # Project the 3D points to 2D
    img_point1 = project_3d_to_2d(camera_matrix, rvecs, tvecs, point1)[0,0]
    img_point2 = project_3d_to_2d(camera_matrix, rvecs, tvecs, point2)[0,0]
    
    print (img_point1)
    # Draw the line on the image
    cv2.line(image, tuple(map(int, img_point1)), tuple(map(int, img_point2)), color, 2)
    
    return image

def get_calib_mat(image_points, image_size) :
    print(image_points)
    world_points = [(-L/2., B/2., H), (-L/2., 0, H), (-L/2., -B/2., H), (L/2., -B/2., H), (L/2., 0, H), (L/2., B/2., H),
                 (0, B/2.+d1, H), (0, B/2.+d1, H+h), (0, -B/2.-d1, H+h), (0, -B/2.-d1, H),]
                # (10, 10, 0), (11, 11, 0)]

    print(world_points)
    
    camera_matrix, rvecs, tvecs = calibrate_camera(image_points[:-1], world_points[:-1], image_size)
    print(rvecs, tvecs)
    return camera_matrix, rvecs, tvecs

def draw_calib_lines(image, camera_matrix, rvecs, tvecs):
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., B/2., H), (L/2., B/2., H))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., -B/2., H), (L/2., -B/2., H))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., B/2., H), (-L/2., -B/2., H))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (L/2., B/2., H), (L/2., -B/2., H))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (L/2., 0., H), (-L/2., 0, H))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (0, B/2.+d1, H), (0, -B/2.-d1, H),color=(255,0,0))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (0, B/2.+d1, H), (0, B/2.+d1, H+h),color=(255,0,0))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (0, -B/2.-d1, H), (0, -B/2.-d1, H+h),color=(255,0,0))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (0, B/2.+d1, H+h), (0, -B/2.-d1, H+h),color=(255,0,0))

    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., -B/2, H), (-L/2., -B/2., 0),color=(0,255,0))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., B/2, H), (-L/2., B/2, 0),color=(0,255,0))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (L/2., -B/2, H), (L/2., -B/2, 0),color=(0,255,0))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (L/2., B/2, H), (L/2., B/2, 0),color=(0,255,0))

    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., B/2, 0), (L/2., B/2, 0),color=(0,255,0))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., -B/2, 0), (L/2., -B/2, 0),color=(0,255,0))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., B/2, 0), (-L/2., -B/2, 0),color=(0,255,0))
    image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (L/2., B/2, 0), (L/2., -B/2, 0),color=(0,255,0))
    return image

def run_inference(model, image):
        
    # Run inference
    results = model(image)
    
    # Extract bounding boxes
    bboxes = []
    for result in results:
        for box in result.boxes.xyxy:
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            bboxes.append((x_min, y_min, x_max, y_max))
    
    return bboxes

# Example usage
model_path = "dataset/runs/detect/train/weights/best.pt"

# Load the trained YOLO model
model_bb = YOLO(model_path)



parser = argparse.ArgumentParser()
parser.add_argument('--file', default='videos/vid1.mp4', help='Location of target video file')
parser.add_argument('--start_id', default=0, help='Start id for saving')

args = parser.parse_args()
video_path = args.file

if not os.path.isfile(video_path) or not video_path.endswith('.mp4'):
    print("Not a valid video path! Please modify path in parser.py --label_video_path")
    sys.exit(1)

# create labels in dataset/train/labels folder and save images in dataset/train/images folder
# Format: bl, bc, br, tr, tc, tl, clb, clt, crt, crb
# Where b: bottom, t: top, l: left, r: right, c: center

start_id = int(args.start_id)

if VIDEO_ID is not None :
    video_path = VIDEO_ID + '/' + VIDEO_ID + '.mp4'
# acquire video info
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(n_frames)
# # # # # # # # # # # # # # # #
# e: exit program             #
# s: save info                #
# n: next frame               #
# p: previous frame           #
# f: to first frame           #
# l: to last frame            #
# d: display labels           #
# r: reset labels             #
# >: fast forward 36 frames   #
# <: fast backward 36 frames  #
# # # # # # # # # # # # # # # #

model = modify_resnet(output_size=20)
model.load_state_dict(torch.load('models/best_model.pth'))
# model.load_state_dict(torch.load('models/model_epoch_49.pth'))
model.eval()
    
info = []

def get_corners(model, image, l, r, t, d):
    h = image.shape[0]
    w = image.shape[1]
    
    l_orig = l
    r_orig = r
    t_orig = t
    d_orig = d
    if (d-t) > (r-l):
        r += int(((d-t) - (r-l))/2)
        l = r - (d-t)
        if l < 0:
            l = 0
            r = d-t
        if r > w:
            r = w
            l = w - (d-t)
    else:
        d += int(((r-l) - (d-t))/2)
        t = d - (r-l)
        if t < 0:
            t = 0
            d = r-l
        if d > h:
            d = h
            t = h - (r-l)
    

    image_ = image.copy()[t:d, l:r,:]

    image_[:int(t_orig-t),:,:] = 0
    image_[-int(d-d_orig+1):,:,:] = 0
    image_[:,:int(l_orig-l),:] = 0
    image_[:,-int(r-r_orig+1):,:] = 0
    
    # reshape image to (224, 224)
    image_ = cv2.resize(image_, (224, 224), interpolation=cv2.INTER_AREA)

    # transform image to (3,224,224)

    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    # normalize image and convert to float
    image_ = image_.transpose((2, 0, 1))
    image_ = image_ / 255.0
    image_ = image_.astype(np.float32)
    
    pred_label = model(torch.tensor(image_).unsqueeze(0)).detach().cpu().numpy()
    pred_label = pred_label.reshape(-1, 2)
    pred_label[:,0] = pred_label[:,0] * (r-l) + l
    pred_label[:,1] = pred_label[:,1] * (d-t) + t

    for k in pred_label:
        x = int(k[0])
        y = int(k[1])
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
    return image, l, r, t, d, pred_label
    
def corner_label(event, x, y, flags, param):
    global info, image
    if event == cv2.EVENT_LBUTTONDOWN:
        info.append([x, y])
        x_pos = int(x)
        y_pos = int(y)
        cv2.circle(image, (x_pos, y_pos), 5, (0, 0, 255), -1)
        print(len(info))
        if len(info) == 2 :
            image, l, r, t, d, _ = get_corners(model, image, info[0][0], info[1][0], info[0][1], info[1][1])
            # Draw rectangle from info[0] to info[1] as diagonals
            cv2.rectangle(image, (l,t), (r,d), (0, 255, 0), 2)
    
        cv2.imshow('imgLabel', image)

saved_success = False
frame_no = 0
curr_id = start_id
_, image = cap.read()
keypts_2d =  np.load(VIDEO_ID + '/' + VIDEO_ID + '/' + VIDEO_ID + '_keypoints_2d.npy')
print(keypts_2d.shape)
# read json file
with open(VIDEO_ID + '/' + VIDEO_ID + '/' + VIDEO_ID + '_metadata.json') as f:
    metadata = json.load(f)
print(len(metadata))
exit(0)
curr_id = 0
# show_image(image, 0, info[0]['x'], info[0]['y'])
while True:
    leave = 'y'
    while int(metadata[curr_id]['frame']) == frame_no:
        curr_id += 1
    # print(image.shape)
    cv2.imshow('imgLabel', image)
    cv2.setMouseCallback('imgLabel', corner_label)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        bounding_boxes = run_inference(model_bb, image)
        print("Detected bounding boxes:", bounding_boxes)
        
        if len(bounding_boxes) >= 1:
            l = bounding_boxes[0][0] - margin
            r = bounding_boxes[0][2] + margin
            t = bounding_boxes[0][1] - margin
            d = bounding_boxes[0][3] + margin
            image, l, r, t, d, corners = get_corners(model, image, l, r, t, d)

            # Convert corners to int array
            corners = np.array(corners, dtype=np.int32)
            cv2.rectangle(image, (l, t), (r, d), (0, 255, 0), 2)
            print(image.shape)
            camera_matrix, rvecs, tvecs = get_calib_mat(corners, (image.shape[1], image.shape[0]))
            image = draw_calib_lines(image, camera_matrix, rvecs, tvecs)
        

    elif key == ord('e'):
        
        if leave == 'y':
            cap.release()
            cv2.destroyAllWindows()
            print("Exit label program")
            sys.exit(1)

    elif key == ord('n'):
        if frame_no >= n_frames-1:
            print("This is the last frame")
            continue
        frame_no += 1
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('p'):
        if frame_no == 0:
            print("This is the first frame")
            continue
        frame_no -= 1
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('f'):
        if frame_no == 0:
            print("This is the first frame")
            continue
        frame_no = 0
        image = go2frame(cap, frame_no, info) 
        print("Frame No.{}".format(frame_no))

    elif key == ord('l'):
        if frame_no == n_frames-1:
            print("This is the last frame")
            continue
        frame_no = n_frames-1
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('>'):
        if frame_no + 36 >= n_frames-1:
            print("Reach last frame")
            frame_no = n_frames-1
        else:
            frame_no += 36
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('<'):
        if frame_no - 36 <= 0:
            print("Reach first frame")
            frame_no = 0
        else:
            frame_no -= 36
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))
    
    elif key == ord('r'):
        info = []
        image = go2frame(cap, frame_no, info)
        print("Reset labels")
    # else:
    #     image = go2frame(cap, frame_no, info)