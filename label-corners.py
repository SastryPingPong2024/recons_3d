import os
import cv2
import sys, getopt
import parser 
from utils import save_info, load_info, go2frame, show_image
import numpy as np
import argparse

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

# acquire video info
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

info = []

def corner_label(event, x, y, flags, param):
    global info, image
    if event == cv2.EVENT_LBUTTONDOWN:
        info.append([x, y])
        x_pos = int(x)
        y_pos = int(y)
        cv2.circle(image, (x_pos, y_pos), 5, (0, 0, 255), -1)
        cv2.imshow('imgLabel', image)

saved_success = False
frame_no = 0
curr_id = start_id
_, image = cap.read()

# show_image(image, 0, info[0]['x'], info[0]['y'])
while True:
    leave = 'y'
    print(image.shape)
    cv2.imshow('imgLabel', image)
    cv2.setMouseCallback('imgLabel', corner_label)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        if not saved_success:
            print("You forget to save file!")
            while True:
                leave = str(input("Really want to leave without saving? [Y/N]"))
                leave = leave.lower()
                if leave != 'y' and leave != 'n':
                    print("Please type 'y/Y' or 'n/N'")
                    continue
                elif leave == 'y':
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Exit label program")
                    sys.exit(1)
                elif leave == 'n':
                    break
       
        if leave == 'y':
            cap.release()
            cv2.destroyAllWindows()
            print("Exit label program")
            sys.exit(1)

    elif key == ord('s'):
        
        if len(info) != 10:
            print("Please label all corners! current: {}".format(len(info)))
            continue
        
        # save current image into dataset/train/images folder
        image_orig = go2frame(cap, frame_no, [], show_text=False)
        cv2.imwrite('dataset/train/images/{}.jpg'.format(curr_id), image_orig)
        
        # save current info into dataset/train/labels folder
        np.savetxt('dataset/train/labels/{}.csv'.format(curr_id), np.array(info))
        saved_success = True
        print("Saved ", curr_id)
        curr_id += 1

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
    else:
        image = go2frame(cap, frame_no, info)