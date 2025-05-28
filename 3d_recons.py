import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import json
import casadi as ca
import pickle
import random

L = 2.74  # Length
B = 1.525  # Breadth
HT = 0.76  # Height
d1 = 0.1525  # Distance from end to net
h = 0.1525  # Height of net
forgiveness = 4
EPS = 0.01
RIGHT_HAND = 4
LEFT_HAND = 7
SEGMENT_LEN_THRES = 0.2
SIGN_THRES = 0.
DIST_THRES = 0.7
DOUBLES_FACTOR = 1.5
MIN_FRAMES = 3
MAX_FRAMES = 50

def find_closest_projected_3d_point(camera_matrix, rvecs, tvecs, image_point, query_point):
    """
    Finds the 3D point that is closest to the query point while ensuring its projection
    onto the image matches the given 2D image point.

    :param camera_matrix: Intrinsic camera matrix (3x3)
    :param rvecs: Rotation vectors (assumed to be a single vector)
    :param tvecs: Translation vectors (assumed to be a single vector)
    :param image_point: 2D image coordinates (x, y)
    :param query_point: 3D query world coordinates (X, Y, Z)
    :return: 3D world coordinates (X', Y', Z')
    """
    # Convert image point to homogeneous coordinates
    image_point_h = np.array([image_point[0], image_point[1], 1.0], dtype=np.float32)

    # Compute the inverse of the camera intrinsic matrix
    camera_matrix_inv = np.linalg.inv(camera_matrix)

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvecs[0])

    # Compute the camera center in world coordinates
    camera_center = -np.dot(R.T, tvecs[0])

    # Compute the ray direction in world coordinates
    ray_dir = np.dot(R.T, np.dot(camera_matrix_inv, image_point_h))
    ray_dir /= np.linalg.norm(ray_dir)  # Normalize direction

    # Compute closest point on the ray to the query point
    query_vector = query_point - camera_center.flatten()
    t = np.dot(query_vector, ray_dir)
    closest_point = camera_center.flatten() + t * ray_dir

    return closest_point

def project_3d_to_2d_casadi(camera_matrix, rvecs, tvecs, world_points, p=False):
    """
    world_points has cvxpy variables, project those points to 2d image plane
    """
    # world_points = cp.reshape(world_points, (3,1))
    rot_mat = cv2.Rodrigues(rvecs[0])[0]
    rotated = rot_mat @ world_points
    image_points = camera_matrix @ (rotated + tvecs[0])
    image_points = image_points[:2]/image_points[2]
    return image_points

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

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='_0pjrW7Viek', help='Location of target video file')
parser.add_argument('--start_id', default=0, help='Start id for saving')

args = parser.parse_args()
video_path = args.folder

VIDEO_ID = video_path.split('/')[-1]

calib_results = np.load(video_path + '/calib___/' + VIDEO_ID + '_calib.pkl', allow_pickle=True)[0]

# ignore first line in np.loadtxt
ball_track_results = np.loadtxt(video_path + '/ball_tracker_10/' + VIDEO_ID + '_ball.csv', delimiter=',', skiprows=1)
# ball_track_results = np.loadtxt(video_path + '/metadata/second_clip_indices.txt', delimiter=',', skiprows=1)

# Read second_clip_indices.txt for start and end frame of ball_track_results
clip_indices = np.loadtxt(video_path + '/metadata/second_clipped_indices.txt', delimiter=',', max_rows=1)

ball_track_results = ball_track_results[int(clip_indices[0]):int(clip_indices[1])]

BTR_N = ball_track_results.shape[0]-1
while ball_track_results[BTR_N,1] == 0:
    BTR_N -= 1
ball_track_results = ball_track_results[:BTR_N+1]
# convert ball_track_results to integers
# ball_track_results = ball_track_results.astype(int)
camera_matrix = calib_results['cam_mat']
# RESULTS
# exit(0)
rvecs = calib_results['rvecs']
tvecs = calib_results['tvecs']
focal_x = camera_matrix[0,0]
focal_y = camera_matrix[1,1]
focal = (focal_x + focal_y) / 2.


_3d_pts = [[-L/2., 0., 0.], [L/2., 0., 0.], [-L/2., 0., 1.], [L/2., 0., 1.]]
_2d_pts = project_3d_to_2d(camera_matrix, rvecs, tvecs, np.array(_3d_pts))

_2d_pts_homo = [[-L/2.,0.], [L/2.,0.], [-L/2.,1.], [L/2.,1.]]

# Get homography matrix
_2d_pts = np.array(_2d_pts, dtype=np.float32).reshape(-1, 2)
_2d_pts_homo = np.array(_2d_pts_homo, dtype=np.float32).reshape(-1, 2)

# Compute the homography matrix
H, status = cv2.findHomography(_2d_pts_homo, _2d_pts)


_3d_pts = [[-L/2., -B/2., 0.], [-L/2., B/2., 0.], [L/2., -B/2., 0.], [L/2., B/2., 0.]]
_2d_pts = project_3d_to_2d(camera_matrix, rvecs, tvecs, np.array(_3d_pts))

_2d_pts_homo = [[-L/2.,-B/2.], [-L/2.,B/2.], [L/2.,-B/2.], [L/2.,B/2.]]

# Get homography matrix
_2d_pts = np.array(_2d_pts, dtype=np.float32).reshape(-1, 2)
_2d_pts_homo = np.array(_2d_pts_homo, dtype=np.float32).reshape(-1, 2)

# Compute the homography matrix
H_ground, status = cv2.findHomography(_2d_pts_homo, _2d_pts)


# Use the homography matrix to project the point to the plane
plane_points = cv2.perspectiveTransform(ball_track_results[None,:,2:], np.linalg.inv(H))[0]

curr_status = 0
prev_pt = plane_points[0]
start_pt = plane_points[0]
start_i = 0
segments = []
valid_left_segments = []
valid_right_segments = []
curr_forgiveness = forgiveness
for i in range(1,len(plane_points)) :
    pt = plane_points[i]
    if pt[0] > prev_pt[0] + EPS:
        if curr_status == -1 :
            segments.append([start_i, i])
            segment_length_x = start_pt[0] - prev_pt[0]
            if segment_length_x > SEGMENT_LEN_THRES and prev_pt[0]*start_pt[0] < SIGN_THRES and i-start_i > MIN_FRAMES and i-start_i < MAX_FRAMES:
                valid_left_segments.append([start_i, i])
            
            start_pt = pt
            start_i = i
            
        curr_status = 1
        curr_forgiveness = forgiveness
    elif pt[0] < prev_pt[0] - EPS:
        if curr_status == 1 :
            segments.append([start_i, i])
            segment_length_x = prev_pt[0] - start_pt[0]
            if segment_length_x > SEGMENT_LEN_THRES and prev_pt[0]*start_pt[0] < SIGN_THRES and i-start_i > MIN_FRAMES and i-start_i < MAX_FRAMES:
                valid_right_segments.append([start_i, i])
            
            start_pt = pt
            start_i = i
        
        curr_status = -1
        curr_forgiveness = forgiveness
    else :
        curr_forgiveness -= 1
        if curr_forgiveness >= 0 :
            continue
        segments.append([start_i, i])
        if curr_status == -1 :
            segment_length_x = start_pt[0] - prev_pt[0]
            if segment_length_x > SEGMENT_LEN_THRES and prev_pt[0]*start_pt[0] < 0. and i-start_i > MIN_FRAMES and i-start_i < MAX_FRAMES:
                valid_left_segments.append([start_i, i])
        if curr_status == 1 :
            segment_length_x = prev_pt[0] - start_pt[0]
            if segment_length_x > SEGMENT_LEN_THRES and prev_pt[0]*start_pt[0] < 0. and i-start_i > MIN_FRAMES and i-start_i < MAX_FRAMES:
                valid_right_segments.append([start_i, i])
        start_pt = pt
        start_i = i
        curr_status = 0
    
            
    prev_pt = pt

if curr_status == 1 :
    segments.append([start_i, i])
    segment_length_x = pt[0] - start_pt[0]
    if segment_length_x > SEGMENT_LEN_THRES and pt[0]*start_pt[0] < 0. and i-start_i > MIN_FRAMES and i-start_i < MAX_FRAMES:
        valid_right_segments.append([start_i, len(plane_points)])
if curr_status == -1 :
    segments.append([start_i, i])
    segment_length_x = start_pt[0] - pt[0]
    if segment_length_x > SEGMENT_LEN_THRES and pt[0]*start_pt[0] < 0. and i-start_i > MIN_FRAMES and i-start_i < MAX_FRAMES:
        valid_left_segments.append([start_i, len(plane_points)])

RESULTS = {}
RESULTS['cam_mat'] = camera_matrix
RESULTS['rvecs'] = rvecs
RESULTS['tvecs'] = tvecs
RESULTS['valid_left_segments'] = valid_left_segments
RESULTS['valid_right_segments'] = valid_right_segments
RESULTS['segments'] = segments
# exit(0)
keypts_2d =  np.load(video_path + '/human_pose_tracker/4DHumans/' + VIDEO_ID + '/' + VIDEO_ID + '_keypoints_2d.npy')
keypts_3d =  np.load(video_path + '/human_pose_tracker/4DHumans/' + VIDEO_ID + '/' + VIDEO_ID + '_keypoints_3d.npy')

# read json file
with open(video_path + '/human_pose_tracker/4DHumans/' + VIDEO_ID + '/' + VIDEO_ID + '_metadata.json') as f:
    metadata = json.load(f)


def optimize_serve_ball_bounces(ball_poses_2d,start_3d,end_3d,cam_mat,rvecs,tvecs,fps=30,g=9.81,bounce_cost_x=0., bounce_cost_y=0., bounce_cost_z=0.) :
    """
    Optimize the ball bounce by using the 3D ball positions and the 2D ball positions
    
    :param ball_poses_2d: 2D ball positions
    :param start_3d: Start 3D position of the ball
    :param end_3d: End 3D position of the ball
    :param cam_mat: Camera matrix
    :param rvecs: Rotation vectors
    :param tvecs: Translation vectors
    :param fps: Frames per second
    :return: Optimized 3D positions of the ball at each frame
    """
    last_x = ball_poses_2d[-1,2]
    last_y = ball_poses_2d[-1,3]
    K = len(ball_poses_2d) - 2
    while K > 0 and (ball_poses_2d[K,2] == last_x and ball_poses_2d[K,3] == last_y) :
        K -= 1
    # print(ball_poses_2d[:,2:])
    # ball_poses_2d = ball_poses_2d[:K+2]
    # Define optimization problem in cvxpy
    dt = 1.0 / fps
    N = len(ball_poses_2d)
    opti = ca.Opti()
    # Decision variables
    ball_bounce_loc1 = opti.variable(2)
    ball_bounce_loc2 = opti.variable(2)
    ball_bounce_t1 = opti.variable()
    ball_bounce_t2 = opti.variable()
    
    # Set initial values for the variables
    opti.set_initial(ball_bounce_t1, N*dt/3.)
    opti.set_initial(ball_bounce_t2, 2*N*dt/3.)
    if start_3d[0] > end_3d[0] :
        opti.set_initial(ball_bounce_loc1, np.array([-L/4.,0.]))
        opti.set_initial(ball_bounce_loc2, np.array([-L/4.,0.]))
    else :
        opti.set_initial(ball_bounce_loc1, np.array([L/4.,0.]))
        opti.set_initial(ball_bounce_loc2, np.array([L/4.,0.]))
    
    if start_3d[0] > end_3d[0] :
        opti.subject_to(ball_bounce_loc2[0] >= -L/2)
        opti.subject_to(ball_bounce_loc2[0] <= 0.)
        opti.subject_to(ball_bounce_loc1[0] >= 0)
        opti.subject_to(ball_bounce_loc1[0] <= L/2.)
    else :
        opti.subject_to(ball_bounce_loc2[0] >= 0.)
        opti.subject_to(ball_bounce_loc2[0] <= L/2.)
        opti.subject_to(ball_bounce_loc1[0] >= -L/2.)
        opti.subject_to(ball_bounce_loc1[0] <= 0)

    opti.subject_to(ball_bounce_loc1[1] >= -B/2)
    opti.subject_to(ball_bounce_loc1[1] <= B/2)
    
    opti.subject_to(ball_bounce_loc2[1] >= -B/2)
    opti.subject_to(ball_bounce_loc2[1] <= B/2)
    
    opti.subject_to(ball_bounce_t2 >= N * dt/2. + dt)
    opti.subject_to(ball_bounce_t2 <= N * dt - 2*dt)
    opti.subject_to(ball_bounce_t1 >= dt)
    opti.subject_to(ball_bounce_t1 <= N * dt/2.)
    
    # Velocity calculations
    ball_vel_x = (ball_bounce_loc1[0] - start_3d[0]) / ball_bounce_t1
    ball_vel_y = (ball_bounce_loc1[1] - start_3d[1]) / ball_bounce_t1
    ball_vel_z = (0.5 * g * ball_bounce_t1**2 + HT - start_3d[2]) / ball_bounce_t1
    
    ball_bounce_vel_x1 = (ball_bounce_loc2[0] - ball_bounce_loc1[0]) / (ball_bounce_t2 - ball_bounce_t1)
    ball_bounce_vel_y1 = (ball_bounce_loc2[1] - ball_bounce_loc1[1]) / (ball_bounce_t2 - ball_bounce_t1)
    ball_bounce_vel_z1 = (0.5 * g * (ball_bounce_t2 - ball_bounce_t1)**2) / (ball_bounce_t2 - ball_bounce_t1)
    
    ball_bounce_vel_x2 = (end_3d[0] - ball_bounce_loc2[0]) / (N * dt - dt - ball_bounce_t2)
    ball_bounce_vel_y2 = (end_3d[1] - ball_bounce_loc2[1]) / (N * dt - dt - ball_bounce_t2)
    ball_bounce_vel_z2 = (0.5 * g * (N * dt - dt - ball_bounce_t2)**2 + end_3d[2] - HT) / (N * dt - dt - ball_bounce_t2)
    
    # Cost function
    cost = 0
    for i in range(1,N-1):
        ball_pos_2d = ca.MX(ball_poses_2d[i, 2:])
        curr_t = i * dt
        step_val1 = 1.0 / (1.0 + ca.exp(-100.0 * (curr_t - ball_bounce_t1)))
        step_val2 = 1.0 / (1.0 + ca.exp(-100.0 * (curr_t - ball_bounce_t2)))
        
        ball_pos_3d_ = ca.vertcat(
            start_3d[0] + ball_vel_x * curr_t,
            start_3d[1] + ball_vel_y * curr_t,
            start_3d[2] + ball_vel_z * curr_t - 0.5 * g * curr_t**2
        )
        
        ball_pos_2d_ = project_3d_to_2d_casadi(cam_mat, rvecs, tvecs, ball_pos_3d_)
        cost += (1.-step_val1) * ca.norm_2(ball_pos_2d - ball_pos_2d_)
        
        ball_pos_3d_1 = ca.vertcat(
            ball_bounce_loc1[0] + ball_bounce_vel_x1 * (curr_t - ball_bounce_t1),
            ball_bounce_loc1[1] + ball_bounce_vel_y1 * (curr_t - ball_bounce_t1),
            HT + ball_bounce_vel_z1 * (curr_t - ball_bounce_t1) - 0.5 * g * (curr_t - ball_bounce_t1)**2
        )
        ball_pos_2d_1 = project_3d_to_2d_casadi(cam_mat, rvecs, tvecs, ball_pos_3d_1)
        cost += step_val1 * (1-step_val2) * ca.norm_2(ball_pos_2d - ball_pos_2d_1)

        ball_pos_3d_2 = ca.vertcat(
            ball_bounce_loc2[0] + ball_bounce_vel_x2 * (curr_t - ball_bounce_t2),
            ball_bounce_loc2[1] + ball_bounce_vel_y2 * (curr_t - ball_bounce_t2),
            HT + ball_bounce_vel_z2 * (curr_t - ball_bounce_t2) - 0.5 * g * (curr_t - ball_bounce_t2)**2
        )
        ball_pos_2d_2 = project_3d_to_2d_casadi(cam_mat, rvecs, tvecs, ball_pos_3d_2)
        cost += step_val2 * ca.norm_2(ball_pos_2d - ball_pos_2d_2)
    cost += bounce_cost_x * (ball_vel_x-ball_bounce_vel_x1)**2 + bounce_cost_y * (ball_vel_y-ball_bounce_vel_y1)**2 + bounce_cost_z * (ball_vel_z-g*ball_bounce_t1+ball_bounce_vel_z1)**2
    cost += bounce_cost_x * (ball_bounce_vel_x1-ball_bounce_vel_x2)**2 + bounce_cost_y * (ball_bounce_vel_y1-ball_bounce_vel_y2)**2 + bounce_cost_z * (ball_bounce_vel_z1-g*(ball_bounce_t2-ball_bounce_t1)+ball_bounce_vel_z2)**2
    # Solve optimization problem
    # Set objective
    opti.minimize(cost)
    
    # opts = {
    # 'ipopt.print_level': 0,
    # 'ipopt.tol': 1e-6,
    # 'ipopt.max_iter': 1000,
    # 'ipopt.bound_relax_factor': 1e-3,  # Relax the box constraints slightly
    # 'ipopt.check_derivatives_for_naninf': 'yes',  # Check derivatives
    # 'ipopt.nlp_scaling_method': 'none',  # No scaling
    # 'ipopt.sb': 'yes',
    # }
    # Solve optimization problem and do not print anything!
    # opti.solver('ipopt', {"ipopt.accept_every_trial_step": "yes", "ipopt.mu_init": 0.03})

    # solver should not print anything
    opti.solver('ipopt', {"print_time": 0, "ipopt.print_level": 0, "ipopt.sb": "yes"})
    try:
        sol = opti.solve()
        ball_bounce_loc1 = sol.value(ball_bounce_loc1)
        ball_bounce_loc2 = sol.value(ball_bounce_loc2)
        ball_bounce_t1 = sol.value(ball_bounce_t1)
        ball_bounce_t2 = sol.value(ball_bounce_t2)
    
    except :
        ball_bounce_loc1 = opti.debug.value(ball_bounce_loc1)
        ball_bounce_loc2 = opti.debug.value(ball_bounce_loc2)
        ball_bounce_t1 = opti.debug.value(ball_bounce_t1)
        ball_bounce_t2 = opti.debug.value(ball_bounce_t2)
        best_obj_val = opti.debug.value(cost)
    # Extract results
    # Use the optimized ball bounce location and time to get the 3D positions of the ball
    ball_pos_pred_3d = []
    # ball_bounce_loc = ball_bounce_loc.value
    # ball_bounce_t = ball_bounce_t.value
    ball_vel_x = (ball_bounce_loc1[0] - start_3d[0])/ball_bounce_t1
    ball_vel_y = (ball_bounce_loc1[1] - start_3d[1])/ball_bounce_t1
    ball_vel_z = (0.5*g*ball_bounce_t1**2 + HT - start_3d[2])/ball_bounce_t1

    ball_bounce_vel_x1 = (ball_bounce_loc2[0] - ball_bounce_loc1[0]) / (ball_bounce_t2 - ball_bounce_t1)
    ball_bounce_vel_y1 = (ball_bounce_loc2[1] - ball_bounce_loc1[1]) / (ball_bounce_t2 - ball_bounce_t1)
    ball_bounce_vel_z1 = (0.5 * g * (ball_bounce_t2 - ball_bounce_t1)**2) / (ball_bounce_t2 - ball_bounce_t1)
    
    ball_bounce_vel_x2 = (end_3d[0] - ball_bounce_loc2[0]) / (N * dt - dt - ball_bounce_t2)
    ball_bounce_vel_y2 = (end_3d[1] - ball_bounce_loc2[1]) / (N * dt - dt - ball_bounce_t2)
    ball_bounce_vel_z2 = (0.5 * g * (N * dt - dt - ball_bounce_t2)**2 + end_3d[2] - HT) / (N * dt - dt - ball_bounce_t2)
    for i in range(N) :
        curr_t = i*dt
        if curr_t < ball_bounce_t1 :
            ball_pos_pred_3d.append([start_3d[0] + ball_vel_x*curr_t, start_3d[1] + ball_vel_y*curr_t, start_3d[2] + ball_vel_z*curr_t - 0.5*g*curr_t**2])
        elif curr_t < ball_bounce_t2 :
            ball_pos_pred_3d.append([ball_bounce_loc1[0] + ball_bounce_vel_x1*(curr_t-ball_bounce_t1), ball_bounce_loc1[1] + ball_bounce_vel_y1*(curr_t-ball_bounce_t1), HT + ball_bounce_vel_z1*(curr_t-ball_bounce_t1) - 0.5*g*(curr_t-ball_bounce_t1)**2])
        else :
            ball_pos_pred_3d.append([ball_bounce_loc2[0] + ball_bounce_vel_x2*(curr_t-ball_bounce_t2), ball_bounce_loc2[1] + ball_bounce_vel_y2*(curr_t-ball_bounce_t2), HT + ball_bounce_vel_z2*(curr_t-ball_bounce_t2) - 0.5*g*(curr_t-ball_bounce_t2)**2])
    return np.array(ball_pos_pred_3d)


def optimize_ball_bounce(ball_poses_2d,start_3d,end_3d,cam_mat,rvecs,tvecs,fps=30,g=9.81,bounce_cost_x=10., bounce_cost_y=10., bounce_cost_z=10.) :
    """
    Optimize the ball bounce by using the 3D ball positions and the 2D ball positions
    
    :param ball_poses_2d: 2D ball positions
    :param start_3d: Start 3D position of the ball
    :param end_3d: End 3D position of the ball
    :param cam_mat: Camera matrix
    :param rvecs: Rotation vectors
    :param tvecs: Translation vectors
    :param fps: Frames per second
    :return: Optimized 3D positions of the ball at each frame
    """
    last_x = ball_poses_2d[-1,2]
    last_y = ball_poses_2d[-1,3]
    K = len(ball_poses_2d) - 2
    while K > 0 and (ball_poses_2d[K,2] == last_x and ball_poses_2d[K,3] == last_y) :
        K -= 1
    # print(ball_poses_2d[:,2:])
    # ball_poses_2d = ball_poses_2d[:K+2]
    # Define optimization problem in cvxpy
    dt = 1.0 / fps
    N = len(ball_poses_2d)
    opti = ca.Opti()
    # Decision variables
    ball_bounce_loc = opti.variable(2)
    ball_bounce_t = opti.variable()
    
    # Set initial values for the variables
    opti.set_initial(ball_bounce_t, N*dt/2.)
    if start_3d[0] > end_3d[0] :
        opti.set_initial(ball_bounce_loc, np.array([-L/4.,0.]))
    else :
        opti.set_initial(ball_bounce_loc, np.array([L/4.,0.]))
    # opti.set_initial(ball_bounce_loc, np.array([0.,0.]))
    if start_3d[0] > end_3d[0] :
        opti.subject_to(ball_bounce_loc[0] >= -L/2.+L/10.)
        opti.subject_to(ball_bounce_loc[0] <= -L/6.)
    else :
        opti.subject_to(ball_bounce_loc[0] >= L/6.)
        opti.subject_to(ball_bounce_loc[0] <= L/2.-L/10.)

    opti.subject_to(ball_bounce_loc[1] >= -0.45*B)
    opti.subject_to(ball_bounce_loc[1] <= 0.45*B)
    opti.subject_to(ball_bounce_t >= 2 * dt)
    opti.subject_to(ball_bounce_t <= N * dt - 3 * dt)
    
    # Velocity calculations
    ball_vel_x = (ball_bounce_loc[0] - start_3d[0]) / ball_bounce_t
    ball_vel_y = (ball_bounce_loc[1] - start_3d[1]) / ball_bounce_t
    ball_vel_z = (0.5 * g * ball_bounce_t**2 + HT - start_3d[2]) / ball_bounce_t
    
    ball_bounce_vel_x = (end_3d[0] - ball_bounce_loc[0]) / (N * dt - dt - ball_bounce_t)
    ball_bounce_vel_y = (end_3d[1] - ball_bounce_loc[1]) / (N * dt - dt - ball_bounce_t)
    ball_bounce_vel_z = (0.5 * g * (N * dt - dt - ball_bounce_t)**2 + end_3d[2] - HT) / (N * dt - dt - ball_bounce_t)
    
    # Cost function
    cost = 0
    for i in range(1,N-1):
        ball_pos_2d = ca.MX(ball_poses_2d[i, 2:])
        curr_t = i * dt
        step_val = 1.0 / (1.0 + ca.exp(-100.0 * (curr_t - ball_bounce_t)))
        
        ball_pos_3d_ = ca.vertcat(
            start_3d[0] + ball_vel_x * curr_t,
            start_3d[1] + ball_vel_y * curr_t,
            start_3d[2] + ball_vel_z * curr_t - 0.5 * g * curr_t**2
        )
        
        ball_pos_2d_ = project_3d_to_2d_casadi(cam_mat, rvecs, tvecs, ball_pos_3d_)
        cost += (1.-step_val) * ca.norm_2(ball_pos_2d - ball_pos_2d_)
        
        ball_pos_3d__ = ca.vertcat(
            ball_bounce_loc[0] + ball_bounce_vel_x * (curr_t - ball_bounce_t),
            ball_bounce_loc[1] + ball_bounce_vel_y * (curr_t - ball_bounce_t),
            HT + ball_bounce_vel_z * (curr_t - ball_bounce_t) - 0.5 * g * (curr_t - ball_bounce_t)**2
        )
        ball_pos_2d__ = project_3d_to_2d_casadi(cam_mat, rvecs, tvecs, ball_pos_3d__)
        cost += step_val * ca.norm_2(ball_pos_2d - ball_pos_2d__)
    cost += bounce_cost_x * (ball_vel_x-ball_bounce_vel_x)**2 + bounce_cost_y * (ball_vel_y-ball_bounce_vel_y)**2 + bounce_cost_z * (ball_vel_z-g*ball_bounce_t+ball_bounce_vel_z)**2
    
    # Solve optimization problem
    # Set objective
    opti.minimize(cost)
    
    # opts = {
    # 'ipopt.print_level': 0,
    # 'ipopt.tol': 1e-6,
    # 'ipopt.max_iter': 1000,
    # 'ipopt.bound_relax_factor': 1e-3,  # Relax the box constraints slightly
    # 'ipopt.check_derivatives_for_naninf': 'yes',  # Check derivatives
    # 'ipopt.nlp_scaling_method': 'none',  # No scaling
    # 'ipopt.sb': 'yes',
    # }
    # Solve optimization problem and do not print anything!
    # opti.solver('ipopt', {"ipopt.accept_every_trial_step": "yes", "ipopt.mu_init": 0.03})

    # solver should not print anything
    opti.solver('ipopt', {"print_time": 0, "ipopt.print_level": 0, "ipopt.sb": "yes"})
    try:
        sol = opti.solve()
        ball_bounce_loc = sol.value(ball_bounce_loc)
        ball_bounce_t = sol.value(ball_bounce_t)
    
    except :
        ball_bounce_loc = opti.debug.value(ball_bounce_loc)
        ball_bounce_t = opti.debug.value(ball_bounce_t)
        best_obj_val = opti.debug.value(cost)
    # Extract results
    # Use the optimized ball bounce location and time to get the 3D positions of the ball
    ball_pos_pred_3d = []
    # ball_bounce_loc = ball_bounce_loc.value
    # ball_bounce_t = ball_bounce_t.value
    ball_vel_x = (ball_bounce_loc[0] - start_3d[0])/ball_bounce_t
    ball_vel_y = (ball_bounce_loc[1] - start_3d[1])/ball_bounce_t
    ball_vel_z = (0.5*g*ball_bounce_t**2 + HT - start_3d[2])/ball_bounce_t
    ball_bounce_vel_x = (end_3d[0] - ball_bounce_loc[0])/(N*dt - dt - ball_bounce_t)
    ball_bounce_vel_y = (end_3d[1] - ball_bounce_loc[1])/(N*dt - dt - ball_bounce_t)
    ball_bounce_vel_z = (0.5*g*(N*dt - dt - ball_bounce_t)**2 + end_3d[2]-HT)/(N*dt - dt - ball_bounce_t)
    for i in range(N) :
        curr_t = i*dt
        if curr_t < ball_bounce_t :
            ball_pos_pred_3d.append([start_3d[0] + ball_vel_x*curr_t, start_3d[1] + ball_vel_y*curr_t, start_3d[2] + ball_vel_z*curr_t - 0.5*g*curr_t**2])
        else :
            ball_pos_pred_3d.append([ball_bounce_loc[0] + ball_bounce_vel_x*(curr_t-ball_bounce_t), ball_bounce_loc[1] + ball_bounce_vel_y*(curr_t-ball_bounce_t), HT + ball_bounce_vel_z*(curr_t-ball_bounce_t) - 0.5*g*(curr_t-ball_bounce_t)**2])
    return np.array(ball_pos_pred_3d)


curr_id = 0
initial_frame_no = 0
cap = cv2.VideoCapture(video_path+'/video_second_clipped/'+VIDEO_ID+'.mp4')
cap.set(1,10)
_, image = cap.read()
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = BTR_N
scale_factor = 800/image.shape[0]
keypts_3d_rot = np.zeros_like(keypts_3d)
for frame_no in tqdm(range(n_frames)):
    while curr_id<len(metadata) and int(metadata[curr_id]['frame']) < frame_no-initial_frame_no:
        curr_id += 1 
    while curr_id < len(metadata) and int(metadata[curr_id]['frame']) == frame_no-initial_frame_no:
        keypts = keypts_3d[curr_id]
        rot_mat = cv2.Rodrigues(rvecs[0])[0]
        keypts_rot = keypts @ rot_mat
        min_i = np.argmin(keypts_rot[:,2])
        _2d_pt = keypts_2d[curr_id][min_i]/scale_factor
        ground_pt = cv2.perspectiveTransform(_2d_pt[None,None,:], np.linalg.inv(H_ground))[0]
        ground_pt_3d = np.array([ground_pt[0,0], ground_pt[0,1], 0.])
        keypts_rot = keypts_rot + ground_pt_3d[None,:] - keypts_rot[min_i:min_i+1]
        keypts_3d_rot[curr_id,:,:] = keypts_rot
        curr_id += 1


def get_left_right_players(keypts_3d,slope=0.5):
    left_costs = []
    right_costs = []
    n_players_left = 1
    n_players_right = 1
    for keypt in keypts_3d:
        x = keypt[0,0]
        y = keypt[0,1]
        z = keypt[0,2]
        left_cost = 0
        right_cost = 0
        if x > 0 :
            left_cost = 1e6
        if x < 0 :
            right_cost = 1e6
        lu_dist = slope*x + y - B/2.
        if lu_dist > 0 :
            lu_dist*=100.
        ld_dist = slope*x - y - B/2.
        if ld_dist > 0 :
            ld_dist*=100.
        left_cost += abs(lu_dist) + abs(ld_dist)
        ru_dist = -slope*x + y - B/2.
        if ru_dist > 0 :
            ru_dist*=100.
        rd_dist = -slope*x - y - B/2.
        if rd_dist > 0 :
            rd_dist*=100.
        right_cost += abs(ru_dist) + abs(rd_dist)
        left_costs.append(left_cost)
        right_costs.append(right_cost)
    left_player = np.argmin(left_costs)
    right_player = np.argmin(right_costs)
    left_costs = np.array(left_costs)
    left_costs.sort()
    right_costs = np.array(right_costs)
    right_costs.sort()
    if left_costs[1] < left_costs[0]*DOUBLES_FACTOR :
        n_players_left = 2
    if right_costs[1] < right_costs[0]*DOUBLES_FACTOR :
        n_players_right = 2
    # print(right_costs)
    return left_player, right_player, n_players_left, n_players_right

curr_id = 0
left_seg_opt_ball_poses = []
n_left_players = []
n_right_players = []
first = True
for seg in valid_left_segments:
    fi = seg[0]
    fl = seg[1]
    while curr_id<len(metadata) and int(metadata[curr_id]['frame']) < fi:
        curr_id += 1 
    n_players = 0
    while curr_id+n_players<len(metadata) and int(metadata[curr_id+n_players]['frame']) == fi:
        n_players += 1
    left_player,right_player, n_left_player, n_right_player = get_left_right_players(keypts_3d_rot[curr_id:curr_id+n_players])
    ball_pos_start_2d = keypts_2d[curr_id+right_player,RIGHT_HAND]/scale_factor
    ball_pos_start_3d = keypts_3d_rot[curr_id+right_player,RIGHT_HAND]
    ball_pos_start_3d_ = find_closest_projected_3d_point(camera_matrix, rvecs, tvecs, ball_track_results[fi,2:], ball_pos_start_3d)
    dist1 = np.linalg.norm(ball_pos_start_3d_ - ball_pos_start_3d)
    
    ball_pos_start_3d = keypts_3d_rot[curr_id+right_player,LEFT_HAND]
    ball_pos_start_3d_left = find_closest_projected_3d_point(camera_matrix, rvecs, tvecs, ball_track_results[fi,2:], ball_pos_start_3d)
    dist1_left = np.linalg.norm(ball_pos_start_3d_left - ball_pos_start_3d)
    
    if dist1_left < dist1 :
        ball_pos_start_3d_ = ball_pos_start_3d_left
        dist1 = dist1_left
    
    while curr_id<len(metadata) and int(metadata[curr_id]['frame']) < fl-1:
        curr_id += 1 
    
    n_players = 0
    while curr_id+n_players<len(metadata) and int(metadata[curr_id+n_players]['frame']) == fl-1:
        n_players += 1
    left_player,right_player,n_left_player,n_right_player = get_left_right_players(keypts_3d_rot[curr_id:curr_id+n_players])
    
    n_left_players.append(n_left_player)
    n_right_players.append(n_right_player)
    ball_pos_end_2d = keypts_2d[curr_id+left_player,RIGHT_HAND]/scale_factor
    ball_pos_end_3d = keypts_3d_rot[curr_id+left_player,RIGHT_HAND]
    ball_pos_end_3d_ = find_closest_projected_3d_point(camera_matrix, rvecs, tvecs, ball_track_results[fl-1,2:], ball_pos_end_3d)
    dist2 = np.linalg.norm(ball_pos_end_3d_ - ball_pos_end_3d)
    
    ball_pos_end_3d = keypts_3d_rot[curr_id+left_player,LEFT_HAND]
    ball_pos_end_3d_left = find_closest_projected_3d_point(camera_matrix, rvecs, tvecs, ball_track_results[fl-1,2:], ball_pos_end_3d)
    dist2_left = np.linalg.norm(ball_pos_end_3d_left - ball_pos_end_3d)
    
    if dist2_left < dist2 :
        ball_pos_end_3d_ = ball_pos_end_3d_left
        dist2 = dist2_left
    
    opt_ball_poses_3d = optimize_ball_bounce(ball_track_results[fi:fl],ball_pos_start_3d_,ball_pos_end_3d_,camera_matrix,rvecs,tvecs,fps)
    print(dist1,dist2)
    if dist1 < DIST_THRES and dist2 < DIST_THRES :
        left_seg_opt_ball_poses.append(opt_ball_poses_3d)
        if first :
            opt_ball_poses_3d_left = optimize_serve_ball_bounces(ball_track_results[fi:fl],ball_pos_start_3d_,ball_pos_end_3d_,camera_matrix,rvecs,tvecs,fps)
            first_left = fi
            first_left_i = len(left_seg_opt_ball_poses)-1
        first = False
    else :
        left_seg_opt_ball_poses.append(None)


curr_id = 0
right_seg_opt_ball_poses = []
first = True
for seg in valid_right_segments:
    fi = seg[0]
    fl = seg[1]
    while curr_id<len(metadata) and int(metadata[curr_id]['frame']) < fi:
        curr_id += 1 
    n_players = 0
    while curr_id+n_players<len(metadata) and int(metadata[curr_id+n_players]['frame']) == fi:
        n_players += 1
    left_player,right_player,n_left_player,n_right_player = get_left_right_players(keypts_3d_rot[curr_id:curr_id+n_players])
    n_left_players.append(n_left_player)
    n_right_players.append(n_right_player)
    ball_pos_start_2d = keypts_2d[curr_id+left_player,RIGHT_HAND]/scale_factor
    ball_pos_start_3d = keypts_3d_rot[curr_id+left_player,RIGHT_HAND]
    ball_pos_start_3d_ = find_closest_projected_3d_point(camera_matrix, rvecs, tvecs, ball_track_results[fi,2:], ball_pos_start_3d)
    dist1 = np.linalg.norm(ball_pos_start_3d_ - ball_pos_start_3d)
    
    ball_pos_start_3d = keypts_3d_rot[curr_id+left_player,LEFT_HAND]
    ball_pos_start_3d_left = find_closest_projected_3d_point(camera_matrix, rvecs, tvecs, ball_track_results[fi,2:], ball_pos_start_3d)
    dist1_left = np.linalg.norm(ball_pos_start_3d_left - ball_pos_start_3d)

    if dist1_left < dist1 :
        ball_pos_start_3d_ = ball_pos_start_3d_left
        dist1 = dist1_left
    
    while curr_id<len(metadata) and int(metadata[curr_id]['frame']) < fl-1:
        curr_id += 1 
    
    n_players = 0
    while curr_id+n_players<len(metadata) and int(metadata[curr_id+n_players]['frame']) == fl-1:
        n_players += 1
    left_player,right_player,n_left_player,n_right_player = get_left_right_players(keypts_3d_rot[curr_id:curr_id+n_players])
    n_left_players.append(n_left_player)
    n_right_players.append(n_right_player)
    ball_pos_end_2d = keypts_2d[curr_id+right_player,RIGHT_HAND]/scale_factor
    ball_pos_end_3d = keypts_3d_rot[curr_id+right_player,RIGHT_HAND]
    ball_pos_end_3d_ = find_closest_projected_3d_point(camera_matrix, rvecs, tvecs, ball_track_results[fl-1,2:], ball_pos_end_3d)
    dist2 = np.linalg.norm(ball_pos_end_3d_ - ball_pos_end_3d)
    
    ball_pos_end_3d = keypts_3d_rot[curr_id+right_player,LEFT_HAND]
    ball_pos_end_3d_left = find_closest_projected_3d_point(camera_matrix, rvecs, tvecs, ball_track_results[fl-1,2:], ball_pos_end_3d)
    dist2_left = np.linalg.norm(ball_pos_end_3d_left - ball_pos_end_3d)

    if dist2_left < dist2 :
        ball_pos_end_3d_ = ball_pos_end_3d_left
        dist2 = dist2_left
    # ball_pos_end_3d[2] /= 1.2
    opt_ball_poses_3d = optimize_ball_bounce(ball_track_results[fi:fl],ball_pos_start_3d_,ball_pos_end_3d_,camera_matrix,rvecs,tvecs,fps)
    print(dist1,dist2)

    if dist1 < DIST_THRES and dist2 < DIST_THRES :
        right_seg_opt_ball_poses.append(opt_ball_poses_3d)
        if first :
            opt_ball_poses_3d_right = optimize_serve_ball_bounces(ball_track_results[fi:fl],ball_pos_start_3d_,ball_pos_end_3d_,camera_matrix,rvecs,tvecs,fps)
            first_right = fi
            first_right_i = len(right_seg_opt_ball_poses)-1
        first = False
    else :
        right_seg_opt_ball_poses.append(None)

if first_left < first_right :
    left_seg_opt_ball_poses[first_left_i] = opt_ball_poses_3d_left
else :
    right_seg_opt_ball_poses[first_right_i] = opt_ball_poses_3d_right
RESULTS['left_seg_opt_ball_poses'] = left_seg_opt_ball_poses
RESULTS['right_seg_opt_ball_poses'] = right_seg_opt_ball_poses
RESULTS['human_poses_3d'] = keypts_3d_rot

new_frames = []
left_player_poses = []
right_player_poses = []
ball_poses_3d = []
curr_id = 0
initial_frame_no = 0
cap.set(1, 0)
for frame_no in range(n_frames):
    ball_poses_3d.append([0.,-1.,-1.,-1.])
    ret, image = cap.read()
    while curr_id<len(metadata) and int(metadata[curr_id]['frame']) < frame_no-initial_frame_no:
        curr_id += 1 
    n_players = 0
    while curr_id+n_players<len(metadata) and int(metadata[curr_id+n_players]['frame']) == frame_no-initial_frame_no:
        n_players += 1
    left_player,right_player,n_left_player,n_right_player = get_left_right_players(keypts_3d_rot[curr_id:curr_id+n_players])
    n_left_players.append(n_left_player)
    n_right_players.append(n_right_player)
    player_no = 0
    while curr_id < len(metadata) and int(metadata[curr_id]['frame']) == frame_no-initial_frame_no:
        keypts = keypts_3d[curr_id]
        rot_mat = cv2.Rodrigues(rvecs[0])[0]
        keypts_rot = keypts @ rot_mat
        min_i = np.argmin(keypts_rot[:,2])
        _2d_pt = keypts_2d[curr_id][min_i]/scale_factor
        cv2.circle(image, (int(_2d_pt[0]), int(_2d_pt[1])), 2, (0,255,0), 2)
        ground_pt = cv2.perspectiveTransform(_2d_pt[None,None,:], np.linalg.inv(H_ground))[0]
        ground_pt_3d = np.array([ground_pt[0,0], ground_pt[0,1], 0.])
        keypts_rot = keypts_rot + ground_pt_3d[None,:] - keypts_rot[min_i:min_i+1]
        # keypts_rot[:,2] /= 1.2
        # Use rvecs, tvecs to transform keypts in camera frame to world frame (NOTE: rvecs, tvecs are for transforming world frame to camera frame)
        # keypts = np.dot(keypts, rot_mat.T)
        # keypts += tvecs[0].T
        # Project 3D points to 2D
        keypts = project_3d_to_2d(camera_matrix, rvecs, tvecs, keypts_rot)
        color = (0, 0, 0)
        if player_no == left_player :
            left_player_poses.append(keypts_rot)
            color = (255,0,0)
        if player_no == right_player :
            right_player_poses.append(keypts_rot)
            color = (0,255,0)
        for keypt in keypts:
            cv2.circle(image, (int(keypt[0,0]), int(keypt[0,1])), 2, color, 2)
        player_no += 1
        curr_id += 1
    left = False
    right = False
    v = 0
    for seg in valid_left_segments:
        if seg[0] <= frame_no < seg[1]:
            left = True
            if left_seg_opt_ball_poses[v] is not None:
                x_curr = left_seg_opt_ball_poses[v][frame_no-seg[0],0]
                y_curr = left_seg_opt_ball_poses[v][frame_no-seg[0],1]  
                z_curr = left_seg_opt_ball_poses[v][frame_no-seg[0],2]
                ball_poses_3d[-1] = [1.,x_curr,y_curr,z_curr]
                for opt_ball_pose in left_seg_opt_ball_poses[v]:
                    x = opt_ball_pose[0]
                    y = opt_ball_pose[1]
                    z = opt_ball_pose[2]
                    _2d = project_3d_to_2d(camera_matrix, rvecs, tvecs, np.array([[x,y,z]]))[0]
                    cv2.circle(image, (int(_2d[0,0]), int(_2d[0,1])), 3, (255,255,0), -1)
                    # cv2.circle(image, (x, y), 3, (255,0,0), -1)
            break
        v += 1
    v = 0


    for seg in valid_right_segments:
        if seg[0] <= frame_no < seg[1]:
            right = True
            if right_seg_opt_ball_poses[v] is not None:
                x_curr = right_seg_opt_ball_poses[v][frame_no-seg[0],0]
                y_curr = right_seg_opt_ball_poses[v][frame_no-seg[0],1]
                z_curr = right_seg_opt_ball_poses[v][frame_no-seg[0],2]
                ball_poses_3d[-1] = [2.,x_curr,y_curr,z_curr]
                for opt_ball_pose in right_seg_opt_ball_poses[v]:
                    x = opt_ball_pose[0]
                    y = opt_ball_pose[1]
                    z = opt_ball_pose[2]
                    _2d = project_3d_to_2d(camera_matrix, rvecs, tvecs, np.array([[x,y,z]]))[0]
                    cv2.circle(image, (int(_2d[0,0]), int(_2d[0,1])), 3, (255,255,0), -1)
            break
        v += 1
    if not left and not right :
        color = (0,0,255)
    elif left and not right :
        color = (255,0,0)
    elif not left and right :
        color = (0,255,0)
    else :
        color = (0,0,255)
    
    # Draw circle at location of ball which is ball_track_results[frame_no,3], ball_track_results[frame_no,4]
    x = int(ball_track_results[frame_no,2])
    y = int(ball_track_results[frame_no,3])
    cv2.circle(image, (x, y), 5, color, -1)
    # Convert _2d_pts to int
    _2d_pts = _2d_pts.astype(int)
    # cv2.circle(image, (_2d_pts[0,0], _2d_pts[0,1]), 10, color, 2)
    # cv2.circle(image, (_2d_pts[1,0], _2d_pts[1,1]), 10, color, 2)
    # cv2.circle(image, (_2d_pts[2,0], _2d_pts[2,1]), 10, color, 2)
    # cv2.circle(image, (_2d_pts[3,0], _2d_pts[3,1]), 10, color, 2)
    # cv2.imshow('imgLabel', image)
    # key = cv2.waitKey(0) & 0xFF
    
    # if key == ord('q'):
    #     break
    new_frames.append(image)

RESULTS['ball_poses_3d'] = ball_poses_3d
RESULTS['left_player_poses'] = left_player_poses
RESULTS['right_player_poses'] = right_player_poses
avg_n_left_players = np.mean(n_left_players)
avg_n_right_players = np.mean(n_right_players)

total_segments = 0
for seg in left_seg_opt_ball_poses:
    if seg is not None:
        total_segments += 1
for seg in right_seg_opt_ball_poses:
    if seg is not None:
        total_segments += 1

if avg_n_left_players > 1.3 and avg_n_right_players > 1.3 :
    print("It's a doubles match")
    exit(0)

if total_segments < 2 :
    print("Not enough segments")
    exit(0)
# create a folder to save the results
if not os.path.exists(video_path + '/3d_recons__'):
    os.makedirs(video_path + '/3d_recons__')

# Save RESULTS as pkl file in video_path + '/3d_recons/' + VIDEO_ID + '_3d_recons.pkl'
with open(video_path + '/3d_recons__/' + VIDEO_ID + '_3d_recons.pkl', 'wb') as f:
    pickle.dump(RESULTS, f)

height, width, layers = new_frames[0].shape
size = (width, height)
out = cv2.VideoWriter(video_path+'/3d_recons__/'+VIDEO_ID+'_3d_recons.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
for i in range(len(new_frames)):
    out.write(new_frames[i])
out.release()
