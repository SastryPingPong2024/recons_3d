import numpy as np
import cv2
import matplotlib.pyplot as plt


# Table tennis table dimensions (in m)
L = 2.74  # Length
B = 1.525  # Breadth
H = 0.76  # Height
d = 0.1525  # Distance from end to net
h = 0.1525  # Height of net
image_path = "images/img1.png"

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

# Example usage
image_points = [(1097, 983), (1577, 1054), (2145, 903), (1698, 854), (1335, 1017),
                (1902, 876), (1383, 909), (1383, 851), (1940, 983), (1941, 920),]
                # (1200, 700), (1300, 750)]

world_points = [(-L/2., B/2., H), (-L/2., -B/2., H), (L/2., -B/2., H), (L/2., B/2., H), (-L/2., 0, H),
                (L/2., 0, H), (0, B/2.+d, H), (0, B/2.+d, H+h), (0, -B/2.-d, H), (0, -B/2.-d, H+h),]
                # (10, 10, 0), (11, 11, 0)]

image_size = (2879, 1539)  # Example image size

camera_matrix, rvecs, tvecs = calibrate_camera(image_points, world_points, image_size)
print(project_3d_to_2d(camera_matrix, rvecs, tvecs, [(-5,0,H),(-4,0,H),(-3,0,H),(-1,0,H),(0,0,H)]))

world_to_camera_coordinates(rvecs, tvecs, (0, 0, 0))
print("Camera Matrix:\n", camera_matrix)
print("Rotation Vectors:\n", rvecs)
print("Translation Vectors:\n", tvecs)

# Read the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Image file not found.")

# Draw a line on the image
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., B/2., H), (L/2., B/2., H))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., -B/2., H), (L/2., -B/2., H))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., B/2., H), (-L/2., -B/2., H))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (L/2., B/2., H), (L/2., -B/2., H))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (L/2., 0., H), (-L/2., 0, H))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (0, B/2.+d, H), (0, -B/2.-d, H),color=(255,0,0))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (0, B/2.+d, H), (0, B/2.+d, H+h),color=(255,0,0))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (0, -B/2.-d, H), (0, -B/2.-d, H+h),color=(255,0,0))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (0, B/2.+d, H+h), (0, -B/2.-d, H+h),color=(255,0,0))

image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., -B/2, H), (-L/2., -B/2., 0),color=(0,255,0))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., B/2, H), (-L/2., B/2, 0),color=(0,255,0))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (L/2., -B/2, H), (L/2., -B/2, 0),color=(0,255,0))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (L/2., B/2, H), (L/2., B/2, 0),color=(0,255,0))

image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., B/2, 0), (L/2., B/2, 0),color=(0,255,0))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., -B/2, 0), (L/2., -B/2, 0),color=(0,255,0))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (-L/2., B/2, 0), (-L/2., -B/2, 0),color=(0,255,0))
image = draw_3d_line_on_image(image, camera_matrix, rvecs, tvecs, (L/2., B/2, 0), (L/2., -B/2, 0),color=(0,255,0))
# Save the image to output.png
cv2.imwrite("output.png", image)
