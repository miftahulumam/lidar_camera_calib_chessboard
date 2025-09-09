import open3d as o3d
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import random


# import npz file containing chessboard points and camera parameters
root_dir = os.getcwd()
data_file = os.path.join(root_dir, "results", "intrinsics_and_cb_points", "intrinsics_and_cb_points.npz")

data = np.load(data_file, allow_pickle=True)
found_images = data['found_images']
obj_points_list = data['obj_points_cam_list']
img_points_list = data['img_points_list']
K = data['K']
distortion = data['distortion']

print(len(img_points_list))

# Load corresponding point cloud
pcd_file_list = []
lidar_points_list = []

for img_file in found_images:
    print(f"Processing image file: {img_file}")
    # Extract number from base filename (format: image_####.png)
    base_filename = os.path.basename(img_file)
    number_str = base_filename.split('_')[1].split('.')[0]
    print(f"Extracted number string: {number_str}")

    pcd_file_list.append(os.path.join(root_dir, "results", "3d_corners", f"points_{number_str}.pcd"))
    print(f"Corresponding point cloud file: {pcd_file_list[-1]}")

    pcd = o3d.io.read_point_cloud(pcd_file_list[-1]) 
    lidar_points_list.append(pcd.points)
    print(f"Loaded point cloud with {len(pcd.points)} points")

lidar_points_list = np.array(lidar_points_list)

print(img_points_list.shape)
print(obj_points_list.shape)
print(lidar_points_list.shape)

# Perform calibration to find extrinsic parameters
# Using OpenCV's solvePnPRansac
rvecs = []
tvecs = []

img_points_list = img_points_list.reshape(-1, 2)
obj_points_list = obj_points_list.reshape(-1, 3)
lidar_points_list = lidar_points_list.reshape(-1, 3)

print(lidar_points_list.shape)
print(obj_points_list.shape)
print(img_points_list.shape)

success, rvecs, tvecs, inliers = cv2.solvePnPRansac(lidar_points_list,
                                                    img_points_list,
                                                    K,
                                                    distortion,
                                                    flags=cv2.SOLVEPNP_EPNP,
                                                    reprojectionError=3.0,
                                                    iterationsCount=10000,
                                                    confidence=0.99)

# print("Rotation Vectors:\n", rvecs)
# print("Translation Vectors:\n", tvecs)

# Convert rotation vector to rotation matrix
R, _ = cv2.Rodrigues(rvecs)

# Form the extrinsic matrix
extrinsic_matrix = np.eye(4)
extrinsic_matrix[:3, :3] = R
extrinsic_matrix[:3, 3] = tvecs.flatten()

print("Extrinsic Matrix:\n", extrinsic_matrix)

print('K:\n', K)

# calculate reprojection error
proj_points, _ = cv2.projectPoints(lidar_points_list[inliers[:,0]], rvecs, tvecs, K, distortion)
proj_points = proj_points.reshape(-1, 2).astype(np.float32)  
error = cv2.norm(img_points_list[inliers[:,0]], proj_points, cv2.NORM_L2) / len(proj_points)
print("Reprojection Error: ", error)

# Number of inliers
print("Number of inliers:", len(inliers))

# Refine using all inliers
success, rvecs, tvecs = cv2.solvePnP(lidar_points_list[inliers[:,0]], 
                                       img_points_list[inliers[:,0]], 
                                       K, distortion, 
                                       rvecs, tvecs, 
                                       useExtrinsicGuess=True, 
                                       flags=cv2.SOLVEPNP_ITERATIVE)   

# print("Refined Rotation Vectors:\n", rvecs)
# print("Refined Translation Vectors:\n", tvecs)

# Convert rotation vector to rotation matrix
R, _ = cv2.Rodrigues(rvecs)
# Form the extrinsic matrix
extrinsic_matrix = np.eye(4)
extrinsic_matrix[:3, :3] = R
extrinsic_matrix[:3, 3] = tvecs.flatten()

print("Refined Extrinsic Matrix:\n", extrinsic_matrix)
# calculate reprojection error
proj_points, _ = cv2.projectPoints(lidar_points_list[inliers[:,0]], rvecs, tvecs, K, distortion)
proj_points = proj_points.reshape(-1, 2).astype(np.float32)  
error = cv2.norm(img_points_list[inliers[:,0]], proj_points, cv2.NORM_L2) / len(proj_points)
print("Refined Reprojection Error: ", error)    



