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

# Load selected point cloud
pcd_file = os.path.join(root_dir, "results", "3d_corners", "points_275491298040.pcd")
print(f"Processing point cloud file: {pcd_file}")
pcd = o3d.io.read_point_cloud(pcd_file)
lidar_points = np.array(pcd.points)
print(f"Loaded point cloud with {len(pcd.points)} points")

# visualize point cloud and chessboard points
centroid_fname = os.path.join(root_dir, "results", "centroids", os.path.basename(pcd_file))
centr_pcd = o3d.io.read_point_cloud(centroid_fname)
segmented_cb_fname = os.path.join(root_dir, "CalibData", "checkerboard_pcd", os.path.basename(pcd_file))
checkerboard_pcd = o3d.io.read_point_cloud(segmented_cb_fname)
o3d.visualization.draw_geometries([pcd, checkerboard_pcd, centr_pcd]) 

# get corresponding image points
base_filename = os.path.basename(pcd_file)
number_str = base_filename.split('_')[1].split('.')[0]
print(f"Extracted number string: {number_str}")
img_file = os.path.join(root_dir, "CalibData", "images", f"image_{number_str}.png")
print(f"Using image file: {img_file}")
img_points = img_points_list[found_images.tolist().index(img_file)]
obj_points = obj_points_list[found_images.tolist().index(img_file)]

# Perform calibration to find extrinsic parameters
# Using OpenCV's solvePnP
success, rvecs, tvecs = cv2.solvePnP(lidar_points, img_points, K, distortion)
# print("Rotation Vector:\n", rvecs)
# print("Translation Vector:\n", tvecs)

R, _ = cv2.Rodrigues(rvecs)
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = tvecs.flatten()
print("Transformation Matrix:\n", T)
