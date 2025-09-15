import open3d as o3d
import os
import glob
import random
import numpy as np

# point cloud view configuration
front = [ 0.99981446187670631, 0.018359800296648025, 0.0058274827549867209 ]
lookat = [ -0.43004493388802895, 0.15504261360253963, 3.2573070358044451 ]
up = [ -0.0046039441797200466, -0.065993489415235682, 0.99780943223282426 ]
zoom = 0.02

root_dir = os.getcwd()
pcd_dir = os.path.join(root_dir, "CalibData", "pointclouds")
pcd_list = glob.glob(os.path.join(pcd_dir, "*.pcd"))

# result = o3d.io.read_point_cloud(random.choice(pcd_list))

# o3d.visualization.draw_geometries([result],
#                                   front=front,
#                                   lookat=lookat,
#                                   up=up,
#                                   zoom=zoom)

# visualize results
# pcd_vis_fname = os.path.join(root_dir, "CalibData", "checkerboard_pcd", "points_246293975320.pcd")
# print(f"Visualizing {pcd_vis_fname}")
# cropped_pcd_fname = os.path.join(root_dir, "CalibData", "cropped_pcd", os.path.basename(pcd_vis_fname))
# checkerboard_pcd_fname = os.path.join(root_dir, "CalibData", "checkerboard_pcd", os.path.basename(pcd_vis_fname))

# pcd_vis = o3d.io.read_point_cloud(pcd_vis_fname)
# cropped_pcd_vis = o3d.io.read_point_cloud(cropped_pcd_fname)
# checkerboard_pcd_vis = o3d.io.read_point_cloud(checkerboard_pcd_fname)

# o3d.visualization.draw_geometries([pcd_vis],
#                                   zoom=zoom, 
#                                   front=front,
#                                   lookat=lookat,
#                                   up=up)      

# o3d.visualization.draw_geometries([cropped_pcd_vis],
#                                   zoom=zoom, 
#                                   front=front,
#                                   lookat=lookat,
#                                   up=up)   

# o3d.visualization.draw_geometries([checkerboard_pcd_vis],
#                                   zoom=zoom, 
#                                   front=front,
#                                   lookat=lookat,
#                                   up=up) 

root_dir = os.getcwd()
data_file = os.path.join(root_dir, "results", "intrinsics_and_cb_points", "intrinsics_and_cb_points.npz")

data = np.load(data_file, allow_pickle=True)
found_images = data['found_images']
obj_points_list = data['obj_points_cam_list']
img_points_list = data['img_points_list']
K = data['K']
distortion = data['distortion']

print("Number of image with corners found:",len(found_images))

# Load corresponding point cloud
pcd_file_list = []
lidar_points_list = []

for img_file in found_images:
    print(f"Processing image file: {img_file}")
    # Extract number from base filename (format: image_####.png)
    base_filename = os.path.basename(img_file)
    number_str = base_filename.split('_')[1].split('.')[0]
    print(f"Extracted number string: {number_str}")

    pcd_file_list.append(os.path.join(root_dir, "CalibData", "checkerboard_pcd", f"points_{number_str}.pcd"))
    print(f"Corresponding point cloud file: {pcd_file_list[-1]}")

    pcd = o3d.io.read_point_cloud(pcd_file_list[-1]) 
    lidar_points_list.append(pcd)
    print(f"Loaded point cloud with {len(pcd.points)} points")

# for i in range(len(lidar_points_list)):
#     print(f"Visualizing point cloud from file: {pcd_file_list[i]}")
#     o3d.visualization.draw_geometries([lidar_points_list[i]]) #,
#                                     #   zoom=zoom, 
#                                     #   front=front,
#                                     #   lookat=lookat,
#                                    #   up=up)

# color lidar points uniformly
for pcd in lidar_points_list:
    pcd.paint_uniform_color([0.8, 0.8, 0.8])

# load 3D checkerboard detected points
if not os.path.exists(os.path.join(root_dir, "results", "3d_corners", "all_cb_points_3d.npz")):
    raise FileNotFoundError("3D checkerboard points not found. Please run checkerboard_3d_marker_detection.py first.")

cb_points_data = np.load(os.path.join(root_dir, "results", "3d_corners", "all_cb_points_3d.npz"), allow_pickle=True)
all_cb_points_3d = cb_points_data['all_cb_points_3d']
centroids = cb_points_data['centroids']

line_equations_3d = np.load(os.path.join(root_dir, "results", "3d_corners", "line_equations_3d.npz"), allow_pickle=True)
line_points = line_equations_3d["line_point_list"]
line_directions = line_equations_3d["line_direction_list"]

# load 3D checkerboard corner points point cloud
for i in range(len(pcd_file_list)):
    print(f"Loading 3D checkerboard points from file: {pcd_file_list[i]}")
    cb_pcd_fname = os.path.join(root_dir, "results", "3d_corners", os.path.basename(pcd_file_list[i]))
    cb_pcd = o3d.io.read_point_cloud(cb_pcd_fname)
    print(f"Loaded checkerboard point cloud with {len(cb_pcd.points)} points")

    # load centroids point cloud
    centroid_fname = os.path.join(root_dir, "results", "centroids", os.path.basename(pcd_file_list[i]))
    centr_pcd = o3d.io.read_point_cloud(centroid_fname)

    # load 3D line equations
    line_fname = os.path.join(root_dir, "results", "boundary_line_vectors", os.path.basename(pcd_file_list[i]))
    line_pcd = o3d.io.read_point_cloud(line_fname)

    print(f"Visualizing file: {centroid_fname}")

    # visualize checkerboard points with corresponding point cloud
    o3d.visualization.draw_geometries([lidar_points_list[i], cb_pcd, centr_pcd, line_pcd]) #,
    