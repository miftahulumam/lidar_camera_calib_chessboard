import glob
import os

import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import copy

import random

n_planes = 2

root_dir = os.getcwd()
pcd_dir = os.path.join(root_dir, "CalibData", "pointclouds")
pcd_list = glob.glob(os.path.join(pcd_dir, "*.pcd"))

# point cloud view configuration
front = [ 0.99981446187670631, 0.018359800296648025, 0.0058274827549867209 ]
lookat = [ -0.43004493388802895, 0.15504261360253963, 3.2573070358044451 ]
up = [ -0.0046039441797200466, -0.065993489415235682, 0.99780943223282426 ]
zoom = 0.02

# checkerboard_planes = []
# cropped_pcd = []
# checkerboard_pcd = []

cb_plane_equations = []

# iterate over all point clouds in the directory
for i, pcd_file in enumerate(pcd_list):
    print(f"Processing {i+1}-th file: {pcd_file}")
    # Initialize reconstructed point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Definining region of interest using bounding box
    bb_3d = o3d.geometry.AxisAlignedBoundingBox(min_bound=[-20, -6.4, -2],
                                                max_bound=[0, 2.9, 2])
    pcd = pcd.crop(bb_3d)

    # Write cropped point cloud to file
    if not os.path.exists(os.path.join(root_dir, "CalibData", "cropped_pcd")):
        os.makedirs(os.path.join(root_dir, "CalibData", "cropped_pcd"))

    cropped_fname = os.path.join(root_dir, "CalibData", "cropped_pcd", os.path.basename(pcd_file))
    o3d.io.write_point_cloud(cropped_fname, pcd)

    #### Plane Fitting using RANSAC to detect checkerboard plane
    print("\n========== RANSAC Segmentation ==========\n")
    point_to_plane_dist = 0.03

    # colors = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]]
    colors = np.random.rand(n_planes, 3)
    plane_equations = []
    segmented_planes = []

    pcd_temp = copy.deepcopy(pcd)
    pcd_temp.paint_uniform_color([0.6, 0.6, 0.6])

    for i in range(n_planes):
        est_plane, inliers = pcd_temp.segment_plane(distance_threshold=point_to_plane_dist,
                                                    ransac_n=8, num_iterations=3000)
        
        [a, b, c, d] = est_plane
        print(f"estimated plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        
        plane_equations.append([est_plane])

        inlier_pcd = pcd_temp.select_by_index(inliers)
        outlier_pcd = pcd_temp.select_by_index(inliers, invert=True)

        inlier_pcd.paint_uniform_color(colors[i])

        segmented_planes.append(inlier_pcd)

        pcd_temp = outlier_pcd

    segmented_planes.append(pcd_temp)

    board_points = segmented_planes[1]
    plane_model = plane_equations[1]

    board_points = board_points.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]

    if not os.path.exists(os.path.join(root_dir, "CalibData", "checkerboard_pcd")):
        os.makedirs(os.path.join(root_dir, "CalibData", "checkerboard_pcd"))

    checkerboard_fname = os.path.join(root_dir, "CalibData", "checkerboard_pcd", os.path.basename(pcd_file))
    o3d.io.write_point_cloud(checkerboard_fname, board_points)

    cb_plane_equations.append(plane_model)

# save plane equations list to file
cb_plane_equations = np.array(cb_plane_equations)
print(cb_plane_equations.shape)
np.savez(os.path.join(root_dir, "results", "cb_plane_equations.npz"), cb_plane_equations=cb_plane_equations)

# visualize results
pcd_vis_fname = random.choice(pcd_list)
print(f"Visualizing {pcd_vis_fname}")
cropped_pcd_fname = os.path.join(root_dir, "CalibData", "cropped_pcd", os.path.basename(pcd_vis_fname))
checkerboard_pcd_fname = os.path.join(root_dir, "CalibData", "checkerboard_pcd", os.path.basename(pcd_vis_fname))

pcd_vis = o3d.io.read_point_cloud(pcd_vis_fname)
cropped_pcd_vis = o3d.io.read_point_cloud(cropped_pcd_fname)
checkerboard_pcd_vis = o3d.io.read_point_cloud(checkerboard_pcd_fname)

o3d.visualization.draw_geometries([pcd_vis],
                                  zoom=zoom, 
                                  front=front,
                                  lookat=lookat,
                                  up=up)      

o3d.visualization.draw_geometries([cropped_pcd_vis],
                                  zoom=zoom, 
                                  front=front,
                                  lookat=lookat,
                                  up=up)   

o3d.visualization.draw_geometries([checkerboard_pcd_vis])  

