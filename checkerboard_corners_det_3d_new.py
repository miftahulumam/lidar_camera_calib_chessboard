import open3d as o3d
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

from boundary_detection import approximate_plane_edges, ransac_line_3d

# Include outter corners
NUM_ROWS = 6
NUM_COLS = 5
CB_DIM = (NUM_ROWS, NUM_COLS)
CB_SQUARE_SIDE_LENGTH = 0.20

visualize_all = False

# load checkerboard 3D plane equations
data_file = os.path.join(os.getcwd(), "results", "cb_plane_equations.npz")
data = np.load(data_file, allow_pickle=True)    
cb_plane_equations = data['cb_plane_equations']
print(cb_plane_equations.shape)
print(cb_plane_equations[0])

# load checkerboar point clouds
root_dir = os.getcwd()
pcd_dir = os.path.join(root_dir, "CalibData", "checkerboard_pcd")
pcd_list = glob.glob(os.path.join(pcd_dir, "*.pcd"))

print("Number of checkerboard point clouds:", len(pcd_list))
if len(pcd_list) != cb_plane_equations.shape[0]:
    raise ValueError("Number of point clouds and number of plane equations do not match!")  

# Obtain centroid from all checkerboard point clouds using Open3D
centroids = []
for i, pcd_file in enumerate(pcd_list):
    print(f"Processing {i+1}-th file: {pcd_file}")
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    centroids.append(centroid)

# creating five dimensional meshgrid to map checkerboard points in 3D 
# to their index in 2D (indexing: row (left to right), col (bottom to top))
cb_points = np.zeros((NUM_ROWS*NUM_COLS, 5), np.float32)
cb_points[:, :2] = np.mgrid[int(-NUM_ROWS/2) : int(NUM_ROWS/2) + NUM_ROWS%2,
                            int(NUM_COLS/2) - (1-NUM_COLS%2) : int(-NUM_COLS/2) - 1 : -1].T.reshape(-1, 2)

# (indexing: row (left to right), col (top to bottom))
# cb_points[:, :2] = np.mgrid[int(-NUM_ROWS/2):int(NUM_ROWS/2)+NUM_ROWS%2,
#                             int(-NUM_COLS/2):int(NUM_COLS/2)+NUM_COLS%2].T.reshape(-1, 2)

cb_points_ = cb_points * CB_SQUARE_SIDE_LENGTH

# Getting the 3D coordinates of the checkerboard points in each point cloud
all_cb_points_3d = []

for i in tqdm(range(len(pcd_list))):
    plane_model = cb_plane_equations[i][0]
    centroid = centroids[i]
    
    # Plane normal vector
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # Detect points in checkerboard boundaries
    hull_ls, boundary_pcd = approximate_plane_edges(pcd_list[i])

    # apply ransac to find a boundary line equation (line equation to check checkerboard orientation)
    line, inliers = ransac_line_3d(np.asarray(boundary_pcd.points), threshold=0.01, max_iterations=1000)

    line_point = line[0]
    line_direction = line[1]/np.linalg.norm(line[1])

    # Ensure tangent1 is more horizontal, tangent2 is more vertical
    if abs(line_direction[2]) < 0.5:  # more horizontal
        tangent1 = line_direction
        tangent2 = np.cross(normal, tangent1)
        tangent2 /= np.linalg.norm(tangent2)
    else: # vertical
        tangent2 = line_direction
        tangent1 = np.cross(tangent2, normal)
        tangent1 /= np.linalg.norm(tangent1)

    # # Create two orthogonal vectors in the plane
    # if np.allclose(normal, [0, 0, 1]):
    #     tangent1 = np.array([1, 0, 0])
    # else:
    #     tangent1 = np.cross(normal, [0, 0, 1])
    #     tangent1 /= np.linalg.norm(tangent1)
    
    # tangent2 = np.cross(normal, tangent1)
    # Create a grid of points in the plane

    grid_points = []
    for point in cb_points:
        x_offset = point[0] * CB_SQUARE_SIDE_LENGTH + 0.5*CB_SQUARE_SIDE_LENGTH*(1 - NUM_ROWS%2)
        y_offset = point[1] * CB_SQUARE_SIDE_LENGTH + 0.5*CB_SQUARE_SIDE_LENGTH*(1 - NUM_COLS%2)
        grid_point = centroid + x_offset * tangent1 + y_offset * tangent2
        grid_points.append(grid_point)
    
    grid_points = np.array(grid_points)

    all_cb_points_3d.append(grid_points)
    # print(f"3D Checkerboard points for {i+1}-th point cloud:\n", grid_points)

all_cb_points_3d = np.array(all_cb_points_3d)

# visualize all checkerboard points
all_cb_pcd = o3d.geometry.PointCloud()  
for cb_points_3d in all_cb_points_3d:
    cb_pcd = o3d.geometry.PointCloud()
    cb_pcd.points = o3d.utility.Vector3dVector(cb_points_3d)
    all_cb_pcd += cb_pcd
all_cb_pcd.paint_uniform_color([1.0, 0.0, 0.0])
o3d.visualization.draw_geometries([all_cb_pcd], window_name="All Checkerboard Points")  

# visualize individual checkerboard points with corresponding point cloud (one by one)
zoom = 0.5
front = [0.0, 0.0, -1.0]    
lookat = [0.0, 0.0, 0.0]
up = [0.0, -1.0, 0.0]

if visualize_all:
    for i in range(len(pcd_list)):
        print(f"Visualizing {i+1}-th checkerboard points with corresponding point cloud")
        pcd = o3d.io.read_point_cloud(pcd_list[i])

        # visualize approximated corner points
        cb_pcd = o3d.geometry.PointCloud()
        cb_pcd.points = o3d.utility.Vector3dVector(all_cb_points_3d[i])
        cb_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        # visualize centroid
        # add four additional points close to centroid (centroid marker)
        centr_point = np.array([centroids[i], 
                                centroids[i]+[0.003, 0.003, 0.003], 
                                centroids[i]-[0.003, 0.003, 0.003]])

        centroid_pcd = o3d.geometry.PointCloud()
        # centroid_pcd.points = o3d.utility.Vector3dVector(centroids[i].reshape(1, -1))
        centroid_pcd.points = o3d.utility.Vector3dVector(centr_point)
        centroid_pcd.paint_uniform_color([0.0, 0.5, 0.5])
        
        o3d.visualization.draw_geometries([pcd, cb_pcd, centroid_pcd])

else:
    idx = np.random.randint(0, len(pcd_list))
    print(f"Visualizing {idx+1}-th checkerboard points with corresponding point cloud")
    pcd = o3d.io.read_point_cloud(pcd_list[idx])

    # visualize approximated corner points
    cb_pcd = o3d.geometry.PointCloud()
    cb_pcd.points = o3d.utility.Vector3dVector(all_cb_points_3d[idx])
    cb_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    # visualize centroid
    # add four additional points close to centroid (centroid marker)
    centr_point = np.array([centroids[idx], 
                            centroids[idx]+[0.003, 0.003, 0.003], 
                            centroids[idx]-[0.003, 0.003, 0.003]])

    centroid_pcd = o3d.geometry.PointCloud()
    # centroid_pcd.points = o3d.utility.Vector3dVector(centroids[idx].reshape(1, -1))
    centroid_pcd.points = o3d.utility.Vector3dVector(centr_point)
    centroid_pcd.paint_uniform_color([0.0, 0.5, 0.5])
    
    o3d.visualization.draw_geometries([pcd, cb_pcd, centroid_pcd])    

# save 3D checkerboard points to file
if not os.path.exists(os.path.join(root_dir, "results", "3d_corners")):
    os.makedirs(os.path.join(root_dir, "results", "3d_corners"))

np.savez(os.path.join(root_dir, "results", "3d_corners", "all_cb_points_3d.npz"), 
         centroids=centroids,
         all_cb_points_3d=all_cb_points_3d)

# save checkerboard points in each point cloud to separate files
assert len(pcd_list) == all_cb_points_3d.shape[0]
assert len(pcd_list) == len(centroids)

for i in range(len(pcd_list)): 
    # save checkerboard 3D corner points
    cb_points_3d = all_cb_points_3d[i, :]
    cb_pcd = o3d.geometry.PointCloud()
    cb_pcd.points = o3d.utility.Vector3dVector(cb_points_3d)
    cb_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    if not os.path.exists(os.path.join(root_dir, "results", "3d_corners")):
        os.makedirs(os.path.join(root_dir, "results", "3d_corners"))

    cb_fname = os.path.join(root_dir, "results", "3d_corners", os.path.basename(pcd_list[i]))
    o3d.io.write_point_cloud(cb_fname, cb_pcd)

    # save centroid
    # add four additional points close to centroid (centroid marker)
    centr_point = np.array([centroids[i], 
                            centroids[i]+[0.003, 0.003, 0.003], 
                            centroids[i]-[0.003, 0.003, 0.003]])
    centroid_pcd = o3d.geometry.PointCloud()
    centroid_pcd.points = o3d.utility.Vector3dVector(centr_point)
    centroid_pcd.paint_uniform_color([0.0, 0.5, 0.5])

    if not os.path.exists(os.path.join(root_dir, "results", "centroids")):
        os.makedirs(os.path.join(root_dir, "results", "centroids"))

    centr_fname = os.path.join(root_dir, "results", "centroids", os.path.basename(pcd_list[i]))
    o3d.io.write_point_cloud(centr_fname, centroid_pcd)
